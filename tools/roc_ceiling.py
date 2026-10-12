# -*- coding: utf-8 -*-
"""算天花板：给定现有模型的 ROC，macro_f1 最高能到多少？0.95 需要什么条件？

## 原理

macro_f1 与 (TP, FP, FN) 的关系（逐类等权下严格成立）：

    F1 = 2·TP / (TP + FP + 1000)      （因为 FN = 1000 − TP）

而 (TP, FP) 由 ROC 上的操作点决定：
    FP = 15540 × (1 − UR),  TP = 1000 × known_recall

所以只要能测出 ROC，就能扫出该模型能达到的最高 macro_f1 —— **不用真的提交**。

## 关键：ROC 要分级测

在「训练见过的来源类」和「没见过的类」上分别测，得到上界与下界；
测试集的真实水平落在两者之间（按来源覆盖率加权）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from cardex import paths  # noqa: E402

DS = ROOT / "数据集"
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
N_KNOWN, N_UNKNOWN = 1000.0, 15540.0
CACHE = ROOT / "outputs" / "reports" / "roc_ceiling"


def load_image(p: Path):
    from PIL import Image
    with Image.open(p) as im:
        return im.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=None,
                    help="覆盖模型目录（用于对比新旧模型）；默认读冠军配置")
    ap.add_argument("--ckpt", default=None,
                    help="单个 checkpoint 路径（评测微调后的单模型，与集成对比时注意口径）")
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--image-size", type=int, default=288)
    ap.add_argument("--tag", default="champion")
    ap.add_argument("--tta", default="hflip")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    npz = CACHE / f"{args.tag}_{args.tta}.npz"

    if npz.exists() and not args.force:
        z = np.load(npz, allow_pickle=False)
        mk, ms, mu = z["m_known"], z["m_seen"], z["m_unseen"]
        z.close()
        print(f"[缓存] {npz}")
    else:
        cfg = json.loads((paths.REPORT_DIR / "final_model_choice.json")
                         .read_text(encoding="utf-8"))["champion"]
        if args.ckpt:
            ckpts = [Path(args.ckpt)]
        elif args.ckpt_dir:
            ckpts = [Path(args.ckpt_dir) / m / "best.pt" for m in cfg["members"]]
        else:
            ckpts = [paths.CKPT_DIR / m / "best.pt" for m in cfg["members"]]
        ckpts = [c for c in ckpts if c.exists()]
        if not ckpts:
            raise SystemExit("没有可用的 checkpoint")
        from cardex.inference import CarEnsemble
        ens = CarEnsemble(ckpts, tta_mode=args.tta, amp=True,
                          prior_adjust_tau=float(cfg["best"]["tau"]),
                          unknown_threshold=0.0, device="auto")
        print(f"模型：{[c.parent.name for c in ckpts]}  TTA={args.tta}"
              f"  （{len(ckpts)} 个成员）\n")

        cm = json.loads((ROOT / "configs" / "class_map.json").read_text(encoding="utf-8"))
        targets = [cm["index_to_id"][str(i)] for i in range(50)]
        used = {s["id"] for s in json.loads(
            (ROOT / "configs" / "unknown_sources.json").read_text(encoding="utf-8"))["sources"]}
        info = json.loads((DS / "cls_info" / "class_info.json").read_text(encoding="utf-8"))
        unseen = sorted({str(x["id"]).zfill(4) for x in info} - set(targets) - used)

        rng = random.Random(20250922)
        known = []
        for cid in targets:
            d = ROOT / "data" / "val" / cid
            fs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG) if d.is_dir() else []
            known += rng.sample(fs, min(24, len(fs)))
        seen = []
        for cid in sorted(used):
            fs = sorted(p for p in (DS / "val" / cid).iterdir()
                        if p.suffix.lower() in IMG) if (DS / "val" / cid).is_dir() else []
            if fs:
                seen += rng.sample(fs, min(8, len(fs)))
        uns = []
        for cid in unseen:
            d = DS / "val" / cid
            if d.is_dir():
                fs = [p for p in d.iterdir() if p.suffix.lower() in IMG]
                if fs:
                    uns.append(rng.choice(fs))
        rng.shuffle(uns)
        uns = uns[:700]
        print(f"已知 {len(known)} 张；见过的来源类 {len(seen)} 张；"
              f"没见过的类 {len(uns)} 张（{len({p.parent.name for p in uns})} 个类）\n")

        def run(files, tag):
            out, t0 = [], time.perf_counter()
            with ThreadPoolExecutor(max_workers=8) as pool:
                for s in range(0, len(files), 16):
                    chunk = files[s: s + 16]
                    imgs = list(pool.map(load_image, chunk))
                    out.append(ens.score_logits(imgs, batch_size=16).float().cpu().numpy())
            print(f"  [{tag}] {len(files)} 张  {time.perf_counter() - t0:.0f}s")
            assert len(out) > 0, f"{tag}: 没有推理结果"
            L = np.concatenate(out)
            return L[:, 50] - L[:, :50].max(1)

        mk = run(known, "已知")
        ms = run(seen, "见过的 unknown")
        mu = run(uns, "没见过的 unknown")
        np.savez_compressed(npz, m_known=mk, m_seen=ms, m_unseen=mu)
        print(f"[写出] {npz}\n")

    # ── ROC → macro_f1 ──
    def curve(m_known, m_unk):
        rows = []
        for th in np.arange(-12, 8.01, 0.02):
            kr = float((m_known > th).mean())          # known_rejection_rate
            ur = float((m_unk > th).mean())
            tp = N_KNOWN * (1 - kr)
            fp = N_UNKNOWN * (1 - ur)
            if tp + fp <= 0:
                continue
            f1 = 2 * tp / (tp + fp + N_KNOWN)
            rows.append((float(th), kr, ur, f1, tp, fp))
        return rows

    print("=" * 92)
    print("ROC → 可达 macro_f1（逐类等权解析式 F1 = 2TP/(TP+FP+1000)）")
    print("=" * 92)
    hdr = (f"{'组':<16}{'最优 macro_f1':>15}{'对应 UR':>11}{'对应 KR':>11}"
           f"{'TP':>7}{'FP':>7}")
    print(hdr)
    results = {}
    for name, m_unk in (("没见过的类（下界）", mu), ("见过的类（上界）", ms)):
        rows = curve(mk, m_unk)
        b = max(rows, key=lambda r: r[3])
        results[name] = {"best_f1": b[3], "ur": b[2], "kr": b[1], "tp": b[4], "fp": b[5]}
        print(f"{name:<16}{b[3]:>15.4f}{b[2]:>11.2%}{b[1]:>11.2%}{b[4]:>7.0f}{b[5]:>7.0f}")

    # 覆盖率加权：训练覆盖了 777 个类中的多少 → 测试集合成
    cov = len(json.loads((ROOT / "configs" / "unknown_sources.json")
                         .read_text(encoding="utf-8"))["sources"]) / 777.0
    lb = results["没见过的类（下界）"]["best_f1"]
    ub = results["见过的类（上界）"]["best_f1"]
    synth = cov * ub + (1 - cov) * lb
    print(f"\n当前来源类覆盖率 {cov:.1%} → 合成估计 macro_f1 ≈ {synth:.4f}")
    print(f"（上界 {ub:.4f}、下界 {lb:.4f}，测试集真实值应落在两者之间）")

    print("\n" + "=" * 92)
    print("要 0.95 需要什么？（在「见过的来源类」这条**上界**曲线上找）")
    print("=" * 92)
    ms_rows = curve(mk, ms)
    hit = [r for r in ms_rows if r[3] >= 0.95]
    print(f"  上界曲线最高 macro_f1 = {ub:.4f}")
    if hit:
        b = max(hit, key=lambda r: r[3])
        print(f"  ✅ 可达 0.95：UR={b[2]:.2%}  KR={b[1]:.2%}  TP={b[4]:.0f}  FP={b[5]:.0f}")
    else:
        print(f"  ❌ **即使 100% 覆盖测试 unknown 来源，也到不了 0.95**")
        need_fp = None
        # 反解：F1=0.95 需要什么 (TP, FP)
        print(f"\n  0.95 需要满足 2·TP/(TP+FP+1000) ≥ 0.95：")
        for tp in (950, 900, 850):
            # 2tp/(tp+fp+1000)=0.95 → fp = 2tp/0.95 - tp - 1000
            fp_need = 2 * tp / 0.95 - tp - 1000
            print(f"    若 TP={tp} → FP ≤ {fp_need:.0f}"
                  f"（即 unknown_recall ≥ {1 - fp_need / N_UNKNOWN:.3%}）")
        print(f"\n  而实测上界曲线的最优点只有 TP={results['见过的类（上界）']['tp']:.0f}"
              f" / FP={results['见过的类（上界）']['fp']:.0f}")

    out = ROOT / "outputs" / "reports" / "roc_ceiling.json"
    out.write_text(json.dumps({"tag": args.tag, "coverage": cov,
                               "results": results, "synthetic_estimate": synth},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[写出] {out}")


if __name__ == "__main__":
    main()
