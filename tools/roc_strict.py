# -*- coding: utf-8 -*-
"""严格口径的上下界与合成估计。

## 修正的口径问题

`roc_ceiling.py` 把「见过的来源类」固定成配置里的 98 个，对**微调前的模型**是对的，
但对**微调后的模型**是错的 —— 后者训练了 677 个来源类，
却只用其中 98 个（还是刻意挑的近似款）去测上界，会系统性低估。

## 正确的分组（来源类三分，见 data_ood/split.json）

    orig   : 98 个   配置里原有的来源类
    new    : 579 个  新增用于训练的来源类
    honest : 100 个  完全没参与训练，留作诚实验证

对每个模型：
    seen   = 该模型训练时覆盖的来源类
    unseen = 其余

    原始模型：seen = orig(98)          unseen = new + honest(679)   覆盖率 12.6%
    微调模型：seen = orig + new(677)   unseen = honest(100)         覆盖率 87.1%

    合成估计 = 覆盖率 × F1(seen) + (1 − 覆盖率) × F1(unseen)

## 为什么必须这样拆

测试集的 unknown 来自全部 777 个类。模型的真实表现是「它见过的那些类」
与「它没见过的那些类」按覆盖率加权的结果。覆盖率不同，加权就不同 —— 
两个模型不能套同一个公式。
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
N_TOTAL_SOURCES = 777.0
CACHE = ROOT / "outputs" / "reports" / "roc_strict"


def load_image(p: Path):
    from PIL import Image
    with Image.open(p) as im:
        return im.convert("RGB")


def sample_group(ids: list[str], per: int, rng: random.Random) -> list[Path]:
    out = []
    for cid in ids:
        d = DS / "val" / cid
        if not d.is_dir():
            continue
        fs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG)
        if fs:
            out += rng.sample(fs, min(per, len(fs)))
    return out


def best_f1(m_known: np.ndarray, m_unk: np.ndarray) -> dict:
    """在 ROC 上找 macro_f1 最大的操作点。F1 = 2TP/(TP+FP+1000)。"""
    best = None
    for th in np.arange(-12, 8.01, 0.02):
        kr = float((m_known > th).mean())
        ur = float((m_unk > th).mean())
        tp = N_KNOWN * (1 - kr)
        fp = N_UNKNOWN * (1 - ur)
        if tp + fp <= 0:
            continue
        f1 = 2 * tp / (tp + fp + N_KNOWN)
        if best is None or f1 > best["f1"]:
            best = {"f1": f1, "ur": ur, "kr": kr, "tp": tp, "fp": fp, "theta": float(th)}
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[],
                    help='可重复传入："标签|覆盖率|ckpt1,ckpt2,..."。'
                         "多个 ckpt 用逗号分隔 → 作为一个集成评测。"
                         "不给覆盖率则按 ckpt 个数推断不了，必须显式写。")
    ap.add_argument("--tta", default="hflip")
    ap.add_argument("--per-source", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    split = json.loads((ROOT / "data_ood" / "split.json").read_text(encoding="utf-8"))
    orig = sorted(split["used_sources"])
    new = sorted(split["new_train_sources"])
    honest = sorted(split["honest_val_sources"])
    print(f"来源类三分：orig {len(orig)} / new {len(new)} / honest {len(honest)}"
          f"（合计 {len(orig) + len(new) + len(honest)}）\n")

    CACHE.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20250922)
    cm = json.loads((ROOT / "configs" / "class_map.json").read_text(encoding="utf-8"))
    targets = [cm["index_to_id"][str(i)] for i in range(50)]
    known = []
    for cid in targets:
        d = ROOT / "data" / "val" / cid
        fs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG) if d.is_dir() else []
        known += rng.sample(fs, min(24, len(fs)))
    g_orig = sample_group(orig, args.per_source, rng)
    g_new = sample_group(new, max(1, args.per_source // 8), rng)
    g_honest = sample_group(honest, args.per_source * 3, rng)
    print(f"样本：已知 {len(known)}   orig {len(g_orig)}   new {len(g_new)}   "
          f"honest {len(g_honest)}\n")

    # ── 待评模型 ──
    cfg = json.loads((paths.REPORT_DIR / "final_model_choice.json")
                     .read_text(encoding="utf-8"))["champion"]
    runs: list[tuple[list[str] | None, str, float]] = []
    if args.run:
        for spec in args.run:
            tag, cov, pstr = spec.split("|")
            runs.append(([p for p in pstr.split(",") if p], tag, float(cov)))
    else:
        runs.append((None, "champion集成(原)", 98 / N_TOTAL_SOURCES))

    from cardex.inference import CarEnsemble
    summary = {}
    for ck, tag, cov in runs:
        npz = CACHE / f"{tag}_{args.tta}.npz"
        if npz.exists() and not args.force:
            z = np.load(npz, allow_pickle=False)
            mk, mo, mn, mh = z["mk"], z["mo"], z["mn"], z["mh"]
            n_members = int(z["n_members"]) if "n_members" in z.files else 0
            z.close()
            print(f"[缓存] {tag}")
        else:
            ckpts = [Path(p) for p in ck] if ck else \
                    [paths.CKPT_DIR / m / "best.pt" for m in cfg["members"]]
            ckpts = [c for c in ckpts if c.exists()]
            if not ckpts:
                print(f"[跳过] {tag}：没有可用的 checkpoint")
                continue
            ens = CarEnsemble(ckpts, tta_mode=args.tta, amp=True,
                              prior_adjust_tau=float(cfg["best"]["tau"]),
                              unknown_threshold=0.0, device="auto")
            n_members = len(ckpts)
            print(f"=== {tag}：{n_members} 个成员 {[c.parent.name for c in ckpts]} ===")

            def run(files):
                out = []
                with ThreadPoolExecutor(max_workers=8) as pool:
                    for s in range(0, len(files), 16):
                        chunk = files[s: s + 16]
                        imgs = list(pool.map(load_image, chunk))
                        out.append(ens.score_logits(imgs, batch_size=16).float().cpu().numpy())
                L = np.concatenate(out)
                return L[:, 50] - L[:, :50].max(1)

            t0 = time.perf_counter()
            mk, mo = run(known), run(g_orig)
            mn, mh = run(g_new), run(g_honest)
            print(f"    推理完成 {time.perf_counter() - t0:.0f}s")
            np.savez_compressed(npz, mk=mk, mo=mo, mn=mn, mh=mh,
                                n_members=np.array(n_members))
            del ens

        # 原始模型：seen = orig，unseen = new + honest
        seen_orig = best_f1(mk, mo)
        unseen_orig = best_f1(mk, np.r_[mn, mh])
        # 微调模型：seen = orig + new，unseen = honest
        seen_ft = best_f1(mk, np.r_[mo, mn])
        unseen_ft = best_f1(mk, mh)

        print(f"  按 orig 分组（原始模型口径）：seen {seen_orig['f1']:.4f} / "
              f"unseen {unseen_orig['f1']:.4f}")
        print(f"  按 677 分组（微调模型口径）：seen {seen_ft['f1']:.4f} / "
              f"unseen {unseen_ft['f1']:.4f}")
        synth = cov * seen_ft["f1"] + (1 - cov) * unseen_ft["f1"]
        synth_orig_formula = (98 / N_TOTAL_SOURCES) * seen_orig["f1"] + \
                             (1 - 98 / N_TOTAL_SOURCES) * unseen_orig["f1"]
        print(f"  → **合成估计（覆盖率 {cov:.1%}）= {synth:.4f}**"
              f"   （若按 12.6% 口径算则是 {synth_orig_formula:.4f}）\n")
        summary[tag] = {"coverage": cov, "n_members": n_members,
                        "seen_677": seen_ft, "unseen_honest": unseen_ft,
                        "seen_98": seen_orig, "unseen_679": unseen_orig,
                        "synthetic": synth}

    out = ROOT / "outputs" / "reports" / "roc_strict_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 84)
    print(f"{'模型':<26}{'覆盖率':>9}{'seen F1':>10}{'unseen F1':>11}{'合成估计':>11}")
    print("=" * 84)
    for tag, s in summary.items():
        print(f"{tag:<26}{s['coverage']:>9.1%}{s['seen_677']['f1']:>10.4f}"
              f"{s['unseen_honest']['f1']:>11.4f}{s['synthetic']:>11.4f}")
    print(f"\n[写出] {out}")


if __name__ == "__main__":
    main()
