# -*- coding: utf-8 -*-
"""量化：unknown 头在「训练见过的 98 个来源类」与「没见过的 679 个类」上差多少。

## 为什么问这个

教师 README 要求「ID 与本组登记的最终选类一致」——**每组自选 50 类**，
所以组与组之间难度本就不同。但除选题之外，还有一个可以量化的自身差距：

    我们只用了 98 个 unknown 来源类 / 1,960 张图，
    而教师数据集里还有 679 个未使用类 / 62,543 张可用。

如果模型在「见过的 98 类」上表现远好于「没见过的 679 类」，
说明 unknown 头过拟合到来源类，**扩充来源类是最直接的修复**。

## 对比设计

  A. 泄漏组：data/val/unknown（588 张，全部来自训练见过的 98 个来源类）
  B. 诚实组：679 个未见类的 val 图（同一 TTA 口径）

两者都当 unknown，各自的 margin 分布直接可比。
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


def load_image(p: Path):
    from PIL import Image
    with Image.open(p) as im:
        return im.convert("RGB")


def auc(s: np.ndarray, is_pos: np.ndarray) -> float:
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1 = int(is_pos.sum()); n0 = len(s) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[is_pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", default="hflip")
    ap.add_argument("--n-honest", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = json.loads((paths.REPORT_DIR / "final_model_choice.json").read_text(encoding="utf-8"))["champion"]
    ckpts = [paths.CKPT_DIR / m / "best.pt" for m in cfg["members"]]
    from cardex.inference import CarEnsemble
    ens = CarEnsemble(ckpts, tta_mode=args.tta, amp=True,
                      prior_adjust_tau=float(cfg["best"]["tau"]),
                      unknown_threshold=0.0, device="auto")
    print(f"冠军 {cfg['tag']}  TTA={args.tta}\n")

    # ── A. 泄漏组：训练见过的 98 个来源类 ──
    leak_dir = ROOT / "data" / "val" / "unknown"
    leak = sorted(p for p in leak_dir.iterdir() if p.suffix.lower() in IMG)
    print(f"A. 泄漏组（训练见过的 98 个来源类）：{len(leak)} 张")

    # ── B. 诚实组：679 个未见类 ──
    cm = json.loads((ROOT / "configs" / "class_map.json").read_text(encoding="utf-8"))
    targets = {cm["index_to_id"][str(i)] for i in range(50)}
    used = {s["id"] for s in json.loads(
        (ROOT / "configs" / "unknown_sources.json").read_text(encoding="utf-8"))["sources"]}
    info = json.loads((DS / "cls_info" / "class_info.json").read_text(encoding="utf-8"))
    honest_ids = sorted({str(x["id"]).zfill(4) for x in info} - targets - used)
    rng = random.Random(20250922)
    cand = []
    for c in honest_ids:
        d = DS / "val" / c
        if d.is_dir():
            fs = [p for p in d.iterdir() if p.suffix.lower() in IMG]
            if fs:
                cand.append(rng.choice(fs))
    rng.shuffle(cand)
    honest = cand[: args.n_honest]
    print(f"B. 诚实组（未见类，每类取 1 张铺开）：{len(honest)} 张，来自 "
          f"{len({p.parent.name for p in honest})} 个类\n")

    # ── C. 已知类参考（算 AUC 用）──
    known = []
    for cid in sorted(targets):
        d = ROOT / "data" / "val" / cid
        if d.is_dir():
            fs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG)
            known += rng.sample(fs, min(12, len(fs)))
    print(f"C. 已知类参考：{len(known)} 张\n")

    def run(files, tag):
        out, t0 = [], time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for s in range(0, len(files), args.batch):
                chunk = files[s: s + args.batch]
                imgs = list(pool.map(load_image, chunk))
                out.append(ens.score_logits(imgs, batch_size=args.batch).float().cpu().numpy())
        L = np.concatenate(out)
        print(f"  [{tag}] {len(files)} 张，用时 {time.perf_counter() - t0:.0f}s")
        return L

    Lk = run(known, "已知")
    La = run(leak, "泄漏 unknown")
    Lb = run(honest, "诚实 unknown")

    def margin(L):
        return L[:, 50] - L[:, :50].max(1)

    mk, ma, mb = margin(Lk), margin(La), margin(Lb)
    print("\n" + "=" * 78)
    print("margin 分布对比（margin = z_unknown − max(z_已知)，越大越像 unknown）")
    print("=" * 78)
    print(f"{'组':<26}{'均值':>9}{'中位':>9}{'10%':>9}{'90%':>9}")
    for name, m in (("已知类（应偏低）", mk), ("泄漏 unknown（98 类）", ma),
                    ("诚实 unknown（679 类）", mb)):
        print(f"{name:<26}{m.mean():>9.2f}{np.median(m):>9.2f}"
              f"{np.percentile(m, 10):>9.2f}{np.percentile(m, 90):>9.2f}")

    print("\n" + "=" * 78)
    print("OOD 判别 AUC（已知 vs 各 unknown 组）")
    print("=" * 78)
    a_leak = auc(np.r_[mk, ma], np.r_[np.zeros(len(mk), bool), np.ones(len(ma), bool)])
    a_hon = auc(np.r_[mk, mb], np.r_[np.zeros(len(mk), bool), np.ones(len(mb), bool)])
    print(f"  已知 vs 泄漏 unknown（训练见过）：AUC = {a_leak:.4f}")
    print(f"  已知 vs 诚实 unknown（没见过的）：AUC = {a_hon:.4f}")
    print(f"  差距：{a_leak - a_hon:+.4f}")
    print(f"\n  → {'✅ 存在明显过拟合：unknown 头对训练用过的来源类判别力显著更强' if a_leak - a_hon > 0.01 else '过拟合不明显'}")

    # 在相同 known_rejection 下比较 unknown_recall
    print("\n" + "=" * 78)
    print("固定已知拒识率，比较能拒掉多少 unknown")
    print("=" * 78)
    print(f"{'known_rejection':>16}{'泄漏组 UR':>14}{'诚实组 UR':>14}{'差':>10}")
    for kr in (0.05, 0.10, 0.20, 0.30):
        th = np.quantile(mk, 1 - kr)
        ua = float((ma > th).mean()); ub = float((mb > th).mean())
        print(f"{kr:>16.0%}{ua:>14.2%}{ub:>14.2%}{ua - ub:>+10.2%}")

    out = ROOT / "outputs" / "reports" / "unknown_head_overfit_check.json"
    out.write_text(json.dumps({
        "auc_known_vs_seen_sources": a_leak,
        "auc_known_vs_unseen_classes": a_hon,
        "gap": a_leak - a_hon,
        "n_leak": len(leak), "n_honest": len(honest), "n_known": len(known),
        "margin_means": {"known": float(mk.mean()), "leak": float(ma.mean()),
                         "honest": float(mb.mean())},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[写出] {out}")


if __name__ == "__main__":
    main()
