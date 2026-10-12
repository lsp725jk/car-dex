# -*- coding: utf-8 -*-
"""按测试集分布加权标定 θ（微调模型专用）。

## 为什么不能只用诚实 val

微调模型训练了 677 个来源类，而测试集的 unknown 正是从这 777 个类来的：
**87.1% 落在它见过的来源类里，只有 12.9% 没见过。**

如果只用「诚实 val（没见过）」标定 θ，会按 100% 没见过的难度去调，
把阈值推得过保守。反之只用「泄漏 val（见过）」又会过于乐观。

## 正确做法：加权混合

    known   : 1,200 个 margin，权重 1.0
    unseen  : 从 100 个诚实来源类采，**总权重 0.1287**
    seen    : 从 677 个训练来源类采，**总权重 0.8713**

这样 UR(θ) / KR(θ) 直接就是测试集口径的估计，不需要任何偏移校正
（前提：标定与测试用同一 TTA 档位，本脚本沿用 hflip）。

    合成 F1(θ) = 2·TP / (TP + FP + 1000)
    TP = 1000 × (1 − KR),  FP = 15540 × (1 − UR)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "reports" / "roc_strict"
N_KNOWN, N_UNKNOWN = 1000.0, 15540.0
N_TOTAL_SOURCES = 777.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ft_plus_orig2")
    ap.add_argument("--tta", default="hflip")
    ap.add_argument("--coverage", type=float, default=None,
                    help="见过的来源类占比；默认按 split.json 算")
    ap.add_argument("--lo", type=float, default=-12.0)
    ap.add_argument("--hi", type=float, default=4.0)
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()

    npz = CACHE / f"{args.tag}_{args.tta}.npz"
    if not npz.exists():
        raise SystemExit(f"找不到 {npz}，先跑 tools/roc_strict.py")
    z = np.load(npz, allow_pickle=False)
    mk, mo, mn, mh = z["mk"], z["mo"], z["mn"], z["mh"]
    z.close()

    if args.coverage is None:
        split = json.loads((ROOT / "data_ood" / "split.json").read_text(encoding="utf-8"))
        cov = len(split["train_sources_all"]) / N_TOTAL_SOURCES
    else:
        cov = args.coverage
    print(f"缓存：{npz.name}")
    print(f"样本：已知 {len(mk)}  seen(orig) {len(mo)}  seen(new) {len(mn)}  unseen {len(mh)}")
    print(f"见过的来源类覆盖率 = {cov:.2%}\n")

    seen = np.r_[mo, mn]
    unseen = mh
    # 每个样本的权重，使 seen 总权重 = cov、unseen 总权重 = 1 − cov
    w_seen = cov / len(seen)
    w_unseen = (1 - cov) / len(unseen)
    print(f"权重：seen 每样本 {w_seen:.3e}（{len(seen)} 个，合计 {cov:.3f}）")
    print(f"      unseen 每样本 {w_unseen:.3e}（{len(unseen)} 个，合计 {1 - cov:.3f}）\n")

    rows = []
    for th in np.arange(args.lo, args.hi + 1e-9, args.step):
        kr = float((mk > th).mean())
        ur = w_seen * float((seen > th).sum()) + w_unseen * float((unseen > th).sum())
        tp = N_KNOWN * (1 - kr)
        fp = N_UNKNOWN * (1 - ur)
        if tp + fp <= 0:
            continue
        f1 = 2 * tp / (tp + fp + N_KNOWN)
        rows.append({"theta": float(th), "kr": kr, "ur": ur, "tp": tp, "fp": fp, "f1": f1})

    best = max(rows, key=lambda r: r["f1"])
    print("=" * 88)
    print(f"{'θ':>8}{'known_rejection':>18}{'unknown_recall':>17}{'TP':>8}{'FP':>8}"
          f"{'合成 macro_f1':>16}")
    print("=" * 88)
    for i, r in enumerate(rows):
        if i % 8:
            continue
        mark = "  ← 最优" if r is best else ""
        print(f"{r['theta']:>8.2f}{r['kr']:>18.2%}{r['ur']:>17.2%}{r['tp']:>8.0f}"
              f"{r['fp']:>8.0f}{r['f1']:>16.4f}{mark}")

    print(f"\n**最优 θ = {best['theta']:+.2f}**")
    print(f"  预期：known_rejection {best['kr']:.2%}  unknown_recall {best['ur']:.2%}")
    print(f"        TP≈{best['tp']:.0f}  FP≈{best['fp']:.0f}  合成 macro_f1 ≈ {best['f1']:.4f}")

    # 平台区间（F1 在最优值 0.005 以内的 θ 范围）
    plat = [r["theta"] for r in rows if r["f1"] >= best["f1"] - 0.005]
    if plat:
        print(f"  平台期（F1 在最优 −0.005 内）：θ ∈ [{min(plat):+.2f}, {max(plat):+.2f}]")

    out = ROOT / "outputs" / "reports" / f"theta_mixture_{args.tag}.json"
    out.write_text(json.dumps({"tag": args.tag, "tta": args.tta, "coverage": cov,
                               "best": best, "plateau": [min(plat), max(plat)] if plat else None,
                               "curve": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[写出] {out}")


if __name__ == "__main__":
    main()
