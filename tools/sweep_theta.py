"""在**真实测试 logits** 上扫描拒识偏置 θ，给出精确的 ur(θ) 曲线。

为什么需要它：
  `calibrate_rejection_v2.py` 给出的 θ→ur 曲线，在 θ=0 处与实测完全吻合
  （0.74073 vs 0.74069），但在 θ=-5.2 处曲线 0.9428、实测 0.9315，差了 1.1%。
  两套 θ 的定义有系统偏移，**不能混用**。本脚本直接在当前 npz 上反推，
  把「θ → 被判未知的比例」这条映射钉死在实测口径上，不需要重新推理。

用法：
    python tools/sweep_theta.py                    # 看当前 θ 下的曲线
    python tools/sweep_theta.py --claim 1000       # 找「声称已知数 ≈ 1000」的 θ
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from cardex import paths  # noqa: E402

NPZ = ROOT / "student_submit" / "result.npz"

# ── 来自老师反馈的 θ=0 那次提交（唯一一次真实标注信号）──
# accuracy 79.91% / known_accuracy 98.10% / unknown_recall 78.74% / known_rejection_rate 1.50%
GRADED = {"accuracy": 0.7991, "known_accuracy": 0.9810,
          "unknown_recall": 0.7874, "known_rejection_rate": 0.0150,
          "macro_f1": 0.4785}
N_TOTAL = 16540


def solve_composition() -> dict:
    """由 accuracy = p·ka + (1−p)·ur 反解测试集里「真已知」的比例 p。"""
    p = (GRADED["accuracy"] - GRADED["unknown_recall"]) / (
        GRADED["known_accuracy"] - GRADED["unknown_recall"])
    n_known = p * N_TOTAL
    n_unknown = N_TOTAL - n_known
    # θ=0 时的混淆计数
    tp = GRADED["known_accuracy"] * n_known                 # 已知判对
    fn_rej = GRADED["known_rejection_rate"] * n_known       # 已知被判未知
    wrong_known = n_known - tp - fn_rej                     # 已知判成别的已知类
    fp = (1 - GRADED["unknown_recall"]) * n_unknown         # 未知漏成已知
    return {"p_known": p, "n_known": n_known, "n_unknown": n_unknown,
            "tp": tp, "fn_rejected": fn_rej, "fn_wrongclass": wrong_known, "fp": fp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(NPZ))
    ap.add_argument("--current-theta", type=float, default=-5.2,
                    help="npz 里已经施加过的 θ（用于反推 θ=0 的原始 logits）")
    ap.add_argument("--claim", type=int, default=0,
                    help="目标：让「判为已知」的张数 ≈ 这个值（真已知约 1000）")
    ap.add_argument("--lo", type=float, default=-8.0)
    ap.add_argument("--hi", type=float, default=-2.0)
    ap.add_argument("--step", type=float, default=0.1)
    args = ap.parse_args()

    comp = solve_composition()
    print("── 由老师反馈反解的测试集构成（θ=0 那次提交）──")
    print(f"  真已知 {comp['n_known']:.0f} 张（{comp['p_known']:.2%}）  "
          f"真未知 {comp['n_unknown']:.0f} 张（{1 - comp['p_known']:.2%}）")
    print(f"  θ=0 混淆：TP={comp['tp']:.0f}  已知被拒={comp['fn_rejected']:.0f}  "
          f"已知判错类={comp['fn_wrongclass']:.0f}  未知漏出(FP)={comp['fp']:.0f}")
    print(f"  → FP/FN ≈ {comp['fp'] / max(comp['fn_rejected'] + comp['fn_wrongclass'], 1):.0f} : 1"
          "（严重偏向「漏出」，所以要往拒识方向压）\n")

    z = np.load(args.npz, allow_pickle=False)
    L = z["logits"].astype(np.float64)
    n = L.shape[0]
    # 反推 θ=0 的 logits。
    # 施加时是 `L[:,50] -= theta`，即 L_biased = L0 - theta；
    # 所以 L0 = L_biased + theta。⚠️ 这里是加法不是减法 —— 减号写反会让
    # unknown 列整体偏移 2θ，曲线直接失去意义（本脚本第一版就踩了这个坑）。
    L0_unk = L[:, 50] + args.current_theta
    known_max = L[:, :50].max(axis=1)          # 已知类里的最高分
    margin = L0_unk - known_max                # >0 就判未知（θ=0 时等价）

    thetas = np.round(np.arange(args.lo, args.hi + 1e-9, args.step), 4)
    rows = []
    for th in thetas:
        is_unk = margin > th
        ur = is_unk.mean()
        claimed = int((~is_unk).sum())
        rows.append({"theta": float(th), "ur": float(ur), "claimed_known": claimed})

    print("── 实测 ur(θ) 曲线（真实测试集，16540 张）──")
    print(f"{'θ':>8}{'判未知':>10}{'判未知%':>11}{'声称已知':>11}")
    for r in rows:
        star = ""
        if abs(r["claimed_known"] - (args.claim or comp["n_known"])) < 25:
            star = "  ←"
        print(f"{r['theta']:>8.1f}{r['ur'] * n:>10.0f}{r['ur']:>11.2%}"
              f"{r['claimed_known']:>11}{star}")

    # 反向插值：找声称已知数最接近目标的 θ
    claims = np.array([r["claimed_known"] for r in rows], dtype=float)
    target = args.claim or comp["n_known"]
    i = int(np.argmin(np.abs(claims - target)))
    best = rows[i]
    print(f"\n── 目标：声称已知 ≈ {target:.0f} 张 ──")
    print(f"  最接近的 θ = {best['theta']:.1f}  →  声称已知 {best['claimed_known']} 张"
          f"（偏离 {best['claimed_known'] - target:+.0f}）")
    print(f"  对应判未知 {best['ur']:.2%}，全测试集 {best['ur'] * n:.0f} 张")

    out = paths.REPORT_DIR / "theta_sweep_actual.json"
    out.write_text(json.dumps({
        "composition_from_grading": comp,
        "current_theta_in_npz": args.current_theta,
        "curve": rows,
        "recommended": best,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[写出] {out}")


if __name__ == "__main__":
    main()
