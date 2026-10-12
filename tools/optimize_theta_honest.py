# -*- coding: utf-8 -*-
"""用「诚实 OOD 验证集」投影到测试集先验，求最优拒识偏置 θ。

## 方法

验证集上直接算 macro-F1 是错的 —— 它的 unknown 只占 50%，而测试集占 94%。
同一个漏出率，在 94% unknown 的测试集上产生的 FP 是验证集的 `β/α` 倍。

所以对每个已知类 c 的混淆计数做**重要性加权**：

    α = 测试集真已知数 / 验证集已知数
    β = 测试集真未知数 / 验证集未知数
    F1_c = 2α·TP_c / (2α·TP_c + β·FP_c + α·FN_c)
    macro-F1 = mean_c F1_c

β/α ≈ 15.5，也就是 FP 的代价被放大 15.5 倍 —— 这正是 47.85 分那个坑的数学形式。

## 自检

θ=0 时本脚本算出的值应当 ≈ 老师实测的 macro_f1 = 0.4785。
对不上就说明验证集不代表测试集，后面的结论全部不可信。

## 用法

    python tools/optimize_theta_honest.py --npz outputs/reports/ood_val_logits/<tag>.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# 由老师 θ=0 那次成绩反解出的测试集构成
N_KNOWN_TEST, N_UNKNOWN_TEST = 1000.0, 15540.0
REPORTED_F1_AT_THETA0 = 0.47850900161045806
REPORTED = {"accuracy": 0.799093107617896, "macro_f1": REPORTED_F1_AT_THETA0,
            "known_accuracy": 0.981, "unknown_recall": 0.7873873873873873,
            "known_rejection_rate": 0.015}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--lo", type=float, default=-8.0)
    ap.add_argument("--hi", type=float, default=1.0)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--anchor-tta-offset", type=float, default=0.0,
                    help="若 val 与 test 的 TTA 口径不同，用这个常量对齐（见 --find-offset）")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    L, y = z["logits"].astype(np.float64), z["labels"]
    z.close()
    is_unk = y == 50
    n_k, n_u = float((~is_unk).sum()), float(is_unk.sum())
    margin = L[:, 50] - L[:, :50].max(1)
    print(f"验证集：{len(y)} 张（已知 {int(n_k)} / 未知 {int(n_u)}），"
          f"未知占比 {is_unk.mean():.1%}")

    alpha = N_KNOWN_TEST / n_k
    beta = N_UNKNOWN_TEST / n_u
    print(f"重要性权重：α={alpha:.4f}（已知）  β={beta:.4f}（未知）  "
          f"β/α={beta / alpha:.1f}× —— FP 的代价倍数\n")

    ths = np.round(np.arange(args.lo, args.hi + 1e-9, args.step), 4)
    rows = []
    for th in ths:
        pred_unk = margin > (th + args.anchor_tta_offset)
        # 每个已知类的混淆计数
        f1s, tps, fps, fns = [], [], [], []
        for c in range(50):
            gold = (~is_unk) & (y == c)
            tp = int((gold & ~pred_unk).sum())
            fn = int((gold & pred_unk).sum())
            fp = int((is_unk & ~pred_unk & (L[:, :50].argmax(1) == c)).sum())
            tps.append(tp); fps.append(fp); fns.append(fn)
            a_tp, a_fn, b_fp = alpha * tp, alpha * fn, beta * fp
            if a_tp + a_fn + b_fp == 0:
                continue                      # 该类在验证集里没有样本，跳过
            f1s.append(2 * a_tp / (2 * a_tp + b_fp + a_fn))
        macro = float(np.mean(f1s)) if f1s else 0.0
        # 全局指标（未加权，用于对照）
        ur = float((pred_unk & is_unk).sum() / n_u)
        kr = float((pred_unk & ~is_unk).sum() / n_k)
        rows.append({"theta": float(th), "macro_f1_weighted": macro,
                     "unknown_recall": ur, "known_rejection_rate": kr,
                     "fp_true_scale": float(beta * np.sum(fps)),
                     "fn_true_scale": float(alpha * np.sum(fns))})

    # ── 自检：θ=0 应当复现老师实测的 0.4785 ──
    at0 = min(rows, key=lambda r: abs(r["theta"]))
    print("=" * 78)
    print("自检：θ=0 时投影出的 macro-F1 是否复现老师的实测值？")
    print(f"  投影值 {at0['macro_f1_weighted']:.4f}   老师实测 {REPORTED_F1_AT_THETA0:.4f}"
          f"   差 {at0['macro_f1_weighted'] - REPORTED_F1_AT_THETA0:+.4f}")
    ok = abs(at0["macro_f1_weighted"] - REPORTED_F1_AT_THETA0) < 0.06
    print(f"  → {'✅ 验证集能代表测试集，后续结论可信' if ok else '⚠️ 偏差较大，结论只能当方向参考'}")
    print(f"  投影 unknown_recall {at0['unknown_recall']:.4f}（实测 {REPORTED['unknown_recall']:.4f}）"
          f"   known_rejection {at0['known_rejection_rate']:.4f}（实测 {REPORTED['known_rejection_rate']:.4f}）")
    print("=" * 78)

    best = max(rows, key=lambda r: r["macro_f1_weighted"])
    print(f"\n{'θ':>7}{'投影 macro-F1':>16}{'unknown_recall':>16}{'known_reject':>14}"
          f"{'FP(实测尺度)':>14}{'FN(实测尺度)':>14}")
    for r in rows:
        mark = ""
        if r is best:
            mark = "  ← 最优"
        elif abs(r["theta"] - (-5.6)) < 0.05:
            mark = "  ← 当前已施加"
        if r["macro_f1_weighted"] < 0.2 and r["theta"] > -1:
            continue
        print(f"{r['theta']:>7.1f}{r['macro_f1_weighted']:>16.4f}{r['unknown_recall']:>16.2%}"
              f"{r['known_rejection_rate']:>14.2%}{r['fp_true_scale']:>14.0f}"
              f"{r['fn_true_scale']:>14.0f}{mark}")

    print(f"\n最优 θ = {best['theta']:+.1f}  投影 macro-F1 = {best['macro_f1_weighted']:.4f}")
    cur = min(rows, key=lambda r: abs(r["theta"] - (-5.6)))
    print(f"当前 θ = -5.6   投影 macro-F1 = {cur['macro_f1_weighted']:.4f}"
          f"   （差 {best['macro_f1_weighted'] - cur['macro_f1_weighted']:+.4f}）")

    out = ROOT / "outputs" / "reports" / "theta_honest_optimum.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "val_npz": str(args.npz), "alpha": alpha, "beta": beta,
        "self_check_theta0_projected": at0["macro_f1_weighted"],
        "self_check_reported": REPORTED_F1_AT_THETA0,
        "best": best, "current_minus_5_6": cur, "curve": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[写出] {out}")


if __name__ == "__main__":
    main()
