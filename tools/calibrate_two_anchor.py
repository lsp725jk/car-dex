# -*- coding: utf-8 -*-
"""用**两次真实评分**标定 val↔test 的口径偏移，再求最优 θ。

## 为什么需要它

诚实 OOD 验证集（`hflip`）的 θ 轴和测试集 logits（`multicrop`）的 θ 轴差一个常量。
这个偏移不能靠猜 —— 但现在有**两个真实锚点**可以把它钉死。

## 两个锚点（老师实测）

| 提交 | θ | known_rejection_rate | unknown_recall | macro_f1 |
| --- | --- | --- | --- | --- |
| 09/22 10:16 | 0 | 1.50% | 78.74% | 0.4785 |
| 09/22 10:35 | −5.6 | 28.10% | 98.17% | 0.7321 |

用 `known_rejection_rate` 做锚（它不像 unknown_recall 那样在 98% 附近饱和，更有分辨力）：
在验证集曲线上找 KR 相同的点，两者之差就是偏移。

## 输出

把验证集曲线上每个 θ 映射到测试集口径，并按测试集先验做重要性加权重算 macro-F1。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = ROOT / "outputs" / "reports" / "ood_val_logits" / "hflip_k1250_u1250.npz"

N_KNOWN_TEST, N_UNKNOWN_TEST = 1000.0, 15540.0
# 两次真实评分
ANCHORS = [
    {"theta_test": 0.0, "kr": 0.0150, "ur": 0.7874, "macro_f1": 0.4785},
    {"theta_test": -5.6, "kr": 0.2810, "ur": 0.9817, "macro_f1": 0.7321},
]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(DEFAULT_NPZ))
    ap.add_argument("--out", default="theta_two_anchor_optimum.json")
    ap.add_argument("--w", type=float, default=1.0,
                    help="决策规则的权重：score = z_unknown − w·max(z_known)。"
                         "w=1 是生产在用的 margin 规则")
    args = ap.parse_args()
    NPZ = Path(args.npz)
    print(f"验证集：{NPZ.name}   决策规则 w={args.w}\n")

    z = np.load(NPZ, allow_pickle=False)
    L, y = z["logits"].astype(np.float64), z["labels"]
    z.close()
    is_unk = y == 50
    n_k, n_u = float((~is_unk).sum()), float(is_unk.sum())
    margin = L[:, 50] - args.w * L[:, :50].max(1)

    alpha, beta = N_KNOWN_TEST / n_k, N_UNKNOWN_TEST / n_u

    # ── 1. 验证集曲线 ──
    ths = np.round(np.arange(-8.0, 0.61, 0.05), 4)
    kr_curve, ur_curve = [], []
    for th in ths:
        pu = margin > th
        kr_curve.append(float((pu & ~is_unk).sum() / n_k))
        ur_curve.append(float((pu & is_unk).sum() / n_u))
    kr_curve, ur_curve = np.array(kr_curve), np.array(ur_curve)

    def val_theta_for_kr(target: float) -> float:
        """在验证集曲线上找 KR 最接近 target 的 θ（单调插值）。"""
        i = int(np.argmin(np.abs(kr_curve - target)))
        return float(ths[i])

    # ── 2. 标定偏移 ──
    print("=" * 78)
    print("标定 val ↔ test 的 θ 偏移（用 known_rejection_rate 对齐）")
    print("=" * 78)
    offsets = []
    for a in ANCHORS:
        vth = val_theta_for_kr(a["kr"])
        off = a["theta_test"] - vth          # test θ = val θ + off
        offsets.append(off)
        print(f"  test θ={a['theta_test']:+.1f}  KR={a['kr']:.2%}"
              f"  →  val 同 KR 处 θ={vth:+.2f}"
              f"  →  偏移 {off:+.2f}")
    off = float(np.mean(offsets))
    print(f"\n  取平均偏移：**test θ = val θ {off:+.2f}**"
          f"（两个锚点差 {abs(offsets[0] - offsets[1]):.2f}）")

    # ── 3. 映射后重算 macro-F1 ──
    rows = []
    for th in ths:
        pu = margin > th
        tps, fps, fns = [], [], []
        argk = L[:, :50].argmax(1)
        for c in range(50):
            gold = (~is_unk) & (y == c)
            tp = int((gold & ~pu).sum())
            fn = int((gold & pu).sum())
            fp = int((is_unk & ~pu & (argk == c)).sum())
            tps.append(tp); fps.append(fp); fns.append(fn)
        f1s = []
        for tp, fp, fn in zip(tps, fps, fns):
            a_tp, a_fn, b_fp = alpha * tp, alpha * fn, beta * fp
            if a_tp + a_fn + b_fp == 0:
                continue
            f1s.append(2 * a_tp / (2 * a_tp + b_fp + a_fn))
        rows.append({
            "theta_val": float(th),
            "theta_test": float(th + off),
            "macro_f1": float(np.mean(f1s)),
            "kr": float((pu & ~is_unk).sum() / n_k),
            "ur": float((pu & is_unk).sum() / n_u),
            "fp_test": float(beta * sum(fps)),
            "fn_test": float(alpha * sum(fns)),
        })

    # ── 4. 自检：两个锚点都要能复现 ──
    print("\n" + "=" * 78)
    print("自检：把两个真实评分点代入，投影值能否复现？")
    print("=" * 78)
    errs = []
    for a in ANCHORS:
        r = min(rows, key=lambda r: abs(r["theta_test"] - a["theta_test"]))
        d = r["macro_f1"] - a["macro_f1"]
        errs.append(d)
        print(f"  test θ={a['theta_test']:+.1f}  投影 {r['macro_f1']:.4f}"
              f"   实测 {a['macro_f1']:.4f}   差 {d:+.4f}")
    print(f"  平均绝对误差 {np.mean(np.abs(errs)):.4f}")

    best = max(rows, key=lambda r: r["macro_f1"])
    cur = min(rows, key=lambda r: abs(r["theta_test"] - (-5.6)))
    print("\n" + "=" * 78)
    print(f"{'θ(val)':>8}{'θ(test)':>9}{'投影 macro-F1':>15}{'KR':>9}{'UR':>9}"
          f"{'FP':>7}{'FN':>7}")
    print("=" * 78)
    for r in rows:
        if r["theta_val"] * 10 % 5:
            continue
        mark = ""
        if r is best:
            mark = "  ← 最优"
        elif abs(r["theta_test"] - (-5.6)) < 0.06:
            mark = "  ← 当前"
        print(f"{r['theta_val']:>8.1f}{r['theta_test']:>9.2f}{r['macro_f1']:>15.4f}"
              f"{r['kr']:>9.2%}{r['ur']:>9.2%}{r['fp_test']:>7.0f}{r['fn_test']:>7.0f}{mark}")

    print(f"\n最优：val θ={best['theta_val']:+.2f}  →  **test θ={best['theta_test']:+.2f}**"
          f"   投影 macro-F1 = {best['macro_f1']:.4f}")
    print(f"当前：θ=-5.60  投影 {cur['macro_f1']:.4f}"
          f"   预期提升 {best['macro_f1'] - cur['macro_f1']:+.4f}")

    out = ROOT / "outputs" / "reports" / args.out
    out.write_text(json.dumps({
        "val_npz": str(NPZ), "anchors": ANCHORS,
        "val_to_test_offset": off, "per_anchor_offset": offsets,
        "self_check": {"projected": errs, "mean_abs_err": float(np.mean(np.abs(errs)))},
        "best": best, "current": cur, "curve": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[写出] {out}")


if __name__ == "__main__":
    main()
