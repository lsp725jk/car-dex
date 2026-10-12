# -*- coding: utf-8 -*-
"""双参数平移模型（符号已修正）。

## 符号约定（务必看清，上一版就是这里反了）

    margin  m = z_unknown − max(z_已知类)        （先验校正后）
    **判 unknown  ⟺  m > θ**

    已知误拒率  s(θ) = P(m > θ | 真已知)     ← 存活函数，不是累积分布
    unknown 召回 R(θ) = P(m > θ | 真 unknown)

## 模型

测试集的 margin = 验证集的 margin **加一个平移量**：

    s_test(θ) = surv_known_val(θ − d_k)
    R_test(θ) = surv_unknown_val(θ − d_u)

d_k、d_u 分别由评测给出的两个标定点确定（θ = 0 处，即原 argmax 规则）：
    s_test(0) = 0.015      R_test(0) = 0.7874

脚本会**打印校准复核**，确认两个目标值都对上再往下算。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from cardex import paths  # noqa: E402
from cardex.data import CarDataset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEST_KNOWN, TEST_UNK = 1000, 15540
P_UNK = TEST_UNK / (TEST_KNOWN + TEST_UNK)
CAL_S, CAL_R = 0.015, 0.7874


def margins(logp, tau, log_prior, unk=50):
    z = logp - tau * log_prior[None, :]
    return z[:, unk] - np.delete(z, unk, axis=1).max(axis=1)


def macro_f1(R, s):
    per = TEST_KNOWN / 50
    fp = TEST_UNK * (1 - R) / 50
    fn = per * s + per * 0.004
    tp = per - fn
    f1k = 2 * tp / (2 * tp + fp + fn)
    tp_u, fp_u = TEST_UNK * R, TEST_KNOWN * s
    prec = tp_u / max(1e-9, tp_u + fp_u)
    f1u = 2 * prec * R / max(1e-9, prec + R)
    return (50 * f1k + f1u) / 51, f1k, f1u


def load_val_logits(champ):
    fc = paths.RUN_DIR / "finalize_cache"
    import hashlib
    lps = []
    for m in champ["members"]:
        ck = paths.CKPT_DIR / m / "best.pt"
        st = ck.stat()
        fp = hashlib.md5(f"{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:10]
        lps.append(np.load(fc / f"logits_{m}_val_ttamc_{fp}.npy"))
    return np.mean(lps, axis=0)


def main() -> None:
    cm = paths.load_class_map()
    champ = json.loads((paths.REPORT_DIR / "final_model_choice.json").read_text(encoding="utf-8"))["champion"]
    tau0 = float(champ["best"]["tau"])

    st = json.loads((paths.CONFIG_DIR / "data_stats.json").read_text(encoding="utf-8"))
    n_c = [st["per_class"]["train"][cm["index_to_id"][str(i)]]["n"] for i in range(50)]
    n_c.append(st["per_class"]["train"]["unknown"]["n"])
    arr = np.array(n_c, dtype=np.float64)
    log_prior = np.log(arr / arr.sum())

    logp_val = load_val_logits(champ)
    y_val = np.array([s[1] for s in CarDataset(paths.DATA_DIR / "val").samples])
    mv = margins(logp_val, tau0, log_prior)
    mk, mu = mv[y_val != 50], mv[y_val == 50]

    print("=" * 86)
    print("第一步：验证集在 τ=0.2、θ=0（即 m>0 判 unknown）处")
    print("=" * 86)
    s_val, R_val = float((mk > 0).mean()), float((mu > 0).mean())
    print(f"  验证集： s = {s_val:.4f}   R = {R_val:.4f}")
    print(f"  测试集： s = {CAL_S:.4f}   R = {CAL_R:.4f}   ← 评测给出")
    print(f"  推断：   unknown 分布**下移**（margin 变小，更像已知类）")

    # 求平移量：模型为 test_m = val_m − d，故 surv_test(θ) = P(val_m > θ + d)
    # 要 surv_test(0) = target，即 P(val_m > d) = target
    #   → d 就是"上面只剩 target 比例"的那个分位点 = quantile(val_m, 1 − target)
    def solve_shift(samples, target):
        return float(np.quantile(samples, 1 - target))

    dk = solve_shift(mk, CAL_S)
    du = solve_shift(mu, CAL_R)

    # 校准复核
    surv_k = lambda th: float((mk - dk > th).mean())
    surv_u = lambda th: float((mu - du > th).mean())
    print(f"\n  平移量：d_k = {dk:+.4f}（已知类）   d_u = {du:+.4f}（unknown）")
    print(f"  校准复核：s_test(0) = {surv_k(0):.4f}（目标 {CAL_S}）  "
          f"R_test(0) = {surv_u(0):.4f}（目标 {CAL_R}）")
    ok_cal = abs(surv_k(0) - CAL_S) < 1e-3 and abs(surv_u(0) - CAL_R) < 1e-3
    print(f"  → {'✅ 校准通过' if ok_cal else '❌ 校准失败，结果不可用'}")
    if not ok_cal:
        sys.exit(1)

    # ---- 扫描 ----
    grid = np.arange(-12.0, 12.01, 0.05)
    rows = []
    for th in grid:
        s = float((mk - dk > th).mean())
        R = float((mu - du > th).mean())
        ur = R * P_UNK + s * (1 - P_UNK)
        f1, f1k, f1u = macro_f1(R, s)
        rows.append((f1, float(th), ur, R, s, f1k, f1u))

    print("\n" + "=" * 86)
    print("第二步：扫描 θ（已按评测值校准到测试集先验）")
    print("=" * 86)
    print(f"  {'θ':>7} {'预测unk':>9} {'R':>7} {'s':>7} {'每类FP':>7} "
          f"{'F1已知':>8} {'F1未知':>8} {'MacroF1':>9}")
    for f1, th, ur, R, s, f1k, f1u in rows[:: max(1, len(rows) // 26)]:
        print(f"  {th:>7.2f} {ur:>8.1%} {R:>7.3f} {s:>7.3f} {TEST_UNK * (1 - R) / 50:>7.1f} "
              f"{f1k:>8.4f} {f1u:>8.4f} {f1:>9.4f}")

    best = max(rows, key=lambda r: r[0])
    zero = min(rows, key=lambda r: abs(r[1]))
    print("\n" + "=" * 86)
    print(f"  原点 θ=0（当前提交）：预测 unknown {zero[2]:.1%}  R={zero[3]:.3f} s={zero[4]:.3f}"
          f"  估算 F1 = {zero[0]:.4f}   实测 = 0.4785")
    print(f"  ★ 最优 θ = {best[1]:.2f}：预测 unknown {best[2]:.1%}  "
          f"R={best[3]:.3f} s={best[4]:.3f}  估算 F1 = {best[0]:.4f}")
    bias = 0.4785 - zero[0]
    print(f"  → 提升 {best[0] - zero[0]:+.4f}；偏置校正后预计 ≈ {best[0] + bias:.4f}")
    print("=" * 86)

    out = paths.REPORT_DIR / "rejection_calibration.json"
    out.write_text(json.dumps({
        "calibration": {"tau": tau0, "d_known": dk, "d_unknown": du,
                        "val_s": s_val, "val_R": R_val,
                        "check_s": surv_k(0), "check_R": surv_u(0)},
        "origin": {"theta": 0.0, "pred_unknown": zero[2], "R": zero[3], "s": zero[4],
                   "est_f1": zero[0], "actual_f1": 0.47850900161045806},
        "best": {"theta": best[1], "pred_unknown": best[2], "R": best[3], "s": best[4],
                 "est_f1": best[0], "est_f1_bias_corrected": best[0] + bias},
        "curve": [{"theta": th, "ur": ur, "R": R, "s": s, "f1": f1}
                  for f1, th, ur, R, s, _, _ in rows[::4]],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
