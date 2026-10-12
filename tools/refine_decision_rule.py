# -*- coding: utf-8 -*-
"""两件事一起做：
  ① 交叉验证：用解析式 F1 = 2·TP/(claimed+1000) 复算最优 θ（不依赖逐类建模）
  ② 换更强的决策规则：score = z_unknown − w·max(z_known)，扫 w（现用 w=1，从未验证过）

## ① 的原理

    claimed(θ) = TP + FP        （测试集上可精确算出，因为 logits 在手）
    FN(θ)      = 1000 − TP
    ⇒ F1 = 2TP / (2TP + FP + FN) = 2TP / (claimed + 1000)

逐类等权时严格成立。它把"逐类集中度"这个难建模的东西整个消掉了，
只用两个量：claimed（测试集精确）和 TP（从验证集借）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "outputs" / "reports" / "ood_val_logits" / "hflip_k1250_u1250.npz"
TEST_NPZ = ROOT / "student_submit" / "result.npz"
CURRENT_THETA = -5.6          # npz 里已施加的 θ
N_KNOWN_TEST = 1000.0
OFFSET = 0.55                 # test θ = val θ + 0.55（两锚点标定）


def main() -> None:
    zv = np.load(VAL, allow_pickle=False)
    Lv, yv = zv["logits"].astype(np.float64), zv["labels"]
    zv.close()
    is_unk = yv == 50
    n_k, n_u = float((~is_unk).sum()), float(is_unk.sum())

    zt = np.load(TEST_NPZ, allow_pickle=False)
    Lt = zt["logits"].astype(np.float64)
    zt.close()
    # 反推 θ=0 的测试 logits
    Lt = Lt.copy()
    Lt[:, 50] += CURRENT_THETA
    n_test = Lt.shape[0]

    def scores(L):
        unk = L[:, 50]
        kn = L[:, :50]
        return unk, kn.max(1)

    u_v, k_v = scores(Lv)
    u_t, k_t = scores(Lt)

    print("=" * 82)
    print("② 决策规则：score = z_unknown − w·max(z_known)，扫 w")
    print("=" * 82)
    print(f"{'w':>6}{'最优 θ_test':>13}{'claimed':>10}{'TP':>8}{'FP':>8}{'FN':>8}"
          f"{'F1(解析)':>11}")

    best_all = None
    for w in (0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0):
        s_v, s_t = u_v - w * k_v, u_t - w * k_t
        # 验证集上求 KR / UR 曲线；⚠️ 必须把 test θ 换算成 val 口径（t - OFFSET），
        # 否则 kr/ur 取的是错位的操作点，w 之间根本不可比（第一版就漏了这一步）。
        ths = np.round(np.arange(-12, 4.01, 0.05), 4)
        kr = np.array([((s_v > (t - OFFSET)) & ~is_unk).sum() / n_k for t in ths])
        ur = np.array([((s_v > (t - OFFSET)) & is_unk).sum() / n_u for t in ths])
        # 测试集上 claimed_known(θ)（精确）
        # ⚠️ 是「声称已知」= 总数 − 判未知数。第一版忘了减，直接拿判未知数当 claimed，
        # 结果算出 F1=1.0 这种一眼假的数。
        claimed = np.array([n_test - (s_t > t).sum() for t in ths], dtype=float)

        row_best = None
        for i, t in enumerate(ths):
            fn = N_KNOWN_TEST * kr[i]
            tp = N_KNOWN_TEST - fn
            f1 = 2 * tp / (claimed[i] + N_KNOWN_TEST)
            # 合法性：claimed 里含 FP，FP 不能为负
            fp = claimed[i] - tp
            if fp < -1:
                continue
            if row_best is None or f1 > row_best[1]:
                row_best = (float(t), float(f1), float(claimed[i]), tp, fp, fn)
        t_opt, f1, cl, tp, fp, fn = row_best
        print(f"{w:>6.1f}{t_opt:>13.2f}{cl:>10.0f}{tp:>8.0f}{fp:>8.0f}{fn:>8.0f}{f1:>11.4f}")
        if best_all is None or f1 > best_all[1]:
            best_all = (w, f1, t_opt, cl, tp, fp, fn)

    w_b, f1_b, t_b, cl_b, tp_b, fp_b, fn_b = best_all
    print(f"\n最优规则：w = {w_b}   θ_test = {t_b:+.2f}"
          f"   → TP={tp_b:.0f} FP={fp_b:.0f} FN={fn_b:.0f}  解析 F1 = {f1_b:.4f}")

    # ── 当前 θ=-5.6 的对照 ──
    s_v1, s_t1 = u_v - k_v, u_t - k_t
    cur_claimed = float(n_test - (s_t1 > -5.6).sum())
    # 用 val 求当前 θ 对应的 FN：test θ=-5.6 → val θ=-6.15
    kr_cur = float(((s_v1 > -6.15) & ~is_unk).sum() / n_k)
    fn_cur = N_KNOWN_TEST * kr_cur
    tp_cur = N_KNOWN_TEST - fn_cur
    f1_cur = 2 * tp_cur / (cur_claimed + N_KNOWN_TEST)
    print(f"当前规则：w = 1.0  θ_test = -5.60")
    print(f"  claimed={cur_claimed:.0f}（测试集精确）  TP≈{tp_cur:.0f}  FN≈{fn_cur:.0f}"
          f"  FP≈{cur_claimed - tp_cur:.0f}")
    print(f"  解析 F1 = {f1_cur:.4f}   实测 macro_f1 = 0.7321   差 {f1_cur - 0.7321:+.4f}")

    print("\n" + "=" * 82)
    print("① 解析式 F1 = 2·TP/(claimed+1000) 下的最优 θ（固定 w=1）")
    print("=" * 82)
    ths = np.round(np.arange(-9, 2.01, 0.1), 4)
    claimed = np.array([n_test - (s_t1 > t).sum() for t in ths], dtype=float)
    krv = np.array([((s_v1 > (t - OFFSET)) & ~is_unk).sum() / n_k for t in ths])
    tpv = N_KNOWN_TEST * (1 - krv)
    f1v = 2 * tpv / (claimed + N_KNOWN_TEST)
    i_best = int(np.argmax(f1v))
    print(f"{'θ_test':>9}{'claimed':>10}{'TP':>8}{'FN':>8}{'FP':>8}{'F1解析':>10}")
    for i, t in enumerate(ths):
        if abs(t * 10) % 5:
            continue
        mark = "  ← 最优" if i == i_best else ("  ← 当前" if abs(t + 5.6) < 0.06 else "")
        print(f"{t:>9.1f}{claimed[i]:>10.0f}{tpv[i]:>8.0f}{N_KNOWN_TEST - tpv[i]:>8.0f}"
              f"{claimed[i] - tpv[i]:>8.0f}{f1v[i]:>10.4f}{mark}")

    out = ROOT / "outputs" / "reports" / "theta_analytic_optimum.json"
    out.write_text(json.dumps({
        "w_sweep_best": {"w": w_b, "theta_test": t_b, "f1_analytic": f1_b,
                         "claimed": cl_b, "tp": tp_b, "fp": fp_b, "fn": fn_b},
        "current_w1_theta_-5.6": {"claimed": cur_claimed, "tp": tp_cur,
                                  "fn": fn_cur, "f1_analytic": f1_cur,
                                  "actual_macro_f1": 0.7321},
        "w1_curve": [{"theta": float(t), "claimed": float(c), "tp": float(p),
                      "f1": float(f)} for t, c, p, f in zip(ths, claimed, tpv, f1v)],
        "offset": OFFSET,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[写出] {out}")


if __name__ == "__main__":
    main()
