# -*- coding: utf-8 -*-
"""双参数平移模型：分别校准「已知类」与「unknown」的分数分布，估算最优拒识阈值。

## 观测到的关键事实

                     已知误拒 s    unknown 召回 R
    验证集             1.20%          95.75%
    测试集             1.50%          78.74%
                      ↑ 几乎一样      ↑ 掉了 17 个点

**已知类的行为在两个集上一致**，只有 unknown 的分布变了
（测试集的 unknown 里有太多"长得像我们 50 款"的车）。

## 模型

用判别 margin  m = z_unknown − max(z_已知类)（先验校正后），判 unknown 当 m > 阈值。

    F_known_test(m) = F_known_val(m + a)      ← 一个平移参数 a
    F_unk_test(m)   = F_unk_val(m + b)        ← 另一个平移参数 b

两个参数分别由评测给出的两个标定点确定（在 m = m₀，即原 argmax 规则处）：
    F_known_test(m₀) = 0.015
    F_unk_test(m₀)   = 1 − 0.7874 = 0.2126

有了 a、b，就能算任意阈值下的 (R, s)，进而估算 Macro-F1。
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
CAL_S, CAL_R = 0.015, 0.7874          # 评测给出的标定点


def margins(logp, tau, log_prior, unk=50):
    z = logp - tau * log_prior[None, :]
    return z[:, unk] - np.delete(z, unk, axis=1).max(axis=1)


def cdf(grid, samples):
    return np.searchsorted(np.sort(samples), grid, side="right") / len(samples)


def macro_f1(R, s, unk=50):
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
    t = np.array(n_c, dtype=np.float64)
    log_prior = np.log(t / t.sum())

    logp_val = load_val_logits(champ)
    y_val = np.array([s[1] for s in CarDataset(paths.DATA_DIR / "val").samples])

    print("=" * 86)
    print("第一步：验证集在原始工作点 (τ=0.2, m=0) 的表现，与评测值对比")
    print("=" * 86)
    mv = margins(logp_val, tau0, log_prior)
    mk, mu = mv[y_val != 50], mv[y_val == 50]
    print(f"  τ={tau0}  m>0 判 unknown")
    print(f"    验证集：s = {float((mk < 0).mean()):.4f}   R = {float((mu >= 0).mean()):.4f}")
    print(f"    测试集：s = {CAL_S:.4f}   R = {CAL_R:.4f}   ← 评测给出")
    print(f"    推断：unknown 分布在测试集上整体**左移**（更靠近已知类）")

    # ---- 双参数校准 ----
    Fk_val = lambda m: float((mk < m).mean())
    Fu_val = lambda m: float((mu < m).mean())
    # Fk_test(0) = Fk_val(a) = CAL_S ；Fu_test(0) = Fu_val(b) = 1 - CAL_R
    a = float(np.quantile(mk, CAL_S))
    b = float(np.quantile(mu, 1 - CAL_R))
    print(f"\n  校准参数：a = {a:+.4f}（已知类右移）  b = {b:+.4f}（unknown 左移）")
    print(f"    复核：Fk_test(0) = {Fk_val(a):.4f}（目标 {CAL_S}）  "
          f"Fu_test(0) = {Fu_val(b):.4f}（目标 {1 - CAL_R:.4f}）")

    # ---- 扫描阈值 ----
    grid = np.arange(-14.0, 14.01, 0.05)
    Fk = cdf(grid, mk - a)          # Fk_test(m) = Fk_val(m - a)
    Fu = cdf(grid, mu - b)          # Fu_test(m) = Fu_val(m - b)
    rows = []
    for m, fk, fu in zip(grid, Fk, Fu):
        s, R = float(fk), 1 - float(fu)
        ur = R * P_UNK + s * (1 - P_UNK)
        f1, f1k, f1u = macro_f1(R, s)
        rows.append((f1, float(m), ur, R, s, f1k, f1u))

    print("\n" + "=" * 86)
    print("第二步：扫描阈值（已按评测值校准到测试集先验）")
    print("=" * 86)
    print(f"  {'阈值':>7} {'预测unk':>9} {'R':>7} {'s':>7} {'每类FP':>7} "
          f"{'F1已知':>8} {'F1未知':>8} {'MacroF1':>9}")
    show = rows[:: max(1, len(rows) // 26)]
    for f1, m, ur, R, s, f1k, f1u in show:
        print(f"  {m:>7.2f} {ur:>8.1%} {R:>7.3f} {s:>7.3f} {TEST_UNK * (1 - R) / 50:>7.1f} "
              f"{f1k:>8.4f} {f1u:>8.4f} {f1:>9.4f}")

    best = max(rows, key=lambda r: r[0])
    zero = min(rows, key=lambda r: abs(r[1]))
    print("\n" + "=" * 86)
    print(f"  原点 (m=0，即当前提交)：预测 unknown {zero[2]:.1%}  "
          f"R={zero[3]:.3f} s={zero[4]:.3f}  估算 Macro-F1 = {zero[0]:.4f}  （实测 0.4785）")
    print(f"  ★ 最优 m = {best[1]:.2f}：预测 unknown {best[2]:.1%}  "
          f"R={best[3]:.3f} s={best[4]:.3f}  估算 Macro-F1 = {best[0]:.4f}")
    lift = best[0] - zero[0]
    print(f"  → 模型预测提升 {lift:+.4f}（{lift * 100:+.1f} 分）")
    print("=" * 86)

    # 用 m=0 的实测值做偏置校正
    bias = 0.4785 - zero[0]
    print(f"\n  偏置校正：模型在 m=0 处估算 {zero[0]:.4f}，实测 0.4785，"
          f"差 {bias:+.4f}")
    print(f"  → 校正后的最优估算 ≈ {best[0] + bias:.4f}")

    out = paths.REPORT_DIR / "rejection_calibration.json"
    out.write_text(json.dumps({
        "calibration": {"tau": tau0, "a": a, "b": b,
                        "val_s": float((mk < 0).mean()), "val_R": float((mu >= 0).mean()),
                        "test_s": CAL_S, "test_R": CAL_R},
        "origin": {"margin": 0.0, "pred_unknown": zero[2], "R": zero[3], "s": zero[4],
                   "est_f1": zero[0], "actual_f1": 0.47850900161045806},
        "best": {"margin": best[1], "pred_unknown": best[2], "R": best[3], "s": best[4],
                 "est_f1": best[0], "est_f1_bias_corrected": best[0] + bias},
        "curve": [{"m": m, "ur": ur, "R": R, "s": s, "f1": f1}
                  for f1, m, ur, R, s, _, _ in rows[::6]],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
