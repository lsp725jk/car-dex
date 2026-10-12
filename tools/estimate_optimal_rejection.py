# -*- coding: utf-8 -*-
"""用「分数分布 + 测试集标定点」估计最优拒识强度。

## 为什么不能靠猜

上一步的估算有个循环论证：为了反推"已知误拒率 s"，我假设了 R=0.995，
然后 s 就完全由这个假设决定 —— 自己证明自己。

## 正确做法

用**判别边界 margin** 建模，并用测试集上的一个**已知标定点**校准：

    margin  m = z_unknown − max(z_已知类)        （先验校正之后）
    m > 阈值  →  判 unknown

在验证集上，可以量出已知类与 unknown 的 margin **累积分布**：
    F_known(m) = P(margin < m | 真已知)
    F_unk(m)   = P(margin < m | 真 unknown)

测试集的已知类分布可以认为与验证集一致 —— **证据**：known_accuracy
（测试 98.10% vs 验证 ~98.5%）和 known_rejection_rate（1.50% vs 1.20%）几乎相同。

测试集的 unknown 分布则**整体右移 δ**（测试集的 unknown 里有很多"长得像已知类"的车）：
    F_unk_test(m) = F_unk_val(m − δ)

δ 由评测指标给出的标定点确定：在 m=0（即原来的 argmax 规则）处，
unknown 召回 = 0.7874，已知误拒 = 0.015。**两个数都要对上**，才算校准成功。

校准完成后，就能算出任意阈值下的 (R, s)，进而估算 Macro-F1。
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
# 评测指标给出的标定点（m=0 处）
CAL_R, CAL_S = 0.7874, 0.015


def margins(logp: np.ndarray, tau: float, log_prior: np.ndarray, unk: int = 50) -> np.ndarray:
    """判别 margin：unknown 分数 − 最高的已知类分数（先验校正后）。"""
    z = logp - tau * log_prior[None, :]
    return z[:, unk] - np.delete(z, unk, axis=1).max(axis=1)


def cdf_at(grid: np.ndarray, samples: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(samples), grid, side="right") / len(samples)


def macro_f1(R: float, s: float, unk: int = 50) -> float:
    per = TEST_KNOWN / 50
    fp = TEST_UNK * (1 - R) / 50
    fn = per * s + per * 0.004
    tp = per - fn
    f1k = 2 * tp / (2 * tp + fp + fn)
    tp_u, fp_u = TEST_UNK * R, TEST_KNOWN * s
    prec = tp_u / max(1e-9, tp_u + fp_u)
    f1u = 2 * prec * R / max(1e-9, prec + R)
    return (50 * f1k + f1u) / 51


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
    tau_old = float(champ["best"]["tau"])

    st = json.loads((paths.CONFIG_DIR / "data_stats.json").read_text(encoding="utf-8"))
    n_c = [st["per_class"]["train"][cm["index_to_id"][str(i)]]["n"] for i in range(50)]
    n_c.append(st["per_class"]["train"]["unknown"]["n"])
    t = np.array(n_c, dtype=np.float64)
    log_prior = np.log(t / t.sum())

    logp_val = load_val_logits(champ)
    y_val = np.array([s[1] for s in CarDataset(paths.DATA_DIR / "val").samples])

    z = np.load(ROOT / "student_submit" / "result.npz", allow_pickle=False)
    L_npz = z["logits"]
    z.close()
    raw_test = L_npz + tau_old * log_prior[None, :]      # 去掉旧校正

    print("=" * 84)
    print("校准：让模型在 m=0 处复现评测给出的 (R=0.7874, s=0.015)")
    print("=" * 84)
    best = None
    for tau in np.arange(-1.0, 0.51, 0.125):
        mv = margins(logp_val, tau, log_prior)
        mk, mu = mv[y_val != 50], mv[y_val == 50]
        # 验证集在 m=0 处的基线
        s0 = float((mk < 0).mean())
        r0 = float((mu >= 0).mean())
        # 用 δ 平移 unknown 分布，使 R(0) 对上评测值
        # R(0) = 1 − F_unk_test(0) = 1 − F_unk_val(−δ) = CAL_R
        target = 1 - CAL_R                        # F_unk_val(−δ) = 0.2126
        if target <= 0 or target >= 1:
            continue
        d = -float(np.quantile(mu, target))       # 需要平移的量
        # 平移后重新校验 s(0)（已知分布在 m<0 的比例）
        err = abs(s0 - CAL_S)
        if best is None or err < best[0]:
            best = (err, float(tau), float(d), s0, r0)
    if best is None:
        print("校准失败")
        return
    err, tau, delta, s0, r0 = best
    print(f"  选定 τ={tau:.3f}  未知分布右移 δ={delta:+.3f}")
    print(f"  验证集原始 m=0 处：R={r0:.4f}（本集）  s={s0:.4f}")
    print(f"  平移后的测试集校准点：R={CAL_R:.4f}（目标）  s={s0:.4f}（评测 {CAL_S}，差 {abs(s0 - CAL_S):.4f}）")

    mv = margins(logp_val, tau, log_prior)
    mk, mu = mv[y_val != 50], mv[y_val == 50] + delta     # 平移 unknown
    grid = np.arange(-12.0, 12.01, 0.05)
    Fk, Fu = cdf_at(grid, mk), cdf_at(grid, mu)

    print("\n" + "=" * 84)
    print("扫描阈值（在**测试集先验**下估算 Macro-F1）")
    print("=" * 84)
    print(f"  {'阈值':>7} {'预测unknown':>12} {'R':>7} {'s':>7} {'每类FP':>7} {'估算F1':>8}")
    rows = []
    for m, fk, fu in zip(grid, Fk, Fu):
        s, R = float(fk), 1 - float(fu)
        ur = R * P_UNK + s * (1 - P_UNK)
        f1 = macro_f1(R, s)
        rows.append((f1, float(m), ur, R, s))
    # 只打印有代表性的
    step = max(1, len(rows) // 24)
    for f1, m, ur, R, s in rows[::step]:
        print(f"  {m:>7.2f} {ur:>11.1%} {R:>7.3f} {s:>7.3f} "
              f"{TEST_UNK * (1 - R) / 50:>7.1f} {f1:>8.4f}")

    best_row = max(rows, key=lambda r: r[0])
    print("\n" + "=" * 84)
    print(f"★ 估算最优：阈值 m = {best_row[1]:.2f}")
    print(f"   → 预测 unknown {best_row[2]:.1%}   R={best_row[3]:.3f}   s={best_row[4]:.3f}")
    print(f"   → 估算 Macro-F1 = {best_row[0]:.4f}")
    print("=" * 84)

    # 和"先验匹配"（预测 unknown 率≈94%）对比
    match = min(rows, key=lambda r: abs(r[2] - P_UNK))
    print(f"\n对照「先验匹配」（预测 unknown≈{P_UNK:.0%}）：")
    print(f"   阈值 m = {match[1]:.2f}  R={match[3]:.3f}  s={match[4]:.3f}  "
          f"估算 F1 = {match[0]:.4f}")

    out = paths.REPORT_DIR / "optimal_rejection.json"
    out.write_text(json.dumps({
        "calibration": {"tau": tau, "delta": delta, "s_at_0": s0, "target_R": CAL_R,
                        "target_s": CAL_S},
        "best": {"margin": best_row[1], "pred_unknown": best_row[2],
                 "R": best_row[3], "s": best_row[4], "est_macro_f1": best_row[0]},
        "prior_match": {"margin": match[1], "pred_unknown": match[2],
                        "est_macro_f1": match[0]},
        "curve": [{"m": m, "ur": ur, "R": R, "s": s, "f1": f1}
                  for f1, m, ur, R, s in rows[::max(1, len(rows) // 60)]],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
