# -*- coding: utf-8 -*-
"""实测用户提出的方案：读 result.npz → 转 logits → 阈值 0.75 后处理。

同时给出**可达上限**：在诚实 val 上把决策规则扫到极限，看 macro-F1 最高能到多少。
用这个回答「能不能到 0.85」。

## 关于「转 logits」

`result.npz` 的 51 列**本来就是 log-softmax**（加 θ 偏置前每行 logsumexp = 0），
即已经是对数概率。再套一层 log 会得到 log(log(p)) —— 无意义且负值变 NaN。
所以这一步实际是什么都不用做。

## 关于「阈值 0.75」

这是 max-prob 置信度阈值，与项目里已有的 margin 规则是**不同的决策规则**，
必须实测才知道好坏。这里把两种解释都测：
  · 对 51 类取 max（含 unknown 列）
  · 对 50 个已知类取 max（不含 unknown 列）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "outputs" / "reports" / "ood_val_logits" / "hflip_multicrop_k1250_u1250.npz"
N_KNOWN_TEST, N_UNKNOWN_TEST = 1000.0, 15540.0
OFFSET = 0.18          # 两锚点标定：test θ = val θ + 0.18（multicrop 口径）


def project(margin_val, is_unk, margin_test, n_test):
    """用诚实 val 的 KR/UR 曲线投影到测试集先验，返回逐 θ 的 macro-F1。"""
    ...


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(VAL))
    ap.add_argument("--test-npz", default=str(ROOT / "student_submit" / "result.npz"))
    ap.add_argument("--current-theta", type=float, default=-4.8)
    ap.add_argument("--threshold", type=float, default=0.75)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    Lv, yv = z["logits"].astype(np.float64), z["labels"]
    z.close()
    is_unk = yv == 50
    n_k, n_u = float((~is_unk).sum()), float(is_unk.sum())
    alpha, beta = N_KNOWN_TEST / n_k, N_UNKNOWN_TEST / n_u

    # ── 验证「npz 里是不是概率」──
    print("=" * 80)
    print("① 核查：result.npz 里到底是概率还是 logits？")
    print("=" * 80)
    zt = np.load(args.test_npz, allow_pickle=False)
    Lt = zt["logits"].astype(np.float64)
    zt.close()
    L0 = Lt.copy()
    L0[:, 50] += args.current_theta          # 反推 θ=0
    lse = np.log(np.exp(L0 - L0.max(1, keepdims=True)).sum(1)) + L0.max(1)
    print(f"  θ=0 时每行 logsumexp：均值 {lse.mean():+.4f}  标准差 {lse.std():.4f}")
    print(f"  取值范围 [{L0.min():.3f}, {L0.max():.3f}]   全部 ≤0：{bool((L0 <= 1e-6).all())}")
    if abs(lse.mean()) < 1e-3 and (L0 <= 1e-6).all():
        print("  → **是 log-softmax（对数概率），不是概率。**\n"
              "     再做一次 log 会得到 log(log(p))，且负值直接 NaN。这一步不需要做。")
    probs_t = np.exp(L0)
    print(f"  exp() 之后：每行和 = {probs_t.sum(1).mean():.6f}（应为 1），"
          f"范围 [{probs_t.min():.2e}, {probs_t.max():.3f}]")

    # ── ② 实测阈值 0.75 规则 ──
    print("\n" + "=" * 80)
    print(f"② 实测「阈值 {args.threshold}」规则（在诚实 val 上）")
    print("=" * 80)
    pv = np.exp(Lv)
    for name, conf in (("对 51 类取 max（含 unknown）", pv.max(1)),
                       ("对 50 个已知类取 max", pv[:, :50].max(1))):
        rej = conf < args.threshold
        ur = float((rej & is_unk).sum() / n_u)
        kr = float((rej & ~is_unk).sum() / n_k)
        claimed = float(((~rej) & ~is_unk).sum() * alpha + ((~rej) & is_unk).sum() * beta)
        tp = float(((~rej) & ~is_unk).sum() * alpha)
        f1 = 2 * tp / (claimed + N_KNOWN_TEST) if claimed > 0 else 0.0
        print(f"\n  {name}：")
        print(f"    判未知 {rej.mean():.2%}   unknown_recall {ur:.2%}   "
              f"known_accuracy {1 - kr:.2%}")
        print(f"    投影到测试集：声称已知 {claimed:.0f} 张   TP≈{tp:.0f}   "
              f"解析 macro-F1 ≈ {f1:.4f}")

    # ── ③ 同样的规则，但把阈值扫一遍（万一是 0.75 这个数取得不好）──
    print(f"\n{'=' * 80}")
    print("③ 阈值扫描：这是规则本身的问题，还是 0.75 取得不好？")
    print("=" * 80)
    conf = pv[:, :50].max(1)
    margin = Lv[:, 50] - Lv[:, :50].max(1)
    rows = []
    for th in np.arange(0.30, 0.99, 0.02):
        rej = conf < th
        n_claim_k = ((~rej) & ~is_unk).sum() * alpha
        n_claim_u = ((~rej) & is_unk).sum() * beta
        claimed = n_claim_k + n_claim_u
        if claimed <= 0:
            continue
        f1 = 2 * n_claim_k / (claimed + N_KNOWN_TEST)
        rows.append((float(th), float((rej & is_unk).sum() / n_u),
                     float((rej & ~is_unk).sum() / n_k), float(f1)))
    best_th = max(rows, key=lambda r: r[3])
    print(f"{'阈值':>7}{'unknown_recall':>16}{'known_accuracy':>16}{'投影 macro-F1':>16}")
    for i, (th, ur, kr, f1) in enumerate(rows):
        # ⚠️ 不能用 abs(th*100) % 6 过滤 —— 浮点下 0.30*100 = 30.000000000000004，
        #    取模几乎永不命中，整张表只剩一行。改用整数步长。
        if i % 3:
            continue
        mark = "  ← 最优" if abs(th - best_th[0]) < 1e-9 else (
            "  ← 你提的 0.75" if abs(th - args.threshold) < 1e-9 else "")
        print(f"{th:>7.2f}{ur:>16.2%}{1 - kr:>16.2%}{f1:>16.4f}{mark}")
    hit = [r for r in rows if abs(r[0] - args.threshold) < 1e-9]
    if not hit:
        near = min(rows, key=lambda r: abs(r[0] - args.threshold))
        print(f"{args.threshold:>7.2f}{near[1]:>16.2%}{1 - near[2]:>16.2%}"
              f"{near[3]:>16.4f}  ← 你提的 0.75（最近点 {near[0]:.2f}）")
    print(f"\n  阈值规则的最优：阈值 {best_th[0]:.2f} → 投影 macro-F1 {best_th[3]:.4f}")

    # ── ④ 可达上限：margin 规则扫到极限 ──
    print(f"\n{'=' * 80}")
    print("④ 可达上限：现用 margin 规则扫到极限（这才是天花板）")
    print("=" * 80)
    mrows = []
    for th in np.arange(-8.0, 1.01, 0.05):
        rej = margin > th
        n_claim_k = ((~rej) & ~is_unk).sum() * alpha
        n_claim_u = ((~rej) & is_unk).sum() * beta
        claimed = n_claim_k + n_claim_u
        if claimed <= 0:
            continue
        f1 = 2 * n_claim_k / (claimed + N_KNOWN_TEST)
        mrows.append((float(th), float((rej & is_unk).sum() / n_u),
                      float((rej & ~is_unk).sum() / n_k), float(f1), claimed))
    best_m = max(mrows, key=lambda r: r[3])
    print(f"{'θ(val)':>8}{'θ(test)':>9}{'unknown_recall':>16}{'known_accuracy':>16}"
          f"{'投影 macro-F1':>16}{'声称已知':>10}")
    for i, (th, ur, kr, f1, cl) in enumerate(mrows):
        if i % 4:
            continue
        mark = ""
        if abs(th - best_m[0]) < 1e-9:
            mark = "  ← 全局最优"
        elif abs(th - (args.current_theta - OFFSET)) < 0.03:
            mark = "  ← 当前 (−4.8)"
        print(f"{th:>8.2f}{th + OFFSET:>9.2f}{ur:>16.2%}{1 - kr:>16.2%}"
              f"{f1:>16.4f}{cl:>10.0f}{mark}")
    print(f"\n  全局最优：θ(test)={best_m[0] + OFFSET:+.2f}  →  "
          f"投影 macro-F1 = {best_m[3]:.4f}")

    print("\n" + "=" * 80)
    print("⑤ 回答「能到 0.85 吗」")
    print("=" * 80)
    gap = 0.85 - best_m[3]
    th075 = ([r[3] for r in rows if abs(r[0] - args.threshold) < 1e-9]
             or [min(rows, key=lambda r: abs(r[0] - args.threshold))[3]])
    print(f"  当前方法可达上限（投影）  ：{best_m[3]:.4f}")
    print(f"  阈值 0.75 方案（投影）    ：{th075[0]:.4f}")
    print(f"  目标                      ：0.8500")
    print(f"  缺口                      ：{gap:+.4f}")
    print(f"\n  → {'可以' if gap <= 0 else '**到不了**'}。"
          f"缺口 {gap:+.4f} 相当于还差 {gap / best_m[3] * 100:.0f}% 的相对提升，"
          f"而这只能靠更好的检测器（重训），后处理已经榨干。")

    out = ROOT / "outputs" / "reports" / "threshold_075_evaluation.json"
    out.write_text(json.dumps({
        "npz_is_log_softmax": bool(abs(lse.mean()) < 1e-3),
        "threshold_rule_best": {"threshold": best_th[0], "projected_macro_f1": best_th[3]},
        "margin_rule_best": {"theta_test": best_m[0] + OFFSET, "projected_macro_f1": best_m[3],
                             "unknown_recall": best_m[1], "known_accuracy": 1 - best_m[2],
                             "claimed": best_m[4]},
        "target": 0.85, "gap": gap,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[写出] {out}")


if __name__ == "__main__":
    main()
