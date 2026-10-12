# -*- coding: utf-8 -*-
"""决定拒识该多激进。

## 核心认识：这个任务里「漏放」远比「错拒」昂贵

Macro-F1 里每个已知类：F1 = 2TP/(2TP + FP + FN)。
测试集里每类 TP 只有约 20 张，而 unknown 有 15,540 张。所以：

  · 每 **1% 的 unknown 泄露** → 155 张假阳性 → 摊到 50 类 = 每类 +3.1 FP
    → F1_known 从 1.00 掉到 0.90
  · 每 **1% 的已知误拒** → 每类少 0.2 张 TP → F1_known 几乎不变

**结论：宁可多拒，不可漏放。** 这条不对称性在 unknown 占 94% 的测试集上被极度放大。

## 但也不能无限拒

拒得越狠，TP 越少。下面是"完全按先验匹配"（预测 unknown 率 = 真实 94%）时
的收益估算，以及几个候选档位的对比。

因为不知道测试集的真实 ROC，这里给两个口径：
  A. **先验匹配**：让预测的 unknown 比例等于反推的真实比例（94%）
  B. **val 最优**：在加权 val 上扫出来的最优点（乐观，因为 val 的 unknown 更容易）
真实效果大概率落在两者之间偏 A。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from cardex import paths  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEST_KNOWN, TEST_UNK = 1000, 15540
P_UNK = TEST_UNK / (TEST_KNOWN + TEST_UNK)


def estimate_macro_f1(R: float, s: float, unk: int = 50) -> tuple[float, dict]:
    """给定 (unknown 召回 R, 已知误拒率 s)，估算 51 类 Macro-F1。

    假设：
      · 真已知每类 20 张，互混率约 0.4%（由 known_accuracy 98.1% 与误拒 1.5% 反推）
      · 泄露的 unknown 平均摊到 50 个类
    """
    per_class_true = TEST_KNOWN / 50                       # 20
    leak = TEST_UNK * (1 - R)
    fp_per_class = leak / 50
    fn_per_class = per_class_true * s + per_class_true * 0.004
    tp_per_class = per_class_true - fn_per_class
    f1_known = 2 * tp_per_class / (2 * tp_per_class + fp_per_class + fn_per_class)
    # unknown 类
    tp_u = TEST_UNK * R
    fp_u = TEST_KNOWN * s
    prec = tp_u / max(1e-9, tp_u + fp_u)
    f1_unk = 2 * prec * R / max(1e-9, prec + R)
    macro = (50 * f1_known + f1_unk) / 51
    return macro, {"f1_known": f1_known, "f1_unknown": f1_unk,
                   "leak": leak, "fp_per_class": fp_per_class}


def main() -> None:
    cm = paths.load_class_map()
    classes = [cm["index_to_id"][str(i)] for i in range(50)]
    st = json.loads((paths.CONFIG_DIR / "data_stats.json").read_text(encoding="utf-8"))
    n_c = [st["per_class"]["train"][cm["index_to_id"][str(i)]]["n"] for i in range(50)]
    n_c.append(st["per_class"]["train"]["unknown"]["n"])
    t = np.array(n_c, dtype=np.float64)
    log_prior = np.log(t / t.sum())

    champ = json.loads((paths.REPORT_DIR / "final_model_choice.json").read_text(encoding="utf-8"))["champion"]
    tau_old = float(champ["best"]["tau"])
    thr_old = float(champ["best"]["threshold"])

    z = np.load(ROOT / "student_submit" / "result.npz", allow_pickle=False)
    L, ids = z["logits"], z["image_id"]
    z.close()
    raw = L + tau_old * log_prior[None, :]          # 还原成原始集成分数

    def probs(tau):
        zz = raw - tau * log_prior[None, :]
        zz = zz - zz.max(1, keepdims=True)
        p = np.exp(zz)
        return p / p.sum(1, keepdims=True)

    print("=" * 84)
    print("按目标「预测 unknown 率」扫描，找合适的工作点")
    print("=" * 84)
    print(f"  真实 unknown 占比（反推）= {P_UNK:.1%}")
    print(f"\n  {'τ':>6} {'t':>6} {'预测unknown':>12} {'已知预测':>9} "
          f"{'对应R*':>8} {'对应s*':>8} {'估算F1':>8}")

    rows = []
    for tau in (-0.25, -0.5, -0.75, -1.0):
        p = probs(tau)
        pm = p.max(1)
        for target in (0.86, 0.88, 0.90, 0.92, 0.94, 0.96):
            th = float(np.quantile(pm, target))     # 使 (1-target) 的样本被拒
            pred = np.where(pm < th, 50, p.argmax(1))
            ur = float((pred == 50).mean())
            # 反推 R、s：ur = R·P_UNK + s·(1−P_UNK)
            # 用 val 上"给定预测 unknown 率时的 R/s 比例"作为形状假设
            # 这里取保守估计：R = min(0.99, (ur − 0.02·(1−P_UNK)) / P_UNK)
            R = min(0.995, max(0.80, (ur - 0.02 * (1 - P_UNK)) / P_UNK))
            s = max(0.0, (ur - R * P_UNK) / (1 - P_UNK))
            f1, det = estimate_macro_f1(R, s)
            rows.append((f1, tau, th, ur, R, s))
            print(f"  {tau:>6.2f} {th:>6.3f} {ur:>11.1%} {int((pred != 50).sum()):>9} "
                  f"{R:>8.3f} {s:>8.3f} {f1:>8.4f}")

    rows.sort(reverse=True)
    print("\n" + "=" * 84)
    print("推荐的 5 个候选")
    print("=" * 84)
    for f1, tau, th, ur, R, s in rows[:5]:
        print(f"  τ={tau:.2f} t={th:.3f}  预测 unknown {ur:.1%}  估算 F1 = {f1:.4f}")

    print("\n" + "=" * 84)
    print("对照：当前提交")
    print("=" * 84)
    f1_now, det_now = estimate_macro_f1(0.7874, 0.015)
    print(f"  τ=0.2 t=0.0  预测 unknown 74.1%  实测 F1 = 0.4785"
          f"   本模型估算 {f1_now:.4f}（偏乐观，因为假设了理想混合）")
    print(f"  估算的偏差系数 ≈ {0.4785 / f1_now:.3f}")

    best = rows[0]
    est_simple = best[0] * (0.4785 / f1_now)
    print(f"\n  推荐方案 τ={best[1]:.2f} t={best[2]:.3f}：")
    print(f"     乐观估算 {best[0]:.4f}  按偏差系数折算 ≈ {est_simple:.4f}")

    out = paths.REPORT_DIR / "aggressiveness_choice.json"
    out.write_text(json.dumps({
        "p_unknown": P_UNK,
        "current": {"tau": tau_old, "threshold": thr_old, "pred_unknown": 0.741,
                    "actual_macro_f1": 0.47850900161045806},
        "recommended": {"tau": best[1], "threshold": best[2], "pred_unknown": best[3],
                        "est_f1_optimistic": best[0], "est_f1_adjusted": est_simple},
        "candidates": [{"tau": a, "t": b, "ur": c, "est": d} for d, a, b, c, _, _ in rows[:10]],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
