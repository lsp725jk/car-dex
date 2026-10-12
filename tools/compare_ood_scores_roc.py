# -*- coding: utf-8 -*-
"""比较多种 OOD 打分的 ROC —— 回答「能否在保住 unknown_recall 的同时提高 known_accuracy」。

## 为什么这是唯一有意义的方向

    known_accuracy = 1 − known_rejection_rate   （实测：模型从不把已知类认成另一个已知类）

所以 known_accuracy 和 unknown_recall 是同一条 ROC 曲线上的两个点，**换阈值是零和的**。
想两头都变好，只有一条路：**换一个 ROC 更优（曲线更靠左上）的打分函数**。

## 好消息：不需要重新推理

所有候选打分都只是那 51 列 log-softmax 的不同组合，
而 16,540×51 的测试 logits 已经在 `result.npz` 里了 → 换打分是**纯 numpy 操作**。

## 判据

固定 unknown_recall = 98%（当前水平），看哪个打分的 known_accuracy 最高。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "outputs" / "reports" / "ood_val_logits" / "hflip_multicrop_k1250_u1250.npz"
TARGET_UR = 0.98


def build_scores(L: np.ndarray) -> dict[str, np.ndarray]:
    """从 (N,51) 的 log-softmax 构造候选打分。分数越大 → 越像 unknown。"""
    unk = L[:, 50]
    kn = L[:, :50]
    kmax = kn.max(1)
    # 已知类的总证据（log 域求和）
    ksum = np.log(np.exp(kn - kn.max(1, keepdims=True)).sum(1)) + kn.max(1)
    # 已知类的熵（大 → 模型在已知类之间犹豫）
    p = np.exp(kn - ksum[:, None])
    kent = -(p * np.log(p + 1e-12)).sum(1)
    # top1 − top2 间隔（小 → 两个已知类在打架）
    srt = np.sort(kn, axis=1)
    gap = srt[:, -1] - srt[:, -2]

    out = {
        "margin_max（现用）": unk - kmax,
        "margin_sum（对比总证据）": unk - ksum,
        "z_unknown（只看未知列）": unk,
        "neg_max_known（只看已知置信度）": -kmax,
        "known_entropy（已知类熵）": kent,
        "neg_top1top2_gap（间隔取负）": -gap,
    }
    # 线性组合：在 max 与 sum 两个口径之间插值
    for lam in (0.25, 0.5, 1.0, 2.0):
        out[f"margin_max + {lam}·margin_sum"] = (unk - kmax) + lam * (unk - ksum)
    # 组合：margin 与 熵
    for lam in (0.5, 1.0, 2.0):
        out[f"margin_max + {lam}·known_entropy"] = (unk - kmax) + lam * kent
    return out


def roc_at(scores: np.ndarray, is_unk: np.ndarray, target_ur: float) -> tuple[float, float]:
    """在 unknown_recall 固定为 target_ur 时，返回 (known_rejection_rate, 实际UR)。"""
    n_u = is_unk.sum()
    # 阈值取 unknown 分数的分位数，使 UR 恰好 = target
    th = np.quantile(scores[is_unk], 1 - target_ur)
    pu = scores > th
    ur = (pu & is_unk).sum() / n_u
    kr = (pu & ~is_unk).sum() / (~is_unk).sum()
    return float(kr), float(ur)


def auc(scores: np.ndarray, is_unk: np.ndarray) -> float:
    """秩和法算 AUC（大样本友好）。"""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_u = int(is_unk.sum()); n_k = len(scores) - n_u
    return float((ranks[is_unk].sum() - n_u * (n_u + 1) / 2) / (n_u * n_k))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(VAL))
    ap.add_argument("--bootstrap", type=int, default=400)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    L, y = z["logits"].astype(np.float64), z["labels"]
    z.close()
    is_unk = y == 50
    print(f"验证集：{Path(args.npz).name}")
    print(f"诚实 OOD val：{len(y)} 张（已知 {int((~is_unk).sum())} / 未知 {int(is_unk.sum())}）")
    print(f"固定 unknown_recall = {TARGET_UR:.0%}，比较各打分的 known_accuracy\n")

    sc = build_scores(L)
    rows = []
    for name, s in sc.items():
        kr, ur = roc_at(s, is_unk, TARGET_UR)
        rows.append({"score": name, "auc": auc(s, is_unk), "kr": kr, "ur": ur,
                     "known_accuracy": 1 - kr})
    rows.sort(key=lambda r: r["known_accuracy"], reverse=True)

    print(f"{'打分函数':<34}{'AUC':>8}{'UR':>9}{'KR':>9}{'known_accuracy':>16}")
    print("-" * 78)
    base = next(r for r in rows if r["score"].startswith("margin_max（"))
    for r in rows:
        mark = ""
        if r is rows[0]:
            mark = "  ← 最好"
        elif r["score"] == base["score"]:
            mark = "  ← 现用"
        gain = r["known_accuracy"] - base["known_accuracy"]
        gs = f"{gain:+.2%}" if r["score"] != base["score"] else ""
        print(f"{r['score']:<34}{r['auc']:>8.4f}{r['ur']:>9.2%}{r['kr']:>9.2%}"
              f"{r['known_accuracy']:>16.2%} {gs}{mark}")

    print(f"\n现用打分在 UR={TARGET_UR:.0%} 时 known_accuracy = {base['known_accuracy']:.2%}")
    print(f"最优打分                       = {rows[0]['known_accuracy']:.2%}"
          f"   （提升 {rows[0]['known_accuracy'] - base['known_accuracy']:+.2%}）")

    # 顺便看几个 UR 档位下的表现
    print("\n各 UR 档位下的 known_accuracy（看最优打分是否稳定占优）：")
    print(f"{'UR 目标':>9}{'现用':>12}{'最优':>12}{'差':>10}")
    for tur in (0.94, 0.96, 0.97, 0.98, 0.99):
        b_kr, b_ur = roc_at(sc[base["score"]], is_unk, tur)
        t_kr, t_ur = roc_at(sc[rows[0]["score"]], is_unk, tur)
        print(f"{tur:>9.0%}{1 - b_kr:>12.2%}{1 - t_kr:>12.2%}"
              f"{(1 - t_kr) - (1 - b_kr):>+10.2%}")

    # ── bootstrap：最优与现用的差是否显著？ ──
    rng = np.random.default_rng(20250922)
    n = len(is_unk)
    diffs = []
    s_best, s_base = sc[rows[0]["score"]], sc[base["score"]]
    for _ in range(args.bootstrap):
        idx = rng.integers(0, n, n)
        iu = is_unk[idx]
        if iu.sum() < 50 or (~iu).sum() < 50:
            continue
        b_kr, _ = roc_at(s_base[idx], iu, TARGET_UR)
        t_kr, _ = roc_at(s_best[idx], iu, TARGET_UR)
        diffs.append((1 - t_kr) - (1 - b_kr))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\nbootstrap（{len(diffs)} 次重采样）最优−现用的 known_accuracy 差：")
    print(f"  均值 {diffs.mean():+.2%}   95% 置信区间 [{lo:+.2%}, {hi:+.2%}]")
    print(f"  → {'❌ 区间跨 0，提升不显著（在噪声内）' if lo < 0 < hi else '✅ 提升显著'}")

    out = ROOT / "outputs" / "reports" / "ood_score_roc_comparison.json"
    out.write_text(json.dumps({"val_npz": str(args.npz), "target_ur": TARGET_UR,
                               "rows": rows, "baseline": base["score"], "best": rows[0]["score"],
                               "bootstrap_diff_mean": float(diffs.mean()),
                               "bootstrap_ci95": [float(lo), float(hi)]},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[写出] {out}")


if __name__ == "__main__":
    main()
