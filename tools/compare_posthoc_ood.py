# -*- coding: utf-8 -*-
"""对比「事后再造 OOD 打分」与现有 unknown 头。

## 候选打分

  1. unknown_head       z_unknown − max(z_已知)      ← 现用（原始 logits 下与 log-softmax 下等价）
  2. energy             logsumexp(z_已知50类)         ← 经典 Energy OOD，需要原始 logits
  3. msp                max softmax(50 个已知类)
  4. proto_cos          到 50 个类原型的最大余弦相似度
  5. knn1_cos           到最近训练样本的余弦相似度
  6. mahalanobis        到最近类均值的马氏距离（**带收缩**）
  7. 以上两两线性组合

## 判据

固定 unknown_recall，比 known_accuracy；配 bootstrap 置信区间。
只报「显著」的赢家，避免把噪声当成改进。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "outputs" / "reports" / "ood_features"
TARGET_UR = 0.98


def l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def build_scores(L: np.ndarray, F: np.ndarray,
                 Ft: np.ndarray, yt: np.ndarray) -> dict[str, np.ndarray]:
    """L:(N,51) 原始 logits  F:(N,M,512) 特征  Ft:(T,M,512) 训练特征  yt:(T,) 标签。"""
    unk = L[:, 50]
    kn = L[:, :50]
    out: dict[str, np.ndarray] = {}

    # 1. 现用：unknown 头（原始 logits 下 lse 相消，与 log-softmax 口径一致）
    out["unknown_head（现用）"] = unk - kn.max(1)

    # 2. Energy：logsumexp over 50 known
    m = kn.max(1, keepdims=True)
    energy = (np.log(np.exp(kn - m).sum(1)) + m[:, 0])
    out["energy（logsumexp 已知50类）"] = -energy          # 分数越大越像 OOD

    # 3. MSP：50 个已知类的最大 softmax
    p = np.exp(kn - energy[:, None])
    out["msp（最大已知类 softmax）"] = -p.max(1)

    N, M, D = F.shape
    # ── 逐成员算相似度再平均（集成的正确做法）──
    # ⚠️ 特征维度是 2048，而训练样本只有 ~2000 个 —— 样本协方差**秩 ≤ 1999 < 2048，直接奇异**。
    #    所以 Mahalanobis 必须先在 PCA 空间里做（降到几百维），否则算出来的是正则项的假象。
    NPC = 256
    proto = np.zeros((M, 50, D), dtype=np.float64)
    for j in range(M):
        for c in range(50):
            sel = yt == c
            if sel.sum():
                proto[j, c] = l2n(Ft[sel, j].astype(np.float64).mean(0))
    Ft_n = l2n(Ft.astype(np.float64))                      # (T,M,D)

    cos_proto = np.zeros((N, M)); cos_knn1 = np.zeros((N, M)); maha = np.zeros((N, M))
    for j in range(M):
        # ⚠️ F 的形状是 (N, M, D)：F[:, j] 才是「第 j 个成员、全部 N 个样本」。
        #    写成 F[j] 会取到第 j 个样本的 (M, D)，静默broadcast成错的形状。
        Fn = l2n(F[:, j].astype(np.float64))               # (N,D)
        cos_proto[:, j] = (Fn @ proto[j].T).max(1)         # 到最近类原型
        cos_knn1[:, j] = (Fn @ Ft_n[:, j].T).max(1)        # 到最近训练样本

        # ---- PCA 降维后再算 Mahalanobis ----
        Xt = Ft[:, j].astype(np.float64)
        mean = Xt.mean(0)
        Xc = Xt - mean
        # 用 SVD 取前 NPC 个主成分
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        V = Vt[:NPC].T                                     # (D, NPC)
        Zt, Zv = Xc @ V, (F[:, j].astype(np.float64) - mean) @ V
        means = np.zeros((50, NPC))
        for c in range(50):
            sel = yt == c
            if sel.sum():
                means[c] = Zt[sel].mean(0)
        Zc = Zt - means[yt]
        cov = (Zc.T @ Zc) / max(1, len(Zc) - 1)
        shrink = 0.1
        cov = (1 - shrink) * cov + shrink * np.diag(np.diag(cov))
        cov += 1e-6 * np.eye(NPC)
        try:
            Lc = np.linalg.cholesky(cov)
            diff = (Zv[:, None, :] - means[None, :, :]).reshape(-1, NPC)
            sol = np.linalg.solve(Lc, diff.T).T
            dist = (sol ** 2).sum(-1).reshape(N, 50)       # 沿 PCA 维度求和 = 马氏距离平方
        except np.linalg.LinAlgError:
            dist = ((Zv[:, None, :] - means[None, :, :]) ** 2).sum(-1)
        # 分数越大越像 OOD ⇒ 距离**越大**越像 OOD，所以取正号。
        # ⚠️ 写成 -dist 会让整个打分反向（AUC 变成 1−真值），
        #    在 ROC 表里表现为「AUC≈0.05、known_accuracy=0%」这种一眼假的结果。
        maha[:, j] = dist.min(1)

    out["proto_cos（类原型余弦）"] = -cos_proto.mean(1)
    out["knn1_cos（最近样本余弦）"] = -cos_knn1.mean(1)
    out["mahalanobis（收缩协方差）"] = maha.mean(1)

    # 7. 组合：energy 与 unknown 头、proto 与 unknown 头
    def zs(x):
        return (x - x.mean()) / (x.std() + 1e-12)
    out["unknown_head + energy"] = zs(out["unknown_head（现用）"]) + zs(out["energy（logsumexp 已知50类）"])
    out["unknown_head + proto_cos"] = zs(out["unknown_head（现用）"]) + zs(out["proto_cos（类原型余弦）"])
    out["unknown_head + mahalanobis"] = zs(out["unknown_head（现用）"]) + zs(out["mahalanobis（收缩协方差）"])
    out["unknown_head + knn1_cos"] = zs(out["unknown_head（现用）"]) + zs(out["knn1_cos（最近样本余弦）"])
    out["energy + proto_cos"] = zs(out["energy（logsumexp 已知50类）"]) + zs(out["proto_cos（类原型余弦）"])
    out["三者之和"] = (zs(out["unknown_head（现用）"]) + zs(out["energy（logsumexp 已知50类）"])
                   + zs(out["proto_cos（类原型余弦）"]))
    out["四者之和"] = (zs(out["unknown_head（现用）"]) + zs(out["energy（logsumexp 已知50类）"])
                   + zs(out["proto_cos（类原型余弦）"]) + zs(out["mahalanobis（收缩协方差）"]))
    return out


def roc_at(s: np.ndarray, is_unk: np.ndarray, target_ur: float):
    th = np.quantile(s[is_unk], 1 - target_ur)
    pu = s > th
    return (float((pu & ~is_unk).sum() / (~is_unk).sum()),
            float((pu & is_unk).sum() / is_unk.sum()))


def auc(s: np.ndarray, is_unk: np.ndarray) -> float:
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    nu = int(is_unk.sum()); nk = len(s) - nu
    return float((ranks[is_unk].sum() - nu * (nu + 1) / 2) / (nu * nk))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(FEAT_DIR))
    ap.add_argument("--bootstrap", type=int, default=400)
    args = ap.parse_args()
    d = Path(args.dir)

    zv = np.load(d / "honest_val.npz", allow_pickle=False)
    L, F, y = zv["logits"].astype(np.float64), zv["feats"].astype(np.float64), zv["labels"]
    zv.close()
    zt = np.load(d / "train_known.npz", allow_pickle=False)
    Ft, yt = zt["feats"].astype(np.float64), zt["labels"]
    zt.close()
    is_unk = y == 50
    print(f"诚实 val {len(y)} 张（已知 {int((~is_unk).sum())} / 未知 {int(is_unk.sum())}）")
    print(f"训练特征 {len(yt)} 张；特征维度 {F.shape[2]}；集成成员 {F.shape[1]}\n")

    sc = build_scores(L, F, Ft, yt)
    rows = []
    for k, s in sc.items():
        kr, ur = roc_at(s, is_unk, TARGET_UR)
        rows.append({"score": k, "auc": auc(s, is_unk), "kr": kr, "ur": ur,
                     "known_accuracy": 1 - kr})
    rows.sort(key=lambda r: r["known_accuracy"], reverse=True)
    base = next(r for r in rows if r["score"].startswith("unknown_head（"))

    print(f"固定 unknown_recall = {TARGET_UR:.0%}")
    print(f"{'打分':<30}{'AUC':>9}{'UR':>9}{'KR':>9}{'known_accuracy':>16}{'vs 现用':>11}")
    print("-" * 86)
    for r in rows:
        mark = "  ← 现用" if r["score"] == base["score"] else ""
        gain = r["known_accuracy"] - base["known_accuracy"]
        gs = "" if not mark else ""
        print(f"{r['score']:<30}{r['auc']:>9.4f}{r['ur']:>9.2%}{r['kr']:>9.2%}"
              f"{r['known_accuracy']:>16.2%}{gain:>+11.2%}{mark}{gs}")

    # ── bootstrap：只对打败现用的候选做显著性检验 ──
    rng = np.random.default_rng(20250922)
    n = len(is_unk)
    print(f"\nbootstrap（{args.bootstrap} 次）——只有置信区间不跨 0 才算真赢：")
    winners = []
    for r in rows:
        if r["score"] == base["score"]:
            continue
        # ⚠️ 变量名要与含义一致：s_cand 是候选、s_ref 是现用。
        #    第一版写成 s_b, s_c = 候选, 现用 但按 (基线 − 候选) 算，
        #    结果整个 bootstrap 表的正负号都是反的，很容易误读成「组合变差了」。
        s_cand, s_ref = sc[r["score"]], sc[base["score"]]
        diffs = []
        for _ in range(args.bootstrap):
            idx = rng.integers(0, n, n)
            iu = is_unk[idx]
            if iu.sum() < 50 or (~iu).sum() < 50:
                continue
            ref_kr, _ = roc_at(s_ref[idx], iu, TARGET_UR)
            cand_kr, _ = roc_at(s_cand[idx], iu, TARGET_UR)
            diffs.append((1 - cand_kr) - (1 - ref_kr))
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        sig = "✅ 显著" if lo > 0 else "❌ 在噪声内"
        print(f"  {r['score']:<30} 差 {diffs.mean():+.2%}  "
              f"95%CI [{lo:+.2%}, {hi:+.2%}]  {sig}")
        if lo > 0:
            winners.append(r["score"])

    print(f"\n显著优于 unknown 头的打分：{winners if winners else '（无）'}")

    # ── 多档位一致性：真提升应当在所有 UR 上都占优，而不只是某一个点 ──
    print("\n各 UR 档位下的 known_accuracy（候选 vs 现用）：")
    cand_names = [r["score"] for r in rows[:3] if r["score"] != base["score"]]
    hdr = "".join(f"{c[:16]:>18}" for c in cand_names)
    print(f"{'UR':>8}{'现用':>12}{hdr}")
    for tur in (0.94, 0.96, 0.97, 0.98, 0.99):
        b_kr, _ = roc_at(sc[base["score"]], is_unk, tur)
        line = f"{tur:>8.0%}{1 - b_kr:>12.2%}"
        for c in cand_names:
            c_kr, _ = roc_at(sc[c], is_unk, tur)
            line += f"{1 - c_kr:>12.2%}({(1 - c_kr) - (1 - b_kr):+.1%})"
        print(line)

    out = d / "score_comparison.json"
    out.write_text(json.dumps({"target_ur": TARGET_UR, "rows": rows,
                               "significant_winners": winners},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[写出] {out}")


if __name__ == "__main__":
    main()
