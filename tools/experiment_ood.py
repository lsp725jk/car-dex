# -*- coding: utf-8 -*-
"""OOD 检测方法对比：max-softmax vs Energy vs Mahalanobis。

## 背景：为什么"拒识"需要一个专门的分数

我们的任务有两件事要同时做好：
  ① **分类**：把图片分到 51 个类之一；
  ② **拒识**：判断"这张图根本不属于这 51 类"。

第 ② 件事在学术上叫 **OOD 检测（Out-of-Distribution Detection）**。
它和分类是**两个不同的问题** —— 一个模型可以分类很准，但完全不会拒识。

## 三种打分方式

**[1] Max-Softmax（我们原来用的）**
    score = max_c softmax(z)_c ，低于阈值就拒识。
    这是最朴素的做法。**已知有明显缺陷**：softmax 是归一化的，
    即使所有 logit 都很小（模型对每一类都没信心），归一化之后仍可能出现
    一个"相对最高"的概率，于是 OOD 样本被误判为已知类。

**[2] Energy（本篇重点）**
    E(x) = −T · logsumexp(z / T)
    由 Liu et al. 2020 (NeurIPS) 提出。**关键区别：它不做归一化**，
    直接用 logits 的绝对量纲 —— 模型见过的东西 logits 大、能量低；
    没见过的东西 logits 小、能量高。理论上比 max-softmax 更适合 OOD。

**[3] Mahalanobis（特征空间方法）**
    在**倒数第二层的特征空间**里，为每个类拟合一个高斯分布，
    取"到最近类中心的马氏距离"作为 OOD 分数。
    它利用的是**特征空间的几何结构**，和上面两个只看输出层的方法互补。

## 怎么比较才公平

不能只比"谁能把 unknown 挑出来"（那可以靠调阈值作弊），
要比**最终 51 类 Macro-F1** —— 也就是分类和拒识**合起来**的效果。
所以三种方法各自扫自己的阈值，取最优 Macro-F1。

另外报告 **AUROC**（把已知类当正类、unknown 当负类的可分性），
这个指标与阈值无关，反映"这个分数本身有多好"。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from cardex import paths  # noqa: E402
from cardex.data import CarDataset, build_transform  # noqa: E402
from cardex.inference import CarEnsemble  # noqa: E402
from cardex.metrics import compute_metrics  # noqa: E402
from cardex.utils import autocast_dtype, get_device  # noqa: E402

CACHE = paths.RUN_DIR / "ood_cache"


def class_names():
    cm = paths.load_class_map()
    return [cm["index_to_name"][str(i)] for i in range(50)] + [cm["unknown_label"]]


def prior_log():
    cm = paths.load_class_map()
    st = json.loads((paths.CONFIG_DIR / "data_stats.json").read_text(encoding="utf-8"))
    n = [st["per_class"]["train"][cm["index_to_id"][str(i)]]["n"] for i in range(50)]
    n.append(st["per_class"]["train"]["unknown"]["n"])
    t = np.array(n, dtype=np.float64)
    return np.log(t / t.sum())


# --------------------------------------------------------------------------- #
# 1. 取原始 logits + 特征
# --------------------------------------------------------------------------- #
def collect(members, tta_mode, batch, device, split="val", limit: int | None = None):
    """返回 (raw_logits, features, y)。都是 (N, ...) 的 numpy。

    `limit` 限制**每类取多少张** —— 拟合 Mahalanobis 的类均值和共享协方差
    不需要全量训练集（每类 40 张、共约 2000 张已经够稳），
    但能省下大量推理时间。验证集必须全量（要算指标）。
    """
    tag = f"{split}_{'_'.join(members)}_{tta_mode.replace('+', '')}" + (f"_l{limit}" if limit else "")
    fl, ff, fy = CACHE / f"{tag}_logits.npy", CACHE / f"{tag}_feat.npy", CACHE / f"{tag}_y.npy"
    if fl.exists() and ff.exists() and fy.exists():
        return np.load(fl), np.load(ff), np.load(fy)
    CACHE.mkdir(parents=True, exist_ok=True)

    ds = CarDataset(paths.DATA_DIR / split,
                    limit_per_class=limit, seed=42)
    y = np.array([s[1] for s in ds.samples])
    ens = CarEnsemble([paths.CKPT_DIR / m / "best.pt" for m in members],
                      tta_mode=tta_mode, amp=True, prior_adjust_tau=0.0,
                      unknown_threshold=0.0, device=device)
    print(f"  取 {split} 的 logits + 特征（{len(ds.samples)} 张，{tta_mode}"
          + (f"，每类限 {limit} 张" if limit else "") + "）…")
    L, F = [], []
    for i in range(0, len(ds.samples), batch):
        chunk = ds.samples[i : i + batch]
        imgs = []
        for p, _ in chunk:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB").copy())
        with torch.no_grad():
            ls, fs = [], []
            for mem in ens.members:
                # 原始 logits（Energy 需要量纲）
                ls.append(mem.tta_logits(imgs, return_raw=True))
                # 倒数第二层特征（Mahalanobis 需要）
                fs.append(_features(mem, imgs))
            L.append(torch.stack(ls).mean(0).float().cpu().numpy())
            F.append(torch.stack(fs).mean(0).float().cpu().numpy())
        if i % 400 == 0:
            print(f"    {i}/{len(ds.samples)}")
    logits, feat = np.concatenate(L), np.concatenate(F)
    np.save(fl, logits), np.save(ff, feat), np.save(fy, y)
    print(f"    logits {logits.shape}  features {feat.shape}")
    return logits, feat, y


@torch.no_grad()
def _features(mem, imgs) -> torch.Tensor:
    """取骨干最后一层池化后的特征（ResNet 是 avgpool 的输出，2048 维）。"""
    x = torch.stack([mem.transform(im.convert("RGB")) for im in imgs]).to(mem.device)
    with torch.autocast(mem.device.type, dtype=autocast_dtype(mem.device), enabled=mem.amp):
        # 手工走一遍 forward 到 avgpool：nn.Sequential 之外的部分
        m = mem.model
        out = m.conv1(x); out = m.bn1(out); out = m.relu(out); out = m.maxpool(out)
        out = m.layer1(out); out = m.layer2(out); out = m.layer3(out); out = m.layer4(out)
        out = m.avgpool(out)
    return torch.flatten(out, 1).float()


# --------------------------------------------------------------------------- #
# 2. 三种 OOD 分数
# --------------------------------------------------------------------------- #
def score_maxsoftmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return p.max(axis=1)                      # 越**小**越像 OOD


def score_energy(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    m = (z / T).max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(z / T - m).sum(axis=1, keepdims=True))).ravel()
    return -T * lse                           # 越**大**越像 OOD


def fit_mahalanobis(feat_tr: np.ndarray, y_tr: np.ndarray, n_class: int = 51):
    """按类拟合高斯：共享协方差 + 每类均值。"""
    d = feat_tr.shape[1]
    mus = np.zeros((n_class, d), dtype=np.float64)
    for c in range(n_class):
        m = y_tr == c
        mus[c] = feat_tr[m].mean(0) if m.any() else feat_tr.mean(0)
    # 共享协方差（用残差估计）
    res = feat_tr.astype(np.float64) - mus[y_tr]
    cov = res.T @ res / max(1, len(feat_tr) - n_class)
    cov += np.eye(d) * 1e-3 * np.trace(cov) / d          # 抖动，保证可逆
    prec = np.linalg.inv(cov)
    return mus, prec


def score_mahalanobis(feat: np.ndarray, mus, prec) -> np.ndarray:
    """到**最近类中心**的马氏距离：越**大**越像 OOD。"""
    f = feat.astype(np.float64)
    # 展开 (x-mu)^T P (x-mu) = x^T P x - 2 mu^T P x + mu^T P mu
    Px = f @ prec
    xPx = np.einsum("ij,ij->i", Px, f)
    muPmu = np.einsum("ij,ij->i", mus @ prec, mus)
    muPx = f @ prec @ mus.T                       # (N, C)
    d2 = xPx[:, None] - 2 * muPx + muPmu[None, :]
    return d2.min(axis=1)


# --------------------------------------------------------------------------- #
# 3. 评估
# --------------------------------------------------------------------------- #
def evaluate(z, ood_score, higher_is_ood, y, names, log_prior, tau):
    """扫阈值，取最优 Macro-F1。返回 (f1, best_threshold, auroc)。"""
    zc = z - tau * log_prior[None, :]
    pred0 = zc.argmax(axis=1)
    lo, hi = float(np.min(ood_score)), float(np.max(ood_score))
    best = (-1.0, 0.0)
    for t in np.linspace(lo, hi, 160):
        rej = ood_score > t if higher_is_ood else ood_score < t
        pr = np.where(rej, 50, pred0)
        f = compute_metrics(y, pr, names)["macro_f1"]
        if f > best[0]:
            best = (f, float(t))
    # AUROC：已知类为正类。分数统一成"越大越像 OOD"
    s = ood_score if higher_is_ood else -ood_score
    is_ood = (y == 50).astype(int)
    auroc = float(roc_auc_score(is_ood, s))
    return best[0], best[1], auroc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="*", default=["r50_288_from_scratch", "r50_352"])
    ap.add_argument("--tta-mode", default="hflip+multicrop")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--train-per-class", type=int, default=40,
                    help="拟合 Mahalanobis 时每类用多少张训练图（不需要全量）")
    args = ap.parse_args()

    names = class_names()
    lp = prior_log()
    dev = get_device(args.device)
    print(f"成员 {args.members}   TTA {args.tta_mode}   设备 {dev}")

    print("\n=== 取数（验证集 + 训练集特征）===")
    z_val, f_val, y_val = collect(args.members, args.tta_mode, args.batch, dev, "val")
    z_tr, f_tr, y_tr = collect(args.members, args.tta_mode, args.batch, dev, "train",
                               limit=args.train_per_class)

    z = z_val - args.tau * lp[None, :]

    print("\n" + "=" * 84)
    print("[1] Max-Softmax（原方法）")
    print("=" * 84)
    s_ms = score_maxsoftmax(z)
    f1, t, auc = evaluate(z_val, s_ms, False, y_val, names, lp, args.tau)
    print(f"  Macro-F1 = {f1:.4f}   最优阈值 = {t:.4f}   AUROC = {auc:.4f}")

    print("\n" + "=" * 84)
    print("[2] Energy-based")
    print("=" * 84)
    energy_best = ("", -1.0, 0.0, 0.0, 1.0)
    for T in (0.5, 1.0, 2.0, 5.0):
        s_e = score_energy(z, T)
        f1e, te, auce = evaluate(z_val, s_e, True, y_val, names, lp, args.tau)
        print(f"  T={T:<4} Macro-F1 = {f1e:.4f}   阈值 = {te:+.3f}   AUROC = {auce:.4f}")
        if f1e > energy_best[1]:
            energy_best = (f"Energy T={T}", f1e, te, auce, T)

    print("\n" + "=" * 84)
    print("[3] Mahalanobis（特征空间）")
    print("=" * 84)
    print("  在训练集特征上拟合每类高斯…")
    mus, prec = fit_mahalanobis(f_tr, y_tr)
    s_m = score_mahalanobis(f_val, mus, prec)
    f1m, tm, aucm = evaluate(z_val, s_m, True, y_val, names, lp, args.tau)
    print(f"  Macro-F1 = {f1m:.4f}   阈值 = {tm:.1f}   AUROC = {aucm:.4f}")

    print("\n" + "=" * 84)
    print("[4] 融合：Energy 与 Mahalanobis 各自标准化后相加")
    print("=" * 84)
    def std(v):
        return (v - v.mean()) / (v.std() + 1e-9)
    s_fuse = std(score_energy(z, 1.0)) + std(s_m)
    f1f, tf, aucf = evaluate(z_val, s_fuse, True, y_val, names, lp, args.tau)
    print(f"  Macro-F1 = {f1f:.4f}   阈值 = {tf:+.3f}   AUROC = {aucf:.4f}")

    print("\n" + "=" * 84)
    print("汇总（按 Macro-F1 排序）")
    print("=" * 84)
    rows = [
        ("Max-Softmax（原方法）", f1, auc),
        (energy_best[0], energy_best[1], energy_best[3]),
        ("Mahalanobis", f1m, aucm),
        ("Energy + Mahalanobis", f1f, aucf),
    ]
    print(f"  {'方法':<24} {'Macro-F1':>10} {'AUROC':>9}")
    for name, f, a in sorted(rows, key=lambda r: -r[1]):
        flag = "  ★" if f == max(r[1] for r in rows) else ""
        print(f"  {name:<24} {f:>10.4f} {a:>9.4f}{flag}")

    best_name, best_f1 = max(rows, key=lambda r: r[1])[:2]
    print(f"\n  → 最优方案：{best_name}   Macro-F1 = {best_f1:.4f}"
          f"（原方法 {f1:.4f}，提升 {best_f1 - f1:+.4f}）")

    out = paths.REPORT_DIR / "ood_comparison.json"
    out.write_text(json.dumps({
        "max_softmax": {"macro_f1": f1, "threshold": t, "auroc": auc},
        "energy_best": {"tag": energy_best[0], "macro_f1": energy_best[1],
                        "threshold": energy_best[2], "auroc": energy_best[3],
                        "T": energy_best[4]},
        "mahalanobis": {"macro_f1": f1m, "threshold": tm, "auroc": aucm},
        "fused": {"macro_f1": f1f, "threshold": tf, "auroc": aucf},
        "best": {"name": best_name, "macro_f1": best_f1},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
