# -*- coding: utf-8 -*-
"""错误结构分析：搞清楚 0.9646 的分数**卡在哪里**，别瞎调参。

回答三个问题：
  1. unknown 类拖了多少后腿？把它做到完美能加几分？
  2. 哪些已知类在互相混淆？（高 precision + 低 recall = 漏判）
  3. **人类视角的地板在哪** —— 有多少张图是"连人都难分"的？
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402

from cardex import paths  # noqa: E402
from cardex.metrics import compute_metrics  # noqa: E402


def load_champion_logprobs() -> np.ndarray:
    """从 finalize 缓存里重建冠军集成的 log-softmax（几何平均）。"""
    champ = json.loads((paths.REPORT_DIR / "final_model_choice.json").read_text(encoding="utf-8"))["champion"]
    cache = paths.RUN_DIR / "finalize_cache"
    import hashlib
    lps = []
    for m in champ["members"]:
        ck = paths.CKPT_DIR / m / "best.pt"
        st = ck.stat()
        fp = hashlib.md5(f"{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:10]
        lps.append(np.load(cache / f"logits_{m}_val_tta_{fp}.npy"))
    return np.mean(lps, axis=0), champ


def main() -> None:
    cm = paths.load_class_map()
    # index_to_name 只有 0..49 这 50 个已知类，unknown（索引 50）是单独字段
    names = [cm["index_to_name"][str(i)] for i in range(50)] + [cm["unknown_label"]]
    y_true = np.load(paths.RUN_DIR / "finalize_cache" / "y_true_val.npy") \
        if (paths.RUN_DIR / "finalize_cache" / "y_true_val.npy").exists() else None

    logp, champ = load_champion_logprobs()
    print(f"冠军：{champ['tag']}")
    print(f"logprobs 形状 {logp.shape}")
    tau, thr = float(champ["best"]["tau"]), float(champ["best"]["threshold"])

    # 真实标签：按 CarDataset.samples 的顺序（和 logits 的行一一对应）
    from cardex.data import CarDataset, build_transform
    ds = CarDataset(paths.DATA_DIR / "val", transform=build_transform("eval", 224))
    y_true = np.array([s[1] for s in ds.samples])
    print(f"验证集 {len(y_true)} 张")
    assert len(y_true) == logp.shape[0], f"标签数 {len(y_true)} 与 logits 行数 {logp.shape[0]} 不一致"

    # 先验校正 + 拒识
    stats = json.loads((paths.CONFIG_DIR / "data_stats.json").read_text(encoding="utf-8"))
    n_c = [stats["per_class"]["train"][cm["index_to_id"][str(i)]]["n"] for i in range(50)]
    n_c.append(stats["per_class"]["train"]["unknown"]["n"])
    t = np.array(n_c, dtype=np.float64)
    log_prior = np.log(t / t.sum())

    z = logp - tau * log_prior[None, :]
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    pred = p.argmax(axis=1)
    if thr > 0:
        pred = np.where(p.max(axis=1) < thr, 50, pred)

    m = compute_metrics(y_true, pred, names)
    base = m["macro_f1"]
    print(f"\n当前 Macro-F1 = {base:.4f}")

    f1s = np.array([c["f1"] for c in m["per_class"]])
    print(f"\n{'=' * 78}\n[1] unknown 的天花板\n{'=' * 78}")
    print(f"  unknown F1 = {f1s[50]:.4f}（验证集 588 张，占 24%）")
    gain_unk = (1.0 - f1s[50]) / 51
    print(f"  就算 unknown 做到完美 F1=1.0，也只加 {gain_unk:+.4f} → {base + gain_unk:.4f}")
    print(f"  **结论：unknown 不是主要瓶颈，别把力气全花在拒识上**")

    print(f"\n{'=' * 78}\n[2] 已知类的地板\n{'=' * 78}")
    known_f1 = f1s[:50]
    print(f"  50 个已知类：最低 {known_f1.min():.4f}  中位 {np.median(known_f1):.4f}  最高 {known_f1.max():.4f}")
    print(f"  完美的 50 类 + 现在的 unknown → {(50 + f1s[50]) / 51:.4f}")
    print(f"  完美的全部 51 类 → 1.0000")

    # 各档位提升能带来多少
    print(f"\n  把最低 N 个类提升到 X 的收益：")
    for n in (5, 10, 15, 20):
        idx = np.argsort(known_f1)[:n]
        for target in (0.95, 0.98, 1.0):
            g = sum(max(0.0, target - known_f1[i]) for i in idx) / 51
            print(f"    最低 {n:2} 个类 → {target:.2f}：{g:+.4f}  → {base + g:.4f}")

    print(f"\n{'=' * 78}\n[3] 混淆结构：谁被认成了谁\n{'=' * 78}")
    cmx = np.array(m["confusion_matrix"])
    off = cmx.copy()
    np.fill_diagonal(off, 0)
    pairs = []
    for i in range(51):
        for j in range(51):
            if off[i, j] > 0:
                pairs.append((off[i, j], names[i], names[j], i, j))
    pairs.sort(reverse=True)
    print(f"  {'真类 → 预测类':<44} {'张数':>4}")
    for n, a, b, i, j in pairs[:18]:
        tag = "  ← unknown 未拒识" if j == 50 else ("  ← 被误拒成 unknown" if i == 50 else "")
        print(f"  {a + ' → ' + b:<44} {n:>4}{tag}")

    # 拒绝相关的损失
    print(f"\n{'=' * 78}\n[4] 拒识带来的两笔账\n{'=' * 78}")
    rej_true = (y_true != 50) & (pred == 50)
    unk_miss = (y_true == 50) & (pred != 50)
    print(f"  已知类被误拒（本该认出来却拒了）：{rej_true.sum()} 张")
    print(f"  unknown 没拒掉（瞎认成某个车型）  ：{unk_miss.sum()} 张")
    print(f"  理想情况两者都是 0 → unknown F1 = 1.0，收益 {gain_unk:+.4f}")
    # 如果完全不拒识（thr=0）会怎样
    pred_no = logp.argmax(axis=1)
    m2 = compute_metrics(y_true, pred_no, names)
    print(f"\n  参考：完全不拒识（t=0）Macro-F1 = {m2['macro_f1']:.4f}")
    best = (0, base, -1)
    for tt in np.arange(0.0, 0.61, 0.02):
        pr = np.where(p.max(axis=1) < tt, 50, p.argmax(axis=1))
        s = compute_metrics(y_true, pr, names)["macro_f1"]
        if s > best[1]:
            best = (tt, s, 0)
    print(f"  当前 τ={tau} 下最好的 t = {best[0]:.2f} → {best[1]:.4f}（现在用的是 {thr}）")


if __name__ == "__main__":
    main()
