# -*- coding: utf-8 -*-
"""诊断：为什么 val 0.9845 却只拿到 48 分？并算出修复方案。

## 诊断结论（由评测指标反推）

设 p = 测试集中真·已知类的比例：
    accuracy = p × known_accuracy + (1 − p) × unknown_recall
    0.799093 = p × 0.981 + (1 − p) × 0.787387   →   p = 0.0605

即测试集只有 **6.05% 是已知类**，93.95% 是 unknown。
而我们的预测是 25.9% 已知 / 74.1% unknown —— **多认了 3,289 张**。

## 为什么这会毁掉 Macro-F1

Macro-F1 里每一类的 F1 是 2TP/(2TP+FP+FN)。关键在于 **FP 与 TP 的相对量级**：

    val （unknown:known = 1:3.1）
        unknown 泄露率 r → 每类 FP = 588r/50 ≈ 11.8r，每类 TP ≈ 36
        r=0.21 时 F1 ≈ 0.97          ← 看起来很好

    test（unknown:known = 1:15.5）
        同一个 r → 每类 FP = 15540r/50 ≈ 311r，每类 TP ≈ 20
        r=0.21 时 F1 ≈ 0.37          ← 崩了

**同样的模型、同样的泄露率，只因 unknown:known 比例从 1:3 变成 1:15，Macro-F1 就从 0.97 掉到 0.37。**

所以：**在 val（24% unknown）上标定出的最优阈值，对这个测试集是完全错的。**

## 修复思路

不能直接拿测试集调阈值（没有标签），但可以**用验证集模拟测试集的先验**：
给 val 里的 unknown 样本乘一个权重 W，使加权的 unknown:known 比例等于测试集的比例：

    W = (15540/1000) / (588/1834) = 48.5

然后在**加权 F1** 上重扫 (τ, t)。这样就等价于"在测试集的类别先验下重新标定"。
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


def weighted_f1(y_true, y_pred, w, n_class=51):
    """加权 Macro-F1：每个样本带权重，等价于改变类别的相对数量。"""
    f1s = []
    for c in range(n_class):
        t = y_true == c
        p = y_pred == c
        tp = float(w[t & p].sum())
        fp = float(w[~t & p].sum())
        fn = float(w[t & ~p].sum())
        if tp + fp + fn == 0:
            f1s.append(0.0)                      # 该类完全没出现，记 0（与官方口径一致）
        else:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)), f1s


def load_champion_val_logits(champ: dict) -> np.ndarray:
    """取冠军在验证集上的 log-softmax（未做先验校正的原始值）。

    优先用 generalization_cache 里现成的集成结果；
    没有就用 finalize_cache 里各成员的缓存**现场重建**——融合方式与
    `src/finalize.py` 完全一致（各成员 log-softmax 取平均），
    口径不能自己另发明一套。
    """
    cache = paths.RUN_DIR / "generalization_cache"
    tag = "champ_" + "_".join(champ["members"]) + "_hflipmulticrop.npy"
    f = cache / tag
    if f.exists():
        print(f"载入集成缓存 {f.name}")
        return np.load(f)

    fc = paths.RUN_DIR / "finalize_cache"
    import hashlib
    lps = []
    for m in champ["members"]:
        ck = paths.CKPT_DIR / m / "best.pt"
        st = ck.stat()
        fp = hashlib.md5(f"{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:10]
        p = fc / f"logits_{m}_val_ttamc_{fp}.npy"
        if not p.exists():
            raise FileNotFoundError(f"缺少成员缓存：{p.name}")
        print(f"  成员 {m:<24} {p.name}")
        lps.append(np.load(p))
    return np.mean(lps, axis=0)


def main() -> None:
    champ = json.loads((paths.REPORT_DIR / "final_model_choice.json").read_text(encoding="utf-8"))["champion"]
    logp = load_champion_val_logits(champ)
    print(f"验证集 logits {logp.shape}")

    cm = paths.load_class_map()
    names = [cm["index_to_name"][str(i)] for i in range(50)] + [cm["unknown_label"]]
    ds = CarDataset(paths.DATA_DIR / "val")
    y = np.array([s[1] for s in ds.samples])
    print(f"验证集 {len(y)} 张：已知 {int((y != 50).sum())} / unknown {int((y == 50).sum())}")

    # 先验
    st = json.loads((paths.CONFIG_DIR / "data_stats.json").read_text(encoding="utf-8"))
    n_c = [st["per_class"]["train"][cm["index_to_id"][str(i)]]["n"] for i in range(50)]
    n_c.append(st["per_class"]["train"]["unknown"]["n"])
    t = np.array(n_c, dtype=np.float64)
    log_prior = np.log(t / t.sum())

    # ---- 权重：把 val 的 unknown:known 拉到测试集的比例 ----
    n_known_val = int((y != 50).sum())
    n_unk_val = int((y == 50).sum())
    TEST_KNOWN, TEST_UNK = 1000, 15540          # 由评测指标反推
    W = (TEST_UNK / TEST_KNOWN) / (n_unk_val / n_known_val)
    w = np.where(y == 50, W, 1.0)
    print(f"\n权重 W={W:.1f}（让 val 的 unknown:known 比例 = {TEST_UNK / TEST_KNOWN:.1f}，"
          f"实际 val 是 {n_unk_val / n_known_val:.2f}）")

    # ---- 当前配置的表现 ----
    print("\n" + "=" * 78)
    print("当前配置 t=0.0 在两种先验下的对比")
    print("=" * 78)
    cur_tau, cur_t = float(champ["best"]["tau"]), float(champ["best"]["threshold"])
    z = logp - cur_tau * log_prior[None, :]
    pred = z.argmax(1)
    f1_plain, _ = weighted_f1(y, pred, np.ones_like(w))
    f1_sim, _ = weighted_f1(y, pred, w)
    print(f"  τ={cur_tau} t={cur_t}：原始 val F1 = {f1_plain:.4f}（我们报的 0.98）")
    print(f"  同一组预测、换成测试集先验：模拟 F1 = {f1_sim:.4f}  ← 官方给的是 0.4785")
    print(f"  unknown 召回 {float((pred[y == 50] == 50).mean()):.4f}"
          f"  已知误拒 {float((pred[y != 50] == 50).mean()):.4f}")

    # ---- 在模拟先验下重扫阈值 ----
    print("\n" + "=" * 78)
    print("在【测试集先验】下重新标定 (τ, t)")
    print("=" * 78)
    best = (-1.0, 0.0, 0.0)
    rows = []
    for tau in np.arange(-2.0, 3.01, 0.25):
        zz = logp - tau * log_prior[None, :]
        zz = zz - zz.max(1, keepdims=True)
        p = np.exp(zz)
        p /= p.sum(1, keepdims=True)
        pm, pa = p.max(1), p.argmax(1)
        for th in np.arange(0.0, 0.96, 0.02):
            pr = np.where(pm < th, 50, pa)
            f1, _ = weighted_f1(y, pr, w)
            rows.append((f1, float(tau), float(th)))
            if f1 > best[0]:
                best = (f1, float(tau), float(th))
    rows.sort(reverse=True)
    print(f"  {'模拟 F1':>8} {'τ':>6} {'t':>6}   未知预测率  未知召回  已知误拒")
    for f1, tau, th in rows[:12]:
        zz = logp - tau * log_prior[None, :]
        zz = zz - zz.max(1, keepdims=True)
        p = np.exp(zz)
        p /= p.sum(1, keepdims=True)
        pr = np.where(p.max(1) < th, 50, p.argmax(1))
        ur = float((pr == 50).mean())
        urec = float((pr[y == 50] == 50).mean())
        krej = float((pr[y != 50] == 50).mean())
        print(f"  {f1:>8.4f} {tau:>6.2f} {th:>6.2f}   {ur:>8.1%} {urec:>9.1%} {krej:>9.1%}")

    print(f"\n★ 测试集先验下的最优：τ={best[1]:.2f}  t={best[2]:.2f}  模拟 F1 = {best[0]:.4f}")
    print(f"  （当前 τ={cur_tau} t={cur_t} 只有 {f1_sim:.4f} → 预计提升 "
          f"{(best[0] - f1_sim) * 100:.1f} 分）")

    out = ROOT / "outputs" / "reports" / "prior_shift_fix.json"
    out.write_text(json.dumps({
        "diagnosis": {
            "test_known_ratio": TEST_KNOWN / (TEST_KNOWN + TEST_UNK),
            "val_known_ratio": n_known_val / len(y),
            "weight_W": W,
            "current_sim_f1": f1_sim,
            "official_macro_f1": 0.47850900161045806,
        },
        "recommended": {"tau": best[1], "threshold": best[2], "sim_f1": best[0]},
        "top10": [{"sim_f1": a, "tau": b, "t": c} for a, b, c in rows[:10]],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
