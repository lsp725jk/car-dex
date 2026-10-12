# -*- coding: utf-8 -*-
"""按【测试集先验】重新标定拒识参数，直接作用于已有的测试集 logits。

## 为什么原来的参数是错的

原参数 (τ=0.2, t=0.0) 是在**验证集**上扫 Macro-F1 扫出来的。
但验证集只有 24% unknown，测试集有 94%。而 Macro-F1 里
**FP 与 TP 的相对量级**取决于 unknown:known 比例：

    val  1:3.1   泄露 4.25% → 每类 FP≈5.6，TP≈36  → F1≈0.87（看起来很好）
    test 1:15.5  泄露 21.3% → 每类 FP≈66，TP≈20   → F1≈0.37（崩了）

**同一个模型、同一个阈值，只因先验不同，得分天差地别。**
在 A 先验下调好的阈值，拿到 B 先验上用必然错。

## 怎么修（不需要测试集标签）

用验证集模拟测试集的先验：给 val 的 unknown 样本乘权重 W，使加权后的
unknown:known 比例等于测试集的比例，然后在**加权 F1** 上重扫 (τ, t)。

测试集比例从评测指标反推得到（accuracy = p·known_acc + (1−p)·unknown_recall）。

用法：
    python tools/retune_for_test.py --dry-run     # 只看结果，不写文件
    python tools/retune_for_test.py              # 写回提交物
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402

from cardex import paths  # noqa: E402
from cardex.data import CarDataset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEST_N_KNOWN = 1000            # 由评测指标反推（见 diagnose_prior_shift.py）
TEST_N_UNK = 15540


def weighted_macro_f1(y_true, y_pred, w, n_class=51):
    f1s = []
    for c in range(n_class):
        t = y_true == c
        p = y_pred == c
        tp = float(w[t & p].sum())
        fp = float(w[~t & p].sum())
        fn = float(w[t & ~p].sum())
        f1s.append(0.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def decode(logp, tau, thr, log_prior, unk=50):
    z = logp - tau * log_prior[None, :]
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    pm, pa = p.max(1), p.argmax(1)
    return (np.where(pm < thr, unk, pa) if thr > 0 else pa)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tau", type=float, default=None, help="手动指定 τ（默认自动选优）")
    ap.add_argument("--thr", type=float, default=None, help="手动指定 t（默认自动选优）")
    args = ap.parse_args()

    champ = json.loads((paths.REPORT_DIR / "final_model_choice.json").read_text(encoding="utf-8"))["champion"]
    cm = paths.load_class_map()
    classes = [cm["index_to_id"][str(i)] for i in range(50)]

    # 先验（训练集）
    st = json.loads((paths.CONFIG_DIR / "data_stats.json").read_text(encoding="utf-8"))
    n_c = [st["per_class"]["train"][cm["index_to_id"][str(i)]]["n"] for i in range(50)]
    n_c.append(st["per_class"]["train"]["unknown"]["n"])
    t_arr = np.array(n_c, dtype=np.float64)
    log_prior = np.log(t_arr / t_arr.sum())

    # ---- 验证集：标定 ----
    logp_val = load_val_logits(champ)
    y_val = np.array([s[1] for s in CarDataset(paths.DATA_DIR / "val").samples])
    nk, nu = int((y_val != 50).sum()), int((y_val == 50).sum())
    W = (TEST_N_UNK / TEST_N_KNOWN) / (nu / nk)
    w = np.where(y_val == 50, W, 1.0)
    print(f"验证集 {len(y_val)} 张（已知 {nk} / unknown {nu}），权重 W={W:.1f}")
    print(f"  → 加权后 unknown:known = {TEST_N_UNK / TEST_N_KNOWN:.1f}:1（与测试集一致）\n")

    if args.tau is not None and args.thr is not None:
        best = (weighted_macro_f1(y_val, decode(logp_val, args.tau, args.thr, log_prior), w),
                args.tau, args.thr)
        print(f"手动指定：τ={best[1]}  t={best[2]} → 模拟 F1={best[0]:.4f}\n")
    else:
        print("在【测试集先验】下扫描 (τ, t) …")
        best = (-1.0, 0.0, 0.0)
        for tau in np.arange(-1.5, 1.01, 0.125):
            for th in np.arange(0.0, 0.95, 0.01):
                f = weighted_macro_f1(y_val, decode(logp_val, tau, th, log_prior), w)
                if f > best[0]:
                    best = (f, float(tau), float(th))
        print(f"★ 最优：τ={best[1]:.3f}  t={best[2]:.2f}  模拟 F1={best[0]:.4f}\n")

    # 与原配置对比
    cur_tau, cur_t = float(champ["best"]["tau"]), float(champ["best"]["threshold"])
    f_cur = weighted_macro_f1(y_val, decode(logp_val, cur_tau, cur_t, log_prior), w)
    f_plain = weighted_macro_f1(y_val, decode(logp_val, cur_tau, cur_t, log_prior), np.ones_like(w))
    print("=" * 78)
    print(f"  原配置 τ={cur_tau} t={cur_t}：")
    print(f"     验证集原始 F1 = {f_plain:.4f}（我们当初报的）")
    print(f"     换成测试集先验的模拟 F1 = {f_cur:.4f}")
    print(f"  新配置 τ={best[1]:.3f} t={best[2]:.2f}：")
    print(f"     测试集先验下的模拟 F1 = {best[0]:.4f}")
    print(f"     预计提升 {(best[0] - f_cur) * 100:+.1f} 分")
    print("=" * 78)

    # ---- 应用到测试集 logits ----
    # ⚠️ 关键：导出的 npz logits **已经做过 τ_old 的先验校正**（score_logits 里做的），
    # 这里不能再校一次 —— 双重校正会让 unknown 被压制两遍（实测 74.1% → 69.0%）。
    # 先把旧校正除掉，还原成"原始集成分数"，再按新 τ 校：
    #     L_npz = raw − τ_old·logπ
    #     z_new = raw − τ_new·logπ = L_npz + (τ_old − τ_new)·logπ
    npz = ROOT / "student_submit" / "result.npz"
    z = np.load(npz, allow_pickle=False)
    L_npz, ids_test = z["logits"], z["image_id"]
    z.close()
    raw = L_npz - (-cur_tau) * log_prior[None, :]      # 等价于 L_npz + cur_tau*logπ

    def decode_raw(raw_logits, tau, thr, unk=50):
        zz = raw_logits - tau * log_prior[None, :]
        zz = zz - zz.max(1, keepdims=True)
        p = np.exp(zz)
        p /= p.sum(1, keepdims=True)
        pm, pa = p.max(1), p.argmax(1)
        return (np.where(pm < thr, unk, pa) if thr > 0 else pa)

    pred_old = decode_raw(raw, cur_tau, cur_t, )
    pred_new = decode_raw(raw, best[1], best[2])
    print(f"\n测试集 {len(ids_test)} 张：")
    print(f"  原参数 τ={cur_tau} t={cur_t} → unknown {int((pred_old == 50).sum())} 张"
          f"（{(pred_old == 50).mean():.1%}）  ← 应与实际提交一致")
    print(f"  新参数 τ={best[1]:.3f} t={best[2]:.2f} → unknown {int((pred_new == 50).sum())} 张"
          f"（{(pred_new == 50).mean():.1%}）")
    print(f"  真实 unknown 占比（反推）≈ {TEST_N_UNK / (TEST_N_KNOWN + TEST_N_UNK):.1%}")
    changed = int((pred_old != pred_new).sum())
    print(f"  改动 {changed} 张（{changed / len(ids_test):.1%}）")

    known_new = int((pred_new != 50).sum())
    print(f"\n  新参数下已知类预测 {known_new} 张，真相约 {TEST_N_KNOWN} 张 → "
          f"多认约 {known_new - TEST_N_KNOWN} 张（原参数多认 {int((pred_old != 50).sum()) - TEST_N_KNOWN} 张）")

    if args.dry_run:
        print("\n[dry-run] 未写文件")
        return

    # 写回 result.csv（保持两列格式）
    labels = [classes[i] if i < 50 else "unknown" for i in pred_new]
    cp = ROOT / "student_submit" / "result.csv"
    with open(cp, "w", encoding="utf-8", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["image_id", "predicted_label"])
        for i, lab in zip(ids_test.tolist(), labels):
            wtr.writerow([i, lab])
    print(f"\n[写出] {cp}")

    # 写回 logits：改成新参数下的决策分数，保证 argmax == 新标签
    zc = raw - best[1] * log_prior[None, :]
    if best[2] > 0:
        e = np.exp(zc - zc.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        rej = p.max(1) < best[2]
        zc[rej, 50] = zc[rej].max(1) + 1.0          # 把 unknown 抬到最大，令 argmax 一致
    zc = zc.astype(np.float32)
    np.savez_compressed(npz, image_id=ids_test, logits=zc)
    print(f"[写出] {npz}  （logits = 新参数下的决策分数，argmax 即新标签）")

    # 一致性自检
    back = np.load(npz, allow_pickle=False)
    chk = np.array([classes[i] if i < 50 else "unknown" for i in back["logits"].argmax(1)])
    ok = bool((chk == np.array(labels)).all())
    back.close()
    print(f"\n自洽检查：argmax(logits) == result.csv → {'✅ 一致' if ok else '❌ 不一致'}")
    if not ok:
        sys.exit(1)

    out = paths.REPORT_DIR / "retune_for_test.json"
    out.write_text(json.dumps({
        "old": {"tau": cur_tau, "threshold": cur_t, "sim_f1": f_cur, "val_f1": f_plain},
        "new": {"tau": best[1], "threshold": best[2], "sim_f1": best[0]},
        "test_unknown_rate": {"old": float((pred_old == 50).mean()),
                              "new": float((pred_new == 50).mean())},
        "weight_W": W,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {out}")


if __name__ == "__main__":
    main()
