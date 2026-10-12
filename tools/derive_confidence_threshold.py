# -*- coding: utf-8 -*-
"""任务二 - 由验证集确定「低置信度提示阈值」

任务书 5.5 节要求：
  「当置信度较低时，可补充展示 Top-K、请用户确认或提示重新拍摄，也可设置额外的拒识阈值；
    阈值须通过验证集确定，并在报告中说明。」

本项目一共有**两个**与置信度相关的阈值，二者用途完全不同：

  ┌ 拒识阈值 t（unknown_threshold = 0.25）
  │   作用：max(softmax) < t 时把预测改写为 unknown。这是**模型决策**的一部分，
  │         已由 src/calibrate.py 在验证集上扫 204 组 (τ, t) 选出。
  │
  └ 提示阈值 c（本脚本求的值）
      作用：预测**仍然是某个具体车型**，但置信度偏低时，智能体应当
            「补充展示 Top-K、请用户确认、提示重新拍摄」。
            它不改变模型的判定结果，只影响**智能体怎么跟用户说话**。

本脚本用缓存的验证集 logits 做选择性预测（selective prediction）分析：
把样本按置信度从高到低排序，看在各个阈值下「被自动接受的样本」的准确率是多少，
从而选出一个既能自动接受大多数样本、又能把错误率压到很低的阈值。

用法：
    python tools/derive_confidence_threshold.py --run improved_resnet50_224 --tta-hflip
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cardex import paths  # noqa: E402

TAU, THR = 0.05, 0.25          # 最终交付的集成配置（见 final_model_choice.json）
GRID = [round(x, 3) for x in np.arange(0.30, 1.001, 0.01)]


def find_logits(run: str, tta: bool) -> Path | None:
    cands = [
        paths.RUN_DIR / run / f"logits_val{'_tta' if tta else ''}.npz",
        paths.RUN_DIR / "finalize_cache" / f"logits_{run}_val{'_tta' if tta else ''}.npy",
    ]
    for c in cands:
        if c.exists():
            return c
    return None


def load(run: str, tta: bool):
    p = find_logits(run, tta)
    if p is None:
        raise SystemExit(f"找不到 {run} 的 logits 缓存，请先跑 src/calibrate.py 或 src/finalize.py")
    if p.suffix == ".npz":
        z = np.load(p, allow_pickle=True)
        return z["logits"], z["y_true"], p
    return np.load(p), None, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="improved_resnet50_224")
    ap.add_argument("--history", default="improved_resnet18_224", help="用于对比的第二个模型")
    ap.add_argument("--tta-hflip", action="store_true", default=True)
    ap.add_argument("--no-tta", dest="tta_hflip", action="store_false")
    ap.add_argument("--target-accuracy", type=float, default=0.985,
                    help="被自动接受的样本希望达到的准确率")
    args = ap.parse_args()

    # 若 logits 是 log_softmax 形式（finalize_cache），直接可用；否则转成 log_softmax
    lp_raw, y_from_npz, src = load(args.run, args.tta_hflip)
    if y_from_npz is not None:
        y_true = np.asarray(y_from_npz)
    else:
        # 从 data/val 目录结构还原真值，顺序必须与 CarDataset.samples 一致
        from cardex.data import CarDataset, build_transform
        ds = CarDataset(paths.DATA_DIR / "val", transform=build_transform("eval", 224, {}))
        y_true = np.array([lab for _, lab in ds.samples])

    lp = lp_raw if lp_raw.min() < 0 else lp_raw - np.log(np.exp(lp_raw).sum(axis=1, keepdims=True))

    # 应用与交付一致的先验校正
    stats = json.loads(paths.DATA_STATS_JSON.read_text(encoding="utf-8"))
    cm = paths.load_class_map()
    n_c = [stats["per_class"]["train"][cm["index_to_id"][str(i)]]["n"] for i in range(50)]
    n_c.append(stats["per_class"]["train"]["unknown"]["n"])
    log_prior = np.log(np.array(n_c, float) / sum(n_c))

    adj = lp - TAU * log_prior[None, :]
    adj = adj - adj.max(axis=1, keepdims=True)
    probs = np.exp(adj)
    probs /= probs.sum(axis=1, keepdims=True)

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    pred[conf < THR] = 50                     # 与交付流程一致的拒识
    correct = pred == y_true

    print("=" * 88)
    print("低置信度提示阈值推导（验证集选择性预测分析）")
    print("=" * 88)
    print(f"logits 来源 : {src}")
    print(f"样本数      : {len(y_true)}    已应用的决策标定：τ={TAU:+.2f}  拒识阈值 t={THR:.2f}")
    print(f"整体         : Macro-F1 口径下的预测，整体准确率 {correct.mean():.4f}")
    print()
    print(f"{'提示阈值 c':>10}{'自动接受':>10}{'覆盖率':>9}{'接受样本准确率':>15}{'低置信样本准确率':>17}")
    print("-" * 88)

    rows = []
    for c in GRID:
        acc_mask = conf >= c
        cov = acc_mask.mean()
        acc_hi = correct[acc_mask].mean() if acc_mask.any() else float("nan")
        lo = ~acc_mask
        acc_lo = correct[lo].mean() if lo.any() else float("nan")
        rows.append({"c": c, "coverage": round(float(cov), 4),
                     "accepted_accuracy": round(float(acc_hi), 4),
                     "low_conf_accuracy": round(float(acc_lo), 4)})
        if abs(c * 100 - round(c * 100)) < 1e-9 and round(c * 100) % 5 == 0:
            print(f"{c:>10.2f}{int(acc_mask.sum()):>10}{cov:>9.3f}{acc_hi:>15.4f}{acc_lo:>17.4f}")

    # 选：在满足目标准确率的前提下，取覆盖率最大的 c（即最低的 c）
    ok = [r for r in rows if r["accepted_accuracy"] >= args.target_accuracy]
    best = ok[0] if ok else rows[-1]
    print("-" * 88)
    print(f"目标：被自动接受的样本准确率 ≥ {args.target_accuracy:.1%}")
    print(f"→ 推荐提示阈值 c = {best['c']:.2f}"
          f"（自动接受 {best['coverage']:.1%} 的样本，这批样本准确率 {best['accepted_accuracy']:.4f}）")
    print(f"   剩余 {1 - best['coverage']:.1%} 的样本会被判为「低置信度」，"
          f"其准确率仅 {best['low_conf_accuracy']:.4f} —— 确实值得让用户确认")

    out = {
        "purpose": "低置信度提示阈值：预测仍为具体车型但置信度偏低时，智能体展示 Top-K / 请用户确认 / 提示重拍",
        "distinct_from": {
            "unknown_threshold": THR,
            "note": "t 是模型决策的一部分（<t 则改判 unknown）；c 只影响智能体的措辞，不改变判定结果",
        },
        "prior_adjust_tau": TAU,
        "n_val": int(len(y_true)),
        "target_accuracy": args.target_accuracy,
        "recommended_display_threshold": best["c"],
        "at_recommended": best,
        "sweep": rows,
        "source_logits": str(src),
    }
    dst = PROJECT_ROOT / "agent" / "data" / "confidence_threshold.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[写出] {dst}")


if __name__ == "__main__":
    main()
