# -*- coding: utf-8 -*-
"""任务一 - 决策校准：先验校正 (prior adjustment) + 拒识阈值 (threshold)

## 为什么需要这一步

本项目评分口径是 **51 类等权 Macro-F1**，但训练集里 unknown 有 1,960 张，
而单个目标车型平均只有 110 张。即使不显式加权，交叉熵也会让模型学到
**训练集先验** P(y)，它反映的是"图片有多少"，而不是"类别有多少"。
后果就是模型**过度预测 unknown**：基线模型把 大众_朗逸 的 25%、日产_轩逸 的 37%
判成了 unknown，直接拉低这些车型的 F1。

## 两种互补的补救手段

1. **先验校正（logit adjustment）**
   在推理时把训练先验从 logits 中减掉：
       z'_c = z_c - tau * log(pi_c),   pi_c = n_c / N
   tau=0 表示不做校正（保留训练先验）；tau=1 表示完全按"类别等概率"决策，
   与 Macro-F1 的等权口径一致。tau 由**验证集**扫描确定。

2. **拒识阈值**
   当 max softmax < t 时把预测改写为 unknown，用于进一步压制"低置信度硬猜"。
   阈值同样由**验证集**扫描确定（任务书 5.5 节要求）。

本脚本只做一次前向传播（把 logits 缓存成 .npy），随后在 numpy 上做二维网格扫描，
因此可以同时评估几百种 (tau, threshold) 组合。

用法：
    python src/calibrate.py --run baseline_resnet18_160
    python src/calibrate.py --checkpoint outputs/checkpoints/xxx/best.pt --tta-hflip
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cardex import paths  # noqa: E402
from cardex.data import CarDataset, build_transform  # noqa: E402
from cardex.engine import load_model_from_checkpoint  # noqa: E402
from cardex.utils import configure_threads, get_logger, save_json  # noqa: E402

TAU_GRID = [-1.0, -0.75, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, 0.0,
            0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
THR_GRID = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def collect_logits(ckpt_path: Path, split: str, batch_size: int, tta_hflip: bool,
                   amp: bool, logger) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    model, ckpt = load_model_from_checkpoint(ckpt_path)
    image_size = int(ckpt["image_size"])
    ds = CarDataset(paths.DATA_DIR / split,
                    transform=build_transform("eval", image_size, ckpt.get("preprocess")))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    logger.info("前向传播：%s %d 张，模型 %s @ %d，TTA=%s",
                split, len(ds), ckpt["arch"], image_size, tta_hflip)

    def forward(x):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            out = model(x)
        return out.float()

    logits_all, ys, names = [], [], []
    model.eval()
    with torch.no_grad():
        for images, targets, batch_paths in loader:
            logits = forward(images)
            if tta_hflip:
                logits = (torch.log_softmax(logits, dim=1)
                          + torch.log_softmax(forward(torch.flip(images, dims=[3])), dim=1)) / 2
            logits_all.append(logits.numpy())
            ys.extend(targets.tolist())
            names.extend(list(batch_paths))
    return np.concatenate(logits_all), np.array(ys), names, ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="outputs/checkpoints/<run>")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tta-hflip", action="store_true")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--cache", default=None, help="logits 缓存路径（默认写在 run 目录）")
    args = ap.parse_args()

    configure_threads(args.threads)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        run_name = ckpt_path.parent.name
    elif args.run:
        run_name = args.run
        ckpt_path = paths.CKPT_DIR / run_name / "best.pt"
    else:
        raise SystemExit("必须指定 --run 或 --checkpoint")

    out_dir = paths.RUN_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"calibrate.{run_name}", out_dir / "calibrate.log")

    suffix = "_tta" if args.tta_hflip else ""
    cache_path = (Path(args.cache) if args.cache
                  else out_dir / f"logits_{args.split}{suffix}.npz")
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        logits, y_true, names = z["logits"], z["y_true"], list(z["names"])
        ckpt = json.loads(str(z["ckpt"]))
        logger.info("复用已有 logits 缓存：%s", cache_path)
    else:
        logits, y_true, names, ckpt = collect_logits(
            ckpt_path, args.split, args.batch_size, args.tta_hflip, bool(ckpt_amp(ckpt_path)), logger)
        np.savez_compressed(cache_path, logits=logits, y_true=y_true, names=np.array(names),
                            ckpt=json.dumps({k: v for k, v in ckpt.items()
                                             if k not in ("model_state", "train_config")},
                                            ensure_ascii=False, default=str))
        logger.info("[写出] %s", cache_path)

    label_names: list[str] = paths.label_names()
    n_classes = len(label_names)
    unk = n_classes - 1
    labels = list(range(n_classes))

    # 训练集先验 pi_c = n_c / N（unknown 为 1960，目标车型为其实际训练样本数）
    stats = json.loads(paths.DATA_STATS_JSON.read_text(encoding="utf-8"))
    n_c = np.zeros(n_classes, dtype=np.float64)
    for i in range(50):
        cid = paths.load_class_map()["index_to_id"][str(i)]
        n_c[i] = stats["per_class"]["train"][cid]["n"]
    n_c[unk] = stats["per_class"]["train"]["unknown"]["n"]
    prior = n_c / n_c.sum()
    log_prior = np.log(np.maximum(prior, 1e-12))
    logger.info("训练集先验：unknown=%.4f  目标车型平均=%.4f  最大 log 先验差=%.3f",
                prior[unk], prior[:50].mean(), (log_prior[unk] - log_prior[:50].min()))

    # ---------------- 二维扫描 ----------------
    rows: list[dict] = []
    for tau in TAU_GRID:
        adj = logits - tau * log_prior[None, :]
        adj = adj - adj.max(axis=1, keepdims=True)
        probs = np.exp(adj)
        probs /= probs.sum(axis=1, keepdims=True)
        conf = probs.max(axis=1)
        pred_base = probs.argmax(axis=1)
        for thr in THR_GRID:
            pred = pred_base.copy()
            if thr > 0:
                pred[conf < thr] = unk
            unk_true = y_true == unk
            known_true = ~unk_true
            rows.append(
                {
                    "tau": tau,
                    "threshold": thr,
                    "macro_f1": round(float(f1_score(y_true, pred, labels=labels,
                                                     average="macro", zero_division=0)), 5),
                    "accuracy": round(float((pred == y_true).mean()), 5),
                    "unknown_detection_rate": round(float((pred[unk_true] == unk).mean()), 5),
                    "known_false_reject_rate": round(float((pred[known_true] == unk).mean()), 5),
                }
            )

    best = max(rows, key=lambda r: r["macro_f1"])
    baseline_row = next(r for r in rows if r["tau"] == 0.0 and r["threshold"] == 0.0)
    best_tau_only = max([r for r in rows if r["threshold"] == 0.0], key=lambda r: r["macro_f1"])

    with open(out_dir / "calibration_sweep.csv", "w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    save_json(
        {
            "run": run_name,
            "checkpoint": str(ckpt_path),
            "split": args.split,
            "tta_hflip": args.tta_hflip,
            "n_samples": int(len(y_true)),
            "no_calibration": baseline_row,
            "best_tau_only": best_tau_only,
            "best_joint": best,
            "recommended": {
                "prior_adjust_tau": best["tau"],
                "unknown_threshold": best["threshold"],
                "note": "由验证集 Macro-F1 选出，供 Web 推理与批量推理使用",
            },
            "sweep": rows,
        },
        out_dir / "calibration.json",
    )

    logger.info("=" * 82)
    logger.info("未校准（tau=0, thr=0）        : Macro-F1=%.4f  Acc=%.4f  unk检出=%.3f  已知误拒=%.3f",
                baseline_row["macro_f1"], baseline_row["accuracy"],
                baseline_row["unknown_detection_rate"], baseline_row["known_false_reject_rate"])
    logger.info("仅先验校正（tau=%.2f）       : Macro-F1=%.4f  Acc=%.4f  unk检出=%.3f  已知误拒=%.3f",
                best_tau_only["tau"], best_tau_only["macro_f1"], best_tau_only["accuracy"],
                best_tau_only["unknown_detection_rate"], best_tau_only["known_false_reject_rate"])
    logger.info("先验校正 + 拒识阈值 (%.2f, %.2f): Macro-F1=%.4f  Acc=%.4f  unk检出=%.3f  已知误拒=%.3f",
                best["tau"], best["threshold"], best["macro_f1"], best["accuracy"],
                best["unknown_detection_rate"], best["known_false_reject_rate"])
    logger.info("-" * 82)
    logger.info("tau 扫描（threshold=0）：")
    for r in [x for x in rows if x["threshold"] == 0.0]:
        logger.info("  tau=%+.2f  Macro-F1=%.4f  已知误拒=%.3f  unk检出=%.3f",
                    r["tau"], r["macro_f1"], r["known_false_reject_rate"],
                    r["unknown_detection_rate"])
    logger.info("[写出] %s 与 %s", out_dir / "calibration_sweep.csv", out_dir / "calibration.json")


def ckpt_amp(ckpt_path: Path) -> bool:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return bool(ck.get("train_config", {}).get("amp", False))


if __name__ == "__main__":
    main()
