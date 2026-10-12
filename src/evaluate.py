# -*- coding: utf-8 -*-
"""任务一 - 验证集评估与错误分析

输出（写入 outputs/runs/<run>/）：
    eval_report.json            完整指标（51 类 Macro-F1、Accuracy、unknown 检出率等）
    eval_classification_report.txt  逐类 Precision / Recall / F1 / support
    eval_confusion_matrix.csv   51x51 混淆矩阵
    top_confusions.csv          最严重的类别混淆对
    error_samples.csv           典型错误样本清单（含真实类别、预测类别、置信度、图片路径）
    low_confidence_samples.csv  低置信度样本清单（用于确认拒识阈值）

用法：
    python src/evaluate.py --checkpoint outputs/checkpoints/<run>/best.pt
    python src/evaluate.py --checkpoint ... --split val --tta-hflip
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cardex import paths  # noqa: E402
from cardex.data import CarDataset, build_transform  # noqa: E402
from cardex.engine import evaluate_checkpoint, load_model_from_checkpoint  # noqa: E402
from cardex.metrics import top_confusions  # noqa: E402
from cardex.utils import configure_threads, get_logger, save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tta-hflip", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--low-conf-threshold", type=float, default=0.60)
    ap.add_argument("--error-limit", type=int, default=200)
    ap.add_argument("--amp", dest="amp", action="store_true", default=None,
                    help="强制开启 CPU bfloat16 推理")
    ap.add_argument("--no-amp", dest="amp", action="store_false",
                    help="强制使用 float32 推理")
    args = ap.parse_args()

    configure_threads(args.threads)
    logger = get_logger("evaluate")

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else paths.RUN_DIR / ckpt_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_model_from_checkpoint(ckpt_path)
    # 推理精度默认与训练时保持一致，否则 bf16/float32 的细微数值差异会让
    # 个别样本的 argmax 翻转，导致"训练日志里的 Macro-F1"与"重新评估的 Macro-F1"
    # 出现 ~0.0005 的不一致。
    amp = bool(ckpt.get("train_config", {}).get("amp", False)) if args.amp is None else args.amp
    logger.info("推理精度：%s（%s）", "bfloat16 autocast" if amp else "float32",
                "与训练一致" if args.amp is None else "命令行指定")
    label_names: list[str] = list(ckpt["label_names"])
    label_ids = paths.label_ids()
    image_size = int(ckpt["image_size"])
    pre = ckpt.get("preprocess", {})

    ds = CarDataset(paths.DATA_DIR / args.split, transform=build_transform("eval", image_size, pre))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info("评估 %s：%d 张图片 / %d 个类别目录，模型 %s @ %d",
                args.split, len(ds), len(ds.class_dirs), ckpt["arch"], image_size)

    result = evaluate_checkpoint(ckpt_path, loader, len(label_names), label_names,
                                 tta_hflip=args.tta_hflip, amp=amp)
    y_true, y_pred = result["y_true"], result["y_pred"]
    conf = result["confidence"]
    paths_seen = result["paths"]
    cm = np.array(result["confusion_matrix"])

    # ---------- 指标落盘 ----------
    metrics = {k: v for k, v in result.items()
               if k not in ("y_true", "y_pred", "confidence", "paths", "confusion_matrix",
                            "report_text", "per_class")}
    metrics["per_class"] = result["per_class"]
    metrics["split"] = args.split
    metrics["checkpoint"] = str(ckpt_path)
    metrics["tta_hflip"] = args.tta_hflip
    metrics["amp"] = amp
    save_json(metrics, out_dir / "eval_report.json")
    (out_dir / "eval_classification_report.txt").write_text(result["report_text"], encoding="utf-8")

    with open(out_dir / "eval_confusion_matrix.csv", "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["true\\pred"] + label_names)
        for i, name in enumerate(label_names):
            w.writerow([name] + cm[i].tolist())

    confusions = top_confusions(cm, label_names, top_k=30)
    with open(out_dir / "top_confusions.csv", "w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["true", "pred", "count", "true_total", "rate"])
        w.writeheader()
        w.writerows(confusions)

    # ---------- 错误样本 ----------
    errors = np.where(y_true != y_pred)[0]
    order = np.argsort(-conf[errors])
    with open(out_dir / "error_samples.csv", "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["image_id", "image_path", "true_label", "true_id", "pred_label", "pred_id",
                    "confidence"])
        for i in errors[order][: args.error_limit]:
            w.writerow([Path(paths_seen[i]).name, paths_seen[i],
                        label_names[y_true[i]], label_ids[y_true[i]],
                        label_names[y_pred[i]], label_ids[y_pred[i]], round(float(conf[i]), 6)])

    with open(out_dir / "low_confidence_samples.csv", "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["image_id", "true_label", "pred_label", "confidence", "correct"])
        low = np.where(conf < args.low_conf_threshold)[0]
        for i in low[np.argsort(conf[low])]:
            w.writerow([Path(paths_seen[i]).name, label_names[y_true[i]], label_names[y_pred[i]],
                        round(float(conf[i]), 6), bool(y_true[i] == y_pred[i])])

    # ---------- 控制台摘要 ----------
    unk = len(label_names) - 1
    logger.info("=" * 74)
    logger.info("Macro-F1 (51 类，唯一评分指标) = %.4f", result["macro_f1"])
    logger.info("Accuracy                        = %.4f", result["accuracy"])
    logger.info("unknown 检出率                  = %.4f",
                result["unknown_detection_rate"])
    logger.info("已知类误拒率（被误判为 unknown）= %.4f", result["known_false_reject_rate"])
    logger.info("低置信度样本（<%.2f）           = %d / %d", args.low_conf_threshold,
                len(low), len(y_true))
    logger.info("-" * 74)
    logger.info("F1 最低的 10 个类别：")
    for row in sorted(result["per_class"], key=lambda r: r["f1"])[:10]:
        logger.info("  %-34s F1=%.3f  P=%.3f  R=%.3f  n=%d", row["label"],
                    row["f1"], row["precision"], row["recall"], row["support"])
    logger.info("F1 最高的 5 个类别：")
    for row in sorted(result["per_class"], key=lambda r: -r["f1"])[:5]:
        logger.info("  %-34s F1=%.3f  n=%d", row["label"], row["f1"], row["support"])
    logger.info("-" * 74)
    logger.info("主要混淆对（真实 → 预测）：")
    for row in confusions[:12]:
        logger.info("  %-30s → %-30s %3d 次 (%.0f%% of 该类)", row["true"], row["pred"],
                    row["count"], 100 * row["rate"])

    # unknown 的错误去向
    unk_true_idx = np.where(y_true == unk)[0]
    if len(unk_true_idx):
        c = Counter(label_names[p] for p in y_pred[unk_true_idx] if p != unk)
        logger.info("-" * 74)
        logger.info("真实 unknown 被误判成的车型 Top-10：")
        for name, n in c.most_common(10):
            logger.info("  %-34s %d 次", name, n)

    logger.info("=" * 74)
    logger.info("结果目录：%s", out_dir)
    print(json.dumps({"macro_f1": result["macro_f1"], "accuracy": result["accuracy"],
                      "unknown_detection_rate": result["unknown_detection_rate"],
                      "known_false_reject_rate": result["known_false_reject_rate"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
