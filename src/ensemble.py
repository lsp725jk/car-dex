# -*- coding: utf-8 -*-
"""任务一 - 多模型软投票集成：评估 + 生成 predictions.csv

用法：
    # 在验证集上评估集成效果
    python src/ensemble.py \
        --checkpoints outputs/checkpoints/improved_resnet18_224/best.pt \
                      outputs/checkpoints/improved_resnet50_224/best.pt \
        --split val --tta-hflip

    # 对教师指定测试目录生成 predictions.csv
    python src/ensemble.py --checkpoints ... --input-dir <测试目录> --output predictions.csv

    # 只推理前 200 张，快速冒烟
    python src/ensemble.py --checkpoints ... --limit 200
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from cardex import paths  # noqa: E402
from cardex.data import CarDataset, build_transform  # noqa: E402
from cardex.inference import CarEnsemble, list_images  # noqa: E402
from cardex.metrics import compute_metrics, top_confusions  # noqa: E402
from cardex.utils import configure_threads, get_logger, save_json  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--split", default=None, choices=["train", "val"])
    ap.add_argument("--input-dir", default=None)
    ap.add_argument("--output", default=None, help="给出则写出 predictions.csv")
    ap.add_argument("--out-dir", default=str(paths.RUN_DIR / "ensemble"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tta-hflip", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    configure_threads(args.threads)
    logger = get_logger("ensemble", paths.RUN_DIR / "ensemble" / "ensemble.log")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 收集图片与真实标签 ----------
    # 顺序必须与 CarDataset / DataLoader 一致：按类别目录名排序，目录内按文件名排序。
    # 不要对全部路径做全局排序 —— unknown/ 下的文件名沿用来来源类别 ID，
    # 全局排序会把它们插到其他类别中间，与模型输出顺序错位。
    if args.split:
        ds = CarDataset(paths.DATA_DIR / args.split,
                        transform=build_transform("eval", 224, {}))
        pairs: list[tuple[Path, int | None]] = [(Path(p), lab) for p, lab in ds.samples]
    else:
        if not args.input_dir:
            raise SystemExit("必须指定 --split 或 --input-dir")
        pairs = [(p, None) for p in list_images(Path(args.input_dir))]
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit("没有找到图片")

    logger.info("集成成员 %d 个：", len(args.checkpoints))
    for p in args.checkpoints:
        logger.info("  - %s", p)
    logger.info("待推理 %d 张（split=%s, tta_hflip=%s）", len(pairs), args.split, args.tta_hflip)

    t0 = time.perf_counter()
    ens = CarEnsemble([Path(p) for p in args.checkpoints], tta_hflip=args.tta_hflip,
                      weights=args.weights)
    label_names = ens.label_names

    records: list[dict] = []
    for start in range(0, len(pairs), args.batch_size):
        chunk = pairs[start : start + args.batch_size]
        images = []
        for p, _ in chunk:
            try:
                with Image.open(p) as im:
                    images.append(im.convert("RGB").copy())
            except Exception as exc:
                logger.warning("图片读取失败 %s：%s", p, exc)
                images.append(Image.new("RGB", (224, 224), (128, 128, 128)))
        preds = ens.predict_batch(images, batch_size=args.batch_size)
        for (p, y), pred in zip(chunk, preds):
            records.append({"path": str(p), "name": p.name, "y_true": y, **pred})
        if start % (args.batch_size * 10) == 0:
            logger.info("  %d/%d  (%.0fs)", min(start + args.batch_size, len(pairs)),
                        len(pairs), time.perf_counter() - t0)

    dt = time.perf_counter() - t0
    logger.info("推理完成，用时 %.1fs（%.1f 张/秒）", dt, len(records) / dt)

    # ---------- 评估 ----------
    result = None
    if args.split:
        y_true = np.array([r["y_true"] for r in records])
        y_pred = np.array([r["index"] for r in records])
        result = compute_metrics(y_true, y_pred, label_names)
        logger.info("=" * 74)
        logger.info("【集成】Macro-F1 = %.4f   Accuracy = %.4f", result["macro_f1"], result["accuracy"])
        logger.info("unknown 检出率 = %.4f   已知类误拒率 = %.4f",
                    result["unknown_detection_rate"], result["known_false_reject_rate"])
        logger.info("-" * 74)
        logger.info("F1 最低的 8 类：")
        for row in sorted(result["per_class"], key=lambda r: r["f1"])[:8]:
            logger.info("  %-34s F1=%.3f (P=%.3f R=%.3f n=%d)", row["label"], row["f1"],
                        row["precision"], row["recall"], row["support"])
        cm_arr = np.array(result["confusion_matrix"])
        (out_dir / "ensemble_classification_report.txt").write_text(
            result["report_text"], encoding="utf-8")
        with open(out_dir / "ensemble_confusion_matrix.csv", "w", encoding="utf-8", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["true\\pred"] + label_names)
            for i, name in enumerate(label_names):
                w.writerow([name] + cm_arr[i].tolist())
        with open(out_dir / "ensemble_top_confusions.csv", "w", encoding="utf-8", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=["true", "pred", "count", "true_total", "rate"])
            w.writeheader()
            w.writerows(top_confusions(cm_arr, label_names, top_k=30))
        with open(out_dir / "ensemble_error_samples.csv", "w", encoding="utf-8", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["image_id", "image_path", "true_label", "pred_label", "confidence"])
            for r in records:
                if r["index"] != r["y_true"]:
                    w.writerow([r["name"], r["path"], label_names[r["y_true"]],
                                label_names[r["index"]], round(r["confidence"], 6)])
        save_json(
            {
                "checkpoints": args.checkpoints,
                "weights": args.weights,
                "tta_hflip": args.tta_hflip,
                "split": args.split,
                "macro_f1": result["macro_f1"],
                "accuracy": result["accuracy"],
                "unknown_detection_rate": result["unknown_detection_rate"],
                "known_false_reject_rate": result["known_false_reject_rate"],
                "per_class": result["per_class"],
                "seconds": round(dt, 1),
            },
            out_dir / "ensemble_report.json",
        )
        logger.info("[写出] %s", out_dir)

    # ---------- predictions.csv ----------
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["image_id", "predicted_label", "confidence"])
            for r in records:
                w.writerow([r["name"], r["label_id"], round(float(r["confidence"]), 6)])
        logger.info("[写出] %s（%d 条记录）", out, len(records))


if __name__ == "__main__":
    main()
