# -*- coding: utf-8 -*-
"""任务一 - 批量推理：对指定目录生成 predictions.csv

predictions.csv 格式（任务书第八节）：
    UTF-8 编码，表头固定为 image_id,predicted_label,confidence
    image_id        包含扩展名的图片文件名
    predicted_label classes.txt 中保留前导零的四位车型 ID，或小写 unknown
    confidence      0~1 的有效数值

用法：
    python src/predict_batch.py --checkpoint outputs/checkpoints/<run>/best.pt \
        --input-dir <教师指定测试图片目录> --output predictions.csv
    python src/predict_batch.py --checkpoint ... --split val      # 演示：对本地验证集推理
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
from cardex.inference import CarClassifier, list_images  # noqa: E402
from cardex.utils import get_logger  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def collect_split_images(split: str) -> list[Path]:
    """按目录结构收集某个划分（train/val）下的全部图片。"""
    base = paths.DATA_DIR / split
    if not base.exists():
        raise SystemExit(f"目录不存在：{base}")
    return sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input-dir", default=None, help="待推理图片目录（递归）")
    ap.add_argument("--split", default=None, choices=["train", "val"],
                    help="改为对本地 data/<split> 推理")
    ap.add_argument("--output", default=str(PROJECT_ROOT / "predictions.csv"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tta-hflip", action="store_true")
    ap.add_argument("--tau", type=float, default=None,
                    help="先验校正强度；默认读取同 run 的 calibration.json")
    ap.add_argument("--threshold", type=float, default=None,
                    help="拒识阈值；默认读取同 run 的 calibration.json")
    ap.add_argument("--no-calibration", action="store_true", help="强制关闭标定（tau=0, t=0）")
    ap.add_argument("--topk-csv", default=None, help="可选：同时导出 Top-5 明细")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logger = get_logger("predict_batch")
    if args.split:
        image_paths = collect_split_images(args.split)
    else:
        if not args.input_dir:
            raise SystemExit("必须指定 --input-dir 或 --split")
        image_paths = list_images(Path(args.input_dir))
    if not image_paths:
        raise SystemExit("没有找到任何图片")

    # ---------------- 判定标定参数 ----------------
    tau, thr = 0.0, 0.0
    calib_src = "未使用标定（tau=0, threshold=0）"
    if not args.no_calibration:
        if args.tau is not None or args.threshold is not None:
            tau = args.tau or 0.0
            thr = args.threshold or 0.0
            calib_src = "命令行指定"
        else:
            calib_json = paths.RUN_DIR / Path(args.checkpoint).parent.name / "calibration.json"
            if calib_json.exists():
                cal = json.loads(calib_json.read_text(encoding="utf-8"))
                rec = cal.get("recommended", {})
                tau = float(rec.get("prior_adjust_tau", 0.0))
                thr = float(rec.get("unknown_threshold", 0.0))
                calib_src = f"读取 {calib_json}（验证集 Macro-F1={cal['best_joint']['macro_f1']}）"
            else:
                logger.warning("未找到 calibration.json，使用默认 tau=0, threshold=0")
    logger.info("决策标定：tau=%+.2f  unknown_threshold=%.2f  —— %s", tau, thr, calib_src)

    logger.info("加载模型：%s", args.checkpoint)
    t0 = time.perf_counter()
    clf = CarClassifier(Path(args.checkpoint), tta_hflip=args.tta_hflip,
                        prior_adjust_tau=tau, unknown_threshold=thr)
    logger.info("模型：%s  输入尺寸=%s  类别数=%d", clf.ckpt["arch"],
                clf.ckpt.get("image_size"), len(clf.label_names))
    logger.info("待推理图片 %d 张", len(image_paths))

    rows: list[dict] = []
    topk_rows: list[dict] = []
    batch_size = args.batch_size
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        images, names = [], []
        for p in chunk:
            try:
                with Image.open(p) as im:
                    images.append(im.convert("RGB").copy())
                names.append(p.name)
            except Exception as exc:  # 图片读取失败：如实记录，不静默丢弃
                logger.warning("图片读取失败：%s (%s)", p, exc)
                names.append(p.name)
                images.append(Image.new("RGB", (224, 224), (128, 128, 128)))
        preds = clf.predict_batch(images, batch_size=batch_size)
        for name, pred in zip(names, preds):
            rows.append(
                {
                    "image_id": name,
                    "predicted_label": pred["label_id"],
                    "confidence": round(float(pred["confidence"]), 6),
                }
            )
        if start % (batch_size * 10) == 0:
            logger.info("  已推理 %d/%d  (%s)", min(start + batch_size, len(image_paths)),
                        len(image_paths), f"{time.perf_counter() - t0:.0f}s")

    # Top-K 明细（可选，便于报告分析）
    if args.topk_csv:
        tk = Path(args.topk_csv)
        tk.parent.mkdir(parents=True, exist_ok=True)
        with open(tk, "w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["image_id", "rank", "label_id", "label_name", "probability"])
            for p in image_paths:
                res = clf.predict_path(p, top_k=5)
                for rank, item in enumerate(res["top_k"], 1):
                    writer.writerow([p.name, rank, item["label_id"], item["label_name"],
                                     round(item["probability"], 6)])
        logger.info("[写出] %s", tk)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "predicted_label", "confidence"])
        writer.writeheader()
        writer.writerows(rows)

    # 自检：表头、记录数、取值范围、标签合法性
    labels_ok = set(paths.read_classes_txt()) | {paths.UNKNOWN_LABEL}
    bad_label = [r for r in rows if r["predicted_label"] not in labels_ok]
    bad_conf = [r for r in rows if not (0.0 <= float(r["confidence"]) <= 1.0)]
    dup = len(rows) - len({r["image_id"] for r in rows})
    confs = np.array([float(r["confidence"]) for r in rows])
    logger.info("=" * 60)
    logger.info("[写出] %s", out)
    logger.info("记录数 %d；重复 image_id %d；非法标签 %d；越界 confidence %d",
                len(rows), dup, len(bad_label), len(bad_conf))
    logger.info("confidence：min=%.4f  mean=%.4f  max=%.4f", confs.min(), confs.mean(), confs.max())
    logger.info("预测为 unknown 的张数：%d (%.1f%%)",
                sum(1 for r in rows if r["predicted_label"] == "unknown"),
                100 * sum(1 for r in rows if r["predicted_label"] == "unknown") / len(rows))
    logger.info("总用时 %.1fs", time.perf_counter() - t0)

    meta = {
        "checkpoint": str(args.checkpoint),
        "input": args.input_dir or f"data/{args.split}",
        "images": len(rows),
        "tta_hflip": args.tta_hflip,
        "prior_adjust_tau": tau,
        "unknown_threshold": thr,
        "calibration_source": calib_src,
        "output": str(out),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
