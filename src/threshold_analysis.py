# -*- coding: utf-8 -*-
"""任务一 - 拒识阈值分析（通过验证集确定阈值）

任务书 5.5 节要求：低置信度阈值必须通过验证集确定并在报告中说明。
本脚本在验证集上扫描阈值 t：
    当 max_prob < t 时把预测改写为 unknown，其余保持模型预测；
统计每个 t 下的 51 类 Macro-F1、unknown 检出率、已知类误拒率，选出最优阈值。

用法：
    python src/threshold_analysis.py --run improved_resnet50_224
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cardex import paths  # noqa: E402
from cardex.data import CarDataset, build_transform  # noqa: E402
from cardex.engine import load_model_from_checkpoint  # noqa: E402
from cardex.utils import configure_threads, get_logger, save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="outputs/checkpoints/<run> 下的运行名")
    ap.add_argument("--checkpoint", default=None, help="默认使用 best.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tta-hflip", action="store_true")
    ap.add_argument("--thresholds", default="0,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    args = ap.parse_args()

    configure_threads(16)
    logger = get_logger("threshold")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else paths.CKPT_DIR / args.run / "best.pt"
    model, ckpt = load_model_from_checkpoint(ckpt_path)
    label_names: list[str] = list(ckpt["label_names"])
    n_classes = len(label_names)
    unk = n_classes - 1
    image_size = int(ckpt["image_size"])

    ds = CarDataset(paths.DATA_DIR / args.split,
                    transform=build_transform("eval", image_size, ckpt.get("preprocess")))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    import torch  # noqa: E402

    model.eval()
    ys, preds, confs = [], [], []
    with torch.no_grad():
        for images, targets, _ in loader:
            logits = model(images)
            if args.tta_hflip:
                logits = (torch.log_softmax(logits, dim=1)
                          + torch.log_softmax(model(torch.flip(images, dims=[3])), dim=1)) / 2
            probs = torch.softmax(logits, dim=1)
            c, p = probs.max(dim=1)
            ys.extend(targets.tolist())
            preds.extend(p.tolist())
            confs.extend(c.tolist())

    y_true = np.array(ys)
    y_pred = np.array(preds)
    conf = np.array(confs)
    labels = list(range(n_classes))

    rows = []
    for t in [float(x) for x in args.thresholds.split(",") if x.strip()]:
        y_rej = y_pred.copy()
        if t > 0:
            y_rej[conf < t] = unk
        macro_f1 = float(f1_score(y_true, y_rej, labels=labels, average="macro", zero_division=0))
        unk_true = y_true == unk
        known_true = ~unk_true
        rows.append(
            {
                "threshold": t,
                "macro_f1": round(macro_f1, 5),
                "unknown_detection_rate": round(float((y_rej[unk_true] == unk).mean()), 5),
                "known_false_reject_rate": round(float((y_rej[known_true] == unk).mean()), 5),
                "overall_accuracy": round(float((y_rej == y_true).mean()), 5),
            }
        )

    best = max(rows, key=lambda r: r["macro_f1"])
    out_dir = paths.RUN_DIR / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "threshold_sweep.csv", "w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    save_json({"best": best, "sweep": rows, "checkpoint": str(ckpt_path),
               "split": args.split, "tta_hflip": args.tta_hflip},
              out_dir / "threshold_sweep.json")

    logger.info("阈值扫描（验证集 %s，%d 张）：", args.split, len(y_true))
    logger.info("%8s %10s %14s %16s %10s", "阈值", "Macro-F1", "unknown检出率", "已知类误拒率", "整体准确率")
    for r in rows:
        mark = "  ★" if r is best else ""
        logger.info("%8.2f %10.4f %14.4f %16.4f %10.4f%s", r["threshold"], r["macro_f1"],
                    r["unknown_detection_rate"], r["known_false_reject_rate"],
                    r["overall_accuracy"], mark)
    logger.info("最优拒识阈值 t=%.2f，验证集 Macro-F1=%.4f", best["threshold"], best["macro_f1"])
    logger.info("[写出] %s", out_dir / "threshold_sweep.csv")


if __name__ == "__main__":
    main()
