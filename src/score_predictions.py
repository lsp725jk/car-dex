# -*- coding: utf-8 -*-
"""任务一 - 用本地验证集真值核算 predictions.csv 的 Macro-F1

作用：确认"报告里写的验证集 Macro-F1"与"提交的 predictions.csv 实际算出来的分数"完全一致，
避免出现"报告分数和提交结果对不上"的情况。

前提：predictions.csv 是对 data/val（或 data/train）生成的，真值来自目录结构。
      若 predictions.csv 是对教师指定的外部测试目录生成的，则本地没有真值，本脚本会给出提示。

用法：
    python src/score_predictions.py                       # 默认核算 predictions.csv vs data/val
    python src/score_predictions.py --predictions x.csv --split val
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from sklearn.metrics import classification_report, f1_score  # noqa: E402

from cardex import paths  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=str(paths.PROJECT_ROOT / "predictions.csv"))
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--out", default=str(paths.REPORT_DIR / "predictions_score.json"))
    args = ap.parse_args()

    cm = paths.load_class_map()
    label_names = paths.label_names()
    truth: dict[str, int] = {}
    for d in (paths.DATA_DIR / args.split).iterdir():
        if not d.is_dir():
            continue
        label = 50 if d.name == "unknown" else cm["id_to_index"].get(d.name)
        if label is None:
            continue
        for f in d.iterdir():
            if f.suffix.lower() in IMAGE_EXT:
                truth[f.name] = label

    pv = Path(args.predictions)
    rows = list(csv.DictReader(pv.open(encoding="utf-8")))
    missing = [r["image_id"] for r in rows if r["image_id"] not in truth]
    covered = [r for r in rows if r["image_id"] in truth]

    print(f"predictions.csv : {pv}（{len(rows)} 条记录）")
    print(f"本地 {args.split} 划分真值 : {len(truth)} 张")
    print(f"可匹配记录      : {len(covered)} 张；未匹配 {len(missing)} 张")
    if missing:
        print(f"  → 未匹配的 image_id 前 5 个：{missing[:5]}")
        print("  → 若 predictions.csv 是对教师指定的外部测试目录生成的，本地无真值属于正常情况。")
    if not covered:
        raise SystemExit("没有任何记录能与本地真值匹配，无法核算。")

    y_true = np.array([truth[r["image_id"]] for r in covered])
    y_pred = np.array([
        50 if r["predicted_label"] == "unknown" else cm["id_to_index"].get(r["predicted_label"], 50)
        for r in covered
    ])
    labels = list(range(51))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    unk = 50
    result = {
        "predictions": str(pv),
        "split": args.split,
        "n_records": len(rows),
        "n_matched": len(covered),
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "unknown_detection_rate": round(float((y_pred[y_true == unk] == unk).mean()), 4),
        "known_false_reject_rate": round(float((y_pred[y_true != unk] == unk).mean()), 4),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    report_text = classification_report(y_true, y_pred, labels=labels,
                                        target_names=label_names, zero_division=0, digits=3)
    (paths.REPORT_DIR / f"predictions_classification_report_{args.split}.txt").write_text(
        report_text, encoding="utf-8")
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[写出] {args.out}")


if __name__ == "__main__":
    main()
