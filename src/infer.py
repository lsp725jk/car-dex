# -*- coding: utf-8 -*-
"""任务一 - 单张图片推理演示 / 命令行工具

用法：
    python src/infer.py --checkpoint outputs/checkpoints/<run>/best.pt --image path/to/car.jpg
    python src/infer.py --checkpoint ... --image path/to/car.jpg --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardex.inference import CarClassifier  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--tta-hflip", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    clf = CarClassifier(Path(args.checkpoint), tta_hflip=args.tta_hflip)
    result = clf.predict_path(Path(args.image), top_k=args.top_k)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"图片        : {result['image_path']}")
    print(f"预测类别 ID : {result['label_id']}")
    print(f"预测车型名  : {result['label_name']}")
    print(f"置信度      : {result['confidence']:.4f}")
    if result["is_unknown"]:
        print("提示        : 未识别为本组支持的 50 个车型")
    print("Top-%d：" % args.top_k)
    for item in result["top_k"]:
        print(f"  {item['label_id']:<9}{item['label_name']:<34}{item['probability']:.4f}")


if __name__ == "__main__":
    main()
