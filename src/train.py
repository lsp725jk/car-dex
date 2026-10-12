# -*- coding: utf-8 -*-
"""任务一 - 训练入口。

用法：
    python src/train.py --config configs/train_baseline.json
    python src/train.py --config configs/train_improved.json --epochs 4 --limit-train-per-class 12   # 冒烟测试
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardex import engine  # noqa: E402
from cardex.utils import get_logger  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="训练配置文件（JSON）")
    ap.add_argument("--run-name", default=None, help="覆盖 run_name")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--limit-train-per-class", type=int, default=None)
    ap.add_argument("--limit-val-per-class", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="覆盖任意配置项，如 --set mixup_alpha=0.2")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for key in ("run_name", "epochs", "batch_size", "lr", "arch", "image_size", "threads",
                "limit_train_per_class", "limit_val_per_class"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    for item in args.set:
        k, _, v = item.partition("=")
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v

    logger = get_logger("train", None)
    logger.info("配置：%s", json.dumps(cfg, ensure_ascii=False))
    summary = engine.train_model(cfg, logger=logger)
    print(json.dumps({k: summary[k] for k in ("run_name", "best", "total_time_s")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
