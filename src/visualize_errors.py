# -*- coding: utf-8 -*-
"""任务一 - 错误样本可视化：定位"类内多代同堂"导致的高置信度错误

对指定类别，把训练集图片与被误判的验证集图片并排展示，用于人工判断错误原因。

用法：
    python src/visualize_errors.py --run baseline_resnet18_160 --class-id 0975
    python src/visualize_errors.py --run baseline_resnet18_160 --class-id 0035 --n 8
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from cardex import paths  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--class-id", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cm = paths.load_class_map()
    cid = args.class_id
    name = cm["index_to_name"][str(cm["id_to_index"][cid])] if cid in cm["id_to_index"] else cid

    run_dir = paths.RUN_DIR / args.run
    err_csv = run_dir / "error_samples.csv"
    if not err_csv.exists():
        raise SystemExit(f"找不到 {err_csv}，请先运行 src/evaluate.py")

    with open(err_csv, encoding="utf-8") as fp:
        errors = [r for r in csv.DictReader(fp) if r["true_id"] == cid]

    train_dir = paths.DATA_DIR / "train" / cid
    train_files = sorted(train_dir.iterdir())
    train_pick = rng.sample(train_files, min(args.n, len(train_files)))
    err_pick = errors[: args.n]

    ncols = max(len(train_pick), len(err_pick))
    fig, axes = plt.subplots(2, ncols, figsize=(ncols * 2.2, 5.0))
    axes = axes.reshape(2, ncols)
    for j, f in enumerate(train_pick):
        with Image.open(f) as im:
            axes[0, j].imshow(im.convert("RGB"))
        axes[0, j].axis("off")
        axes[0, j].set_title("训练集", fontsize=7)
    for j in range(len(train_pick), ncols):
        axes[0, j].axis("off")
    for j, row in enumerate(err_pick):
        try:
            with Image.open(row["image_path"]) as im:
                axes[1, j].imshow(im.convert("RGB"))
        except Exception:
            pass
        axes[1, j].axis("off")
        axes[1, j].set_title(f"验证误判→{row['pred_label']}\nconf={float(row['confidence']):.2f}",
                             fontsize=6)
    for j in range(len(err_pick), ncols):
        axes[1, j].axis("off")
    fig.suptitle(f"{cid} {name}：训练集样例（上） vs 验证集被误判样例（下）", fontsize=11)
    fig.tight_layout()
    out = Path(args.out) if args.out else paths.REPORT_DIR / f"error_case_{cid}.png"
    fig.savefig(out, dpi=130)
    print(f"[写出] {out}")
    print(f"该类验证集共 {len(errors)} 个错误样本，错误去向：")
    from collections import Counter

    for pred, n in Counter(r["pred_label"] for r in errors).most_common(8):
        print(f"  → {pred:<34}{n}")


if __name__ == "__main__":
    main()
