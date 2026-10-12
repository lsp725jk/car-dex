# -*- coding: utf-8 -*-
"""任务一 - 步骤3：从教师数据集压缩包中抽出本项目所需的图片

只抽取本项目用到的图片，不解压整个 6.6 GB 压缩包。

目录结构（与任务书 4.1 节一致）：
    data/train/<四位车型ID>/   50 个目标车型的训练图片（教师 train 划分，全量）
    data/train/unknown/        组外来源类别的训练图片（采样）
    data/val/<四位车型ID>/     50 个目标车型的验证图片（教师 val 划分，全量）
    data/val/unknown/          组外来源类别的验证图片（采样）

约束：
  · train 与 val 来自压缩包内不同的划分，天然不重复；
  · unknown 只取来源类别的 train 划分进 train/unknown，val 划分进 val/unknown；
  · 完全不使用测试图片（压缩包内本身不含测试划分）。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
import zipfile
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = Path(r"C:\Users\y\Downloads\数据集.zip")
DATA_DIR = PROJECT_ROOT / "data"
CLASS_MAP = PROJECT_ROOT / "configs" / "class_map.json"
UNKNOWN_JSON = PROJECT_ROOT / "configs" / "unknown_sources.json"
MANIFEST_CSV = PROJECT_ROOT / "outputs" / "reports" / "data_manifest.csv"
STATS_JSON = PROJECT_ROOT / "configs" / "data_stats.json"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_zip_index(zip_path: Path) -> dict[tuple[str, str], list[str]]:
    """扫描压缩包目录表，返回 {(split, class_id): [内部路径, ...]}，顺序即压缩包内顺序。"""
    index: dict[tuple[str, str], list[str]] = defaultdict(list)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.startswith("__MACOSX"):
                continue
            parts = name.split("/")
            if len(parts) != 3 or parts[0] not in ("train", "val"):
                continue
            if Path(parts[2]).suffix.lower() not in IMAGE_EXT:
                continue
            index[(parts[0], parts[1])].append(name)
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="清空 data/ 后重新抽取")
    ap.add_argument("--seed", type=int, default=20250914)
    args = ap.parse_args()

    if args.force and DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    class_map = json.loads(CLASS_MAP.read_text(encoding="utf-8"))
    unknown_cfg = json.loads(UNKNOWN_JSON.read_text(encoding="utf-8"))
    known_ids: list[str] = [class_map["index_to_id"][str(i)] for i in range(50)]
    name_by_id = class_map["index_to_name"]

    print("扫描压缩包目录表 ...")
    t0 = time.perf_counter()
    index = build_zip_index(ZIP_PATH)
    print(f"  完成，用时 {time.perf_counter() - t0:.1f}s，覆盖 {len(index)} 个 (split, class) 组合")

    rng = random.Random(args.seed)

    # ---------- 规划抽取清单 ----------
    # 目标： (split, class_id, [zip 路径...])
    plan: list[tuple[str, str, str, list[str]]] = []  # split, dst_class, src_class, members

    missing_known: list[str] = []
    for cid in known_ids:
        for split in ("train", "val"):
            members = index.get((split, cid), [])
            if not members:
                missing_known.append(f"{split}/{cid}")
                continue
            plan.append((split, cid, cid, members))
    if missing_known:
        raise SystemExit(f"以下已知类别缺少图片：{missing_known[:20]}")

    n_train_per = unknown_cfg["train_per_source"]
    n_val_per = unknown_cfg["val_per_source"]
    src_stats: list[dict] = []
    for src in unknown_cfg["sources"]:
        sid = src["id"]
        row = {"source_id": sid, "name": src["name"], "brand": src["brand"],
               "category": src["category"], "kind": src["source_kind"],
               "target_class_id": src["target_class_id"] or ""}
        for split, limit, key in (("train", n_train_per, "n_train"), ("val", n_val_per, "n_val")):
            members = index.get((split, sid), [])
            # 固定种子抽样，保证可复现
            picked = sorted(rng.sample(members, min(limit, len(members)))) if members else []
            row[key] = len(picked)
            if picked:
                plan.append((split, "unknown", sid, picked))
        src_stats.append(row)

    total = sum(len(m) for _, _, _, m in plan)
    print(f"计划抽取 {total} 张图片（{len(plan)} 个 (split, class) 任务）")

    # ---------- 执行抽取 ----------
    written = 0
    manifest_rows: list[dict] = []
    t0 = time.perf_counter()
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for i, (split, dst_class, src_class, members) in enumerate(plan, 1):
            out_dir = DATA_DIR / split / dst_class
            out_dir.mkdir(parents=True, exist_ok=True)
            for member in members:
                fname = member.split("/")[-1]
                dst = out_dir / fname
                if dst.exists() and dst.stat().st_size > 0:
                    written += 1
                else:
                    data = zf.read(member)
                    dst.write_bytes(data)
                    written += 1
                manifest_rows.append(
                    {"split": split, "class_id": dst_class, "source_class_id": src_class,
                     "file_name": fname, "bytes": dst.stat().st_size}
                )
            if i % 20 == 0 or i == len(plan):
                el = time.perf_counter() - t0
                rate = written / el if el else 0
                print(f"  [{i}/{len(plan)}] {written} 张已写出  ({rate:.0f} 张/秒)")

    print(f"抽取完成：{written} 张，用时 {time.perf_counter() - t0:.1f}s")

    # ---------- 清单与统计 ----------
    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_CSV, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["split", "class_id", "source_class_id",
                                                "file_name", "bytes"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    stats: dict[str, dict] = {"train": {}, "val": {}}
    for split in ("train", "val"):
        base = DATA_DIR / split
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXT)
            entry = {"n": n, "label": d.name}
            if d.name != "unknown":
                entry["name"] = name_by_id.get(d.name, "")
                entry["index"] = class_map["id_to_index"][d.name]
            stats[split][d.name] = entry

    summary = {
        "source_zip": str(ZIP_PATH),
        "seed": args.seed,
        "unknown_train_per_source": n_train_per,
        "unknown_val_per_source": n_val_per,
        "unknown_source_count": len(unknown_cfg["sources"]),
        "unknown_hard_negative_count": unknown_cfg["hard_negative_count"],
        "unknown_other_brand_count": unknown_cfg["other_brand_count"],
        "train_total": sum(v["n"] for v in stats["train"].values()),
        "val_total": sum(v["n"] for v in stats["val"].values()),
        "train_known_total": sum(v["n"] for k, v in stats["train"].items() if k != "unknown"),
        "train_unknown_total": stats["train"].get("unknown", {}).get("n", 0),
        "val_known_total": sum(v["n"] for k, v in stats["val"].items() if k != "unknown"),
        "val_unknown_total": stats["val"].get("unknown", {}).get("n", 0),
        "per_class": stats,
        "unknown_sources": src_stats,
    }
    summary["unknown_train_ratio"] = round(
        summary["train_unknown_total"] / max(1, summary["train_total"]), 4
    )
    STATS_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"train 合计 {summary['train_total']} 张"
          f"（已知 {summary['train_known_total']} + unknown {summary['train_unknown_total']}）")
    print(f"val   合计 {summary['val_total']} 张"
          f"（已知 {summary['val_known_total']} + unknown {summary['val_unknown_total']}）")
    print(f"unknown 训练占比 {summary['unknown_train_ratio']:.1%}")
    print(f"[写出] {MANIFEST_CSV}")
    print(f"[写出] {STATS_JSON}")


if __name__ == "__main__":
    main()
