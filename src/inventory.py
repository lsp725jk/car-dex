# -*- coding: utf-8 -*-
"""任务一 - 步骤1：数据集清点

读取教师提供的数据集压缩包（只读中央目录，不解压图片），
结合 cls_info/class_info.json 生成 827 个类别的统计清单：

  - 类别 ID / name_from_new / name_from_old / type_info
  - train、val 各自的图片数量
  - 车型名称中的品牌（取 name_from_new 中 "_" 之前的部分）
  - 车系类型（type_info 的第三段，如 Light Truck / Sedan / SUV）

输出：
  outputs/reports/class_inventory.csv   全部 827 类明细
  outputs/reports/inventory_summary.txt 汇总统计（含品牌分布）
"""
from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = Path(r"C:\Users\y\Downloads\数据集.zip")
CLASS_INFO = PROJECT_ROOT / "configs" / "class_info.json"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_class_info() -> list[dict]:
    with open(CLASS_INFO, encoding="utf-8") as fp:
        return json.load(fp)


def scan_zip(zip_path: Path) -> tuple[Counter, Counter]:
    """返回 (train_count, val_count)，键为四位类别 ID。"""
    train_count: Counter = Counter()
    val_count: Counter = Counter()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.startswith("__MACOSX"):
                continue
            parts = name.split("/")
            # 形如 train/0000/xxx.jpg
            if len(parts) != 3 or parts[0] not in ("train", "val"):
                continue
            if Path(parts[2]).suffix.lower() not in IMAGE_EXT:
                continue
            (train_count if parts[0] == "train" else val_count)[parts[1]] += 1
    return train_count, val_count


def parse_brand(name_from_new: str) -> str:
    return name_from_new.split("_")[0] if "_" in name_from_new else name_from_new


def parse_type(type_info: str) -> str:
    """type_info 形如 '卡车####轻卡==Truck####Light Truck'，取最后一段作为车身类型。"""
    if "==" not in type_info:
        return ""
    tail = type_info.split("==")[-1]
    return tail.split("####")[-1].strip()


def parse_category(type_info: str) -> str:
    """取 '####' 之前的中文大类，如 卡车。"""
    if "==" not in type_info:
        return ""
    head = type_info.split("==")[0]
    return head.split("####")[0].strip()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    classes = load_class_info()
    train_count, val_count = scan_zip(ZIP_PATH)

    rows = []
    for item in classes:
        cid = item["id"]
        rows.append(
            {
                "id": cid,
                "name_from_new": item["name_from_new"],
                "brand": parse_brand(item["name_from_new"]),
                "category": parse_category(item["type_info"]),
                "body_type": parse_type(item["type_info"]),
                "type_info": item["type_info"],
                "name_from_old": item["name_from_old"],
                "train": train_count.get(cid, 0),
                "val": val_count.get(cid, 0),
                "total": train_count.get(cid, 0) + val_count.get(cid, 0),
            }
        )

    csv_path = REPORT_DIR / "class_inventory.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    valid = [r for r in rows if r["total"] > 0]
    missing = [r for r in rows if r["total"] == 0]

    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add("数据集清点报告")
    add("=" * 78)
    add(f"压缩包            : {ZIP_PATH}")
    add(f"class_info.json   : {len(classes)} 个类别")
    add(f"train 图片总数    : {sum(train_count.values())}")
    add(f"val   图片总数    : {sum(val_count.values())}")
    add(f"图片总数          : {sum(train_count.values()) + sum(val_count.values())}")
    add(f"压缩包中出现的类别: {len(valid)}")
    add(f"无图片的类别      : {len(missing)}")
    if missing:
        add("  缺失 ID 示例    : " + ", ".join(r["id"] for r in missing[:30]))
    add("")

    tr = [r["train"] for r in valid]
    va = [r["val"] for r in valid]
    for label, arr in (("train/类", tr), ("val/类", va)):
        arr_sorted = sorted(arr)
        n = len(arr_sorted)
        add(
            f"{label:<10} min={arr_sorted[0]:3d}  p25={arr_sorted[n // 4]:3d}  "
            f"median={arr_sorted[n // 2]:3d}  p75={arr_sorted[3 * n // 4]:3d}  max={arr_sorted[-1]:3d}  "
            f"mean={sum(arr) / n:.1f}"
        )
    add("")

    add("-" * 78)
    add("按中文大类分布（类别数 / 图片数）")
    add("-" * 78)
    cat_stat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in valid:
        cat_stat[r["category"]][0] += 1
        cat_stat[r["category"]][1] += r["total"]
    for cat, (cnt, imgs) in sorted(cat_stat.items(), key=lambda kv: -kv[1][0]):
        add(f"  {cat:<12} 类别 {cnt:4d}   图片 {imgs:7d}   平均 {imgs / cnt:6.1f}")
    add("")

    add("-" * 78)
    add("按车身类型分布")
    add("-" * 78)
    body_stat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in valid:
        body_stat[r["body_type"]][0] += 1
        body_stat[r["body_type"]][1] += r["total"]
    for body, (cnt, imgs) in sorted(body_stat.items(), key=lambda kv: -kv[1][0])[:30]:
        add(f"  {body:<20} 类别 {cnt:4d}   图片 {imgs:7d}")
    add("")

    add("-" * 78)
    add("品牌数量 Top 40（可用于挑选多品牌知识库）")
    add("-" * 78)
    brand_stat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in valid:
        brand_stat[r["brand"]][0] += 1
        brand_stat[r["brand"]][1] += r["total"]
    add(f"  有效品牌总数: {len(brand_stat)}")
    for brand, (cnt, imgs) in sorted(brand_stat.items(), key=lambda kv: -kv[1][0])[:40]:
        add(f"  {brand:<18} 类别 {cnt:4d}   图片 {imgs:7d}")

    txt_path = REPORT_DIR / "inventory_summary.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[写出] {csv_path}")
    print(f"[写出] {txt_path}")


if __name__ == "__main__":
    main()
