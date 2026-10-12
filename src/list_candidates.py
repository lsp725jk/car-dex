# -*- coding: utf-8 -*-
"""列出候选类别明细，便于按“个人知识库”主题挑选 50 类。

用法：
    python src/list_candidates.py            # 输出全部 827 类（按大类分组）
    python src/list_candidates.py 奔驰 宝马   # 只看指定品牌
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "outputs" / "reports" / "class_inventory.csv"
OUT_PATH = PROJECT_ROOT / "outputs" / "reports" / "candidates.txt"


def main() -> None:
    brands = set(sys.argv[1:])
    with open(CSV_PATH, encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))

    if brands:
        rows = [r for r in rows if r["brand"] in brands]

    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["category"]].append(r)

    lines: list[str] = []
    add = lines.append
    for cat in sorted(grouped, key=lambda c: -len(grouped[c])):
        add("=" * 100)
        add(f"【{cat}】 共 {len(grouped[cat])} 类")
        add("=" * 100)
        add(f"{'ID':<6}{'车型名称':<34}{'品牌':<12}{'车身':<22}{'train':>6}{'val':>5}{'total':>7}")
        add("-" * 100)
        for r in sorted(grouped[cat], key=lambda x: (-int(x["total"]), x["id"])):
            name = r["name_from_new"]
            add(
                f"{r['id']:<6}{name:<34}{r['brand']:<12}{r['body_type']:<22}"
                f"{r['train']:>6}{r['val']:>5}{r['total']:>7}"
            )
        add("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"写出 {OUT_PATH}  共 {len(rows)} 行")


if __name__ == "__main__":
    main()
