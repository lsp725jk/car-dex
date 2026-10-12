# -*- coding: utf-8 -*-
"""侦察：教师完整数据集里，还有多少「非目标类」可以用来扩充 unknown。

背景：现状 unknown 训练集只有 98 个来源类 / 1,960 张，
而测试集的 unknown 来自剩下 777 个类 —— 来源分布窄是 unknown_recall
从验证集 95.75% 掉到测试集 78.74% 的主因。

本脚本算出可扩充的空间，并按「品牌 / 车身大类」统计多样性。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "数据集"
INFO = DS / "cls_info" / "class_info.json"


def main() -> None:
    info = json.loads(INFO.read_text(encoding="utf-8"))
    print(f"class_info.json 顶层类型：{type(info).__name__}")
    if isinstance(info, dict):
        print(f"  顶层键（前 8）：{list(info.keys())[:8]}")
        # 找到类列表所在
        cand = None
        for k, v in info.items():
            if isinstance(v, list) and len(v) > 100:
                cand = (k, v)
                break
            if isinstance(v, dict) and len(v) > 100:
                cand = (k, list(v.values()))
                break
        key, items = cand
        print(f"  → 类别表在 '{key}'，共 {len(items)} 项")
        print(f"  样例：{json.dumps(items[0], ensure_ascii=False)}")
    else:
        items = info
        print(f"  共 {len(items)} 项；样例：{json.dumps(items[0], ensure_ascii=False)}")

    # 目标类 & 已用来源类
    cm = json.loads((ROOT / "configs" / "class_map.json").read_text(encoding="utf-8"))
    targets = set(cm["index_to_id"].values())
    us = json.loads((ROOT / "configs" / "unknown_sources.json").read_text(encoding="utf-8"))
    used = {s["id"] for s in us["sources"]}
    print(f"\n目标类 {len(targets)} 个；已用 unknown 来源类 {len(used)} 个")

    def cid(x):
        return str(x.get("id", x.get("class_id", ""))).zfill(4)

    def brand(x):
        # name_from_new 形如 "江铃_凯运"
        return str(x.get("name_from_new", "?_?")).split("_")[0]

    def cat(x):
        # type_info 形如 "卡车####轻卡==Truck####Light Truck"
        t = str(x.get("type_info", ""))
        return t.split("####")[0].split("==")[0] or "?"

    def body(x):
        t = str(x.get("type_info", ""))
        parts = t.split("####")
        if len(parts) < 2:
            return "?"
        return parts[1].split("==")[0] or "?"

    all_ids = {cid(x) for x in items}
    print(f"教师类别总数 {len(all_ids)}")
    rest = all_ids - targets - used
    print(f"目标类 ∪ 已用来源类 = {len(targets | used)}")
    print(f"→ **尚未使用的类别 {len(rest)} 个**")

    # 统计 train/val 图片数
    tr, va = DS / "train", DS / "val"
    tr_pool = va_pool = 0
    per_class = {}
    for d in tr.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            per_class.setdefault(d.name, {})["train"] = len(list(d.glob("*")))
    for d in va.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            per_class.setdefault(d.name, {})["val"] = len(list(d.glob("*")))

    for c in rest:
        per_class.setdefault(c, {})
        tr_pool += per_class[c].get("train", 0)
        va_pool += per_class[c].get("val", 0)
    print(f"\n未使用类别的可用图片：train {tr_pool:,} 张 / val {va_pool:,} 张")

    cur_tr = sum(per_class.get(c, {}).get("train", 0) for c in used)
    print(f"当前 98 个来源类的可用总量：train {cur_tr:,} 张"
          f"（实际采样 {us.get('train_per_source')}/类 = {us['train_per_source'] * len(used):,} 张）")

    # 每类可用张数分布（决定能扩到多宽）
    avail = np.array([per_class.get(c, {}).get("train", 0) for c in rest])
    print(f"\n未使用类别的 train 张数分布："
          f"min {avail.min()}  25% {np.percentile(avail, 25):.0f}  "
          f"中位 {np.median(avail):.0f}  75% {np.percentile(avail, 75):.0f}  max {avail.max()}")
    print(f"  可用 ≥10 张的类别：{int((avail >= 10).sum())} 个")
    print(f"  可用 ≥20 张的类别：{int((avail >= 20).sum())} 个")

    # 多样性统计
    meta = {cid(x): x for x in items}
    br = Counter(brand(meta[c]) for c in rest if c in meta)
    ct = Counter(cat(meta[c]) for c in rest if c in meta)
    bd = Counter(body(meta[c]) for c in rest if c in meta)
    print(f"\n未使用类别的品牌数 {len(br)}；车身大类 {len(ct)}；细分车型 {len(bd)}")
    print(f"  大类分布：{dict(ct.most_common())}")
    print(f"  品牌 Top12：{br.most_common(12)}")

    ub = Counter(brand(meta[c]) for c in used if c in meta)
    uc = Counter(cat(meta[c]) for c in used if c in meta)
    print(f"\n已用 98 个来源类：品牌数 {len(ub)}；大类 {dict(uc.most_common())}")

    out = {
        "n_teacher_classes": len(all_ids),
        "n_targets": len(targets),
        "n_used_sources": len(used),
        "n_unused_classes": len(rest),
        "unused_available": {"train": tr_pool, "val": va_pool},
        "current_sources_available_train": cur_tr,
        "unused_per_class_train": {c: per_class.get(c, {}).get("train", 0) for c in sorted(rest)},
        "unused_brands": len(br), "unused_categories": dict(ct), "unused_body_types": len(bd),
        "used_brands": len(ub), "used_categories": dict(uc),
        "unused_ids": sorted(rest),
    }
    o = ROOT / "outputs" / "reports" / "unknown_expansion_recon.json"
    o.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[写出] {o}")


if __name__ == "__main__":
    main()
