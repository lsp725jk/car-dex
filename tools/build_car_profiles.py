# -*- coding: utf-8 -*-
"""任务二 - 构建「车型档案库」

任务二需要一个「车型信息查询」工具。教师数据集只提供 **类别 ID 与车型名称**，
没有任何排量/价格/尺寸等参数，因此本项目**不编造参数**，只把已有的真实数据聚合起来，
再补一段明确标注来源的概述。

数据来源（全部为本项目已有的真实文件）：
  configs/class_info.json      → 车型 ID、名称（name_from_new）、类型信息（type_info）
  configs/class_map.json       → 模型索引、车身大类
  configs/selection.json       → 本组选类时逐类写下的「选择理由」（人工撰写）
  configs/data_stats.json      → 该车型在本组的训练/验证样本量
  outputs/reports/final_model_choice.md → （可选）该车型的验证集 F1

输出：agent/data/car_profiles.json

用法：
    python tools/build_car_profiles.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONFIG_DIR = PROJECT_ROOT / "configs"
OUT_PATH = PROJECT_ROOT / "agent" / "data" / "car_profiles.json"

# 车身大类 → 中文可读描述（用于生成概述，不是编造参数）
GROUP_DESC = {
    "轿车": "轿车",
    "SUV": "SUV（运动型多用途车）",
    "MPV": "MPV（多用途乘用车）",
    "跑车": "跑车",
    "微面": "微型客车（微面）",
    "轻客": "轻型客车",
    "卡车": "卡车",
}

BODY_DETAIL_CN = {
    "Compact Sedan": "紧凑型三厢轿车",
    "Mid-size Sedan": "中型三厢轿车",
    "Upper Mid-size Sedan": "中大型三厢轿车",
    "Large Sedan": "大型豪华轿车",
    "Small Sedan": "小型轿车",
    "Micro Sedan": "微型轿车",
    "Compact SUV": "紧凑型 SUV",
    "Mid-size SUV": "中型 SUV",
    "Upper Mid-size SUV": "中大型 SUV",
    "Large SUV": "大型 SUV",
    "Small SUV": "小型 SUV",
    "Sports Car": "跑车",
    "Compact MPV": "紧凑型 MPV",
    "Mid-size MPV": "中型 MPV",
    "Upper Mid-size MPV": "中大型 MPV",
    "Small MPV": "小型 MPV",
    "Van": "厢式微客",
    "Bus": "轻型客车",
    "Pickup Truck": "皮卡",
    "Light Truck": "轻卡",
    "Micro Truck": "微卡",
    "Heavy Truck": "重卡",
}


def parse_type_info(ti: str) -> tuple[str, str]:
    """type_info 形如 '卡车####轻卡==Truck####Light Truck' → (中文大类, 英文细分)"""
    if "==" not in ti:
        return "", ""
    head = ti.split("==")[0]
    tail = ti.split("==")[-1]
    return head.split("####")[0].strip(), tail.split("####")[-1].strip()


def compose_intro(name: str, brand: str, group: str, detail_cn: str, n_train: int, n_val: int) -> str:
    """生成一段**只包含可核实事实**的概述。

    刻意不写排量、马力、价格、尺寸等参数——教师数据集没有这些信息，
    编造参数会违反任务书「工具应真实执行对应操作」的要求。
    """
    parts = [f"{name} 是 {brand} 旗下的一款{detail_cn or GROUP_DESC.get(group, group)}。"]
    parts.append(f"本项目把它选入「识车档案」私人资料库的 50 款目标车型之一。")
    parts.append(f"训练集 {n_train} 张、验证集 {n_val} 张，模型输出类别索引见 class_map.json。")
    return "".join(parts)


def main() -> None:
    class_info = {r["id"]: r for r in json.loads((CONFIG_DIR / "class_info.json").read_text(encoding="utf-8"))}
    class_map = json.loads((CONFIG_DIR / "class_map.json").read_text(encoding="utf-8"))
    selection = json.loads((CONFIG_DIR / "selection.json").read_text(encoding="utf-8"))
    stats = json.loads((CONFIG_DIR / "data_stats.json").read_text(encoding="utf-8"))

    reason_by_id = {c["id"]: c.get("reason", "") for c in selection["classes"]}
    group_by_id = {c["id"]: c.get("group", "") for c in selection["classes"]}

    ids = [class_map["index_to_id"][str(i)] for i in range(50)]

    profiles: dict[str, dict] = {}
    for idx, cid in enumerate(ids):
        info = class_info[cid]
        name = info["name_from_new"]
        brand_cn, detail_en = parse_type_info(info["type_info"])
        group = group_by_id.get(cid) or class_map["index_to_group"].get(str(idx), "")
        detail_cn = BODY_DETAIL_CN.get(detail_en, detail_en)
        n_train = stats["per_class"]["train"].get(cid, {}).get("n", 0)
        n_val = stats["per_class"]["val"].get(cid, {}).get("n", 0)

        profiles[cid] = {
            "car_id": cid,
            "index": idx,
            "name": name,
            "brand": name.split("_")[0] if "_" in name else name,
            "body_group": group,
            "body_detail": detail_cn,
            "body_detail_en": detail_en,
            "type_category": brand_cn,
            "intro": compose_intro(name, name.split("_")[0] if "_" in name else name,
                                   group, detail_cn, n_train, n_val),
            # 选类时人工写下的理由，作为「看点/标签」
            "highlights": [s.strip() for s in re.split(r"[，,；;]", reason_by_id.get(cid, "")) if s.strip()][:5],
            "selection_reason": reason_by_id.get(cid, ""),
            "train_samples": n_train,
            "val_samples": n_val,
            "aliases": [a for a in {name, name.replace("_", " "), name.split("_")[-1]} if a],
            "data_source": (
                "车型 ID 与名称来自教师 class_info.json 的 id / name_from_new 字段；"
                "车身类型来自 type_info；看点来自本组 configs/selection.json 的选类理由；"
                "样本量来自 configs/data_stats.json。"
                "本档案不含排量、功率、价格、尺寸等参数——教师数据集未提供这些信息，故未编造。"
            ),
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "count": len(profiles),
            "generated_by": "tools/build_car_profiles.py",
            "note": "私人车型资料库的初始条目。仅使用教师提供与本组人工撰写的真实数据，不含编造参数。",
        },
        "profiles": profiles,
        # 便于按品牌/大类检索的索引
        "by_brand": {},
        "by_group": {},
    }
    for cid, p in profiles.items():
        payload["by_brand"].setdefault(p["brand"], []).append(cid)
        payload["by_group"].setdefault(p["body_group"], []).append(cid)

    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[写出] {OUT_PATH}  （{len(profiles)} 个车型档案）")
    print(f"       品牌 {len(payload['by_brand'])} 个，车身大类 {len(payload['by_group'])} 个")
    print()
    print("抽样预览：")
    for cid in ["0975", "0003", "0283" if "0283" in profiles else ids[0]]:
        p = profiles.get(cid)
        if not p:
            continue
        print(f"  [{p['car_id']}] {p['name']}  ({p['brand']} / {p['body_group']} / {p['body_detail']})")
        print(f"        看点：{'；'.join(p['highlights'])}")
        print(f"        概述：{p['intro']}")


if __name__ == "__main__":
    main()
