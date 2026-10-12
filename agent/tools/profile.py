# -*- coding: utf-8 -*-
"""车型资料库工具：`query_car_info` / `list_supported_cars`

数据来自 `agent/data/car_profiles.json`（由 `tools/build_car_profiles.py` 从
教师 `class_info.json` + 本组 `selection.json` + `data_stats.json` 聚合而成）。

**不编造参数**：教师数据集只提供类别 ID 与车型名称，没有排量/功率/价格/尺寸，
所以本工具只返回真实存在的信息，并在 `data_source` 字段里写明每个字段的来源。
"""
from __future__ import annotations

import json
import threading
from typing import Any

from ..errors import Code, ToolError
from .registry import ToolContext, ToolDefinition

_LOCK = threading.Lock()
_CACHE: dict[str, Any] | None = None
_CACHE_PATH: str | None = None


def load_profiles(cfg) -> dict:
    global _CACHE, _CACHE_PATH
    path = str(cfg.profiles_file)
    with _LOCK:
        if _CACHE is not None and _CACHE_PATH == path:
            return _CACHE
        if not cfg.profiles_file.exists():
            raise ToolError(
                Code.CAR_NOT_FOUND,
                "车型档案库不存在",
                hint="请先运行：python tools/build_car_profiles.py",
            )
        _CACHE = json.loads(cfg.profiles_file.read_text(encoding="utf-8"))
        _CACHE_PATH = path
        return _CACHE


def resolve_car(cfg, key: str) -> dict:
    """把用户/模型给的各种写法解析成唯一的车型档案。

    支持：四位 ID（'0975'）、完整名称（'日产_轩逸'）、车型名（'轩逸'）、
    带品牌或空格的写法（'日产 轩逸'）、大小写不敏感的英文别名。
    解析不到唯一结果时抛出 CAR_NOT_FOUND，并把候选一并返回，便于模型修正。
    """
    data = load_profiles(cfg)
    profiles: dict[str, dict] = data["profiles"]
    key = (key or "").strip()
    if not key:
        raise ToolError(Code.INVALID_ARGUMENTS, "car_id_or_name 不能为空")

    if key in profiles:                       # 精确 ID
        return profiles[key]

    k = key.replace(" ", "").replace("　", "").lower()
    exact, partial = [], []
    for cid, p in profiles.items():
        names = [p["name"].replace(" ", "").lower(), p["name"].split("_")[-1].lower()]
        if k in names:
            exact.append(p)
        elif any(k in n for n in names) or k in p["brand"].lower():
            partial.append(p)

    hits = exact or partial
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ToolError(
            Code.CAR_NOT_FOUND,
            f"资料库里没有与「{key}」匹配的车型",
            hint="本资料库只收录 50 款车型，可用 list_supported_cars 查看全部清单",
        )
    raise ToolError(
        Code.CAR_NOT_FOUND,
        f"「{key}」匹配到 {len(hits)} 款车型，无法确定是哪一款",
        hint="请让用户确认，或使用四位 car_id 精确指定",
        candidates=[{"car_id": h["car_id"], "car_name": h["name"], "brand": h["brand"]} for h in hits[:8]],
    )


# --------------------------------------------------------------------------- #
# query_car_info
# --------------------------------------------------------------------------- #
def _handler_query(args: dict, ctx: ToolContext) -> dict:
    cfg = ctx.config
    ctx.check_abort()
    p = resolve_car(cfg, args.get("car_id_or_name", ""))

    # 档次（配置版本）数据是**联网抓取沉淀下来**的，可能还没有。
    # 有就一并返回，并说清来源与更新时间；没有就明确告知"可以联网去查"，不要瞎猜。
    trim_entry = None
    try:
        from .trims import get_car_trims, summarize_trims
        trim_entry = get_car_trims(cfg, p["car_id"])
    except Exception:                                # noqa: BLE001
        pass

    out = {
        "ok": True,
        "car_id": p["car_id"],
        "car_name": p["name"],
        "brand": p["brand"],
        "body_group": p["body_group"],            # 轿车 / SUV / MPV / 跑车 / 微面 / 轻客
        "body_detail": p["body_detail"],
        "intro": p["intro"],
        "highlights": p["highlights"],
        "aliases": p["aliases"],
        "dataset": {
            "train_samples": p["train_samples"],
            "val_samples": p["val_samples"],
            "model_index": p["index"],
        },
        "data_source": p["data_source"],
    }

    if trim_entry and trim_entry.get("trims"):
        from .trims import summarize_trims as _sum
        out["trims"] = trim_entry["trims"]
        out["trim_count"] = len(trim_entry["trims"])
        out["trim_summary"] = _sum(trim_entry)
        out["trim_sources"] = trim_entry.get("sources", [])
        out["trim_updated_at"] = trim_entry.get("updated_at")
        out["trim_note"] = ("档次/价格/参数来自联网抓取（见 trim_sources），"
                            "是**参考数据**，以厂商官方配置表为准。"
                            "要看各档次差别用 compare_car_trims。")
    else:
        out["trims"] = []
        out["trim_count"] = 0
        out["trim_note"] = ("档案库里还没有这款车的档次/价格数据 —— 这部分教师数据集没有提供，"
                            "需要联网查。若用户问价格或配置差别，请先开启联网搜索，"
                            "用 web_search + browse_page 去查，再用 save_car_trims 存进来，"
                            "**不要凭记忆或猜测回答价格**。")
    return out


# --------------------------------------------------------------------------- #
# list_supported_cars
# --------------------------------------------------------------------------- #
def _handler_list(args: dict, ctx: ToolContext) -> dict:
    cfg = ctx.config
    ctx.check_abort()
    data = load_profiles(cfg)
    profiles: dict[str, dict] = data["profiles"]

    group = args.get("body_group")
    brand = args.get("brand")
    items = list(profiles.values())
    if group:
        items = [p for p in items if p["body_group"] == group]
    if brand:
        items = [p for p in items if p["brand"] == brand]

    groups = sorted({p["body_group"] for p in profiles.values()})
    brands = sorted({p["brand"] for p in profiles.values()})

    return {
        "ok": True,
        "count": len(items),
        "total_supported": len(profiles),
        "filter": {"body_group": group, "brand": brand},
        "available_body_groups": groups,
        "available_brands": brands,
        "cars": [
            {"car_id": p["car_id"], "car_name": p["name"], "brand": p["brand"],
             "body_group": p["body_group"], "body_detail": p["body_detail"]}
            for p in sorted(items, key=lambda x: x["car_id"])
        ],
        "note": "这是本组视觉模型支持的 50 款车型清单；不在其中的车辆会被识别为 unknown。",
    }


PROFILE_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="query_car_info",
        description=(
            "查询某一款车型的档案：品牌、车身大类与细分、简介、看点标签、数据集样本量，"
            "以及**已积累的档次数据**（各配置版本的价格与参数，如果之前联网查过并存进了档案库）。"
            "识别出车型之后、或用户想看某款车资料时调用。"
            "参数可以是四位 car_id（如 \"0975\"），也可以是车型名（如 \"轩逸\"）。"
            "只能查询本组支持的 50 款车型，查不到时本工具会如实报错，不要编造资料。\n"
            "注意：如果返回的 `trims` 为空，说明这款车的价格/配置还没查过 —— "
            "教师数据集不含这类信息，需要联网查（web_search + browse_page + save_car_trims），"
            "**不要凭记忆给价格**。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "car_id_or_name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 40,
                    "description": "四位类别 ID 或车型名称，例如 \"0975\" / \"日产_轩逸\" / \"轩逸\"",
                },
            },
            "required": ["car_id_or_name"],
            "additionalProperties": False,
        },
        handler=_handler_query,
        timeout_s=10.0,
        tags=("knowledge",),
    ),
    ToolDefinition(
        name="list_supported_cars",
        description=(
            "列出本组视觉模型支持的 50 款车型清单，可按车身大类（轿车/SUV/MPV/跑车/微面/轻客）"
            "或品牌筛选。用户问「你们能识别哪些车」「有哪些 SUV」时调用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "body_group": {
                    "type": "string",
                    "enum": ["轿车", "SUV", "MPV", "跑车", "微面", "轻客"],
                    "description": "可选，按车身大类筛选",
                },
                "brand": {"type": "string", "description": "可选，按品牌筛选，如 \"日产\""},
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=_handler_list,
        timeout_s=10.0,
        tags=("knowledge",),
    ),
]
