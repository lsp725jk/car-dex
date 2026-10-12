# -*- coding: utf-8 -*-
"""路径与类别映射的统一定义。"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
RUN_DIR = OUTPUT_DIR / "runs"

CLASSES_TXT = PROJECT_ROOT / "classes.txt"
CLASS_MAP_JSON = CONFIG_DIR / "class_map.json"
CLASS_INFO_JSON = CONFIG_DIR / "class_info.json"
DATA_STATS_JSON = CONFIG_DIR / "data_stats.json"
UNKNOWN_SOURCES_JSON = CONFIG_DIR / "unknown_sources.json"

UNKNOWN_LABEL = "unknown"


def load_class_map() -> dict:
    return json.loads(CLASS_MAP_JSON.read_text(encoding="utf-8"))


def label_names() -> list[str]:
    """返回长度 51 的标签名列表，索引与模型输出一致。"""
    cm = load_class_map()
    names = [cm["index_to_name"][str(i)] for i in range(cm["num_classes"] - 1)]
    names.append(UNKNOWN_LABEL)
    return names


def label_ids() -> list[str]:
    """返回长度 51 的标签 ID 列表（unknown 位置为字符串 unknown）。"""
    cm = load_class_map()
    ids = [cm["index_to_id"][str(i)] for i in range(cm["num_classes"] - 1)]
    ids.append(UNKNOWN_LABEL)
    return ids


def read_classes_txt() -> list[str]:
    return [line.strip() for line in CLASSES_TXT.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_dirs() -> None:
    for d in (REPORT_DIR, CKPT_DIR, RUN_DIR):
        d.mkdir(parents=True, exist_ok=True)
