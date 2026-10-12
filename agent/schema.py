# -*- coding: utf-8 -*-
"""轻量 JSON Schema 校验器（只覆盖本项目用到的关键字）。

为什么不直接用 `jsonschema` 库：本项目要给老师一个"装上依赖就能跑"的系统，
少一个依赖少一处失败点；而且工具参数 schema 是自己写的，
用到的关键字就那么几个，自己实现反而更可控、报错信息也更贴合工具场景。

支持的关键字：
    type           object / array / string / number / integer / boolean / null
    properties, required, additionalProperties
    items, minItems, maxItems, uniqueItems
    enum, const
    minimum, maximum, exclusiveMinimum, exclusiveMaximum
    minLength, maxLength, pattern
    minProperties, maxProperties
    anyOf, oneOf
    default（仅记录，不参与校验）
"""
from __future__ import annotations

import re
from typing import Any

JsonValue = Any


class SchemaError(ValueError):
    """校验失败。`path` 指出出错位置，便于直接回给模型。"""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}" if path else message)
        self.path = path
        self.message = message


_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def _type_ok(value: JsonValue, t: str) -> bool:
    if t == "integer":
        # 注意 bool 是 int 的子类，必须排除
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    if t in _TYPE_MAP:
        return isinstance(value, _TYPE_MAP[t])
    return True


def validate(value: JsonValue, schema: dict, path: str = "$") -> None:
    """校验失败抛 SchemaError；成功返回 None。"""
    if not isinstance(schema, dict):
        return

    # ---- 组合关键字 ----
    if "anyOf" in schema:
        errs = []
        for sub in schema["anyOf"]:
            try:
                validate(value, sub, path)
                return
            except SchemaError as e:
                errs.append(e.message)
        raise SchemaError(path, "不满足 anyOf 中的任何一个：" + "；".join(errs))

    if "oneOf" in schema:
        matched = 0
        for sub in schema["oneOf"]:
            try:
                validate(value, sub, path)
                matched += 1
            except SchemaError:
                pass
        if matched != 1:
            raise SchemaError(path, f"oneOf 要求恰好匹配一种类型，实际匹配 {matched} 种")
        return

    if "const" in schema and value != schema["const"]:
        raise SchemaError(path, f"必须是 {schema['const']!r}")

    if "enum" in schema and value not in schema["enum"]:
        raise SchemaError(path, f"必须是以下之一：{schema['enum']}")

    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_type_ok(value, x) for x in types):
            raise SchemaError(path, f"类型应为 {'/'.join(types)}，实际是 {type(value).__name__}")

    # ---- 数值 ----
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaError(path, f"不能小于 {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaError(path, f"不能大于 {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise SchemaError(path, f"必须大于 {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            raise SchemaError(path, f"必须小于 {schema['exclusiveMaximum']}")

    # ---- 字符串 ----
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise SchemaError(path, f"长度不能少于 {schema['minLength']} 个字符")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise SchemaError(path, f"长度不能超过 {schema['maxLength']} 个字符")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise SchemaError(path, f"不匹配正则 {schema['pattern']}")

    # ---- 数组 ----
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise SchemaError(path, f"至少需要 {schema['minItems']} 个元素")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise SchemaError(path, f"最多允许 {schema['maxItems']} 个元素")
        if schema.get("uniqueItems"):
            seen = []
            for v in value:
                if v in seen:
                    raise SchemaError(path, f"元素不能重复：{v!r}")
                seen.append(v)
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, v in enumerate(value):
                validate(v, item_schema, f"{path}[{i}]")

    # ---- 对象 ----
    if isinstance(value, dict):
        props = schema.get("properties", {})
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            raise SchemaError(path, f"至少需要 {schema['minProperties']} 个字段")
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            raise SchemaError(path, f"最多允许 {schema['maxProperties']} 个字段")

        for key in schema.get("required", []):
            if key not in value:
                raise SchemaError(path, f"缺少必填参数 `{key}`")

        if schema.get("additionalProperties") is False:
            extra = [k for k in value if k not in props]
            if extra:
                raise SchemaError(path, f"出现了未定义的参数 {extra}，只允许 {sorted(props)}")

        for key, sub in props.items():
            if key in value:
                validate(value[key], sub, f"{path}.{key}")


def format_schema_for_prompt(schema: dict) -> str:
    """把 schema 压成一行人类可读的参数说明（用于离线替身与调试输出）。"""
    props = schema.get("properties", {})
    req = set(schema.get("required", []))
    bits = []
    for k, v in props.items():
        t = v.get("type", "?")
        if isinstance(t, list):
            t = "/".join(t)
        mark = "" if k in req else "?"
        extra = ""
        if "enum" in v:
            extra = f"∈{v['enum']}"
        bits.append(f"{mark}{k}:{t}{extra}")
    return ", ".join(bits)
