# -*- coding: utf-8 -*-
"""看车档案工具：`add_car_record` / `search_my_records` / `update_record_note` /
`delete_car_record` / `my_garage_stats`

这是「私人车型资料库 / 对话式看车档案」选题的**业务落点**：
视觉识别的结果不是终点，而是被写进用户自己的档案，之后能按车型、品牌、车身类型检索。

存储：SQLite（`agent/data/garage.db`），按 `session_id` 隔离不同会话的档案。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any

from ..errors import Code, ToolError
from .profile import load_profiles, resolve_car
from .registry import ToolContext, ToolDefinition

_LOCK = threading.Lock()
_DB_PATH: str | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS car_records (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    car_id      TEXT    NOT NULL,
    car_name    TEXT    NOT NULL,
    brand       TEXT,
    body_group  TEXT,
    confidence  REAL,
    image_path  TEXT,
    note        TEXT,
    source      TEXT    NOT NULL DEFAULT 'vision',
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_session ON car_records(session_id);
CREATE INDEX IF NOT EXISTS idx_records_car     ON car_records(session_id, car_id);
"""

# 后加的列：老库要能平滑升级（CREATE TABLE IF NOT EXISTS 不会给已有表加列）
MIGRATIONS = [
    ("trim", "ALTER TABLE car_records ADD COLUMN trim TEXT"),
    ("price_wan", "ALTER TABLE car_records ADD COLUMN price_wan REAL"),
]


def _migrate(conn: sqlite3.Connection) -> None:
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(car_records)").fetchall()}
    for name, sql in MIGRATIONS:
        if name not in cols:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass
    conn.commit()


def get_conn(cfg) -> sqlite3.Connection:
    global _DB_PATH
    path = str(cfg.garage_db)
    cfg.garage_db.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        if _DB_PATH != path:
            conn = sqlite3.connect(path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.executescript(SCHEMA)
            conn.commit()
            # 复用同一个连接会导致并发问题，这里改为每次新建
            conn.close()
            _DB_PATH = path
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def db(cfg):
    conn = get_conn(cfg)
    try:
        _migrate(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _resolve_target(args: dict, ctx: ToolContext) -> tuple[str, str, float | None, str]:
    """确定要写入档案的车型。

    多轮对话的关键：用户第二轮说「帮我记录下来」时不会再上传图片，
    此时模型可以**省略 car_id**，本工具回退到本会话最近一次识别结果
    （`ctx.session.last_prediction`），从而无需重新识别。
    """
    car_id = (args.get("car_id") or "").strip()
    confidence = args.get("confidence")

    if car_id:
        if car_id.lower() == "unknown":
            raise ToolError(
                Code.UNKNOWN_CAR,
                "不能把 unknown 作为具体车型写入档案",
                hint="先请用户手动指定是哪一款车型，再用该车型的 car_id 调用本工具。"
                     "如果只是想备忘，可以把「未能识别」写进 note 里并指定一个已知车型。",
            )
        p = resolve_car(ctx.config, car_id)
        return p["car_id"], p["name"], confidence, "explicit"

    last = getattr(ctx.session, "last_prediction", None)
    if last and not last.get("is_unknown"):
        return last["car_id"], last["car_name"], last.get("confidence"), "from_last_prediction"

    if last and last.get("is_unknown"):
        raise ToolError(
            Code.UNKNOWN_CAR,
            "上一次识别结果是 unknown，无法据此写入档案",
            hint="请先请用户手动选择车型，或重新上传一张更清晰的照片。",
        )
    raise ToolError(
        Code.DEPENDENCY_MISSING,
        "没有可以写入的车型",
        hint="请先调用 classify_car 识别图片，或在参数里显式给出 car_id。",
    )


# --------------------------------------------------------------------------- #
# 工具实现
# --------------------------------------------------------------------------- #
def _handler_add(args: dict, ctx: ToolContext) -> dict:
    cfg = ctx.config
    ctx.check_abort()
    car_id, car_name, confidence, how = _resolve_target(args, ctx)
    profiles = load_profiles(cfg)["profiles"]
    meta = profiles.get(car_id, {})

    note = (args.get("note") or "").strip()
    # 用户自己看中的是哪一档 —— 从档案库里已有的档次里核对一下，对不上就如实标注
    trim = (args.get("trim") or "").strip()[:80]
    price_wan = None
    trim_matched = None
    if trim:
        try:
            from .trims import get_car_trims
            entry = get_car_trims(cfg, car_id) or {}
            for t in entry.get("trims") or []:
                if trim == t.get("name") or trim in (t.get("name") or ""):
                    trim_matched = t.get("name")
                    price_wan = t.get("guide_price_wan")
                    break
        except Exception:                            # noqa: BLE001
            pass

    image_path = args.get("image_path") or ctx.image_path
    now = _now()

    with db(cfg) as conn:
        # 同一会话里同一车型 + 同一张图，视为重复
        dup = conn.execute(
            "SELECT id FROM car_records WHERE session_id=? AND car_id=? AND IFNULL(image_path,'')=IFNULL(?,'')",
            (ctx.session_id, car_id, image_path),
        ).fetchone()
        if dup:
            conn.execute(
                "UPDATE car_records SET note=?, updated_at=?, confidence=?,"
                " trim=COALESCE(?, trim), price_wan=COALESCE(?, price_wan) WHERE id=?",
                (note or None, now, confidence, trim or None, price_wan, dup["id"]),
            )
            rid = dup["id"]
            duplicated = True
        else:
            cur = conn.execute(
                "INSERT INTO car_records (session_id, car_id, car_name, brand, body_group,"
                " confidence, image_path, note, trim, price_wan, source, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ctx.session_id, car_id, car_name, meta.get("brand"), meta.get("body_group"),
                 confidence, image_path, note or None, trim or None, price_wan,
                 "vision", now, now),
            )
            rid = cur.lastrowid
            duplicated = False
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM car_records WHERE session_id=?", (ctx.session_id,)
        ).fetchone()["n"]

    out = {
        "ok": True,
        "record_id": rid,
        "duplicated": duplicated,
        "car_id": car_id,
        "car_name": car_name,
        "brand": meta.get("brand"),
        "body_group": meta.get("body_group"),
        "note": note or None,
        "trim": trim or None,
        "confidence": confidence,
        "resolved_from": how,
        "total_records_in_session": total,
        "created_at": now,
        "message": ("该车型已在你的看车档案中，已更新" if duplicated
                    else f"已把 {car_name} 写入你的看车档案"),
    }
    if trim:
        if trim_matched:
            out["trim_matched"] = trim_matched
            out["guide_price_wan"] = price_wan
            out["message"] += f"（档次：{trim_matched}" + (
                f"，指导价 {price_wan:g} 万）" if price_wan else "）")
        else:
            out["trim_unverified"] = True
            out["message"] += f"（档次：{trim}，档案库里还没有这款车的档次数据，未做核对）"
    return out


def _handler_search(args: dict, ctx: ToolContext) -> dict:
    cfg = ctx.config
    ctx.check_abort()
    scope = args.get("scope", "session")
    limit = int(args.get("limit") or 20)

    where, params = [], []
    if scope == "session":
        where.append("session_id=?")
        params.append(ctx.session_id)
    if args.get("car_id"):
        p = resolve_car(cfg, args["car_id"])
        where.append("car_id=?")
        params.append(p["car_id"])
    if args.get("brand"):
        where.append("brand=?")
        params.append(args["brand"])
    if args.get("body_group"):
        where.append("body_group=?")
        params.append(args["body_group"])
    if args.get("keyword"):
        kw = f"%{args['keyword']}%"
        where.append("(car_name LIKE ? OR brand LIKE ? OR IFNULL(note,'') LIKE ?)")
        params += [kw, kw, kw]

    sql = "SELECT * FROM car_records"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY datetime(created_at) DESC, id DESC LIMIT ?"
    params.append(limit)

    with db(cfg) as conn:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM car_records WHERE session_id=?", (ctx.session_id,)
        ).fetchone()["n"]

    return {
        "ok": True,
        "scope": scope,
        "count": len(rows),
        "total_in_session": total,
        "records": [
            {
                "record_id": r["id"],
                "car_id": r["car_id"],
                "car_name": r["car_name"],
                "brand": r["brand"],
                "body_group": r["body_group"],
                "confidence": r["confidence"],
                "note": r["note"],
                "trim": (r["trim"] if "trim" in r.keys() else None),
                "guide_price_wan": (r["price_wan"] if "price_wan" in r.keys() else None),
                "created_at": r["created_at"],
                "image": (r["image_path"].split("/")[-1].split("\\")[-1] if r["image_path"] else None),
            }
            for r in rows
        ],
        "note": "scope=session 只查当前会话的档案；scope=all 查这台机器上所有会话的档案。",
    }


def _handler_update(args: dict, ctx: ToolContext) -> dict:
    cfg = ctx.config
    ctx.check_abort()
    rid = int(args["record_id"])
    note = (args.get("note") or "").strip()
    with db(cfg) as conn:
        row = conn.execute(
            "SELECT * FROM car_records WHERE id=? AND session_id=?", (rid, ctx.session_id)
        ).fetchone()
        if row is None:
            raise ToolError(Code.RECORD_NOT_FOUND, f"档案记录 #{rid} 不存在",
                            hint="可以先用 search_my_records 查看当前会话里的记录编号")
        conn.execute("UPDATE car_records SET note=?, updated_at=? WHERE id=?",
                     (note or None, _now(), rid))
    return {"ok": True, "record_id": rid, "car_name": row["car_name"], "note": note or None,
            "message": f"已更新 {row['car_name']} 的备注"}


def _handler_delete(args: dict, ctx: ToolContext) -> dict:
    cfg = ctx.config
    ctx.check_abort()
    rid = int(args["record_id"])
    with db(cfg) as conn:
        row = conn.execute(
            "SELECT * FROM car_records WHERE id=? AND session_id=?", (rid, ctx.session_id)
        ).fetchone()
        if row is None:
            raise ToolError(Code.RECORD_NOT_FOUND, f"档案记录 #{rid} 不存在",
                            hint="可以先用 search_my_records 查看当前会话里的记录编号")
        conn.execute("DELETE FROM car_records WHERE id=?", (rid,))
    return {"ok": True, "record_id": rid, "car_name": row["car_name"],
            "message": f"已从看车档案中删除 {row['car_name']}"}


def _handler_stats(args: dict, ctx: ToolContext) -> dict:
    cfg = ctx.config
    ctx.check_abort()
    scope = args.get("scope", "session")
    cond, params = ("WHERE session_id=?", [ctx.session_id]) if scope == "session" else ("", [])
    with db(cfg) as conn:
        total = conn.execute(f"SELECT COUNT(*) n FROM car_records {cond}", params).fetchone()["n"]
        by_brand = [dict(r) for r in conn.execute(
            f"SELECT brand, COUNT(*) n FROM car_records {cond} GROUP BY brand ORDER BY n DESC, brand",
            params).fetchall()]
        by_group = [dict(r) for r in conn.execute(
            f"SELECT body_group, COUNT(*) n FROM car_records {cond} "
            f"GROUP BY body_group ORDER BY n DESC", params).fetchall()]
        recent = [dict(r) for r in conn.execute(
            f"SELECT id, car_id, car_name, created_at FROM car_records {cond} "
            f"ORDER BY datetime(created_at) DESC, id DESC LIMIT 5", params).fetchall()]
    supported = len(load_profiles(cfg)["profiles"])
    return {
        "ok": True,
        "scope": scope,
        "total_records": total,
        "distinct_models": len({r["brand"] for r in by_brand}) if by_brand else 0,
        "by_brand": by_brand,
        "by_body_group": by_group,
        "recent": recent,
        "coverage": {
            "supported_models": supported,
            "collected_records": total,
            "coverage_pct": round(100.0 * total / supported, 1) if supported else 0.0,
        },
    }


GARAGE_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="add_car_record",
        description=(
            "把一款车型写入用户自己的「看车档案」。用户说「记录下来」「加入收藏」「记一下」时调用。"
            "如果刚识别过车型，可以省略 car_id —— 会自动沿用本次会话最近一次识别结果，"
            "无需用户重新上传图片。不要把 unknown 写入档案。\n"
            "如果用户说了自己看中的是哪个**档次**（如「舒适版」「豪华版」），"
            "把它填进 trim —— 会去档案库里核对是否存在该档次，并带出指导价。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "car_id": {
                    "type": "string",
                    "description": "四位类别 ID（如 \"0975\"）。省略时沿用本会话最近一次识别结果。",
                },
                "note": {"type": "string", "maxLength": 500,
                         "description": "可选备注，如用户的原话、看车感受、价格印象等"},
                "trim": {
                    "type": "string", "maxLength": 80,
                    "description": ("可选，用户看中的**具体档次/配置版本**，如 \"2026款 1.6L CVT 舒适版\"。"
                                    "只填用户明确说过的；没说过不要替他猜。"),
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1,
                               "description": "可选，识别置信度；由 classify_car 的结果带入"},
                "image_path": {"type": "string", "description": "可选，关联的图片路径"},
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=_handler_add,
        timeout_s=15.0,
        tags=("archive", "write"),
    ),
    ToolDefinition(
        name="search_my_records",
        description=(
            "检索用户的看车档案。可按车型名/品牌/备注关键词、车型 ID、车身大类筛选。"
            "用户问「我记录过哪些车」「有没有记过 SUV」「上次看的轩逸」时调用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "maxLength": 50,
                            "description": "在车型名、品牌、备注里做模糊匹配"},
                "car_id": {"type": "string", "description": "按车型精确筛选"},
                "brand": {"type": "string", "description": "按品牌筛选，如 \"日产\""},
                "body_group": {"type": "string",
                               "enum": ["轿车", "SUV", "MPV", "跑车", "微面", "轻客"],
                               "description": "按车身大类筛选"},
                "scope": {"type": "string", "enum": ["session", "all"],
                          "description": "session=只查当前会话（默认）；all=查所有会话"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "最多返回几条"},
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=_handler_search,
        timeout_s=15.0,
        tags=("archive", "read"),
    ),
    ToolDefinition(
        name="update_record_note",
        description="修改某条看车档案的备注。需要先用 search_my_records 拿到 record_id。",
        parameters={
            "type": "object",
            "properties": {
                "record_id": {"type": "integer", "minimum": 1, "description": "档案记录编号"},
                "note": {"type": "string", "maxLength": 500, "description": "新的备注内容"},
            },
            "required": ["record_id"],
            "additionalProperties": False,
        },
        handler=_handler_update,
        timeout_s=15.0,
        tags=("archive", "write"),
    ),
    ToolDefinition(
        name="delete_car_record",
        description="从看车档案中删除一条记录。需要先用 search_my_records 拿到 record_id。",
        parameters={
            "type": "object",
            "properties": {
                "record_id": {"type": "integer", "minimum": 1, "description": "档案记录编号"},
            },
            "required": ["record_id"],
            "additionalProperties": False,
        },
        handler=_handler_delete,
        timeout_s=15.0,
        tags=("archive", "write"),
    ),
    ToolDefinition(
        name="my_garage_stats",
        description=(
            "统计用户看车档案的构成：总记录数、按品牌和车身大类的分布、最近记录、"
            "以及对 50 款支持车型的覆盖度。用户问「我的档案里都有什么」时调用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "enum": ["session", "all"],
                          "description": "session=当前会话（默认）；all=所有会话"},
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=_handler_stats,
        timeout_s=15.0,
        tags=("archive", "read"),
    ),
]
