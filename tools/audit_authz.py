# -*- coding: utf-8 -*-
"""验证 /api/records 的越权与越界读取问题（只读+可回滚的探测）。

发现的两个可疑点：
  ① DELETE /api/records/<id> 在不带 session_id 时会**跳过归属校验**
  ② GET  /api/records 不带 session_id 时会不会返回**所有会话**的记录
"""
from __future__ import annotations

import json
import sys
import threading

sys.path.insert(0, ".")
sys.path.insert(0, "src")

import requests  # noqa: E402

from agent.config import load_config  # noqa: E402
from agent.loop import Agent  # noqa: E402
from agent.session import SessionStore  # noqa: E402
from agent.tools.registry import ToolContext  # noqa: E402

B = "http://127.0.0.1:5000"
cfg = load_config()
agent = Agent(cfg, store=SessionStore(cfg))


def add_record(sid: str, car_id: str, note: str):
    s = agent.store.get_or_create(sid)
    ctx = ToolContext(session_id=sid, turn_id="t", call_id="c", config=cfg,
                      session=s, signal=threading.Event(), image_path=None)
    r = agent.registry.execute("c", "add_car_record",
                               json.dumps({"car_id": car_id, "note": note}), ctx)
    return r.payload.get("record_id") if r.ok else None


def all_records():
    return requests.get(B + "/api/records", params={"scope": "all", "limit": 200},
                        timeout=15).json()["records"]


def main() -> None:
    a = requests.post(B + "/api/session", timeout=15).json()["session"]["session_id"]
    b = requests.post(B + "/api/session", timeout=15).json()["session"]["session_id"]
    ra = add_record(a, "0632", "A会话-权限探测")
    rb = add_record(b, "0222", "B会话-权限探测")
    print(f"会话A={a}  记录#{ra}")
    print(f"会话B={b}  记录#{rb}\n")

    if rb is None or ra is None:
        print("❌ 记录没建成，测试中止")
        return

    print("=" * 74)
    print("① DELETE 不带 session_id —— 能不能删掉别人的记录？")
    print("=" * 74)
    resp = requests.delete(f"{B}/api/records/{rb}", timeout=15)
    print(f"  HTTP {resp.status_code}  {resp.json()}")
    still = rb in [r["id"] for r in all_records()]
    print(f"  记录 #{rb} 还在吗：{still}")
    print(f"  → {'✅ 被拒绝（安全）' if still else '❌ 被删掉了（越权！）'}")

    print()
    print("=" * 74)
    print("② DELETE 带**别人的** session_id")
    print("=" * 74)
    resp = requests.delete(f"{B}/api/records/{ra}", params={"session_id": b}, timeout=15)
    print(f"  HTTP {resp.status_code}  {resp.json()}")
    still = ra in [r["id"] for r in all_records()]
    print(f"  记录 #{ra} 还在吗：{still}")
    print(f"  → {'✅ 被拒绝（安全）' if still else '❌ 被删掉了（越权！）'}")

    print()
    print("=" * 74)
    print("③ GET /api/records 不带 session_id —— 会不会串会话？")
    print("=" * 74)
    r = requests.get(B + "/api/records", params={"limit": 200}, timeout=15).json()
    sids = {x["session_id"] for x in r["records"]}
    print(f"  scope 默认值 = session，但没给 sid → 返回 {len(r['records'])} 条")
    print(f"  涉及 {len(sids)} 个不同会话")
    print(f"  → {'✅ 已限定单会话' if len(sids) <= 1 else f'❌ 跨会话泄露（{len(sids)} 个）'}")

    print()
    print("=" * 74)
    print("清理探测数据")
    print("=" * 74)
    for rid in (ra, rb):
        requests.delete(f"{B}/api/records/{rid}", timeout=15)
    print("  已清理")


if __name__ == "__main__":
    main()
