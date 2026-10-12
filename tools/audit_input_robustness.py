# -*- coding: utf-8 -*-
"""输入健壮性探测：边界值、类型错误、畸形输入会不会造成 500。"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

import requests  # noqa: E402

B = "http://127.0.0.1:5000"
SID = requests.post(B + "/api/session", timeout=15).json()["session"]["session_id"]
print(f"会话 {SID}\n")

CASES = [
    ("GET",  "/api/records?limit=abc", None, "limit 非数字"),
    ("GET",  "/api/records?limit=-5", None, "limit 负数"),
    ("GET",  "/api/records?limit=99999999999999999999", None, "limit 溢出"),
    ("GET",  "/api/records?scope=" + "x" * 5000, None, "超长 scope"),
    ("GET",  "/api/tools?session_id=" + "y" * 5000, None, "超长 session_id"),
    ("POST", "/api/session", {}, "空 body（应能建会话）"),
    ("POST", "/api/chat", {}, "chat 空 body"),
    ("POST", "/api/chat", {"session_id": "不存在", "message": "hi"}, "不存在的会话"),
    ("POST", "/api/chat", {"session_id": SID, "message": ""}, "空消息"),
    ("POST", "/api/chat", {"session_id": SID, "message": "x" * 3000}, "超长消息"),
    ("POST", f"/api/sessions/{SID}/web-search", {}, "web-search 空 body"),
    ("POST", f"/api/sessions/{SID}/web-search", {"enabled": "yes"}, "enabled 传字符串"),
    ("GET",  "/api/sessions/" + "z" * 300, None, "超长 sid"),
    ("GET",  "/api/events/不存在", None, "不存在会话的事件"),
    ("DELETE", "/api/records/999999999", None, "删除不存在的记录"),
    ("DELETE", "/api/records/abc", None, "record id 非数字"),
]

print(f"{'方法':<7} {'用例':<26} {'状态':<6} 响应摘要")
print("-" * 100)
for method, path, body, desc in CASES:
    try:
        if method == "GET":
            r = requests.get(B + path, timeout=20)
        elif method == "DELETE":
            r = requests.delete(B + path, timeout=20)
        else:
            r = requests.post(B + path, json=body, timeout=20)
        code = r.status_code
        try:
            txt = json.dumps(r.json(), ensure_ascii=False)[:60]
        except Exception:
            txt = r.text[:60].replace("\n", " ")
        flag = "🔴" if code >= 500 else ("🟡" if code == 404 else "  ")
        print(f"{method:<7} {desc:<26} {code:<6} {flag} {txt}")
    except Exception as e:
        print(f"{method:<7} {desc:<26} {'EXC':<6} ❌ {type(e).__name__}: {e}")
