# -*- coding: utf-8 -*-
"""会话：状态 + **append-only 事件日志** + 持久化。

设计参考主流 agent harness 的做法：**会话的真身是一份只追加的事件日志**，
内存里的对象只是这份日志的投影。好处是：

  · 崩溃/刷新后可以把日志重放回原状（Web 端刷新页面 == 恢复会话）；
  · 每一次工具调用、每一个错误都留痕，天然满足任务书"提供工具调用记录"的要求；
  · 报告里可以直接展示某个会话的完整事件时间线。

事件类型（与 Codex rollout 的分层思路对应）：

    session_meta      会话头：创建时间、模型、提示词版本
    turn_context      每一轮开始时的快照：上传的图片、阈值、可用工具
    user_message      用户输入
    assistant_message 模型回复（可能带 tool_calls）
    tool_result       工具执行结果（含耗时与错误码）
    error             本轮异常
    turn_end          本轮结束：最终答复、工具调用次数
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SESSION_DIR_NAME = "sessions"
_INDEX_LOCK = threading.Lock()


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Session:
    session_id: str
    config: Any
    image_path: str | None = None
    messages: list[dict] = field(default_factory=list)
    last_prediction: dict | None = None
    record_count: int = 0
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    turns: int = 0
    # 会话标题：默认取用户的第一句话，用户也可以在侧边栏里改名
    title: str = ""
    # 联网搜索开关：由界面上的「🌐 联网搜索」按钮控制，逐会话独立
    web_search_enabled: bool = False
    _log_path: Path | None = None

    # ------------------------------------------------------------------ #
    def display_title(self) -> str:
        """给侧边栏用的标题。"""
        if self.title:
            return self.title
        for m in self.messages:
            if m.get("role") == "user" and (m.get("content") or "").strip():
                text = re.sub(r"\s+", " ", m["content"]).strip()
                return text[:28] + ("…" if len(text) > 28 else "")
        return "新会话"

    def set_title(self, title: str) -> None:
        self.title = (title or "").strip()[:60]
        self.append_event("session_title", {"title": self.title})

    def set_web_search(self, enabled: bool) -> None:
        self.web_search_enabled = bool(enabled)
        self.append_event("web_search_toggled", {"enabled": self.web_search_enabled})

    # ------------------------------------------------------------------ #
    @property
    def log_path(self) -> Path:
        if self._log_path is None:
            d = self.config.data_dir / SESSION_DIR_NAME
            d.mkdir(parents=True, exist_ok=True)
            self._log_path = d / f"{self.session_id}.jsonl"
        return self._log_path

    def append_event(self, etype: str, payload: dict) -> None:
        """追加一条事件（append + flush，保证崩溃时也能读到最后一条）。"""
        row = {"ts": _now(), "type": etype, "session_id": self.session_id, "payload": payload}
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass
        self.updated_at = row["ts"]

    # ------------------------------------------------------------------ #
    # 模型可见的消息历史
    # ------------------------------------------------------------------ #
    def add_user(self, text: str, image_path: str | None = None) -> None:
        if image_path:
            self.image_path = image_path
        self.messages.append({"role": "user", "content": text})
        self.append_event("user_message", {"text": text, "image": image_path})

    def add_assistant(self, message: dict) -> None:
        self.messages.append(message)

    def add_tool_result(self, call_id: str, name: str, content: str, ok: bool,
                        code: str = "", elapsed_ms: float = 0.0) -> None:
        # `_tool_name` 是本项目内部标记，离线替身与 UI 需要知道这条结果属于哪个工具；
        # 发往真实接口前会被剥离（见 public_messages()）。
        self.messages.append({
            "role": "tool", "tool_call_id": call_id, "content": content,
            "_tool_name": name,
        })
        self.append_event("tool_result", {
            "tool_call_id": call_id, "tool": name, "ok": ok, "code": code,
            "elapsed_ms": elapsed_ms,
            "result": self._parse_for_log(content),
        })

    @staticmethod
    def _parse_for_log(content: str) -> Any:
        """把工具结果转成可写进事件日志的对象。

        事件日志里的 `result` 只是**审计/展示用的副本**，绝不能因为解析失败
        反过来打断一轮真实对话 —— 曾经就出过这个事故：结果被截断成非法 JSON，
        这里的 `json.loads` 抛 JSONDecodeError，整个 turn 直接失败、回复 0 字。
        （截断源头已在 loop.truncate_tool_result() 修好，这里是第二道防线。）
        """
        if not content or not content.lstrip().startswith(("{", "[")):
            return content
        try:
            return json.loads(content)
        except (ValueError, TypeError):
            return content

    def public_messages(self) -> list[dict]:
        """剥离内部字段后发给模型的消息列表。"""
        out = []
        for m in self.messages:
            clean = {k: v for k, v in m.items() if not k.startswith("_")}
            out.append(clean)
        return out

    def reset_history(self) -> None:
        """清空对话历史（保留上传的图片与档案）。"""
        self.messages.clear()
        self.append_event("history_reset", {})

    # ------------------------------------------------------------------ #
    def to_dict(self, *, with_messages: bool = False) -> dict:
        d = {
            "session_id": self.session_id,
            "title": self.display_title(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turns": self.turns,
            "has_image": bool(self.image_path),
            "image_name": Path(self.image_path).name if self.image_path else None,
            "last_prediction": self.last_prediction,
            "record_count": self.record_count,
            "message_count": len(self.messages),
            "web_search_enabled": bool(self.web_search_enabled),
        }
        if with_messages:
            d["messages"] = self.public_messages()
        return d

    # ------------------------------------------------------------------ #
    def events(self, limit: int = 200) -> list[dict]:
        """读取事件日志（供 UI 展示时间线 / 报告引用）。"""
        if not self.log_path.exists():
            return []
        rows = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows[-limit:]


class SessionStore:
    """会话仓库：创建、按 id 取出、持久化索引。"""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.dir = config.data_dir / SESSION_DIR_NAME
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = config.data_dir / "sessions_index.json"
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def create(self, session_id: str | None = None) -> Session:
        sid = session_id or f"s-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
        with self._lock:
            s = Session(session_id=sid, config=self.config)
            s.append_event("session_meta", {
                "model": self.config.model,
                "offline": bool(self.config.offline),
                "vision_checkpoints": [str(p) for p in self.config.checkpoint_paths() if p.exists()],
                "unknown_threshold": self.config.unknown_threshold,
                "display_confidence_threshold": self.config.display_confidence_threshold,
                "prompt_version": "cardex-agent-1.0",
            })
            self._sessions[sid] = s
            self._write_index()
            return s

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]
        # 不在内存里 → 尝试从事件日志重放
        s = self._replay(session_id)
        if s is not None:
            with self._lock:
                self._sessions[session_id] = s
        return s

    def get_or_create(self, session_id: str | None) -> Session:
        if session_id:
            s = self.get(session_id)
            if s is not None:
                return s
            return self.create(session_id)
        return self.create()

    def rebind(self, config: Any) -> None:
        """配置变更后，把仓库与已加载的会话指向新配置（保留内存里的会话状态）。

        界面上「设置」改了模型 / Key / 阈值之后要立刻生效，
        但用户正在聊的那些会话不能因此丢掉。
        """
        self.config = config
        with self._lock:
            for s in self._sessions.values():
                s.config = config

    def list(self, limit: int = 50) -> list[dict]:
        rows = []
        for p in sorted(self.dir.glob("*.jsonl"), key=lambda x: -x.stat().st_mtime)[:limit]:
            sid = p.stem
            s = self.get(sid)
            rows.append(s.to_dict() if s else {"session_id": sid})
        return rows

    # ------------------------------------------------------------------ #
    def _replay(self, session_id: str) -> Session | None:
        path = self.dir / f"{session_id}.jsonl"
        if not path.exists():
            return None
        s = Session(session_id=session_id, config=self.config)
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            t, p = ev.get("type"), ev.get("payload") or {}
            s.updated_at = ev.get("ts", s.updated_at)
            if t == "user_message":
                s.messages.append({"role": "user", "content": p.get("text", "")})
                if p.get("image"):
                    s.image_path = p["image"]
            elif t == "assistant_message":
                msg = p.get("message") or {}
                if msg:
                    s.messages.append(msg)
            elif t == "tool_result":
                s.messages.append({
                    "role": "tool", "tool_call_id": p.get("tool_call_id", ""),
                    "content": json.dumps(p.get("result"), ensure_ascii=False, default=str),
                    "_tool_name": p.get("tool", ""),
                })
            elif t == "turn_end":
                s.turns = int(p.get("turn_index", s.turns + 1))
            elif t == "session_title":
                s.title = p.get("title", "")
            elif t == "web_search_toggled":
                s.web_search_enabled = bool(p.get("enabled"))
        return s

    def delete(self, session_id: str) -> bool:
        """删除会话：内存 + 事件日志 + 索引。"""
        with self._lock:
            self._sessions.pop(session_id, None)
        path = self.dir / f"{session_id}.jsonl"
        existed = path.exists()
        try:
            path.unlink(missing_ok=True)
        except OSError:
            return False
        self._write_index()
        return existed

    def list_summaries(self, limit: int = 100) -> list[dict]:
        """侧边栏用的会话列表：按最近更新倒序，带标题与统计。

        只读事件日志的**头尾**来取时间与轮次，不做全量重放——
        会话多了以后全量重放会明显拖慢侧边栏。
        """
        rows = []
        for p in sorted(self.dir.glob("*.jsonl"), key=lambda x: -x.stat().st_mtime)[:limit]:
            sid = p.stem
            s = self.get(sid)                       # 内存里有就复用，没有才重放
            if s is None:
                rows.append({"session_id": sid, "title": "(无法读取)",
                             "updated_at": "", "turns": 0, "record_count": 0,
                             "message_count": 0, "web_search_enabled": False,
                             "has_image": False})
                continue
            d = s.to_dict()
            d["mtime"] = p.stat().st_mtime
            rows.append(d)
        return rows

    def _write_index(self) -> None:
        try:
            with _INDEX_LOCK:
                data = [s.to_dict() for s in self._sessions.values()]
                self.index_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
