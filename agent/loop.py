# -*- coding: utf-8 -*-
"""Agent Loop —— 任务二的核心。

一轮用户请求（turn）的完整流程，结构对标主流 agent harness：

    用户输入
      ↓  记录 turn 快照（本轮生效的图片、阈值、工具清单 —— 一次 turn 一个策略快照）
      ↓
    ┌─ 步骤（step）循环，最多 max_rounds 轮 ──────────────────────────┐
    │  1. 组装请求：系统提示词（分段） + 会话消息历史 + 工具 schema      │
    │  2. 调用模型                                                     │
    │  3. 没有 tool_calls → 这就是最终回答，结束                        │
    │  4. 追加 assistant 消息（含 tool_calls）                          │
    │  5. 逐个执行工具（走注册表流水线：查表→校验→门禁→执行→后置→记账）   │
    │  6. 结果以 role=tool 回填，回到 1                                 │
    └─────────────────────────────────────────────────────────────────┘
      ↓
    记录 turn_end，返回最终回答 + 本轮工具调用记录

刻意做的几个设计决定：

  · **工具结果会截断**（默认 6000 字符），避免一次返回把上下文打爆；
  · **同一工具以相同方式连续失败 3 次即熔断**，防止模型在同一个坑里来回撞；
  · **失败不伪造成功**：任何工具错误都如实回填 `ok:false` + 错误码 + 修正建议；
  · **图片不经过模型**：模型只看到「本会话已上传图片」这个事实，真正的识别
    必须由 `classify_car` 工具完成（任务书 5.1 节明确要求）。
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import prompts
from .errors import AgentError, Code
from .llm import LLMResponse, get_client
from .session import Session, SessionStore
from .tools import ToolContext, ToolRegistry, build_registry, enabled_tool_names

EventCallback = Callable[[dict], None]
MAX_TOOL_RESULT_CHARS = 6000

# 只读且昂贵的联网工具：同样的参数重复调用没有任何收益，直接复用上次结果。
IDEMPOTENT_WEB_TOOLS = {"web_search", "browse_page"}


# --------------------------------------------------------------------------- #
@dataclass
class ToolCallRecord:
    """一次工具调用的记录（用于 UI 时间线与任务书要求的「工具调用记录」）。"""

    index: int
    call_id: str
    name: str
    arguments: dict
    ok: bool
    code: str = ""
    elapsed_ms: float = 0.0
    result: dict = field(default_factory=dict)

    def brief(self) -> str:
        return f"{self.name}({json.dumps(self.arguments, ensure_ascii=False)})"


@dataclass
class TurnResult:
    turn_id: str
    reply: str
    tool_calls: list[ToolCallRecord]
    rounds: int
    elapsed_ms: float
    ok: bool = True
    error_code: str = ""
    error_message: str = ""
    # 错误的补充信息（hint / status / retryable / attempts …）。
    # 必须一路带到前端：用户看到的「建议」就来自这里，
    # 早先 to_dict() 忘了带它，前端就只剩一句干巴巴的错误描述。
    error_info: dict = field(default_factory=dict)
    simulated: bool = False
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "reply": self.reply,
            "ok": self.ok,
            "error": ({"code": self.error_code, "message": self.error_message,
                       **self.error_info}
                      if self.error_code else None),
            "rounds": self.rounds,
            "elapsed_ms": self.elapsed_ms,
            "simulated": self.simulated,
            "usage": self.usage,
            "tool_calls": [
                {
                    "index": r.index, "call_id": r.call_id, "tool": r.name,
                    "arguments": r.arguments, "ok": r.ok, "code": r.code,
                    "elapsed_ms": r.elapsed_ms, "result": r.result,
                }
                for r in self.tool_calls
            ],
        }


# --------------------------------------------------------------------------- #
def _name_of(p: str | None) -> str | None:
    return Path(p).name if p else None


def truncate(text: str, limit: int) -> str:
    """把过长文本砍到 limit 以内（**只用于纯文本**，不要用在 JSON 上）。

    注意：这个函数会从中间切开并插入一段说明，切出来的东西**不再是合法 JSON**。
    工具结果是 JSON，必须走 `truncate_tool_result()`。
    """
    if len(text) <= limit:
        return text
    keep = limit // 2
    return (text[:keep]
            + f"\n…（结果过长已截断，原长 {len(text)} 字符。"
              f"如需完整内容，请用更精确的参数重新调用该工具）…\n"
            + text[-keep:])


# --------------------------------------------------------------------------- #
# 工具结果的结构化裁剪
# --------------------------------------------------------------------------- #
# 工具结果是以 JSON 字符串回填给模型的。如果按纯文本那样"从中间砍一刀"，
# 切出来的就不再是 JSON —— 下游 `json.loads` 会直接抛 JSONDecodeError，
# 把整轮对话打断（实测：my_garage_stats 记录一多就会命中，约 1/8 概率复现）。
#
# 所以这里做**结构化裁剪**：解析 → 逐级瘦身 → 重新序列化，
# 保证无论怎么裁，回填给模型的东西始终是合法 JSON。
_SHRINK_SCHEDULE = ((2000, 50), (500, 20), (200, 8), (80, 3), (40, 1), (20, 0))


def _shrink_json(data: Any, limit: int) -> tuple[Any, bool]:
    """在保持 JSON 合法性的前提下把结构压到 limit 字符以内。

    逐级收紧两把尺子：`str_cap` 截短长字符串、`keep` 限制列表只留前若干项。
    越靠后的档位丢的信息越多，一旦装得下就停手。
    """
    def dumps(node: Any) -> str:
        return json.dumps(node, ensure_ascii=False, default=str)

    def rewrite(node: Any, str_cap: int, keep: int) -> Any:
        if isinstance(node, dict):
            return {k: rewrite(v, str_cap, keep) for k, v in node.items()}
        if isinstance(node, list):
            return [rewrite(v, str_cap, keep) for v in node[:keep]]
        if isinstance(node, str) and len(node) > str_cap:
            return node[:str_cap] + "…"
        return node

    for str_cap, keep in _SHRINK_SCHEDULE:
        out = rewrite(data, str_cap, keep)
        if len(dumps(out)) <= limit:
            return out, (str_cap, keep) != _SHRINK_SCHEDULE[0]
    return rewrite(data, 20, 0), True


def truncate_tool_result(content: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    """工具结果专用截断：**保证返回的仍是合法 JSON**。

    不是 JSON（理论上不该发生，兜底）时退回文本截断。
    """
    if len(content) <= limit:
        return content
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return truncate(content, limit)

    shrunk, _ = _shrink_json(data, limit)
    dropped = json.dumps(data, ensure_ascii=False, default=str) != json.dumps(
        shrunk, ensure_ascii=False, default=str)
    if isinstance(shrunk, dict):
        # 显式告诉模型"这份结果是裁剪过的"，它才能决定要不要换更精确的参数重调，
        # 而不是把残缺的数据当成全部事实。
        if dropped:
            shrunk["_truncated"] = True
            shrunk["_original_chars"] = len(content)
            shrunk["_notice"] = ("结果过长，已做结构化裁剪（列表只保留前若干项、长文本已截短）。"
                                 "若需要完整内容，请用更精确的参数重新调用本工具。")
    return json.dumps(shrunk, ensure_ascii=False, default=str)


def summarize_tool_result(name: str, payload: dict) -> str:
    """给 UI 时间线用的一句话摘要。"""
    if not payload:
        return ""
    if not payload.get("ok"):
        err = payload.get("error") or {}
        return f"失败：{err.get('code', '')} {str(err.get('message', ''))[:60]}"
    if name == "classify_car":
        pct = float(payload.get("confidence") or 0) * 100
        if payload.get("is_unknown"):
            return "判定为 unknown（不属于本组支持的 50 款车型）"
        flag = "（低置信度）" if payload.get("low_confidence") else ""
        return f'{payload.get("car_name")} · {payload.get("car_id")} · {pct:.1f}%{flag}'
    if name == "query_car_info":
        return f'{payload.get("car_name")}｜{payload.get("brand")}｜{payload.get("body_group")}'
    if name == "list_supported_cars":
        return f'返回 {payload.get("count")} 款车型'
    if name == "add_car_record":
        verb = "已更新" if payload.get("duplicated") else "已写入"
        return f'{verb}档案：{payload.get("car_name")}（记录 #{payload.get("record_id")}）'
    if name == "search_my_records":
        return f'找到 {payload.get("count")} 条记录'
    if name == "my_garage_stats":
        return f'共 {payload.get("total_records")} 条记录'
    if name == "web_search":
        n = payload.get("count")
        if not n:
            return f'搜索「{str(payload.get("query", ""))[:24]}」→ 没有结果'
        return (f'搜索「{str(payload.get("query", ""))[:24]}」→ {n} 条结果'
                f'（{payload.get("provider", "")}）')
    if name == "browse_page":
        flag = "" if payload.get("mode") == "browser" else "（降级：HTTP 抓取）"
        return (f'打开「{str(payload.get("title") or payload.get("url", ""))[:32]}」'
                f'→ {payload.get("chars", 0)} 字{flag}')
    if name == "save_car_trims":
        v = "已核验来源" if payload.get("verified") else "来源未核实"
        return (f'{payload.get("car_name", "")} · 存入 {payload.get("saved", 0)} 个档次'
                f'（共 {payload.get("total_trims", 0)} 个，{v}）')
    if name == "compare_car_trims":
        if not payload.get("has_trim_data"):
            return f'{payload.get("car_name", "")} · 档案库暂无档次数据'
        keys = payload.get("differing_specs") or []
        return (f'{payload.get("car_name", "")} · 对比 {payload.get("compared", 0)} 个档次'
                f'，{len(keys)} 项参数有差异' + (f'（{"、".join(keys[:4])}…）' if keys else ""))
    if name == "update_record_note":
        return f'已更新记录 #{payload.get("record_id")} 的备注'
    if name == "delete_car_record":
        return f'已删除记录 #{payload.get("record_id")}'
    return ""


# --------------------------------------------------------------------------- #
class Agent:
    """识车档案智能体。同一会话串行执行，不同会话可并发。"""

    def __init__(self, config, registry: ToolRegistry | None = None,
                 store: SessionStore | None = None) -> None:
        self.config = config
        self.registry = registry or build_registry(config)
        self.store = store or SessionStore(config)
        self.client = get_client(config)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ------------------------------------------------------------------ #
    def _lock_for(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(session_id, threading.Lock())

    @property
    def is_offline(self) -> bool:
        return bool(self.config.offline)

    def describe_tools(self) -> list[dict]:
        return self.registry.describe()

    # ------------------------------------------------------------------ #
    def run_turn(self, session: Session, user_text: str,
                 image_path: str | None = None,
                 on_event: EventCallback | None = None) -> TurnResult:
        turn_id = f"t-{uuid.uuid4().hex[:8]}"
        t0 = time.perf_counter()
        cfg = self.config
        records: list[ToolCallRecord] = []

        def emit(kind: str, **payload: Any) -> None:
            if on_event is None:
                return
            try:
                on_event({"type": kind, "turn_id": turn_id, **payload})
            except Exception:
                pass

        session.add_user(user_text, image_path)
        session.turns += 1

        # turn 快照：本轮生效的策略与资源（对标 Codex 的 turn_context）
        snapshot = {
            "turn_id": turn_id,
            "index": session.turns,
            "image": _name_of(image_path or session.image_path),
            "unknown_threshold": cfg.unknown_threshold,
            "display_confidence_threshold": cfg.display_confidence_threshold,
            "vision_model": [p.parent.name for p in cfg.checkpoint_paths() if p.exists()],
            "tools": self.registry.names(),
            "offline": bool(cfg.offline),
            "max_rounds": cfg.max_rounds,
        }
        session.append_event("turn_context", snapshot)
        emit("turn_start", turn=snapshot, user_text=user_text)

        reply, rounds, usage = "", 0, {}
        ok, err_code, err_msg = True, "", ""
        err_info: dict = {}
        self._last_rounds = 0

        with self._lock_for(session.session_id):
            try:
                reply, rounds, usage = self._run_steps(session, turn_id, records, emit)
            except AgentError as e:
                ok, err_code, err_msg = False, e.code, e.message
                err_info = dict(e.info or {})
                rounds = self._last_rounds
                session.append_event("error", {"turn_id": turn_id, "code": e.code,
                                               "message": e.message, "info": e.info})
                emit("error", code=e.code, message=e.message, info=e.info)
            except Exception as e:                        # noqa: BLE001
                ok, err_code, err_msg = False, "INTERNAL_ERROR", f"{type(e).__name__}: {e}"
                rounds = self._last_rounds
                session.append_event("error", {"turn_id": turn_id, "code": err_code,
                                               "message": err_msg})
                emit("error", code=err_code, message=err_msg)

        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        session.append_event("turn_end", {
            "turn_id": turn_id, "turn_index": session.turns, "ok": ok, "reply": reply,
            "rounds": rounds, "elapsed_ms": elapsed,
            "tool_calls": [r.brief() for r in records],
            "tools_used": [r.name for r in records],
        })
        emit("turn_end", ok=ok, reply=reply, rounds=rounds, elapsed_ms=elapsed,
             tool_calls=len(records))

        return TurnResult(
            turn_id=turn_id, reply=reply, tool_calls=records, rounds=rounds,
            elapsed_ms=elapsed, ok=ok, error_code=err_code, error_message=err_msg,
            error_info=err_info, simulated=self.is_offline, usage=usage,
        )

    # ------------------------------------------------------------------ #
    def _run_steps(self, session: Session, turn_id: str,
                   records: list[ToolCallRecord],
                   emit: EventCallback) -> tuple[str, int, dict]:
        cfg = self.config
        # 按会话状态挑出本次允许模型看见的工具。
        # 「看不见」比「调了再被拒绝」更好：模型不会白白浪费一轮去试一个它其实没权限用的工具。
        allowed = enabled_tool_names(session)
        schemas = [s for s in self.registry.schemas()
                   if s["function"]["name"] in allowed]
        used = 0
        usage: dict = {}
        # 本轮已经执行过的「工具 + 参数」指纹 → 上次结果。
        # 联网类工具是**只读且昂贵**的，同一个查询重复调用纯属浪费预算和网络——
        # 实测真实模型会在一个轮次里对同一关键词搜 6 遍。
        seen_calls: dict[str, Any] = {}

        for round_no in range(1, cfg.max_rounds + 1):
            self._last_rounds = round_no
            system_text = prompts.build_system_prompt_text(cfg, session)
            messages: list[dict] = [{"role": "system", "content": system_text}]
            messages += session.public_messages()
            # ---- 预算提示：临近上限时明确要求"停止收集、开始作答" ----
            # 没有这一下，模型会一直搜下去，直到撞上 MAX_ROUNDS_EXCEEDED 而交白卷。
            rounds_left = cfg.max_rounds - round_no + 1
            calls_left = cfg.max_tool_calls - used
            if rounds_left <= 2 or calls_left <= 2:
                messages.append({"role": "system", "content": (
                    f"<budget_warning>本轮的执行预算即将用尽（剩余 {rounds_left} 轮、"
                    f"最多还能调用 {calls_left} 次工具）。现在**不要再搜集新信息**，"
                    f"请立刻基于已经拿到的工具结果给出最终回答；"
                    f"如果信息确实不足，就如实说明「查到的信息不足以给出确切结论」，"
                    f"并给出你已知的部分和下一步建议。绝不允许编造数据。</budget_warning>")})
            if cfg.offline:
                # 离线替身需要会话状态才能模拟出合理的工具调用序列
                messages[0] = {"role": "system", "content": system_text,
                               "_sim_state": {
                                   "has_image": bool(session.image_path),
                                   "image_name": _name_of(session.image_path),
                                   "last_prediction": session.last_prediction,
                                   "record_count": session.record_count,
                                   "web_search_enabled": bool(
                                       getattr(session, "web_search_enabled", False)),
                               }}

            emit("model_request", round=round_no, messages=len(messages), tools=len(schemas))
            # notify：让客户端把「网络抖动，正在第 n 次重试」这类中间事件推给前端，
            # 否则用户会盯着一个静止的界面，不知道系统在自救还是在卡死。
            resp: LLMResponse = self.client.complete(
                messages, schemas, notify=lambda kind, **kw: emit(kind, **kw))
            if resp.usage:
                usage = resp.usage

            calls = resp.tool_calls
            if not calls:
                if not (resp.content or "").strip():
                    raise AgentError(Code.MODEL_REFUSED, "模型返回了空内容",
                                     info={"hint": "可重新发起一次请求，或检查模型服务状态"})
                emit("assistant_message", content=resp.content, round=round_no,
                     simulated=resp.simulated)
                session.add_assistant({"role": "assistant", "content": resp.content})
                session.append_event("assistant_message",
                                     {"message": {"role": "assistant", "content": resp.content},
                                      "round": round_no, "final": True})
                return resp.content, round_no, usage

            # 先落库带 tool_calls 的 assistant 消息，再执行工具
            assistant_msg = resp.assistant_message()
            session.add_assistant(assistant_msg)
            session.append_event("assistant_message",
                                 {"message": assistant_msg, "round": round_no, "final": False})
            if (resp.content or "").strip():
                emit("assistant_message", content=resp.content, round=round_no,
                     simulated=resp.simulated)

            if used + len(calls) > cfg.max_tool_calls:
                raise AgentError(
                    Code.MAX_TOOL_CALLS_EXCEEDED,
                    f"本次请求的工具调用次数超过上限（{cfg.max_tool_calls}）",
                    info={"hint": "把任务拆成更小的步骤，或减少一次要求完成的事情"},
                )

            for call in calls:
                used += 1
                args = call.args()
                emit("tool_call", tool=call.name, arguments=args, call_id=call.id, index=used)

                # ---- 重复调用短路：同样的工具 + 同样的参数，直接用上次结果 ----
                # 联网工具是只读且昂贵的，重复调用没有任何收益，
                # 白白吃掉预算、拖长时间。把上次结果原样回给模型，并提示它换个角度。
                fingerprint = ""
                if call.name in IDEMPOTENT_WEB_TOOLS:
                    fingerprint = call.name + "|" + json.dumps(
                        args, ensure_ascii=False, sort_keys=True)
                    if fingerprint in seen_calls:
                        cached = dict(seen_calls[fingerprint])
                        cached["_cached"] = True
                        cached["_cached_notice"] = (
                            "参数与本轮之前的一次调用完全相同，这是**上次的结果**，没有重新执行。"
                            "请不要重复同样的调用：要么换更具体的关键词 / 换一个链接，"
                            "要么直接用已有信息作答。")
                        session.add_tool_result(
                            call.id, call.name,
                            json.dumps(cached, ensure_ascii=False, default=str),
                            True, "CACHED", 0.0)
                        records.append(ToolCallRecord(
                            index=used, call_id=call.id, name=call.name, arguments=args,
                            ok=True, code="CACHED", elapsed_ms=0.0, result=cached))
                        emit("tool_result", tool=call.name, call_id=call.id, ok=True,
                             code="CACHED", elapsed_ms=0.0,
                             summary="重复调用，返回上次结果", result=cached)
                        continue

                ctx = ToolContext(
                    session_id=session.session_id, turn_id=turn_id, call_id=call.id,
                    config=cfg, session=session, signal=threading.Event(),
                    image_path=session.image_path,
                )
                res = self.registry.execute(call.id, call.name, call.arguments, ctx)

                session.add_tool_result(call.id, call.name,
                                        truncate_tool_result(res.content(), MAX_TOOL_RESULT_CHARS),
                                        res.ok, res.code, res.elapsed_ms)
                records.append(ToolCallRecord(
                    index=used, call_id=call.id, name=call.name, arguments=args,
                    ok=res.ok, code=res.code, elapsed_ms=res.elapsed_ms, result=res.payload,
                ))
                self._absorb(session, call.name, res.payload)
                if fingerprint and res.ok:
                    seen_calls[fingerprint] = res.payload
                emit("tool_result", tool=call.name, call_id=call.id, ok=res.ok,
                     code=res.code, elapsed_ms=res.elapsed_ms,
                     summary=summarize_tool_result(call.name, res.payload),
                     result=res.payload)

            if self._should_circuit_break(records):
                raise AgentError(
                    Code.MAX_ROUNDS_EXCEEDED,
                    "同一个工具以相同方式连续失败多次，已停止以避免无效循环",
                    info={"hint": "请查看工具返回的 error.code 与 error.hint，"
                                  "或换一种方式向用户澄清需求"},
                )

        # ---- 预算用尽：先要一次「不带工具的最终回答」，别直接判死 ----
        # 之前这里是直接抛 MAX_ROUNDS_EXCEEDED，结果是**回复 0 字** ——
        # 明明已经抓到一堆资料，却因为"轮数用完了"一个字都不给用户，体验很差。
        # 现在改成再要一次纯文本回答（不给工具，逼它基于已有信息作答），
        # 只有当它连这次也说不出话时，才如实报错。
        emit("budget_final", rounds=cfg.max_rounds, tool_calls=used)
        try:
            final_messages: list[dict] = [
                {"role": "system", "content": prompts.build_system_prompt_text(cfg, session)}]
            final_messages += session.public_messages()
            final_messages.append({"role": "system", "content": (
                f"<budget_exhausted>本次请求的执行预算已用尽（{cfg.max_rounds} 轮 / {used} 次工具调用）。"
                f"现在**不允许再调用任何工具**，请立刻基于已经拿到的工具结果给出最终回答。"
                f"信息不足的部分如实说明「这部分没查到」，并告诉用户下一步可以怎么做。"
                f"绝对不要编造数据。</budget_exhausted>")})
            resp = self.client.complete(final_messages, [])
            content = (resp.content or "").strip()
            if content:
                session.add_assistant({"role": "assistant", "content": content})
                session.append_event("assistant_message",
                                     {"message": {"role": "assistant", "content": content},
                                      "round": cfg.max_rounds, "final": True, "forced": True})
                emit("assistant_message", content=content, round=cfg.max_rounds,
                     simulated=resp.simulated)
                return content, cfg.max_rounds, usage
        except AgentError:
            pass                                     # 连这一下都失败 → 走下面的如实报错

        raise AgentError(
            Code.MAX_ROUNDS_EXCEEDED,
            f"达到最大工具轮数（{cfg.max_rounds}）仍未得到最终回答",
            info={"hint": "把任务拆小一些，或提高 agent/config.py 里的 max_rounds。"
                          "已尝试强制模型基于现有信息作答但仍未成功。"},
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _absorb(session: Session, tool_name: str, payload: dict) -> None:
        """把工具结果吸收进会话状态 —— 这是「多轮对话沿用上一轮结果」的关键。"""
        if not isinstance(payload, dict) or not payload.get("ok"):
            return
        if tool_name == "classify_car":
            session.last_prediction = {
                "car_id": payload.get("car_id"),
                "car_name": payload.get("car_name"),
                "confidence": payload.get("confidence"),
                "is_unknown": payload.get("is_unknown"),
                "low_confidence": payload.get("low_confidence"),
                "decision": (payload.get("decision") or {}).get("action"),
                "image": (payload.get("image") or {}).get("file_name"),
            }
        elif tool_name == "add_car_record":
            session.record_count = int(payload.get("total_records_in_session")
                                       or session.record_count)
        elif tool_name == "search_my_records":
            session.record_count = int(payload.get("total_in_session") or session.record_count)
        elif tool_name == "my_garage_stats":
            session.record_count = int(payload.get("total_records") or session.record_count)

    @staticmethod
    def _should_circuit_break(records: list[ToolCallRecord], threshold: int = 3) -> bool:
        """同一工具 + 同一错误码 + 同一参数，连续出现 threshold 次即熔断。"""
        if len(records) < threshold:
            return False
        tail = records[-threshold:]
        keys = {(r.name, r.code, json.dumps(r.arguments, sort_keys=True, ensure_ascii=False))
                for r in tail}
        return len(keys) == 1 and not tail[-1].ok
