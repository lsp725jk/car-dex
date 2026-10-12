# -*- coding: utf-8 -*-
"""工具注册表与执行流水线。

这是任务二的核心。设计参照主流 agent harness 的做法，把"执行一次工具调用"
拆成**若干可观测、可干预的环节**，而不是一行 `handler(**args)`：

    1. 查表        name → ToolDefinition        （未注册 → UNKNOWN_TOOL）
    2. 解析参数    JSON 字符串 → dict            （解析失败 → INVALID_ARGUMENTS）
    3. 校验参数    JSON Schema                   （不合规 → INVALID_ARGUMENTS）
    4. 前置门禁    pre_execute 钩子              （拒绝 → DENIED_BY_POLICY）
    5. 执行主体    带超时、带取消信号             （超时 → TOOL_TIMEOUT；抛错 → TOOL_FAILED）
    6. 后置处理    post_execute 钩子             （可替换/补充结果）
    7. 记账        写 JSONL 工具调用记录          （任务书 3.2 要求"提供工具调用记录"）

**失败一律如实上报**：任何环节出错都返回 `{"ok": false, "error": {...}}`，
绝不伪造成功——这是任务书"工具应真实执行对应操作，不能用固定文本或伪造结果代替功能"的要求。
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..errors import Code, ToolError, tool_failure
from ..schema import SchemaError, validate

ToolHandler = Callable[[dict, "ToolContext"], dict]


class Aborted(Exception):
    """协作式取消信号。"""


@dataclass
class ToolContext:
    """传给工具主体的执行上下文。

    只读约定：工具**不得**修改 session 的业务状态以外的东西；
    需要写数据一律通过 garage 工具自己的数据层完成。
    """

    session_id: str
    turn_id: str
    call_id: str
    config: Any                       # agent.config.Config
    session: Any                      # agent.session.Session
    signal: threading.Event           # 协作式取消
    image_path: str | None = None     # 本会话最近一次上传的图片路径
    started_at: float = field(default_factory=time.time)

    @property
    def aborted(self) -> bool:
        return self.signal.is_set()

    def check_abort(self) -> None:
        if self.aborted:
            raise Aborted()


@dataclass
class ToolDefinition:
    """一个工具：名字 + 用途 + 参数 schema + 执行主体 + 执行策略。"""

    name: str
    description: str
    parameters: dict
    handler: ToolHandler
    timeout_s: float = 30.0
    # 与主流 harness 的 isConcurrencySafe 对应：为 True 时多个调用可并行。
    # 本项目所有工具都涉及 SQLite 写入或模型推理，默认独占，避免状态竞争。
    concurrency_safe: bool = False
    # 该工具是否需要本会话已上传图片
    requires_image: bool = False
    tags: tuple[str, ...] = ()

    def to_openai_schema(self) -> dict:
        """转成 DeepSeek / OpenAI Function Calling 的工具定义。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    """一次工具调用的最终结果。"""

    call_id: str
    name: str
    ok: bool
    payload: dict                 # 回填给模型的 JSON 对象
    elapsed_ms: float = 0.0
    code: str = ""                # 失败时的错误码
    cached: bool = False

    def content(self) -> str:
        """转成 role=tool 消息的 content（JSON 字符串）。"""
        return json.dumps(self.payload, ensure_ascii=False, default=str)


class ToolRegistry:
    """工具注册表 + 执行器。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._gates: list[Callable[[ToolDefinition, dict, ToolContext], str | None]] = []
        self._post: list[Callable[[ToolDefinition, ToolResult, ToolContext], None]] = []
        self._log_path: Path | None = None
        self._log_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # 注册与查询
    # ------------------------------------------------------------------ #
    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError(f"工具名重复：{definition.name}")
        self._tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def all(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def describe(self) -> list[dict]:
        """给 UI / 报告用的工具清单。"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "timeout_s": t.timeout_s,
                "requires_image": t.requires_image,
                "tags": list(t.tags),
            }
            for t in self._tools.values()
        ]

    # ------------------------------------------------------------------ #
    # 钩子
    # ------------------------------------------------------------------ #
    def add_gate(self, fn: Callable[[ToolDefinition, dict, ToolContext], str | None]) -> None:
        """前置门禁：返回字符串表示拒绝（字符串即拒绝理由），返回 None 表示放行。"""
        self._gates.append(fn)

    def add_post(self, fn: Callable[[ToolDefinition, ToolResult, ToolContext], None]) -> None:
        """后置处理：可以观察或就地修改 ToolResult。"""
        self._post.append(fn)

    def enable_log(self, path: Path) -> None:
        """启用工具调用记录（JSONL，每行一次调用）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path = path

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #
    def execute(self, call_id: str, name: str, raw_arguments: str | dict,
                ctx: ToolContext, timeout_s: float | None = None) -> ToolResult:
        t0 = time.perf_counter()

        def done(payload: dict, ok: bool, code: str = "") -> ToolResult:
            res = ToolResult(
                call_id=call_id, name=name, ok=ok, payload=payload,
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 2), code=code,
            )
            for hook in self._post:
                try:
                    hook(self._tools.get(name), res, ctx)  # type: ignore[arg-type]
                except Exception:
                    pass
            self._write_log(res, ctx)
            return res

        # --- 1. 查表 ---
        tool = self._tools.get(name)
        if tool is None:
            return done(
                tool_failure(Code.UNKNOWN_TOOL,
                             f"没有名为 `{name}` 的工具",
                             hint=f"可用工具：{', '.join(self._tools)}"),
                False, Code.UNKNOWN_TOOL,
            )

        # --- 2. 解析参数 ---
        if isinstance(raw_arguments, str):
            text = raw_arguments.strip() or "{}"
            try:
                args = json.loads(text)
            except json.JSONDecodeError as e:
                return done(
                    tool_failure(Code.INVALID_ARGUMENTS, f"参数不是合法 JSON：{e.msg}",
                                 hint="请输出合法的 JSON 对象作为参数"),
                    False, Code.INVALID_ARGUMENTS,
                )
        else:
            args = dict(raw_arguments or {})
        if not isinstance(args, dict):
            return done(
                tool_failure(Code.INVALID_ARGUMENTS, "参数必须是一个 JSON 对象"),
                False, Code.INVALID_ARGUMENTS,
            )

        # --- 3. 校验参数 ---
        try:
            validate(args, tool.parameters)
        except SchemaError as e:
            return done(
                tool_failure(Code.INVALID_ARGUMENTS, f"参数校验失败 —— {e}",
                             hint=f"该工具的参数定义：{json.dumps(tool.parameters, ensure_ascii=False)}"),
                False, Code.INVALID_ARGUMENTS,
            )

        # --- 4. 前置门禁 ---
        for gate in self._gates:
            try:
                reason = gate(tool, args, ctx)
            except Exception as e:      # 门禁自身出错 → 保守拒绝
                reason = f"权限校验异常：{e}"
            if reason:
                return done(
                    tool_failure(Code.DENIED_BY_POLICY, reason),
                    False, Code.DENIED_BY_POLICY,
                )

        # --- 5. 执行主体（带超时与取消） ---
        if ctx.aborted:
            return done(tool_failure(Code.ABORTED_BEFORE_DISPATCH, "调用在开始执行前被取消"),
                        False, Code.ABORTED_BEFORE_DISPATCH)

        box: dict[str, Any] = {}

        def runner() -> None:
            try:
                box["value"] = tool.handler(args, ctx)
            except ToolError as e:
                box["error"] = tool_failure(e.code, e.message, hint=e.hint, **e.info)
                box["code"] = e.code
            except Aborted:
                box["error"] = tool_failure(Code.ABORTED, "执行过程中被取消")
                box["code"] = Code.ABORTED
            except Exception as e:            # noqa: BLE001 —— 工具内部任何异常都必须兜住
                box["error"] = tool_failure(Code.TOOL_FAILED, f"工具内部错误：{type(e).__name__}: {e}")
                box["code"] = Code.TOOL_FAILED

        limit = timeout_s if timeout_s is not None else tool.timeout_s
        th = threading.Thread(target=runner, name=f"tool-{name}", daemon=True)
        th.start()
        th.join(limit)

        if th.is_alive():
            ctx.signal.set()                  # 通知工具主体尽快退出
            th.join(2.0)
            return done(
                tool_failure(Code.TOOL_TIMEOUT, f"工具 `{name}` 超过 {limit:.1f} 秒未返回",
                             hint="请简化请求或稍后重试"),
                False, Code.TOOL_TIMEOUT,
            )

        if "error" in box:
            return done(box["error"], False, box.get("code", Code.TOOL_FAILED))

        value = box.get("value")
        if not isinstance(value, dict):
            return done(
                tool_failure(Code.TOOL_FAILED, f"工具 `{name}` 没有返回 JSON 对象"),
                False, Code.TOOL_FAILED,
            )
        return done(value, bool(value.get("ok", True)))

    # ------------------------------------------------------------------ #
    def _write_log(self, res: ToolResult, ctx: ToolContext) -> None:
        """任务书要求"提供工具调用记录" —— 每次调用一行 JSONL。"""
        if self._log_path is None:
            return
        row = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": ctx.session_id,
            "turn_id": ctx.turn_id,
            "call_id": res.call_id,
            "tool": res.name,
            "ok": res.ok,
            "code": res.code,
            "elapsed_ms": res.elapsed_ms,
            "result": res.payload,
        }
        try:
            with self._log_lock:
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass
