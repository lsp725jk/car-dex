# -*- coding: utf-8 -*-
"""模型客户端。

两种实现，同一套接口：

  `DeepSeekClient`   真实调用 DeepSeek 开放平台的 `/chat/completions`（任务书规定模型名
                     `deepseek-v4-flash`）。负责把 HTTP 错误、超时、返回体异常
                     翻译成带错误码的 `AgentError`，不吞错、不重试掩盖。

  `ScriptedClient`   **离线替身**。没有 API Key 时用它驱动同一套 Agent Loop，
                     让工具注册表、执行流水线、多轮会话、Web 界面都能被完整验证。
                     它按规则模拟模型的工具调用决策，**不是**真的语言模型，
                     仅用于开发自测与课堂演示脱网环境，绝不能当作模型效果。

`get_client(config)` 按配置自动选择。
"""
from __future__ import annotations

import json
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Config
from .errors import AgentError, Code


# 重试通知回调：Agent 用它把「正在重试」推给 Web 前端
Notify = Callable[..., None]


# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str            # JSON 字符串（与 OpenAI/DeepSeek 协议一致）

    def args(self) -> dict:
        try:
            v = json.loads(self.arguments or "{}")
            return v if isinstance(v, dict) else {}
        except json.JSONDecodeError:
            return {}

    def to_message(self) -> dict:
        return {"id": self.id, "type": "function",
                "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    model: str = ""
    latency_ms: float = 0.0
    simulated: bool = False
    retries: int = 0                       # 本次请求实际重试了几次（0 = 一次就成功）

    def assistant_message(self) -> dict:
        """构造要追加进 messages 的 assistant 消息。"""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [c.to_message() for c in self.tool_calls]
        return msg


# --------------------------------------------------------------------------- #
# 把 requests 的异常翻译成「哪一层坏了」
# --------------------------------------------------------------------------- #
# 为什么值得单独做这件事：家用宽带 / 移动网络的 DNS 会偶发解析失败
# （Windows 上表现为 `[Errno 11001] getaddrinfo failed`），通常一两秒就恢复。
# 直接把 urllib3 的异常字符串抛给用户，用户看到的是这样一坨：
#     ConnectionError: HTTPSConnectionPool(host='api.deepseek.com', port=443):
#     Max retries exceeded ... NameResolutionError ... getaddrinfo failed
# 既看不出「该修什么」，也看不出「这不是你代码的问题」。
#
# 所以这里按网络栈自下而上分层判定，并给出各自的修复建议。
_DNS_MARKERS = (
    "nameresolutionerror", "getaddrinfo failed", "name or service not known",
    "nodename nor servname", "temporary failure in name resolution",
    "no address associated with hostname", "11001",
)
_TLS_MARKERS = ("sslerror", "certificate verify failed", "ssl:", "tlsv1", "wrong version number")
_PROXY_MARKERS = ("proxyerror", "cannot connect to proxy", "proxy")

_PROXY_HINT = ("检查系统代理 / VPN 设置；若在公司或校园网内，可能需要配置 http_proxy / https_proxy")


def _exception_chain_text(exc: BaseException, depth: int = 6) -> str:
    """把异常的 __cause__ / __context__ 链拼成一段小写文本，用于关键词判定。"""
    parts: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and len(parts) < depth and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    return " | ".join(parts).lower()


def classify_request_error(exc: BaseException) -> AgentError:
    """把一个 requests 异常翻译成带错误码、可读原因、修复建议和「能否重试」的 AgentError。"""
    import requests

    full = _exception_chain_text(exc)
    name = type(exc).__name__

    # 顺序很重要：SSL / 代理 都是 ConnectionError 的子类，要先判。
    if isinstance(exc, requests.exceptions.SSLError) or any(m in full for m in _TLS_MARKERS):
        return AgentError(
            Code.NETWORK_TLS_FAILED, f"与 DeepSeek 建立 TLS 连接失败（{name}）",
            info={"retryable": False,
                  "hint": "证书校验或 TLS 握手失败。检查系统时间是否正确、是否有抓包/杀毒软件插入了"
                          "自签名证书；公司网络下的中间人代理也会触发这个问题。",
                  "raw": full[:400]})

    if isinstance(exc, requests.exceptions.ProxyError) or any(m in full for m in _PROXY_MARKERS):
        return AgentError(
            Code.NETWORK_UNREACHABLE, f"通过代理连接 DeepSeek 失败（{name}）",
            info={"retryable": True, "hint": _PROXY_HINT, "raw": full[:400]})

    if any(m in full for m in _DNS_MARKERS):
        return AgentError(
            Code.NETWORK_DNS_FAILED, "域名解析失败：无法把 api.deepseek.com 解析成 IP",
            info={"retryable": True,
                  "hint": "这是本机 DNS 的问题，不是代码或 Key 的问题。通常是网络刚切换（Wi-Fi↔有线）、"
                          "VPN 断连、或运营商 DNS 抖动。一般几秒后自动恢复——系统已自动重试过若干次。"
                          "仍然失败时：① 换网络重试 ② 关闭/切换 VPN ③ 刷新 DNS 缓存"
                          "（ipconfig /flushdns）④ 手动把 DNS 改成 223.5.5.5 或 119.29.29.29。",
                  "raw": full[:400]})

    if isinstance(exc, requests.exceptions.Timeout) or "timeout" in full or "timed out" in full:
        return AgentError(
            Code.API_TIMEOUT, f"请求 DeepSeek 超时（{name}）",
            info={"retryable": True,
                  "hint": "网络慢或被限速。可在 agent/config.py 调大 read_timeout / connect_timeout 后重试。",
                  "raw": full[:400]})

    return AgentError(
        Code.NETWORK_UNREACHABLE, f"无法连接到 DeepSeek 接口（{name}）",
        info={"retryable": True,
              "hint": "TCP 连接没建立起来：检查网络是否可用、是否被防火墙/代理拦截、"
                      "或换一个网络重试。",
              "raw": full[:400]})


def classify_http_error(status: int) -> AgentError:
    """把 HTTP 状态码翻译成错误码：429 与 5xx 可重试，4xx 参数类不重试。"""
    if status == 429:
        return AgentError(Code.API_RATE_LIMITED, "请求过于频繁，已达到接口限额（HTTP 429）",
                          info={"status": status, "retryable": True,
                                "hint": "已被限流，稍后自动重试；持续出现请降低调用频率。"})
    if 500 <= status < 600:
        return AgentError(Code.API_SERVER_ERROR, f"DeepSeek 服务端错误（HTTP {status}）",
                          info={"status": status, "retryable": True,
                                "hint": "这是服务端的问题，不是你的请求有问题；稍后自动重试。"})
    hints = {
        400: "检查模型名、请求参数与消息顺序",
        401: "API Key 无效或已失效（去 platform.deepseek.com 核对）",
        402: "账号余额不足，请先充值",
        403: "账号或模型访问权限不足",
        404: "接口路径不对",
    }
    return AgentError(Code.API_REQUEST_FAILED, f"DeepSeek 接口返回 HTTP {status}",
                      info={"status": status, "retryable": False,
                            "hint": hints.get(status, "查看 DeepSeek 平台控制台")})


# --------------------------------------------------------------------------- #
# 真实客户端
# --------------------------------------------------------------------------- #
class DeepSeekClient:
    """真实调用 `/chat/completions`。

    两个职责要分清：

      1. `_request_once`  发一次请求，把任何失败翻译成带 `retryable` 标记的 AgentError；
      2. `complete`       负责**策略**——只在 `retryable` 时做指数退避重试，并广播重试事件。

    重试的意义：DNS 抖动、限流、服务端 5xx 都是**会自己好的**故障，
    不该让用户白打一遍字。反过来说，400/401 这类参数或鉴权错误重试一万次也没用，
    所以标记为不可重试，立刻失败并给出明确的修复建议。
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    # ------------------------------------------------------------------ #
    def _request_once(self, payload: dict, t0: float) -> LLMResponse:
        import requests

        cfg = self.config
        try:
            resp = requests.post(
                cfg.api_url,
                headers={"Authorization": f"Bearer {cfg.api_key}",
                         "Content-Type": "application/json"},
                json=payload,
                timeout=(cfg.connect_timeout, cfg.read_timeout),
            )
        except Exception as e:                       # noqa: BLE001 —— requests 的异常体系很宽
            raise classify_request_error(e) from e

        if not resp.ok:
            raise classify_http_error(resp.status_code)

        try:
            data = resp.json()
        except ValueError as e:
            raise AgentError(Code.API_RESPONSE_INVALID, "接口返回的不是有效 JSON") from e

        if not isinstance(data, dict) or not data.get("choices"):
            raise AgentError(Code.API_RESPONSE_INVALID, "接口返回体缺少 choices 字段",
                             info={"keys": list(data)[:10] if isinstance(data, dict) else None})

        choice = data["choices"][0]
        raw_msg = choice.get("message") or {}
        calls = []
        for c in raw_msg.get("tool_calls") or []:
            fn = c.get("function") or {}
            calls.append(ToolCall(id=c.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                                  name=fn.get("name", ""),
                                  arguments=fn.get("arguments") or "{}"))

        if choice.get("finish_reason") == "length":
            raise AgentError(
                Code.OUTPUT_TRUNCATED,
                "输出被 max_tokens 截断",
                info={"truncated": True, "retryable": False,
                      "max_tokens": payload.get("max_tokens"),
                      "hint": "如果模型正在生成很大的工具参数（比如一次写入几十个档次），"
                              "要么让它分批调用，要么放大 max_tokens。"})

        return LLMResponse(
            content=raw_msg.get("content") or "",
            tool_calls=calls,
            finish_reason=choice.get("finish_reason", ""),
            usage=data.get("usage") or {},
            model=data.get("model", cfg.model),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    # ------------------------------------------------------------------ #
    def complete(self, messages: list[dict], tools: list[dict] | None = None,
                 tool_choice: str = "auto", notify: Notify | None = None) -> LLMResponse:
        cfg = self.config
        if not cfg.api_key_configured:
            raise AgentError(Code.API_KEY_MISSING,
                             "未配置 DeepSeek API Key",
                             info={"howto": "设置环境变量 DEEPSEEK_API_KEY，"
                                             "或写入 agent/.env.local（该文件已被 .gitignore 忽略）"})

        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        attempts = max(1, int(cfg.max_retries) + 1)
        last: AgentError | None = None
        # 截断补救援冲：第一次撞上 max_tokens 时，放大预算再试一次。
        # 模型生成大参数对象时很容易超，而重试一次几乎总能成功，
        # 比直接把这一轮判死、让用户重说一遍要好得多。
        ceiling = max(int(getattr(cfg, "max_tokens_ceiling", cfg.max_tokens) or cfg.max_tokens),
                      int(cfg.max_tokens))
        escalated = False

        for attempt in range(1, attempts + 1):
            t0 = time.perf_counter()
            try:
                resp = self._request_once(payload, t0)
            except AgentError as e:
                # ---- 输出截断：放大 max_tokens 重试（最多一次，且必须还有重试次数）----
                if (e.code == Code.OUTPUT_TRUNCATED and not escalated
                        and ceiling > cfg.max_tokens and attempt < attempts):
                    escalated = True
                    old = int(payload.get("max_tokens") or cfg.max_tokens)
                    payload["max_tokens"] = ceiling
                    if notify is not None:
                        try:
                            notify("api_retry", attempt=attempt, attempts=attempts,
                                   wait_s=0.0, code=e.code,
                                   message=f"输出被截断，已把 max_tokens 从 {old} 提到 {ceiling} 重试")
                        except Exception:               # noqa: BLE001
                            pass
                    continue

                if not e.info.get("retryable") or attempt >= attempts:
                    if attempt > 1:
                        e.info.setdefault("attempts", attempt)
                    if e.code == Code.OUTPUT_TRUNCATED and payload.get("max_tokens"):
                        e.info.setdefault("hint",
                                          f"已按 {payload['max_tokens']} tokens 重试仍被截断。"
                                          f"请让模型**分批**调用工具（例如一次只写 5 个档次），"
                                          f"或只保留有区分度的参数。")
                    raise
                # 指数退避 + ±25% 抖动，避免多个会话同时重试造成同步冲击
                wait = min(cfg.retry_backoff * (2 ** (attempt - 1)), 8.0)
                wait *= 0.75 + random.random() * 0.5
                if notify is not None:
                    try:
                        notify("api_retry", attempt=attempt, attempts=attempts,
                               wait_s=round(wait, 1), code=e.code, message=e.message)
                    except Exception:               # noqa: BLE001 —— 通知失败不能影响主流程
                        pass
                last = e
                time.sleep(wait)
            else:
                resp.retries = attempt - 1
                if attempt > 1 and notify is not None:
                    try:
                        notify("api_recovered", attempts=attempt)
                    except Exception:               # noqa: BLE001
                        pass
                return resp

        raise last or AgentError(Code.API_REQUEST_FAILED, "请求失败且未捕获到具体原因")


# --------------------------------------------------------------------------- #
# 离线替身
# --------------------------------------------------------------------------- #
class ScriptedClient:
    """**测试替身**：用规则模拟模型的工具调用决策，驱动同一套 Agent Loop。

    它的作用不是"替代模型"，而是在没有 API Key / 断网时验证：
      · 工具注册、参数校验、执行流水线、错误回传是否正常
      · 多轮会话与"沿用上一轮识别结果"是否生效
      · Web 前端的工具调用时间线能否正确渲染

    它会读取会话状态（是否已上传图片、是否已识别、是否已查档案、是否已记录），
    从而走出一条与真实模型高度相似的多步工具链。
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._seq = 0
        self._is_scripted = True

    def _id(self) -> str:
        self._seq += 1
        return f"call_sim_{self._seq:04d}_{uuid.uuid4().hex[:6]}"

    def complete(self, messages: list[dict], tools: list[dict] | None = None,
                 tool_choice: str = "auto", notify: Notify | None = None) -> LLMResponse:
        t0 = time.perf_counter()
        state = messages[0].get("_sim_state", {}) if messages else {}
        user_text = ""
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                user_text = messages[i].get("content") or ""
                last_user_idx = i
                break

        # **只统计本轮（最后一条 user 消息之后）发生过的工具调用**，
        # 否则会把上一轮的调用误当成本轮已经做过的事。
        turn_messages = messages[last_user_idx + 1:] if last_user_idx >= 0 else []
        tool_results = [m for m in turn_messages if m.get("role") == "tool"]
        id2name: dict[str, str] = {}
        for m in turn_messages:
            if m.get("role") == "assistant":
                for c in (m.get("tool_calls") or []):
                    fid = c.get("id")
                    fname = (c.get("function") or {}).get("name")
                    if fid and fname:
                        id2name[fid] = fname
        called = [id2name.get(m.get("tool_call_id", ""), "") for m in tool_results]

        text = user_text.strip()
        has_image = bool(state.get("has_image"))
        cur_image = state.get("image_name")
        pred = state.get("last_prediction") or {}
        identified = bool(pred)
        is_unknown = bool(pred.get("is_unknown"))
        # 换了一张新图片 → 必须重新识别（上一次的结论不是这张图的）
        stale_prediction = bool(identified and pred.get("image") and cur_image
                                and pred.get("image") != cur_image)
        need_classify = has_image and (not identified or stale_prediction)
        queried = "query_car_info" in called
        recorded = "add_car_record" in called

        def call(name: str, args: dict) -> LLMResponse:
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(self._id(), name, json.dumps(args, ensure_ascii=False))],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model=self.config.model + " (offline-sim)",
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                simulated=True,
            )

        def say(content: str) -> LLMResponse:
            return LLMResponse(
                content=content, tool_calls=[], finish_reason="stop",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model=self.config.model + " (offline-sim)",
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                simulated=True,
            )

        # ==================================================================
        # 判定顺序（很关键，决定了多轮对话是否自然）
        #
        #   0. 有图片但还没识别（或换了新图片）→ 必须先 classif y_car
        #      任务书 5.1：视觉识别必须通过智能体工具调用执行
        #   1. 识别结果是 unknown → 如实告知，不继续任何业务操作
        #   2. 明确的文本意图（查档案库 / 列清单 / 搜记录 / 按名查档）
        #      —— 这类请求与"当前这张图"无关，必须先判定
        #   3. 已经识别出具体车型 → 资料 → 收藏 的连续调用链
        #   4. 兜底
        # ==================================================================

        # ---- 0. 先识别 ----
        if need_classify:
            return call("classify_car", {})

        # ---- 1.5 联网（开关打开时才走；关闭时明确引导用户去开）----
        # 走的是一条「搜索 → 打开正文 → 综合」的链，和真实模型的用法一致。
        web_on = bool(state.get("web_search_enabled"))
        if self._wants_web_search(text) and not identified:
            if not web_on:
                return say("这个问题需要联网查。点一下输入框左边的「🌐 联网搜索」按钮开启，"
                           "我就能帮你去公网上查最新信息了。\n\n"
                           "（离线演示模式：文字由内置的脚本化替身生成。）")
            if "browse_page" in called:
                return say(self._compose_web_summary(
                    self._last_tool_payload(tool_results, "web_search", id2name),
                    self._last_tool_payload(tool_results, "browse_page", id2name)))
            if "web_search" in called:
                # 摘要不够 → 打开最相关那条读正文（这正是"反复"的那一步）
                sr = self._last_tool_payload(tool_results, "web_search", id2name)
                hits = sr.get("results") or []
                if hits and hits[0].get("url"):
                    return call("browse_page", {"url": hits[0]["url"], "max_chars": 3000})
                return say(self._compose_web_summary(sr, {}))

            return call("web_search", {"query": self._web_query(text), "max_results": 5})

        # ---- 1.6 档次/价格（离线替身：演示"查档案 → 没有就联网 → 存回档案"的完整链路）----
        if self._wants_trim_info(text) and identified:
            car_id = pred.get("car_id")
            if "compare_car_trims" in called:
                return say(self._compose_trim_summary(
                    self._last_tool_payload(tool_results, "compare_car_trims", id2name)))
            if "save_car_trims" in called:
                return say(self._compose_trim_saved(
                    self._last_tool_payload(tool_results, "save_car_trims", id2name)))
            if "query_car_info" in called:
                info = self._last_tool_payload(tool_results, "query_car_info", id2name)
                if info.get("trim_count"):
                    return call("compare_car_trims", {"car_id_or_name": car_id})
                if not web_on:
                    return say("档案库里没有这款车的档次和价格数据 —— 这部分教师数据集没有提供。"
                               "点一下输入框左边的「🌐 联网搜索」按钮开启，我就能帮你去查，"
                               "查完还可以存进档案库，以后直接就能看。\n\n"
                               "（离线演示模式：文字由内置的脚本化替身生成。）")
                return call("web_search",
                            {"query": f"{pred.get('car_name', '').split('_')[-1]} 各配置 价格 参数",
                             "max_results": 5})
            return call("query_car_info", {"car_id_or_name": car_id})

        # ---- 1. unknown 如实告知 ----
        if identified and is_unknown:
            last = self._last_tool_payload(tool_results, "classify_car", id2name)
            return say(self._compose_unknown_reply(last))

        # ---- 2. 明确的文本意图（优先于"已识别"分支）----
        #   每个分支都先检查"本轮是否已经调过该工具"，避免重复调用打转。
        if self._wants_garage(text):
            if "my_garage_stats" in called:
                return say(self._compose_garage_summary(
                    self._last_tool_payload(tool_results, "my_garage_stats", id2name)))
            return call("my_garage_stats", {"scope": "session"})

        if self._wants_list(text):
            if "list_supported_cars" in called:
                return say(self._compose_list_summary(
                    self._last_tool_payload(tool_results, "list_supported_cars", id2name)))
            args = {}
            for g in ("轿车", "SUV", "MPV", "跑车", "微面", "轻客"):
                if g in text:
                    args["body_group"] = g
                    break
            return call("list_supported_cars", args)

        if self._wants_search(text):
            if "search_my_records" in called:
                return say(self._compose_search_summary(
                    self._last_tool_payload(tool_results, "search_my_records", id2name)))
            kw = self._guess_keyword(text)
            return call("search_my_records", {"keyword": kw} if kw else {})

        if self._wants_query(text) and self._mentions_model(text):
            if "query_car_info" in called:
                return say(self._compose_info_summary(
                    state, self._last_tool_payload(tool_results, "query_car_info", id2name)))
            kw = self._guess_keyword(text)
            if kw:
                return call("query_car_info", {"car_id_or_name": kw})

        # ---- 3. 已识别 → 资料 → 收藏 的链式调用 ----
        if identified:
            if self._wants_info(text) and not queried:
                return call("query_car_info", {"car_id_or_name": pred["car_id"]})
            if self._wants_record(text) and not recorded:
                note = self._guess_note(text)
                return call("add_car_record", {"note": note} if note else {})
            if recorded:
                return say(self._compose_record_summary(
                    state, self._last_tool_payload(tool_results, "add_car_record", id2name)))
            if queried:
                return say(self._compose_info_summary(
                    state, self._last_tool_payload(tool_results, "query_car_info", id2name)))
            if self._wants_record(text):
                return call("add_car_record", {})
            return say(self._compose_seen_summary(state))

        # ---- 4. 兜底 ----
        if self._wants_recognize(text) and not has_image:
            return say("请先在对话框里上传一张车辆照片（点输入框左侧的「上传图片」按钮），"
                       "我再帮你识别这是哪一款车。\n\n"
                       "（离线演示模式：文字由内置的脚本化替身生成，用于验证工具链是否连通。）")

        return say(self._compose_fallback(text, state))

    # ------------------------------------------------------------------ #
    # 规则
    # ------------------------------------------------------------------ #
    @staticmethod
    def _last_tool_payload(tool_messages: list[dict], name: str,
                           id2name: dict[str, str] | None = None) -> dict:
        """取某个工具最近一次返回的结果。"""
        id2name = id2name or {}
        for m in reversed(tool_messages):
            tname = id2name.get(m.get("tool_call_id", ""), "") or m.get("_tool_name", "")
            if tname == name:
                try:
                    return json.loads(m.get("content") or "{}")
                except json.JSONDecodeError:
                    return {}
        return {}

    @staticmethod
    def _wants_recognize(t: str) -> bool:
        return any(k in t for k in ("这是什么", "识别", "看看", "什么车", "哪款"))

    @staticmethod
    def _wants_info(t: str) -> bool:
        """想了解"当前这张图里的车"的资料。注意不要匹配"档案"，
        否则「我的看车档案里有什么」会被误判成查车型资料。"""
        return any(k in t for k in ("介绍", "资料", "详情", "说说", "了解一下",
                                    "这车怎么样", "这车", "什么车", "哪款"))

    @staticmethod
    def _wants_record(t: str) -> bool:
        return any(k in t for k in ("记录", "收藏", "加进", "加入", "记一下", "存一下", "留个"))

    @staticmethod
    def _wants_garage(t: str) -> bool:
        return any(k in t for k in ("我的档案", "看车档案", "我记录过", "我的车库",
                                    "收藏了哪些", "档案里", "统计分析"))

    @staticmethod
    def _wants_list(t: str) -> bool:
        return any(k in t for k in ("支持哪些", "能识别哪些", "有哪些车", "车型清单",
                                    "你们能识别", "支持什么车", "都能识别"))

    @staticmethod
    def _wants_search(t: str) -> bool:
        return any(k in t for k in ("搜", "找找", "有没有记过", "查一下我", "我记过", "我收藏"))

    @staticmethod
    def _wants_query(t: str) -> bool:
        return any(k in t for k in ("查一下", "帮我查", "介绍一下", "这车是"))

    @staticmethod
    def _wants_trim_info(t: str) -> bool:
        """判断用户是不是在问"某个车型的档次 / 配置版本 / 价格"。"""
        return any(k in t for k in (
            "档次", "配置版本", "标准版", "舒适版", "豪华版", "豪华", "尊贵", "旗舰",
            "低配", "高配", "顶配", "中配", "哪一档", "哪个配置", "配置差别", "差别",
            "多少钱", "价格", "报价", "落地价", "差价", "贵多少",
        ))

    @staticmethod
    def _mentions_model(t: str) -> bool:
        """文本里是否出现了具体的车型名/ID（用于"按名查档"）。"""
        if re.search(r"\b\d{4}\b", t):
            return True
        return bool(re.search(r"[\u4e00-\u9fff]{2,8}(轿车|SUV|MPV|客车|之光|凯捷|星瑞)", t))

    @staticmethod
    def _wants_web_search(t: str) -> bool:
        """判断用户是不是在要"本组数据之外、需要联网"的信息。

        刻意用偏窄的关键词：只有明确提到价格/新闻/口碑/销量/上市这类
        "本地档案答不了"的东西，才走联网。否则会把「查一下轩逸的资料」
        也误判成联网请求。
        """
        return any(k in t for k in (
            "联网", "上网搜", "网页", "搜索一下", "百度", "谷歌",
            "多少钱", "价格", "报价", "优惠", "落地价", "降价",
            "新闻", "上市", "改款", "新款", "发布", "召回",
            "口碑", "评价", "车主说", "销量", "投诉",
        ))

    @staticmethod
    def _web_query(text: str) -> str:
        """从整句话里抽出适合搜索的关键词。"""
        cleaned = re.sub(
            r"(帮我|请|联网|上网|搜索一下|搜一下|搜搜|百度一下|百度|谷歌|网页|"
            r"查一下|查询|看看|了解一下|告诉我|现在|最新|最新的|的|吗|呢|？|\?|。|，|,)",
            " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return (cleaned or text.strip())[:80]

    @staticmethod
    def _guess_keyword(t: str) -> str:
        m = re.search(r"\b(\d{4})\b", t)
        if m:
            return m.group(1)
        # 先剥掉疑问与指令性措辞，再取剩下的中文片段当车型名，
        # 避免把「有没有记过轩逸」整句当成关键词
        cleaned = re.sub(
            r"(有没有记过|我记过|我收藏过|我搜|我|这是我|这是|帮我|查一下|介绍一下|"
            r"这是什么|什么车|哪款|这车|怎么样|记过|收藏过|找找|搜一下|搜|看看|一下|"
            r"的|吗|呢|？|\?|。|，|,)", "", t)
        m = re.search(r"([\u4e00-\u9fffA-Za-z0-9]{2,8})", cleaned)
        return m.group(1) if m else ""

    @staticmethod
    def _guess_note(t: str) -> str:
        m = re.search(r"[「“\"']([^」”\"']{1,60})[」”\"']", t)
        return m.group(1) if m else ""

    def _compose_web_summary(self, last: dict, page: dict | None = None) -> str:
        """把「搜索结果 + 打开的页面正文」整理成人话。

        离线替身用；真实模型会自己组织语言，但这条**链**是一样的：
        搜索拿摘要 → 打开最相关的一条读正文 → 综合。
        """
        page = page or {}
        items = last.get("results") or []
        if not items and not page:
            return ("没搜到相关内容。可以换个更具体的关键词再试，"
                    "或者告诉我你想了解这款车的哪个方面。\n\n"
                    "（离线演示模式：文字由内置的脚本化替身生成。）")
        lines = []
        if items:
            lines.append(f"我上网查了一下「{last.get('query', '')}」，找到 {len(items)} 条结果：")
            for i, it in enumerate(items, 1):
                lines.append(f"{i}. **{it.get('title', '')}**")
                if it.get("snippet"):
                    lines.append(f"   {it['snippet']}")
                lines.append(f"   {it.get('url', '')}")
        if page and page.get("ok"):
            lines.append(f"\n我又打开了其中最相关的那一条读正文（《{page.get('title', '')}》，"
                         f"{page.get('chars', 0)} 字）：")
            excerpt = (page.get("text") or "").strip().replace("\n", " ")
            lines.append("> " + excerpt[:400] + ("…" if len(excerpt) > 400 else ""))
        lines.append("\n以上来自公网，属于参考信息，具体请以官方渠道为准。")
        lines.append("\n（离线演示模式：文字由内置的脚本化替身生成。）")
        return "\n".join(lines)

    def _compose_trim_summary(self, last: dict) -> str:
        """档次对比（离线替身用）。"""
        if not last.get("has_trim_data"):
            return (last.get("message") or "档案库里还没有这款车的档次数据。")
        rows = last.get("table") or []
        lines = [f"**{last.get('car_name', '')}** 的档次情况：{last.get('trim_summary', '')}"]
        pr = last.get("price_range") or {}
        if pr.get("guide"):
            lines.append(f"厂商指导价区间：{pr['guide'][0]:g}–{pr['guide'][1]:g} 万")
        if pr.get("dealer"):
            lines.append(f"经销商参考价区间：{pr['dealer'][0]:g}–{pr['dealer'][1]:g} 万")
        keys = last.get("differing_specs") or []
        if keys:
            lines.append(f"\n各档次之间有差别的参数：{'、'.join(keys)}")
        for r in rows:
            price = f"{r['guide_price_wan']:g} 万" if r.get("guide_price_wan") else "（无价格）"
            extra = ""
            if r.get("differing"):
                extra = "；".join(f"{k}={v}" for k, v in r["differing"].items())
            lines.append(f"· **{r['trim']}** — {price}" + (f"（{extra}）" if extra else ""))
        if last.get("sources"):
            lines.append(f"\n数据来源：{last['sources'][-1].get('site') or last['sources'][-1].get('url')}"
                         f"（抓取于 {str(last['sources'][-1].get('fetched_at', ''))[:16]}）")
        lines.append("\n（离线演示模式：文字由内置的脚本化替身生成。）")
        return "\n".join(lines)

    def _compose_trim_saved(self, last: dict) -> str:
        """档次落库后的回执（离线替身用）。"""
        lines = [f"已经把 **{last.get('car_name', '')}** 的 {last.get('saved', 0)} 个档次"
                 f"存进档案库了（共 {last.get('total_trims', 0)} 个）。"]
        if last.get("trim_summary"):
            lines.append(last["trim_summary"])
        if last.get("added"):
            lines.append("新增：" + "、".join(last["added"][:6]))
        if last.get("updated"):
            lines.append("更新：" + "、".join(last["updated"][:6]))
        src = last.get("source") or {}
        lines.append(f"来源：{src.get('site') or src.get('url', '')}"
                     + ("" if last.get("verified") else "（⚠️ 未经核验）"))
        lines.append("\n以后问这款车的价格或配置差别，我可以直接从档案库回答，不用再联网。")
        lines.append("\n（离线演示模式：文字由内置的脚本化替身生成。）")
        return "\n".join(lines)

    def _compose_unknown_reply(self, last: dict) -> str:
        cands = last.get("top_candidates") or []
        lines = ["这张照片里的车**未识别为本组支持的 50 款车型**（模型判定为 unknown）。"]
        if cands:
            top = "、".join(f"{c['car_name']}（{c['confidence']:.1%}）" for c in cands[:3])
            lines.append(f"外形上它和这几款比较接近：{top}，但这只是相似度排序，不是结论。")
        lines.append("如果你知道具体是哪一款，可以直接告诉我车型名，我帮你查档案；"
                     "也可以换一张正侧面、光线更足的照片再试一次。")
        lines.append("\n（当前为**离线演示模式**：没有配置 DEEPSEEK_API_KEY，"
                     "这段文字由内置的脚本化替身生成，用于验证工具链是否连通。）")
        return "\n".join(lines)

    def _compose_garage_summary(self, g: dict) -> str:
        total = g.get("total_records", 0)
        if not total:
            return ("你的看车档案现在还是空的。上传一张车照，说「记录下来」我就帮你存进去。\n\n"
                    "（离线演示模式：文字由脚本化替身生成。）")
        lines = [f"你的看车档案里一共有 **{total}** 条记录。"]
        groups = g.get("by_body_group") or []
        if groups:
            lines.append("按车身类型：" + "、".join(f"{r['body_group']} {r['n']} 条" for r in groups))
        brands = g.get("by_brand") or []
        if brands:
            lines.append("按品牌：" + "、".join(f"{r['brand']} {r['n']} 条" for r in brands[:8]))
        recent = g.get("recent") or []
        if recent:
            lines.append("最近记录：" + "、".join(r["car_name"] for r in recent[:5]))
        cov = g.get("coverage") or {}
        if cov:
            lines.append(f"覆盖度：已收录 {cov.get('collected_records')} / "
                         f"支持的 {cov.get('supported_models')} 款车型"
                         f"（{cov.get('coverage_pct')}%）。")
        lines.append("\n（离线演示模式：文字由脚本化替身生成。）")
        return "\n".join(lines)

    def _compose_list_summary(self, r: dict) -> str:
        cars = r.get("cars") or []
        names = "、".join(f"{c['car_name']}({c['car_id']})" for c in cars[:12])
        more = f"，等共 {len(cars)} 款" if len(cars) > 12 else ""
        f = r.get("filter") or {}
        scope = f"（{f['body_group']}）" if f.get("body_group") else ""
        return (f"本组视觉模型支持 **{r.get('total_supported')}** 款车型{scope}。"
                f"筛选后共 {len(cars)} 款：{names}{more}。\n\n"
                f"（离线演示模式：文字由脚本化替身生成。）")

    def _compose_search_summary(self, r: dict) -> str:
        recs = r.get("records") or []
        if not recs:
            return ("档案里没有匹配的记录。\n\n（离线演示模式：文字由脚本化替身生成。）")
        lines = [f"在档案中找到 {len(recs)} 条匹配记录："]
        for x in recs[:10]:
            note = f"｜备注：{x['note']}" if x.get("note") else ""
            lines.append(f"· #{x['record_id']} {x['car_name']}（{x.get('brand') or ''}"
                         f"{' · ' + x['body_group'] if x.get('body_group') else ''}）"
                         f"{x['created_at']}{note}")
        lines.append("\n（离线演示模式：文字由脚本化替身生成。）")
        return "\n".join(lines)

    def _compose_info_summary(self, state: dict, info: dict) -> str:
        p = state.get("last_prediction") or {}
        name = info.get("car_name") or p.get("car_name") or "该车型"
        hl = info.get("highlights") or []
        lines = [f"这是 **{name}**（{info.get('brand', '')} · {info.get('body_group', '')}"
                 f"{' · ' + info['body_detail'] if info.get('body_detail') else ''}）。"]
        if info.get("intro"):
            lines.append(info["intro"])
        if hl:
            lines.append("看点：" + "；".join(hl))
        if p.get("confidence") is not None:
            lines.append(f"（识别置信度 {float(p['confidence']):.1%}）")
        lines.append("\n（离线演示模式：文字由脚本化替身生成。）")
        return "\n".join(lines)

    def _compose_seen_summary(self, state: dict) -> str:
        p = state.get("last_prediction") or {}
        return (f"识别结果是 **{p.get('car_name')}**（{p.get('car_id')}），"
                f"置信度 {float(p.get('confidence') or 0):.1%}。\n"
                f"你想让我做什么？可以「介绍一下这车」，或者「加进我的收藏」。\n"
                f"\n（离线演示模式：文字由脚本化替身生成。）")

    def _compose_record_summary(self, state: dict, last: dict) -> str:
        p = state.get("last_prediction") or {}
        name = last.get("car_name") or p.get("car_name") or "该车型"
        extra = "（该车型此前已在档案中，这次更新了备注）" if last.get("duplicated") else ""
        return (f"已经把 **{name}** 写入你的看车档案{extra}。\n"
                f"记录编号 #{last.get('record_id')}，当前这个会话一共记录了 "
                f"{last.get('total_records_in_session', '若干')} 款车。\n"
                f"之后你可以直接说「我的看车档案里都有什么」，我帮你按品牌和车身类型汇总。\n"
                f"\n（离线演示模式：文字由脚本化替身生成。）")

    def _compose_fallback(self, text: str, state: dict) -> str:
        tips = [
            "我目前能做这些事：",
            "· 上传一张车辆照片，我识别它是哪一款（本组支持 50 款车型）",
            "· 查某一款车的档案（品牌、车身类型、看点）",
            "· 把车写进你的看车档案，并按品牌 / 车身类型检索",
        ]
        if not state.get("has_image"):
            tips.append("\n先上传一张车照试试吧。")
        return "\n".join(tips) + "\n\n（离线演示模式：文字由脚本化替身生成。）"


def get_client(config: Config):
    """按配置返回模型客户端。"""
    if config.offline:
        return ScriptedClient(config)
    return DeepSeekClient(config)
