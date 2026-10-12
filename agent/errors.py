# -*- coding: utf-8 -*-
"""错误码与异常类型。

设计原则（与主流 agent harness 一致）：**工具失败必须如实上报，不能伪造成功**。
每个失败都带一个稳定的机器可读 `code`，让上层的提示词策略和 UI 能分别处理，
而不是只拿到一句人类可读的报错。

误差分层：
    AgentError        智能体运行时错误（配置、会话、模型）
    ToolError         工具执行错误（参数、超时、业务失败）
"""
from __future__ import annotations

from typing import Any


# --------------------------------------------------------------------------- #
# 错误码（稳定标识，勿随意改动；Web 前端与提示词策略都依赖它们）
# --------------------------------------------------------------------------- #
class Code:
    # ---- 工具调度层 ----
    UNKNOWN_TOOL = "UNKNOWN_TOOL"                  # 模型请求了未注册的工具
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"        # 参数不符合 JSON Schema
    DENIED_BY_POLICY = "DENIED_BY_POLICY"          # 被前置门禁拒绝（权限/阈值策略）
    TOOL_TIMEOUT = "TOOL_TIMEOUT"                  # 超时
    TOOL_FAILED = "TOOL_FAILED"                    # 工具内部抛出异常
    ABORTED = "ABORTED"                            # 执行中被取消
    ABORTED_BEFORE_DISPATCH = "ABORTED_BEFORE_DISPATCH"  # 还没开始就被取消
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"      # 缺少前置条件（如没上传图片）

    # ---- 业务层（视觉） ----
    IMAGE_NOT_FOUND = "IMAGE_NOT_FOUND"            # 图片路径不存在
    IMAGE_UNREADABLE = "IMAGE_UNREADABLE"          # 图片存在但读不出来
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"          # 视觉模型加载失败

    # ---- 业务层（资料库 / 档案） ----
    CAR_NOT_FOUND = "CAR_NOT_FOUND"                # car_id 不在 50 类中
    RECORD_NOT_FOUND = "RECORD_NOT_FOUND"          # 档案记录不存在
    DUPLICATE_RECORD = "DUPLICATE_RECORD"          # 重复收藏
    UNKNOWN_CAR = "UNKNOWN_CAR"                    # 模型判定为 unknown，不能据此继续业务

    # ---- 模型与网络 ----
    API_KEY_MISSING = "API_KEY_MISSING"            # 未配置 API Key
    API_REQUEST_FAILED = "API_REQUEST_FAILED"      # 其他 HTTP / 请求失败
    API_RESPONSE_INVALID = "API_RESPONSE_INVALID"  # 返回体结构不符合预期
    MODEL_REFUSED = "MODEL_REFUSED"                # 模型返回空内容
    # 输出被 max_tokens 截断。**单列一个码**是因为它能自动补救：
    # 模型在生成一个很大的工具参数对象时最容易撞上（实测：让它把整张配置表
    # 塞进 save_car_trims，10 个档次 × 几十项参数，2048 个 token 直接不够）。
    # 客户端见到这个码会自动放大预算重试一次。
    OUTPUT_TRUNCATED = "OUTPUT_TRUNCATED"
    MAX_ROUNDS_EXCEEDED = "MAX_ROUNDS_EXCEEDED"    # 超过最大工具轮数
    MAX_TOOL_CALLS_EXCEEDED = "MAX_TOOL_CALLS_EXCEEDED"

    # ---- 网络分层错误（**可重试**，见 agent/llm.py 的自动重试）----
    # 把"网络出问题"拆成四层，用户一眼能看出该修什么：
    #   DNS 解析 → TCP 连接 → TLS 握手 → HTTP 状态码
    NETWORK_DNS_FAILED = "NETWORK_DNS_FAILED"      # 域名解析失败（getaddrinfo failed）
    NETWORK_UNREACHABLE = "NETWORK_UNREACHABLE"    # TCP 连不上 / 被防火墙或代理拦截
    NETWORK_TLS_FAILED = "NETWORK_TLS_FAILED"      # TLS 握手或证书失败
    API_TIMEOUT = "API_TIMEOUT"                    # 连接或读取超时
    API_RATE_LIMITED = "API_RATE_LIMITED"          # HTTP 429
    API_SERVER_ERROR = "API_SERVER_ERROR"          # HTTP 5xx（服务端问题）


class AgentError(Exception):
    """智能体运行时错误（非工具内部错误）。"""

    def __init__(self, code: str, message: str, *, info: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.info = info or {}

    def to_dict(self) -> dict[str, Any]:
        return {"ok": False, "error": {"code": self.code, "message": self.message, **self.info}}


class ToolError(Exception):
    """工具执行错误。`code` 会原样回传给模型，让模型知道该怎么修正。"""

    def __init__(self, code: str, message: str, *, hint: str = "", info: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint          # 给模型的修正建议（会写进工具结果）
        self.info = info or {}


def tool_failure(code: str, message: str, *, hint: str = "", **extra: Any) -> dict[str, Any]:
    """构造一个「失败」的工具返回体。"""
    payload: dict[str, Any] = {"ok": False, "error": {"code": code, "message": message}}
    if hint:
        payload["error"]["hint"] = hint
    if extra:
        payload["error"].update(extra)
    return payload
