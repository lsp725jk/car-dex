# -*- coding: utf-8 -*-
"""工具包：注册表 + 具体工具。

任务书 5.2 节要求「工具总数不少于 3 个」，本项目的工具清单：

    ┌─ 视觉分类（任务书要求至少 1 个）
    │   classify_car            真实加载并调用本组训练的 51 类模型，返回类别与 confidence
    │
    ├─ 车型知识（个人知识库选题的「检索」能力）
    │   query_car_info          按 ID 或名称查询车型档案（含已积累的档次/价格数据）
    │   list_supported_cars     列出 50 款支持车型，可按车身大类/品牌筛选
    │
    ├─ 档次知识（联网抓取沉淀下来的配置/价格，可反复查阅）
    │   save_car_trims          把网页上读到的档次/价格/参数写进档案库（必须带来源 URL）
    │   compare_car_trims       比较各档次之间的差别（价格跨度 + 配置差异）
    │
    └─ 看车档案（个人知识库选题的「记录」能力）
        add_car_record          把车型写入用户自己的看车档案（可记下是哪一档）
        search_my_records       检索档案（关键词/ID/品牌/车身大类）
        update_record_note      修改档案备注
        delete_car_record       删除档案记录
        my_garage_stats         档案统计与覆盖度

    └─ 联网（可选，由界面上的「联网搜索」按钮逐会话开关）
        web_search              搜索公网信息（价格 / 新闻 / 口碑等本地档案答不了的）
        browse_page             用真实浏览器打开网页，读前端渲染后的正文

共 12 个工具，其中视觉工具 1 个、其他业务工具 11 个，满足「至少 1 + 至少 2」的要求。
"""
from __future__ import annotations

from .browser import BROWSER_TOOLS
from .garage import GARAGE_TOOLS
from .profile import PROFILE_TOOLS
from .registry import Aborted, ToolContext, ToolDefinition, ToolRegistry, ToolResult
from .trims import TRIM_TOOLS
from .vision import VISION_TOOLS
from .websearch import WEBSEARCH_TOOLS

ALL_TOOLS: list[ToolDefinition] = (
    VISION_TOOLS + PROFILE_TOOLS + TRIM_TOOLS + GARAGE_TOOLS
    + WEBSEARCH_TOOLS + BROWSER_TOOLS)

# 需要用户在界面上显式开启才会生效的工具。
# 它们都属于"会往外发请求、会引入公网不可信内容"的能力，所以共用一个开关。
GATED_TOOLS: set[str] = {"web_search", "browse_page"}


def enabled_tool_names(session) -> set[str]:
    """按会话状态算出**本次允许模型看见**的工具名集合。

    联网搜索默认关闭：它是一个"会往外发请求、会产生额外费用、可能引入不可信内容"的能力，
    应该由用户在界面上主动打开，而不是让模型自己决定什么时候联网。
    """
    names = {t.name for t in ALL_TOOLS}
    if not getattr(session, "web_search_enabled", False):
        names -= GATED_TOOLS
    return names


def build_registry(config=None) -> ToolRegistry:
    """构造一个装好全部工具的注册表。

    同时挂上两个钩子，演示「工具执行流水线」的可扩展性：
      · 门禁：需要图片的工具在会话没有图片时直接拒绝，避免把无意义调用送进模型
      · 后置：给失败结果统一补一个 `retryable` 标记，方便上层与提示词策略使用
    """
    reg = ToolRegistry()
    for t in ALL_TOOLS:
        reg.register(t)

    def image_gate(tool: ToolDefinition, args: dict, ctx: ToolContext) -> str | None:
        if tool.requires_image and not (args.get("image_path") or ctx.image_path):
            return ("当前会话还没有上传任何图片，无法执行该工具。"
                    "请提示用户先上传一张车辆照片。")
        return None

    def web_search_gate(tool: ToolDefinition, args: dict, ctx: ToolContext) -> str | None:
        """联网搜索必须由用户在界面上打开。

        正常情况下这个工具根本不会出现在模型的工具清单里；这里是第二道防线——
        万一模型从历史消息里"记住"了它并直接调用，也要挡住，
        并给出可执行的提示，而不是让它静默失败。
        """
        if tool.name in GATED_TOOLS and not getattr(ctx.session, "web_search_enabled", False):
            return ("用户尚未开启联网搜索。请告诉用户：点击输入框左侧的「🌐 联网搜索」按钮"
                    "开启后，你就能帮他查最新的外部信息。不要凭空猜测需要联网才能回答的内容。")
        return None

    def mark_retryable(tool, res: ToolResult, ctx: ToolContext) -> None:
        if res.ok:
            return
        transient = {"TOOL_TIMEOUT", "TOOL_FAILED", "API_REQUEST_FAILED", "IMAGE_UNREADABLE"}
        res.payload.setdefault("retryable", res.code in transient)

    reg.add_gate(image_gate)
    reg.add_gate(web_search_gate)
    reg.add_post(mark_retryable)

    if config is not None:
        reg.enable_log(config.tool_log_file)
    return reg


__all__ = [
    "Aborted", "ToolContext", "ToolDefinition", "ToolRegistry", "ToolResult",
    "ALL_TOOLS", "GATED_TOOLS", "build_registry", "enabled_tool_names",
    "VISION_TOOLS", "PROFILE_TOOLS", "TRIM_TOOLS", "GARAGE_TOOLS",
    "WEBSEARCH_TOOLS", "BROWSER_TOOLS",
]
