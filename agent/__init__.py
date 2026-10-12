# -*- coding: utf-8 -*-
"""任务二：识车档案智能体（CarDex Agent）

一个基于 DeepSeek Function Calling 的**自建智能体**，与任务一的视觉模型对接，
为用户提供「私人车型资料库 / 对话式看车档案」业务能力。

模块划分（借鉴主流 agent harness 的分层思路：**只有循环是逻辑，其余都是可挂载的环节**）：

    config.py     配置：模型名、路径、API Key 来源、各类阈值
    errors.py     错误码与异常类型（工具失败不伪造成功）
    llm.py        模型客户端：真实 DeepSeek 客户端 + 离线脚本化替身
    prompts.py    系统提示词：按「段」组装，而非拼一大坨字符串
    session.py    会话状态 + 事件日志 + SQLite 持久化
    tools/        工具：注册表 + 执行流水线 + 具体工具实现
    loop.py       Agent Loop：调用模型 → 执行工具 → 回填结果 → 重复

上层入口：
    web/app.py    任务三的 Web 系统
    cli.py        命令行调试入口
"""
from __future__ import annotations

__all__ = ["config", "errors", "llm", "prompts", "session", "tools", "loop"]

__version__ = "1.0.0"
