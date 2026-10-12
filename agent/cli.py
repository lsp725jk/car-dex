# -*- coding: utf-8 -*-
"""命令行入口 —— 用于不启动 Web 就能调试智能体。

用法：
    python -m agent.cli                          # 交互式对话
    python -m agent.cli --image 图片路径           # 先"上传"一张图再对话
    python -m agent.cli --once "这是什么车？"       # 单次提问
    python -m agent.cli --tools                  # 只打印工具清单
    python -m agent.cli --events <session_id>    # 打印某会话的事件时间线
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.config import load_config          # noqa: E402
from agent.loop import Agent                  # noqa: E402
from agent.session import SessionStore        # noqa: E402


def print_tools(agent: Agent) -> None:
    tools = agent.describe_tools()
    print(f"共 {len(tools)} 个工具：\n")
    for t in tools:
        props = t["parameters"].get("properties", {})
        req = set(t["parameters"].get("required", []))
        params = "、".join(
            f"{k}{'' if k in req else '?'}" for k in props
        ) or "（无参数）"
        print(f"▪ {t['name']}")
        print(f"    用途：{t['description'][:110]}")
        print(f"    参数：{params}    超时：{t['timeout_s']}s"
              f"{'    需要图片' if t['requires_image'] else ''}")


def make_printer():
    def on_event(ev: dict) -> None:
        t = ev.get("type")
        if t == "tool_call":
            print(f"   ↳ 调用工具 {ev['tool']}({json.dumps(ev['arguments'], ensure_ascii=False)})")
        elif t == "tool_result":
            mark = "✓" if ev.get("ok") else "✗"
            print(f"     {mark} {ev['tool']} [{ev.get('elapsed_ms', 0):.0f}ms] {ev.get('summary', '')}")
        elif t == "error":
            print(f"   ⚠ 错误 {ev.get('code')}: {ev.get('message')}")
    return on_event


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="先上传一张图片")
    ap.add_argument("--once", default=None, help="只问一次就退出")
    ap.add_argument("--tools", action="store_true", help="打印工具清单")
    ap.add_argument("--events", default=None, help="打印某会话的事件时间线")
    ap.add_argument("--session", default=None, help="指定会话 ID（用于恢复）")
    ap.add_argument("--offline", action="store_true", help="强制离线替身模式")
    ap.add_argument("--no-offline", action="store_true", help="强制真实模型（需 API Key）")
    args = ap.parse_args()

    cfg = load_config(True if args.offline else (False if args.no_offline else None))
    agent = Agent(cfg)

    if args.tools:
        print_tools(agent)
        return

    if args.events:
        s = agent.store.get(args.events)
        if s is None:
            print(f"找不到会话 {args.events}")
            return
        for ev in s.events(500):
            p = ev.get("payload") or {}
            brief = json.dumps(p, ensure_ascii=False)
            print(f"[{ev['ts']}] {ev['type']:<18} {brief[:150]}")
        return

    print("=" * 74)
    print("识车档案 CarDex — 智能体命令行")
    print(f"  模型      : {cfg.model}")
    print(f"  模式      : {'离线替身（未配置 DEEPSEEK_API_KEY）' if cfg.offline else '真实 DeepSeek API'}")
    print(f"  API Key   : {cfg.masked_key()}")
    print(f"  视觉模型  : {[p.parent.name for p in cfg.checkpoint_paths() if p.exists()] or '未找到'}")
    print(f"  工具数    : {len(agent.registry.names())}")
    print("=" * 74)

    sess = agent.store.get_or_create(args.session)
    if args.session:
        print(f"已恢复会话 {sess.session_id}（历史消息 {len(sess.messages)} 条）")
    else:
        print(f"新会话 {sess.session_id}")

    if args.image:
        p = Path(args.image)
        if not p.exists():
            print(f"图片不存在：{p}")
            return
        sess.image_path = str(p)
        sess.append_event("image_uploaded", {"path": str(p), "name": p.name})
        print(f"已上传图片：{p.name}")

    on_event = make_printer()

    def ask(text: str) -> None:
        print(f"\n你 > {text}")
        r = agent.run_turn(sess, text, on_event=on_event)
        if r.ok:
            print(f"\n助手 > {r.reply}")
        else:
            print(f"\n助手 > [执行失败] {r.error_code}: {r.error_message}")
        print(f"       （{r.rounds} 轮，{len(r.tool_calls)} 次工具调用，{r.elapsed_ms / 1000:.1f}s）")

    if args.once:
        ask(args.once)
        return

    print("\n输入问题开始对话；:quit 退出，:image <路径> 上传图片，:tools 看工具，:events 看时间线\n")
    while True:
        try:
            text = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in (":quit", ":q", "exit"):
            break
        if text == ":tools":
            print_tools(agent)
            continue
        if text == ":events":
            for ev in sess.events(100):
                print(f"  [{ev['ts']}] {ev['type']}")
            continue
        if text.startswith(":image "):
            p = Path(text[7:].strip())
            if p.exists():
                sess.image_path = str(p)
                sess.append_event("image_uploaded", {"path": str(p), "name": p.name})
                print(f"已上传图片：{p.name}")
            else:
                print(f"图片不存在：{p}")
            continue
        ask(text)

    print(f"\n会话 {sess.session_id} 结束。事件日志：{sess.log_path}")


if __name__ == "__main__":
    main()
