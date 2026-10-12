# -*- coding: utf-8 -*-
"""浏览器工具的完整验证（需要能启动 Chromium，故在放宽沙箱下运行）。"""
import json
import sys
import threading
import time

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from agent.config import load_config
from agent.loop import Agent, summarize_tool_result
from agent.session import SessionStore
from agent.tools.registry import ToolContext
from agent.tools import enabled_tool_names
from agent.prompts import build_system_prompt

cfg = load_config(offline=False)
ag = Agent(cfg, store=SessionStore(cfg))
PASS, FAIL = "[OK]  ", "[FAIL]"

results = []


def check(name, ok, detail=""):
    results.append((ok, name))
    print(f"  {PASS if ok else FAIL} {name}")
    if detail:
        print(f"          {detail}")


def call(sess, tool, args, image=None):
    ctx = ToolContext(session_id=sess.session_id, turn_id="t-test", call_id="c-1",
                      config=cfg, session=sess, signal=threading.Event(),
                      image_path=str(image) if image else None)
    return ag.registry.execute("c-1", tool, json.dumps(args, ensure_ascii=False), ctx)


print("=" * 78)
print("浏览器工具验证")
print("=" * 78)

s = ag.store.create()

# ---- 0. 门控 ----
print("\n[0] 门控：未开启联网时不允许")
r = call(s, "browse_page", {"url": "https://www.autohome.com.cn/448/"})
check("未开启时被门禁拒绝", (not r.ok) and r.code == "DENIED_BY_POLICY", r.code)
check("未开启时工具清单里没有 browse_page",
      "browse_page" not in enabled_tool_names(s),
      f"允许 {len(enabled_tool_names(s))} 个")

s.set_web_search(True)
check("开启后工具清单里有 browse_page", "browse_page" in enabled_tool_names(s),
      f"允许 {len(enabled_tool_names(s))} 个")
p_on = build_system_prompt(cfg, s).text
check("提示词里描述了「搜索→打开→再搜」的循环",
      "循环" in p_on and "browse_page" in p_on and "web_search" in p_on)

# ---- 1. 参数校验 ----
print("\n[1] 参数校验")
check("缺 url → INVALID_ARGUMENTS", call(s, "browse_page", {}).code == "INVALID_ARGUMENTS")
check("url 过长 → INVALID_ARGUMENTS",
      call(s, "browse_page", {"url": "https://x.com/" + "a" * 600}).code == "INVALID_ARGUMENTS")
check("多余参数 → INVALID_ARGUMENTS",
      call(s, "browse_page", {"url": "https://x.com", "zzz": 1}).code == "INVALID_ARGUMENTS")

# ---- 2. 真实浏览器打开 JS 渲染站点 ----
print("\n[2] 真实浏览器抓取（关键：JS 渲染站点）")
t0 = time.perf_counter()
r = call(s, "browse_page", {"url": "https://www.autohome.com.cn/448/", "max_chars": 5000})
first_ms = (time.perf_counter() - t0) * 1000
if r.ok:
    d = r.payload
    print(f"          title = {d.get('title')}")
    print(f"          mode={d.get('mode')}  chars={d.get('chars')}  {d.get('elapsed_ms')}ms")
    txt = d.get("text") or ""
    import re
    prices = re.findall(r"\d+\.\d+-\d+\.\d+万|厂商指导价：[\d.\-]+万", txt)[:5]
    print(f"          价格片段 = {prices}")
    check("mode = browser（真的开了浏览器）", d.get("mode") == "browser", d.get("mode"))
    check("拿到了渲染后的正文", d.get("chars", 0) > 1000, f"{d.get('chars')} 字")
    check("标题是汽车之家轩逸页", "轩逸" in (d.get("title") or ""), d.get("title"))
    check("正文里有厂商指导价（HTTP 抓不到）",
          "厂商指导价" in txt or bool(prices), str(prices))
    check("标注 untrusted（防注入）", d.get("untrusted") is True)
else:
    check("浏览器抓取成功", False, r.code + ": " + str(r.payload)[:200])

# ---- 3. 会话复用（第二次应明显更快）----
print("\n[3] 会话复用")
t0 = time.perf_counter()
r2 = call(s, "browse_page", {"url": "https://www.autohome.com.cn/448/"})
second_ms = (time.perf_counter() - t0) * 1000
check("第二次打开成功", r2.ok, f"{second_ms:.0f}ms（首次 {first_ms:.0f}ms）")
print(f"          复用后耗时 {second_ms:.0f} ms")

# ---- 4. include_links ----
print("\n[4] 附带链接")
r3 = call(s, "browse_page", {"url": "https://www.autohome.com.cn/448/", "include_links": True})
links = (r3.payload.get("links") or []) if r3.ok else []
check("返回了链接列表", len(links) > 3, f"{len(links)} 条")
if links:
    print(f"          示例: {links[0].get('text', '')[:30]} -> {links[0].get('url', '')[:60]}")

# ---- 5. 降级路径 ----
print("\n[5] 浏览器不可用时的降级")
import agent.tools.browser as B
orig = B._candidate_browsers
B._candidate_browsers = lambda: []          # 模拟"找不到浏览器"
B.reset_session()
try:
    r4 = call(s, "browse_page", {"url": "https://example.com"})
    check("自动降级为 HTTP 抓取", r4.ok and r4.payload.get("mode") == "http",
          f"mode={r4.payload.get('mode') if r4.ok else r4.code}")
    if r4.ok:
        check("降级时给出了明确提示",
              "浏览器不可用" in (r4.payload.get("note") or ""))
finally:
    B._candidate_browsers = orig
    B.reset_session()

# ---- 6. 工具卡摘要 ----
print("\n[6] 工具卡摘要")
if r.ok:
    sm = summarize_tool_result("browse_page", r.payload)
    check("摘要文案正确", bool(sm) and "字" in sm, sm)

# ---- 7. 真实模型：搜索 → 打开 → 再搜 ----
print("\n[7] 真实模型端到端（联网多轮：搜索→打开→综合）")
s2 = ag.store.create()
s2.set_web_search(True)
ev = []
r5 = ag.run_turn(s2, "帮我查一下日产轩逸现在的实际成交价大概多少，"
                     "要具体数字，去汽车之家或懂车帝这类网站找，把来源告诉我。",
                 on_event=ev.append)
used = [c.name for c in r5.tool_calls]
print(f"          轮次 {r5.rounds} · 工具 {len(used)} 次 · {r5.elapsed_ms/1000:.1f}s")
print(f"          工具序列: {used}")
print(f"          回复 {len(r5.reply)} 字: {r5.reply[:220].replace(chr(10),' ')}")
check("本轮执行成功", r5.ok, r5.error_code or "")
check("调用了 web_search", "web_search" in used, str(used))
check("调用了 browse_page（真的打开网页读正文）", "browse_page" in used, str(used))
check("回复非空", len(r5.reply) > 80, f"{len(r5.reply)} 字")

# ---- 汇总 ----
ok = sum(1 for k, _ in results if k)
bad = len(results) - ok
print("\n" + "=" * 78)
print(f"通过 {ok} 项，失败 {bad} 项")
for k, n in results:
    if not k:
        print("  - " + n)
print("=" * 78)
sys.exit(1 if bad else 0)
