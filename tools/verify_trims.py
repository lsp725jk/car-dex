# -*- coding: utf-8 -*-
"""两个新功能的端到端验证：
  ① 档案库支持档次（价格/参数/档次间差别）
  ② 搜索到的信息能存进档案
"""
import json
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from agent.config import load_config
from agent.loop import Agent
from agent.session import SessionStore
from agent.tools.trims import load_trims, trims_path

cfg = load_config(offline=False)
ag = Agent(cfg, store=SessionStore(cfg))
results = []


def check(name, ok, detail=""):
    results.append((ok, name))
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {name}")
    if detail:
        print(f"          {detail}")


def drive(sess, text, timeout_note=""):
    ev = []
    t0 = time.perf_counter()
    r = ag.run_turn(sess, text, on_event=ev.append)
    print(f"\n  你 > {text}")
    for e in ev:
        if e["type"] == "tool_call":
            print(f"     ↳ {e['tool']}({json.dumps(e['arguments'], ensure_ascii=False)[:110]})")
        elif e["type"] == "tool_result":
            mk = "OK " if e.get("ok") else "ERR"
            print(f"       [{mk}] {e['tool']} {e.get('elapsed_ms', 0):.0f}ms  {e.get('summary','')[:95]}")
    print(f"  助手 > {r.reply[:700]}")
    print(f"     （{r.rounds} 轮 · {len(r.tool_calls)} 次工具 · {time.perf_counter()-t0:.1f}s）")
    return r, [c.name for c in r.tool_calls]


print("=" * 82)
print("①② 档次知识库 + 搜索结果落库")
print("=" * 82)

# **冷启动**：先把目标车型的档次数据清掉，否则档案库里已有数据时，
# 模型第一轮会直接从库里回答、不再联网抓取，这条链路就测不到了。
#
# 但清掉会**洗掉之前积累的知识**，所以先备份，测完再合并回去
# （走和 save_car_trims 同一套归并规则，不会产生重复项）。
_target = "0975"
_data = load_trims(cfg)
_backup = None
if _target in _data["cars"]:
    _backup = _data["cars"].pop(_target)
    from agent.tools.trims import save_trims
    save_trims(cfg, _data)
    print(f"（已备份并清空 {_target} 的 {len(_backup.get('trims', []))} 个档次，"
          f"确保本轮走完整的「联网→落库」路径；跑完会合并回去）")


def _restore_backup() -> None:
    """把开跑前备份的档次合并回档案库（用同一套归并规则）。"""
    if not _backup:
        return
    from agent.tools.trims import dedupe_car, load_trims as _load, save_trims as _save
    from agent.tools.trims import _same_trim
    d = _load(cfg)
    entry = d["cars"].get(_target)
    if entry is None:
        d["cars"][_target] = _backup
    else:
        for t in _backup.get("trims", []):
            if not any(_same_trim(k["name"], t["name"], k.get("guide_price_wan"),
                                  t.get("guide_price_wan")) for k in entry["trims"]):
                entry["trims"].append(t)
        for s in _backup.get("sources", []):
            if s.get("url") not in {x.get("url") for x in entry.get("sources", [])}:
                entry.setdefault("sources", []).append(s)
    _save(cfg, d)
    dedupe_car(cfg, _target)


before = len(load_trims(cfg)["cars"])

# ---------- 第 1 轮：查档次并存入档案 ----------
print("\n" + "─" * 82)
print("第 1 轮：查「轩逸有哪些配置、各多少钱」，并要求存进档案库")
s = ag.store.create()
s.set_web_search(True)
r1, used1 = drive(s, "日产轩逸有哪些配置版本？各自多少钱？"
                     "帮我上网查清楚，然后存进档案库里，以后我直接问你就行。")

check("第 1 轮执行成功", r1.ok, r1.error_code or "")
check("先查了本地档案库 query_car_info", "query_car_info" in used1, str(used1))
check("联网搜索 web_search", "web_search" in used1, str(used1))
check("真实浏览器打开网页 browse_page", "browse_page" in used1, str(used1))
check("把结果存进档案库 save_car_trims", "save_car_trims" in used1, str(used1))

data = load_trims(cfg)
check("档案库里确实多了这款车", len(data["cars"]) > before or "0975" in data["cars"],
      f"{before} → {len(data['cars'])} 款")
entry = data["cars"].get("0975", {})
trims = entry.get("trims", [])
check("存档的档次数量 > 1", len(trims) > 1, f"{len(trims)} 个")
if trims:
    named = [t["name"] for t in trims][:5]
    print(f"          档次示例: {named}")
    with_price = [t for t in trims if t.get("guide_price_wan")]
    check("档次带上了价格", len(with_price) > 0,
          f"{len(with_price)}/{len(trims)} 个有指导价：" +
          "、".join(f"{t['name'][:16]}={t['guide_price_wan']}万" for t in with_price[:4]))
    with_specs = [t for t in trims if t.get("specs")]
    check("档次带上了参数配置", len(with_specs) > 0,
          f"{len(with_specs)}/{len(trims)} 个有参数；" +
          (f"首个含 {len(with_specs[0]['specs'])} 项：{list(with_specs[0]['specs'])[:5]}"
           if with_specs else ""))
    srcs = entry.get("sources", [])
    check("每条数据都带来源 URL", all(t.get("source", {}).get("url") for t in trims),
          f"sources: {[x.get('url', '')[:52] for x in srcs]}")
    check("来源被核验为「本会话真的打开过」",
          any(x.get("verified") for x in srcs),
          f"verified={[x.get('verified') for x in srcs]}")

# ---------- 第 2 轮：从档案库直接比较（不再联网） ----------
print("\n" + "─" * 82)
print("第 2 轮：换个新会话，直接问档次差别（应该走本地档案，不必联网）")
s2 = ag.store.create()          # 注意：**没有**开联网
r2, used2 = drive(s2, "日产轩逸豪华版比舒适版贵多少？差在哪些配置上？")
check("第 2 轮执行成功", r2.ok, r2.error_code or "")
check("用了 compare_car_trims 做档次对比", "compare_car_trims" in used2, str(used2))
check("未开启联网也能答（数据来自档案库）",
      "web_search" not in used2 and "browse_page" not in used2, str(used2))
check("回复里有具体差价或价格",
      any(k in r2.reply for k in ("万", "元")) and len(r2.reply) > 60,
      f"{len(r2.reply)} 字")

# ---------- 反编造 ----------
print("\n" + "─" * 82)
print("反编造兜底：从没查过的车型，不应凭空给出价格")
s3 = ag.store.create()
r3, used3 = drive(s3, "宝马X7 的各个配置分别多少钱？")
has_fake = any(k in r3.reply for k in ("万左右", "指导价约", "大约", "大概是"))
check("没有查到就如实说明（或引导去联网）",
      any(k in r3.reply for k in ("联网", "没有", "查不到", "档案库里", "未收录", "无法")),
      r3.reply[:150].replace("\n", " "))
check("没有编造具体价格", not has_fake or "万" not in r3.reply[:80],
      f"回复前 80 字：{r3.reply[:80]}".replace("\n", " "))

print("\n" + "=" * 82)
ok = sum(1 for k, _ in results if k)
print(f"通过 {ok} 项，失败 {len(results) - ok} 项")
for k, n in results:
    if not k:
        print("  - " + n)
print("=" * 82)

# 把开跑前备份的档次合并回去，别让测试洗掉积累的知识
_restore_backup()
_final = load_trims(cfg)["cars"].get(_target, {})
print(f"\n档案库文件：{trims_path(cfg)}")
print(f"合并备份后：{_final.get('car_name', _target)} "
      f"{summarize_trims(_final)}（共 {len(_final.get('trims', []))} 个档次）")
sys.exit(1 if ok != len(results) else 0)
