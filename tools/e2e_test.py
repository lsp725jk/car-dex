# -*- coding: utf-8 -*-
"""任务二 / 任务三 端到端测试。

分两段跑：

  阶段 A「工具层」——**不需要模型**，直接调 `ToolRegistry.execute`，
      逐个验证参数校验、前置门禁、错误码、拒识分支。毫秒级、结果确定。

  阶段 B「模型层」——**调用真实 DeepSeek API**，验证任务书要求的智能体行为：
      · 多轮会话（同一 Session 内延续上文）
      · 单轮内 ≥3 个工具串联（识别 → 查档案 → 写档案）
      · 视觉识别确实通过工具调用发生，而不是外部预先算好塞进提示词
      · unknown / 低置信度时如实告知、不编造、不继续业务操作
      · 没有图片却要求识别时的异常反馈

用法：
    python tools/e2e_test.py                  # A + B（需要 API Key）
    python tools/e2e_test.py --stage A        # 只跑工具层
    python tools/e2e_test.py --stage B
    python tools/e2e_test.py --offline        # 用离线替身跑 B（对照，不代表模型效果）
    python tools/e2e_test.py --only S3,S5     # 只跑指定场景
    python tools/e2e_test.py --cleanup        # 跑完删掉测试写入的档案记录
    python tools/e2e_test.py --json           # 机器可读输出

退出码：0 = 全通过；1 = 有失败项。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agent.config import load_config                      # noqa: E402
from agent.loop import Agent, TurnResult                  # noqa: E402
from agent.session import Session, SessionStore           # noqa: E402
from agent.tools.registry import ToolContext, ToolResult  # noqa: E402

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
MARK = {PASS: "[OK]  ", WARN: "[WARN]", FAIL: "[FAIL]"}


# --------------------------------------------------------------------------- #
# 结果收集
# --------------------------------------------------------------------------- #
class Checks:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, level: str, name: str, detail: str = "") -> None:
        self.items.append({"level": level, "name": name, "detail": detail})

    def expect(self, cond: bool, name: str, detail: str = "") -> bool:
        self.add(PASS if cond else FAIL, name, detail)
        return bool(cond)

    def note(self, name: str, detail: str = "") -> None:
        self.add(WARN, name, detail)

    @property
    def failed(self) -> int:
        return sum(1 for i in self.items if i["level"] == FAIL)

    @property
    def passed(self) -> int:
        return sum(1 for i in self.items if i["level"] == PASS)


# --------------------------------------------------------------------------- #
# 素材
# --------------------------------------------------------------------------- #
def pick_images() -> dict[str, Path | None]:
    """从 data/val 里挑一张「支持车型」和一张「非支持车型」的图。"""
    known = unknown = None
    classes = {ln.strip() for ln in (ROOT / "classes.txt").read_text(encoding="utf-8").splitlines() if ln.strip()}
    val = ROOT / "data" / "val"
    if val.exists():
        for cid in sorted(classes):
            d = val / cid
            if d.exists():
                hits = sorted(d.glob("*.jpg")) or sorted(d.glob("*.png"))
                if hits:
                    known = hits[0]
                    break
        ud = val / "unknown"
        if ud.exists():
            hits = sorted(ud.glob("*.jpg")) or sorted(ud.glob("*.png"))
            if hits:
                unknown = hits[0]
    ups = ROOT / "data" / "uploads"
    for p in sorted(ups.glob("*.jpg")) if ups.exists() else []:
        if unknown and known:
            break
        unknown = unknown or p
        known = known or p
    return {"known": known, "unknown": unknown}


def exec_tool(agent: Agent, session: Session, name: str,
              args: dict | str, image: Path | None = None,
              call_id: str = "c-test") -> ToolResult:
    """不经模型，直接执行一次工具调用。"""
    ctx = ToolContext(
        session_id=session.session_id, turn_id="t-test", call_id=call_id,
        config=agent.config, session=session, signal=threading.Event(),
        image_path=str(image) if image else None,
    )
    raw = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    return agent.registry.execute(call_id, name, raw, ctx)


def code_of(res: ToolResult) -> str:
    return (res.payload.get("error") or {}).get("code", "") or res.code


def names_of(r: TurnResult) -> list[str]:
    return [c.name for c in r.tool_calls]


def result_of(r: TurnResult, tool: str) -> dict:
    for c in r.tool_calls:
        if c.name == tool:
            return c.result or {}
    return {}


# =========================================================================== #
# 阶段 A：工具层
# =========================================================================== #
def stage_a(agent: Agent, imgs: dict, ch: Checks) -> None:
    print("\n" + "=" * 78)
    print("阶段 A —— 工具层（不经过模型，结果确定）")
    print("=" * 78)

    sess = agent.store.create()
    known, unknown = imgs["known"], imgs["unknown"]

    # --- A1 视觉工具：支持车型 ---
    if known:
        r = exec_tool(agent, sess, "classify_car", {"top_k": 5}, image=known)
        d = r.payload.get("decision") or {}
        ch.expect(r.ok, "A1 classify_car 识别支持车型成功",
                  f"{known.name} → {r.payload.get('car_name')} "
                  f"{float(r.payload.get('confidence') or 0):.2%} · {r.elapsed_ms:.0f}ms")
        ch.expect(bool(r.payload.get("car_id")) and not r.payload.get("is_unknown"),
                  "A1a 返回具体车型（非 unknown）", f"car_id={r.payload.get('car_id')}")
        ch.expect(len(r.payload.get("top_candidates") or []) == 5,
                  "A1b 返回 Top-5 候选",
                  "、".join(t["car_name"] for t in (r.payload.get("top_candidates") or [])[:3]) + " …")
        ch.expect(d.get("action") in ("accept", "confirm_with_user"),
                  "A1c 给出结构化的下一步建议", f"decision.action={d.get('action')}")
        ch.expect("unknown_threshold" in (r.payload.get("model") or {}),
                  "A1d 结果里带阈值溯源信息",
                  f"t={r.payload['model']['unknown_threshold']} "
                  f"τ={r.payload['model']['prior_adjust_tau']}")
    else:
        ch.note("A1 跳过", "data/val 下没有可用图片")

    # --- A2 视觉工具：非支持车型 → 拒识 ---
    if unknown:
        r = exec_tool(agent, sess, "classify_car", {}, image=unknown, call_id="c-a2")
        d = r.payload.get("decision") or {}
        ch.expect(r.ok and r.payload.get("is_unknown") is True,
                  "A2 非支持车型被正确拒识（is_unknown=True）",
                  f"{unknown.name} → car_id={r.payload.get('car_id')} "
                  f"conf={float(r.payload.get('confidence') or 0):.2%}")
        ch.expect(d.get("action") == "report_unknown",
                  "A2a 拒识时给出 report_unknown 指引", str(d.get("guidance", ""))[:80] + "…")

    # --- A3 没上传图片就调用 ---
    # 预期由注册表的前置门禁拦下（DENIED_BY_POLICY）；若门禁被绕过，
    # 工具主体也必须自己报 DEPENDENCY_MISSING。两条路径都不允许"伪造成功"。
    r = exec_tool(agent, sess, "classify_car", {}, image=None, call_id="c-a3")
    ch.expect(not r.ok and code_of(r) in ("DENIED_BY_POLICY", "DEPENDENCY_MISSING"),
              "A3 无图片时被前置门禁拦下，如实报错（不伪造结果）",
              f"{code_of(r)} · {str((r.payload.get('error') or {}).get('message',''))[:60]}")

    # --- A4 图片路径不存在 ---
    r = exec_tool(agent, sess, "classify_car", {"image_path": str(ROOT / "data" / "no-such-file.jpg")},
                  image=None, call_id="c-a4")
    ch.expect(not r.ok and code_of(r) == "IMAGE_NOT_FOUND",
              "A4 图片不存在时报 IMAGE_NOT_FOUND", code_of(r))

    # --- A5 参数越界 → JSON Schema 拦截 ---
    r = exec_tool(agent, sess, "classify_car", {"top_k": 999}, image=known, call_id="c-a5")
    ch.expect(not r.ok and code_of(r) == "INVALID_ARGUMENTS",
              "A5 top_k 越界被 Schema 拦截（maxItems/maximum）",
              f"{code_of(r)} · {str((r.payload.get('error') or {}).get('message',''))[:60]}")

    # --- A6 多余参数 → additionalProperties: false ---
    r = exec_tool(agent, sess, "classify_car", {"bogus_field": 1}, image=known, call_id="c-a6")
    ch.expect(not r.ok and code_of(r) == "INVALID_ARGUMENTS",
              "A6 多余参数被 Schema 拦截（additionalProperties=false）", code_of(r))

    # --- A7 参数不是合法 JSON ---
    r = exec_tool(agent, sess, "query_car_info", "{不是JSON", call_id="c-a7")
    ch.expect(not r.ok and code_of(r) == "INVALID_ARGUMENTS",
              "A7 参数 JSON 解析失败时报 INVALID_ARGUMENTS", code_of(r))

    # --- A8 调用不存在的工具 ---
    r = exec_tool(agent, sess, "launch_rocket", {}, call_id="c-a8")
    ch.expect(not r.ok and code_of(r) == "UNKNOWN_TOOL",
              "A8 未注册工具报 UNKNOWN_TOOL", code_of(r))

    # --- A9 查不属于支持范围的车型 ---
    r = exec_tool(agent, sess, "query_car_info", {"car_id_or_name": "0022"}, call_id="c-a9")
    ch.expect(not r.ok and code_of(r) in ("CAR_NOT_FOUND", "INVALID_ARGUMENTS"),
              "A9 查询范围外车型被拒绝（不编造资料）",
              f"{code_of(r)} · {str((r.payload.get('error') or {}).get('message',''))[:60]}")

    # --- A10 未识别就写档案 ---
    r = exec_tool(agent, sess, "add_car_record", {}, call_id="c-a10")
    ch.expect(not r.ok and code_of(r) == "DEPENDENCY_MISSING",
              "A10 没有识别结果时不能写档案",
              f"{code_of(r)} · {str((r.payload.get('error') or {}).get('message',''))[:60]}")

    # --- A11 支持车型清单 ---
    r = exec_tool(agent, sess, "list_supported_cars", {"body_group": "SUV"}, call_id="c-a11")
    items = r.payload.get("cars") or r.payload.get("items") or []
    ch.expect(r.ok and len(items) == 21,
              "A11 list_supported_cars 按车身类型筛选（SUV 应为 21 款）",
              f"实际返回 {len(items)} 款")

    # --- A12 unknown 不能写档案（门禁） ---
    tmp = agent.store.create()
    tmp.last_prediction = {"car_id": "unknown", "car_name": "unknown",
                           "confidence": 0.11, "is_unknown": True}
    r = exec_tool(agent, tmp, "add_car_record", {}, call_id="c-a12a")
    ch.expect(not r.ok and code_of(r) == "UNKNOWN_CAR",
              "A12 拒识结果不能写入档案（前置门禁生效）",
              f"{code_of(r)} · {str((r.payload.get('error') or {}).get('message',''))[:60]}")
    r = exec_tool(agent, tmp, "add_car_record", {"car_id": "unknown"}, call_id="c-a12b")
    ch.expect(not r.ok and code_of(r) == "UNKNOWN_CAR",
              "A12b 显式传 car_id=unknown 同样被拒绝", code_of(r))

    # --- A13 低置信度分支（构造：把显示阈值临时抬到 1.0） ---
    if known:
        old = agent.config.display_confidence_threshold
        try:
            agent.config.display_confidence_threshold = 1.01
            r = exec_tool(agent, sess, "classify_car", {}, image=known, call_id="c-a13")
            act = (r.payload.get("decision") or {}).get("action")
            ch.expect(r.ok and r.payload.get("low_confidence") is True and act == "confirm_with_user",
                      "A13 低置信度时切换为「请用户确认」分支",
                      f"阈值临时抬到 1.01 → action={act}，指引："
                      f"{str((r.payload.get('decision') or {}).get('guidance',''))[:46]}…")
        finally:
            agent.config.display_confidence_threshold = old

    # --- A14 档案记录审计 ---
    log = agent.config.tool_log_file
    ch.expect(log.exists(), "A14 工具调用记录已落盘（任务书 3.2 要求）",
              f"{log.relative_to(ROOT)} · {len(log.read_text(encoding='utf-8').splitlines())} 行")


# =========================================================================== #
# 阶段 B：真实模型
# =========================================================================== #
def new_session(agent: Agent, image: Path | None = None) -> Session:
    s = agent.store.create()
    if image:
        s.image_path = str(image)
        s.append_event("image_uploaded", {"path": str(image), "name": image.name})
    return s


def drive(agent: Agent, sess: Session, text: str) -> tuple[TurnResult, list[dict]]:
    events: list[dict] = []
    t0 = time.perf_counter()
    r = agent.run_turn(sess, text, on_event=events.append)
    print(f"\n  你 > {text}")
    for ev in events:
        if ev["type"] == "tool_call":
            print(f"     ↳ {ev['tool']}({json.dumps(ev['arguments'], ensure_ascii=False)[:90]})")
        elif ev["type"] == "tool_result":
            mk = "OK " if ev.get("ok") else "ERR"
            print(f"       [{mk}] {ev['tool']} {ev.get('elapsed_ms', 0):.0f}ms  {ev.get('summary', '')[:80]}")
    tag = "助手 >" if r.ok else f"助手 >[失败 {r.error_code}]"
    print(f"  {tag} {r.reply[:600]}")
    print(f"     （{r.rounds} 轮 · {len(r.tool_calls)} 次工具 · {r.elapsed_ms / 1000:.1f}s · "
          f"累计 {(time.perf_counter() - t0):.1f}s）")
    if not r.ok:
        print(f"     error: {r.error_message}")
    return r, events


def stage_b(agent: Agent, imgs: dict, ch: Checks, only: set[str]) -> None:
    print("\n" + "=" * 78)
    print(f"阶段 B —— 真实模型端到端（{'离线替身对照' if agent.is_offline else 'DeepSeek API'}）")
    print("=" * 78)
    known, unknown = imgs["known"], imgs["unknown"]
    created: list[tuple[str, int]] = []

    def want(sid: str) -> bool:
        return not only or sid in only

    # ---------------- S1 纯文本，不需要工具 ----------------
    if want("S1"):
        print("\n--- S1 纯文本问候（期望：不调工具，直接回答）")
        s = new_session(agent)
        r, _ = drive(agent, s, "你好，介绍一下你能做什么。")
        ch.expect(r.ok and len(r.reply) > 10, "S1 纯文本轮次正常返回", f"{len(r.reply)} 字")
        if r.tool_calls:
            ch.note("S1a 本轮调用了工具（问候语本可不必）", names_of(r).__repr__())
        else:
            ch.expect(True, "S1a 问候语未触发工具调用", "0 次")

    # ---------------- S2 视觉识别 ----------------
    if want("S2") and known:
        print("\n--- S2 上传图片后要求识别（期望：模型自己决定调用 classify_car）")
        s = new_session(agent, known)
        r, _ = drive(agent, s, "我上传了一张车的照片，这是什么车？")
        used = names_of(r)
        ch.expect(r.ok, "S2 识别轮次成功返回")
        ch.expect("classify_car" in used,
                  "S2a 视觉识别确实由模型通过工具调用发起（不是外部预算好塞进提示词）",
                  f"本轮工具：{used}")
        res = result_of(r, "classify_car")
        ch.expect(bool(res.get("car_name")), "S2b 工具返回了具体车型",
                  f"{res.get('car_name')} {float(res.get('confidence') or 0):.2%}")
        ch.expect(bool(s.last_prediction), "S2c 会话状态记录了识别结果（供后续轮次复用）",
                  json.dumps(s.last_prediction, ensure_ascii=False)[:100])

    # ---------------- S3 单轮 ≥3 工具串联 ----------------
    if want("S3") and known:
        print("\n--- S3 单轮内 3 个工具串联：识别 → 查档案 → 写档案（任务书硬性要求）")
        s = new_session(agent, known)
        r, _ = drive(agent, s, "识别一下这张照片里的车，查一下它的详细资料，然后加入我的看车档案。")
        used = names_of(r)
        ch.expect(r.ok, "S3 三工具链轮次成功返回")
        ch.expect(len(used) >= 3, "S3a 单轮内串联调用 ≥3 个工具", f"{len(used)} 次：{used}")
        ch.expect("classify_car" in used, "S3b 链路含视觉识别工具")
        ch.expect(any(n != "classify_car" for n in used),
                  "S3c 链路含视觉之外的业务工具",
                  "、".join(n for n in used if n != "classify_car"))
        ch.expect(all(c.ok for c in r.tool_calls), "S3d 链路上每个工具都执行成功",
                  "；".join(f"{c.name}={'OK' if c.ok else c.code}" for c in r.tool_calls))
        rec = result_of(r, "add_car_record")
        if rec.get("record_id"):
            created.append((s.session_id, int(rec["record_id"])))
            ch.expect(True, "S3e 确实写入了档案（返回记录号）", f"#{rec['record_id']} {rec.get('car_name')}")
        else:
            ch.expect("add_car_record" in used and s.record_count > 0,
                      "S3e 档案写入生效", f"本会话记录数 {s.record_count}")

    # ---------------- S4 同一会话多轮 ----------------
    if want("S4") and known:
        print("\n--- S4 同一 Session 多轮（第 2 轮不再上传图片，应沿用上一轮识别结果）")
        s = new_session(agent, known)
        r1, _ = drive(agent, s, "帮我看看这张图是什么车。")
        first_ok = r1.ok and "classify_car" in names_of(r1)
        ch.expect(first_ok, "S4a 第 1 轮完成识别")
        r2, _ = drive(agent, s, "它是什么品牌、什么车身类型？给我讲讲。")
        ch.expect(r2.ok, "S4b 第 2 轮在同一会话内正常返回")
        ch.expect("classify_car" not in names_of(r2),
                  "S4c 第 2 轮没有重新识别（说明复用了会话里的视觉结果）",
                  f"本轮工具：{names_of(r2) or '（无）'}")
        pred = (s.last_prediction or {}).get("car_name") or ""
        brand = pred.split("_")[0] if pred else ""
        model_part = pred.split("_")[-1] if pred else ""
        carried = any(t and t in r2.reply for t in (pred, brand, model_part))
        ch.expect(carried or not pred,
                  "S4d 第 2 轮回答延续了第 1 轮的车型（未重新识别也能答对）",
                  f"车型 {pred}；命中：{brand}；回复前 50 字：{r2.reply[:50]}")
        ch.expect(s.turns >= 2, "S4e 会话轮次计数递增", f"turns={s.turns}，历史消息 {len(s.messages)} 条")

        r3, _ = drive(agent, s, "把它加进我的档案，备注写「赛道日拍的」。")
        ch.expect(r3.ok, "S4f 第 3 轮继续同一会话正常返回")
        ch.expect("add_car_record" in names_of(r3),
                  "S4g 第 3 轮写入档案（仍然不用重新上传图片）", f"本轮工具：{names_of(r3)}")
        rec = result_of(r3, "add_car_record")
        if rec.get("record_id"):
            created.append((s.session_id, int(rec["record_id"])))
        note_ok = rec.get("note") == "赛道日拍的" or rec.get("duplicated")
        ch.expect(bool(rec.get("record_id")), "S4h 备注随记录一起写入",
                  f"#{rec.get('record_id')} note={rec.get('note')!r}"
                  + ("（已存在，转为更新备注）" if rec.get("duplicated") else ""))

    # ---------------- S5 拒识 unknown ----------------
    if want("S5") and unknown:
        print("\n--- S5 非支持车型（期望：如实告知未识别，不编造、不写档案）")
        s = new_session(agent, unknown)
        r, _ = drive(agent, s, "这张照片里是什么车？帮我记录一下。")
        used = names_of(r)
        ch.expect(r.ok, "S5 未知车型轮次正常返回（不崩溃）")
        ch.expect("classify_car" in used, "S5a 仍然走了视觉识别工具", f"{used}")
        res = result_of(r, "classify_car")
        ch.expect(res.get("is_unknown") is True, "S5b 模型判定为 unknown",
                  f"car_id={res.get('car_id')} conf={float(res.get('confidence') or 0):.2%}")
        ch.expect("add_car_record" not in used,
                  "S5c 拒识后没有继续写入档案（任务书 5.5 要求）", f"本轮工具：{used}")
        # 措辞检查必须宽松：真实模型每次说法都不一样，
        # 断言的目的是"有没有如实说明没认出来"，而不是"用没用某个固定词"。
        NEGATIVE_PHRASES = (
            "未识别", "没认出来", "认不出来", "认不出", "识别不出", "无法识别",
            "不属于", "不在", "不支持", "无法确定", "不确定", "没有把握", "拿不准",
            "unknown", "50 款", "50款",
        )
        hit = any(k in r.reply for k in NEGATIVE_PHRASES)
        ch.expect(hit, "S5d 回答如实说明未识别为本组支持车型", r.reply[:110].replace("\n", " "))

    # ---------------- S6 没有图片却要求识别 ----------------
    if want("S6"):
        print("\n--- S6 未上传图片却要求识别（异常处理）")
        s = new_session(agent)          # 注意：没有图片
        r, _ = drive(agent, s, "帮我识别一下我刚拍的车。")
        ch.expect(r.ok, "S6 无图片时轮次仍然正常收尾（智能体不崩）")
        mentioned = any(k in r.reply for k in ("上传", "图片", "照片", "没有"))
        ch.expect(mentioned, "S6a 提示用户先上传图片", r.reply[:110].replace("\n", " "))
        bad = [c for c in r.tool_calls if not c.ok]
        ch.expect(all(c.code in ("DENIED_BY_POLICY", "DEPENDENCY_MISSING") for c in bad),
                  "S6b 若调用了视觉工具，失败原因被如实回传（前置门禁 / 缺依赖）",
                  "；".join(f"{c.name}:{c.code}" for c in bad) or "（未调用工具）")

    # ---------------- S7 档案检索与统计 ----------------
    if want("S7"):
        print("\n--- S7 档案检索 / 统计（业务操作类工具）")
        s = new_session(agent)
        r, _ = drive(agent, s, "我的看车档案里现在都记了些什么？按品牌给我汇总一下。")
        used = names_of(r)
        ch.expect(r.ok, "S7 档案汇总轮次成功")
        ch.expect(any(n in used for n in ("my_garage_stats", "search_my_records")),
                  "S7a 调用了档案类工具", f"{used}")
        ch.expect(bool(r.reply.strip()), "S7b 基于真实档案数据给出汇总", f"{len(r.reply)} 字")

    # ---------------- S8 边界：范围外车型 ----------------
    if want("S8"):
        print("\n--- S8 询问支持范围之外的车型（期望：说明边界，不硬答）")
        s = new_session(agent)
        r, _ = drive(agent, s, "你能识别奔驰S级和宝马3系吗？")
        ch.expect(r.ok, "S8 范围外提问正常返回")
        ch.expect(bool(r.reply.strip()), "S8a 给出了明确答复", r.reply[:110].replace("\n", " "))

    # ---------------- S9 联网搜索：未开启 ----------------
    if want("S9"):
        print("\n--- S9 联网搜索未开启时提问（期望：引导用户去开启，而不是编造）")
        s = new_session(agent)
        r, _ = drive(agent, s, "日产轩逸现在落地价大概多少？")
        used = names_of(r)
        ch.expect(r.ok, "S9 轮次正常返回")
        ch.expect("web_search" not in used,
                  "S9a 未开启时模型看不到 web_search（本轮不该调它）", f"本轮工具：{used}")
        ch.expect(any(k in r.reply for k in ("联网", "开启", "按钮")),
                  "S9b 回答引导用户去开启联网搜索", r.reply[:110].replace("\n", " "))

    # ---------------- S10 联网搜索：已开启 ----------------
    if want("S10"):
        print("\n--- S10 联网搜索已开启（期望：模型真的调用 web_search 并交代来源）")
        s = new_session(agent)
        s.set_web_search(True)
        r, _ = drive(agent, s, "帮我上网查一下日产轩逸 2024 款的官方指导价，把来源告诉我。")
        used = names_of(r)
        ch.expect(r.ok, "S10 联网搜索轮次成功返回")
        ch.expect("web_search" in used, "S10a 模型确实调用了 web_search 工具", f"本轮工具：{used}")
        res = result_of(r, "web_search")
        ch.expect(bool(res.get("results")), "S10b 搜索返回了真实结果",
                  f'{res.get("provider")} · {res.get("count")} 条 · {res.get("elapsed_ms")} ms')
        ch.expect(res.get("untrusted") is True, "S10c 结果标注为不可信内容（防 prompt injection）")
        ch.expect(any(k in r.reply for k in (".com", ".cn", ".org", "http", "官网", "来源", "来自")),
                  "S10d 回答里交代了信息来源", r.reply[:130].replace("\n", " "))

    # ---------------- 收尾：清理 ----------------
    return created


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="识车档案 CarDex 端到端测试")
    ap.add_argument("--stage", default="A,B", help="要跑的阶段，如 A 或 A,B（默认两段都跑）")
    ap.add_argument("--only", default="", help="只跑指定场景，如 S3,S5")
    ap.add_argument("--offline", action="store_true", help="用离线替身模式跑阶段 B")
    ap.add_argument("--cleanup", action="store_true", help="结束时删除测试写入的档案记录")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    stages = {s.strip().upper() for s in args.stage.split(",") if s.strip()}
    only = {s.strip().upper() for s in args.only.split(",") if s.strip()}

    cfg = load_config(offline=True if args.offline else None)
    agent = Agent(cfg, store=SessionStore(cfg))
    imgs = pick_images()

    print("=" * 78)
    print("识车档案 CarDex —— 端到端测试")
    print("=" * 78)
    print(f"  运行模式  : {'离线替身（不代表模型效果）' if cfg.offline else '真实 DeepSeek API'}")
    print(f"  接口模型  : {cfg.model}")
    print(f"  API Key   : {cfg.masked_key()}")
    print(f"  视觉模型  : {[p.parent.name for p in cfg.checkpoint_paths() if p.exists()]}")
    print(f"  工具箱    : {len(agent.registry.names())} 个 —— {'、'.join(agent.registry.names())}")
    print(f"  支持车型图: {imgs['known']}")
    print(f"  范围外车型图: {imgs['unknown']}")

    ch = Checks()
    created: list[tuple[str, int]] = []
    t0 = time.perf_counter()
    if "A" in stages:
        stage_a(agent, imgs, ch)
    if "B" in stages:
        created = stage_b(agent, imgs, ch, only) or []

    if args.cleanup and created:
        print("\n--- 清理测试写入的档案记录")
        for sid, rid in created:
            s = agent.store.get(sid) or agent.store.create(sid)
            r = exec_tool(agent, s, "delete_car_record", {"record_id": rid}, call_id=f"c-del-{rid}")
            print(f"  {'OK ' if r.ok else 'ERR'} 删除 #{rid}")

    # ---- 汇总 ----
    elapsed = time.perf_counter() - t0
    if args.json:
        print(json.dumps({"pass": ch.passed, "fail": ch.failed,
                          "elapsed_s": round(elapsed, 1), "items": ch.items},
                         ensure_ascii=False, indent=2))
        return 1 if ch.failed else 0

    print("\n" + "=" * 78)
    print("测试结果")
    print("=" * 78)
    for i in ch.items:
        print(f"  {MARK[i['level']]} {i['name']}")
        if i["detail"]:
            print(f"          {i['detail']}")
    print("-" * 78)
    print(f"通过 {ch.passed} 项 · 失败 {ch.failed} 项 · 用时 {elapsed:.1f}s")
    if ch.failed:
        print("存在失败项，请检查上面的 [FAIL] 明细。")
    else:
        print("全部通过。")
    print("-" * 78)
    return 1 if ch.failed else 0


if __name__ == "__main__":
    sys.exit(main())
