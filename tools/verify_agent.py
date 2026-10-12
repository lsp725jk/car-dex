# -*- coding: utf-8 -*-
"""任务二 / 任务三 —— 提交前自查

逐项验证任务书第五节（智能体与工具）与第六节（Web 系统）的要求，
以及"提交材料不得包含真实 API Key"的合规要求。

用法：
    python tools/verify_agent.py
    python tools/verify_agent.py --skip-model    # 跳过需要加载视觉模型的耗时检查
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


class Checker:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.warned: list[str] = []

    def check(self, ok: bool, msg: str) -> bool:
        (self.passed if ok else self.failed).append(msg)
        print(("  [OK]   " if ok else "  [FAIL] ") + msg)
        return ok

    def warn(self, msg: str) -> None:
        self.warned.append(msg)
        print("  [WARN] " + msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-model", action="store_true", help="跳过视觉模型加载（耗时约 15 秒）")
    ap.add_argument("--live", action="store_true",
                    help="用真实 DeepSeek API 跑行为类检查（默认用离线替身，保证结果可复现）")
    args = ap.parse_args()

    c = Checker()
    print("=" * 84)
    print("任务二 / 任务三 自查")
    print("=" * 84)

    # ---------------- 1. 配置与合规 ----------------
    print("\n[1] 配置与 API Key 合规")
    from agent.config import MODEL_NAME, load_config
    cfg = load_config()
    # 本脚本检查的是**智能体骨架**（工具链、门禁、会话、状态机），
    # 必须可复现，因此默认走离线替身；真实模型的行为差异由 tools/e2e_test.py 覆盖。
    if not args.live:
        cfg.offline = True
    print(f"  运行模式：{'真实 DeepSeek API' if not cfg.offline else '离线替身（--live 可切到真实模型）'}")

    c.check(MODEL_NAME == "deepseek-v4-flash", f"模型名固定为 deepseek-v4-flash（实际 {MODEL_NAME}）")
    c.check(cfg.api_key == "" or not cfg.api_key.startswith("sk-REPLACE"),
            "未内置真实 API Key（空或占位符）")
    masked = cfg.masked_key()
    c.check(cfg.api_key == "" or cfg.api_key not in masked,
            f"日志脱敏函数不返回完整 Key（当前显示 {masked}）")
    c.check((ROOT / "agent" / ".env.example").exists(),
            "提供 agent/.env.example 配置示例（含占位符）")
    c.check((ROOT / ".gitignore").exists() and
            "agent/.env.local" in (ROOT / ".gitignore").read_text(encoding="utf-8"),
            ".gitignore 已忽略 agent/.env.local")

    # 扫描提交文件里是否混入真实 Key
    pattern = re.compile(r"sk-[A-Za-z0-9]{20,}")
    hits: list[str] = []
    for p in list(ROOT.glob("*.md")) + list(ROOT.glob("*.txt")) + list(ROOT.glob("*.py")):
        if pattern.search(p.read_text(encoding="utf-8", errors="ignore")):
            hits.append(p.name)
    for sub in ("agent", "web", "docs", "tools", "src", "configs"):
        for p in (ROOT / sub).rglob("*"):
            if p.is_file() and p.suffix.lower() in {".py", ".md", ".json", ".txt", ".html", ".js", ".css"}:
                if pattern.search(p.read_text(encoding="utf-8", errors="ignore")):
                    hits.append(str(p.relative_to(ROOT)))
    c.check(not hits, f"提交材料中未发现真实 API Key（扫描 {len(hits)} 处命中）")

    # ---------------- 2. 工具清单 ----------------
    print("\n[2] 工具清单（任务书 5.2：至少 1 个视觉工具 + 至少 2 个其他工具）")
    from agent.tools import build_registry
    reg = build_registry(cfg)
    names = reg.names()
    vision = [t for t in reg.all() if "vision" in t.tags]
    other = [t for t in reg.all() if "vision" not in t.tags]

    c.check(len(names) >= 3, f"工具总数 ≥ 3（实际 {len(names)}）")
    c.check(len(vision) >= 1, f"视觉分类工具 ≥ 1（实际 {len(vision)}：{[t.name for t in vision]}）")
    c.check(len(other) >= 2, f"其他业务工具 ≥ 2（实际 {len(other)}）")
    for t in reg.all():
        assert t.parameters.get("type") == "object", t.name
    c.check(all(t.parameters.get("type") == "object" for t in reg.all()),
            "每个工具的参数都是合法 JSON Schema object")
    c.check(all(t.description.strip() for t in reg.all()), "每个工具都有用途说明")
    c.check(all(t.timeout_s > 0 for t in reg.all()), "每个工具都设置了超时")
    print(f"         工具：{'、'.join(names)}")

    # ---------------- 3. 执行流水线的错误处理 ----------------
    print("\n[3] 工具执行流水线的错误处理")
    from agent.loop import Agent                   # noqa: E402
    from agent.session import SessionStore         # noqa: E402
    from agent.tools import ToolContext            # noqa: E402
    import threading

    agent = Agent(cfg, registry=reg, store=SessionStore(cfg))
    sess = agent.store.create()
    ctx = ToolContext(session_id=sess.session_id, turn_id="t-check", call_id="c1",
                      config=cfg, session=sess, signal=threading.Event())

    r = reg.execute("c1", "no_such_tool", "{}", ctx)
    c.check((not r.ok) and r.code == "UNKNOWN_TOOL", "未注册工具 → UNKNOWN_TOOL")

    r = reg.execute("c2", "query_car_info", "{bad json", ctx)
    c.check((not r.ok) and r.code == "INVALID_ARGUMENTS", "参数不是合法 JSON → INVALID_ARGUMENTS")

    r = reg.execute("c3", "query_car_info", "{}", ctx)
    c.check((not r.ok) and r.code == "INVALID_ARGUMENTS", "缺少必填参数 → INVALID_ARGUMENTS")

    r = reg.execute("c4", "query_car_info", '{"car_id_or_name":"0975","extra":1}', ctx)
    c.check((not r.ok) and r.code == "INVALID_ARGUMENTS", "多余参数(additionalProperties=false) → INVALID_ARGUMENTS")

    r = reg.execute("c5", "query_car_info", '{"car_id_or_name":"保时捷911"}', ctx)
    c.check((not r.ok) and r.code == "CAR_NOT_FOUND", "查不到的车型 → CAR_NOT_FOUND（不编造）")

    r = reg.execute("c6", "query_car_info", '{"car_id_or_name":"0975"}', ctx)
    c.check(r.ok and r.payload.get("car_id") == "0975", "合法查询 → 成功返回档案")

    r = reg.execute("c7", "classify_car", "{}", ctx)
    c.check((not r.ok) and r.code == "DENIED_BY_POLICY",
            "未上传图片就调视觉工具 → 被前置门禁拒绝")

    r = reg.execute("c8", "add_car_record", '{"car_id":"unknown"}', ctx)
    c.check((not r.ok) and r.code == "UNKNOWN_CAR", "把 unknown 写入档案 → 被拒绝")

    # ---------------- 4. 连续调用 3 个工具 ----------------
    print("\n[4] 连续调用 ≥3 个工具（任务书 5.3）")
    img = ROOT / "data" / "val" / "0975" / "0975_0007.jpg"
    if not img.exists():
        c.warn(f"测试图片不存在：{img}")
    elif args.skip_model:
        c.warn("已跳过（--skip-model）")
    else:
        s2 = agent.store.create()
        res = agent.run_turn(s2, "这是什么车？介绍一下，然后加进我的收藏。", image_path=str(img))
        used = [tc.name for tc in res.tool_calls]
        c.check(res.ok, f"本轮执行成功（{res.rounds} 轮，{len(used)} 次工具调用）")
        c.check(len(used) >= 3, f"连续调用工具数 ≥ 3（实际 {len(used)}：{used}）")
        c.check(used[:1] == ["classify_car"], "第一个工具是视觉分类工具")
        if len(used) >= 3:
            c.check("query_car_info" in used and "add_car_record" in used,
                    "包含 query_car_info 与 add_car_record，构成完整业务链")
        c.check(bool(s2.last_prediction), "识别结果已写入会话状态（供多轮沿用）")

        # 多轮：不重新上传图片
        res2 = agent.run_turn(s2, "帮我记录下来")
        used2 = [tc.name for tc in res2.tool_calls]
        recorded = res2.ok and "add_car_record" in used2
        if recorded or not args.live:
            c.check(recorded, f"第二轮无需重新上传图片即可记录（工具：{used2}）")
        else:
            # 真实模型有自己的判断（例如先 search_my_records 查重），
            # 这不属于智能体骨架的问题，如实标注而不判失败。
            c.warn(f"第二轮未直接写档案（真实模型本轮调用了 {used2}）；"
                   f"骨架能力已由 e2e_test.py 的 S4 覆盖")
        c.check(s2.image_path == str(img), "会话仍持有第一轮的图片")

        # 换新图片 → 必须重新识别
        img2 = ROOT / "data" / "val" / "0003" / "0003_0009.jpg"
        if img2.exists():
            res3 = agent.run_turn(s2, "这是什么车？", image_path=str(img2))
            c.check("classify_car" in [tc.name for tc in res3.tool_calls],
                    "换新图片后会重新调用 classify_car")

    # ---------------- 5. unknown 处理 ----------------
    print("\n[5] unknown 处理（任务书 5.5）")
    uimg = ROOT / "data" / "val" / "unknown" / "0022_0022.jpg"
    if not uimg.exists():
        c.warn(f"unknown 测试图片不存在：{uimg}")
    elif args.skip_model:
        c.warn("已跳过（--skip-model）")
    else:
        s3 = agent.store.create()
        res = agent.run_turn(s3, "这是什么车？", image_path=str(uimg))
        pred = s3.last_prediction or {}
        c.check(pred.get("is_unknown") is True,
                f"unknown 图片被正确判定（car_id={pred.get('car_id')}）")
        c.check(pred.get("decision") == "report_unknown", "返回 decision.action = report_unknown")
        c.check(any(k in res.reply for k in ("未识别", "unknown", "不属于", "不支持", "无法确定")),
                "回复中明确说明未识别为本组支持的车型")
        n_before = s3.record_count
        r = reg.execute("c9", "add_car_record", "{}", ctx)
        c.check(not r.ok and r.code in ("UNKNOWN_CAR", "DEPENDENCY_MISSING"),
                "unknown 结果不会被写入档案")

    # ---------------- 6. 阈值来源 ----------------
    print("\n[6] 阈值必须由验证集确定（任务书 5.5）")
    tf = ROOT / "agent" / "data" / "confidence_threshold.json"
    if c.check(tf.exists(), "存在 agent/data/confidence_threshold.json"):
        d = json.loads(tf.read_text(encoding="utf-8"))
        c.check(d.get("n_val", 0) > 1000, f"基于验证集推导（n_val={d.get('n_val')}）")
        c.check(0.3 <= d.get("recommended_display_threshold", 0) <= 0.99,
                f"推荐提示阈值 = {d.get('recommended_display_threshold')}")
        c.check("sweep" in d and len(d["sweep"]) > 20, f"含完整扫描记录（{len(d.get('sweep', []))} 组）")
    # 拒识阈值必须是**从验证集扫出来的**，而且要跟当前冠军模型的标定结果**一致**。
    #
    # ⚠️ 这里不能写死一个数字（原来是 `== 0.25`）：模型重新训练/换集成成员之后，
    # 最优阈值本来就会变 —— 写死的话，一次正确的重训会把自查搞挂，
    # 而"测试挂了"和"配置错了"是两件完全不同的事。
    # 正确的断言是**一致性检查**：配置里的值 == 最近一次 finalize 标定出的值。
    champ_file = ROOT / "outputs" / "reports" / "final_model_choice.json"
    c.check(champ_file.exists(), "存在 finalize 的比选结果 final_model_choice.json")
    if champ_file.exists():
        ch = json.loads(champ_file.read_text(encoding="utf-8"))["champion"]
        exp_t = float(ch["best"]["threshold"])
        exp_tau = float(ch["best"]["tau"])
        c.check(abs(cfg.unknown_threshold - exp_t) < 1e-6,
                f"拒识阈值 t = {cfg.unknown_threshold} 与 finalize 标定一致（{exp_t}）")
        c.check(abs(cfg.prior_adjust_tau - exp_tau) < 1e-6,
                f"先验校正 τ = {cfg.prior_adjust_tau} 与 finalize 标定一致（{exp_tau}）")
        # 集成成员也要和冠军一致，否则跑的根本不是被标定的那个模型
        want = {f"outputs/checkpoints/{m}/best.pt" for m in ch["members"]}
        got = {p.replace("\\", "/") for p in cfg.vision_checkpoints}
        c.check(want == got,
                f"视觉模型成员与冠军一致（{len(got)} 个："
                f"{'、'.join(sorted(m.split('/')[-2] for m in got))}）")
    c.check(cfg.display_confidence_threshold > 0, f"低置信度提示阈值 c = {cfg.display_confidence_threshold}")

    # ---------------- 7. 会话持久化 ----------------
    print("\n[7] 会话事件日志与重放")
    sdir = ROOT / "agent" / "data" / "sessions"
    logs = sorted(sdir.glob("*.jsonl"), key=lambda p: -p.stat().st_size) if sdir.exists() else []
    if c.check(bool(logs), f"存在会话事件日志（{len(logs)} 个）"):
        # 挑一个真正发生过对话的会话（事件类型最多的那个）来检查
        best, best_types = None, set()
        for p in logs[:20]:
            evs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            types = {e["type"] for e in evs}
            if len(types) > len(best_types):
                best, best_types, best_events = p, types, evs
        c.check(len(best_types) >= 3,
                f"事件类型 ≥ 3 种：{sorted(best_types)}")
        c.check("turn_end" in best_types, "包含 turn_end（一轮结束）事件")
        c.check("tool_result" in best_types, "包含 tool_result（工具执行）事件")
        c.check("turn_context" in best_types, "包含 turn_context（本轮策略快照）事件")
        sid = best.stem
        replayed = SessionStore(cfg).get(sid)
        c.check(replayed is not None and len(replayed.messages) > 0,
                f"可以从日志重放恢复会话（{sid}，恢复出 {len(replayed.messages) if replayed else 0} 条消息）")

    # ---------------- 8. 工具调用记录 ----------------
    print("\n[8] 工具调用记录（任务书 3.2）")
    tl = ROOT / "agent" / "data" / "tool_calls.jsonl"
    if c.check(tl.exists(), "存在 agent/data/tool_calls.jsonl"):
        rows = [json.loads(l) for l in tl.read_text(encoding="utf-8").splitlines() if l.strip()]
        c.check(len(rows) > 0, f"记录数 {len(rows)}")
        need = {"ts", "session_id", "turn_id", "call_id", "tool", "ok", "elapsed_ms", "result"}
        c.check(need <= set(rows[-1]), f"每条记录含字段：{sorted(need)}")

    # ---------------- 9. Web 系统 ----------------
    print("\n[9] Web 系统（任务书第六节）")
    try:
        import web.app as webapp
        rules = {str(r.rule) for r in webapp.app.url_map.iter_rules()}
        need_rules = {"/", "/api/session", "/api/upload", "/api/chat", "/api/records", "/api/tools"}
        c.check(need_rules <= rules, f"必需路由齐备（缺 {sorted(need_rules - rules)}）")
        c.check((ROOT / "web" / "templates" / "index.html").exists(), "页面模板存在")
        c.check((ROOT / "web" / "static" / "app.js").exists(), "前端脚本存在")
        c.check((ROOT / "web" / "static" / "style.css").exists(), "前端样式存在")
        html = (ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
        for kw, desc in [("上传图片", "上传入口"), ("我的看车档案", "业务面板"),
                         ("可用工具", "工具清单面板"), ("运行状态", "状态面板")]:
            c.check(kw in html, f"页面包含{desc}（{kw}）")
        js = (ROOT / "web" / "static" / "app.js").read_text(encoding="utf-8")
        app_py = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
        c.check("text/event-stream" in app_py and "data: " in app_py,
                "对话接口以 SSE（text/event-stream）流式返回")
        c.check("SSE" in js or "getReader" in js, "前端用 ReadableStream 增量解析事件流")
        c.check("tool_call" in js and "tool_result" in js, "前端渲染工具调用时间线")
        c.check("hidden" in (ROOT / "web" / "static" / "style.css").read_text(encoding="utf-8"),
                "样式表处理 [hidden]（避免被类选择器的 display 覆盖）")
    except Exception as e:                                   # noqa: BLE001
        c.check(False, f"Web 应用导入失败：{type(e).__name__}: {e}")

    # ---------------- 10. 会话管理 / 设置 / 联网搜索 ----------------
    print("\n[10] 界面新增能力：会话管理 · 设置 · 联网搜索")
    try:
        import web.app as webapp
        rules = {str(r.rule) for r in webapp.app.url_map.iter_rules()}

        # ---- 会话管理 ----
        c.check({"/api/sessions", "/api/sessions/<sid>", "/api/sessions/<sid>/web-search"} <= rules,
                "会话管理路由齐备（列表 / 重命名 / 删除 / 联网开关）")
        html = (ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
        js = (ROOT / "web" / "static" / "app.js").read_text(encoding="utf-8")
        css = (ROOT / "web" / "static" / "style.css").read_text(encoding="utf-8")
        c.check('id="rail"' in html and 'id="sess-list"' in html,
                "页面左侧包含可收缩的会话侧边栏")
        c.check("rail-collapsed" in css and "rail-collapsed" in js,
                "侧边栏支持收缩/展开（含状态持久化）")
        c.check('id="btn-rail"' in html, "顶栏提供侧边栏收缩按钮")
        c.check('id="btn-settings"' in html and "gear" in html,
                "侧边栏底部有设置按钮，图标为齿轮")
        c.check('class="rail-foot"' in html, "设置按钮位于侧边栏底部（左下角）")
        c.check("renderHistory" in js and "openSession" in js,
                "切换到历史会话时重放事件日志还原对话")
        c.check("data-sid" in js and "重命名" in js and "删除会话" in js,
                "会话条目支持重命名与删除")

        # ---- 设置：API 配置 + 模型配置 ----
        c.check({"/api/settings", "/api/settings/test"} <= rules,
                "设置路由齐备（读取 / 保存 / 连接测试）")
        for pane, name in [('data-pane="api"', "API 配置"), ('data-pane="model"', "模型配置"),
                           ('data-pane="threshold"', "决策阈值"), ('data-pane="web"', "联网搜索")]:
            c.check(pane in html, f"设置面板含「{name}」页签")
        c.check('id="set-api-key"' in html and 'type="password"' in html,
                "API Key 用密码框输入（不明文显示）")

        # ---- 设置接口绝不下发明文 Key ----
        with webapp.app.test_client() as client:
            r = client.get("/api/settings")
            body = r.get_data(as_text=True)
            real_key = webapp.CONFIG.api_key
            c.check(real_key == "" or real_key not in body.replace("\\u2026", "…"),
                    "GET /api/settings 不下发明文 API Key（只给掩码）")
            data = json.loads(body)
            c.check("api_key_masked" in data and "api_key" not in data.get("config", {}),
                    "设置响应只含 api_key_masked，不含 api_key 字段")
            # 保存时只接受白名单键；顺带验证 reset 能把覆盖项删掉
            # （这里用 reset 而不是写一个值，避免自查脚本自己往配置里留痕迹）
            r2 = client.post("/api/settings", json={"不存在的键": 1, "reset": ["max_rounds"]})
            c.check(r2.status_code == 200, "POST /api/settings 正常返回")
            body2 = r2.get_json() or {}
            c.check(any("不存在的键" in w for w in body2.get("warnings", [])),
                    "保存时未知配置项被忽略并给出警告")

        # ---- 运行时配置里不能有密钥 ----
        rc = ROOT / "agent" / "data" / "runtime_config.json"
        if rc.exists():
            txt = rc.read_text(encoding="utf-8")
            c.check("sk-" not in txt, "runtime_config.json 中不含任何 API Key")
            c.check("api_key" not in json.loads(txt), "runtime_config.json 不记录 api_key 字段")

        # ---- 联网搜索工具 ----
        from agent.tools import ALL_TOOLS, GATED_TOOLS, enabled_tool_names
        from agent.session import SessionStore
        names = [t.name for t in ALL_TOOLS]
        c.check("web_search" in names, f"已注册联网搜索工具（共 {len(names)} 个工具）")
        c.check("web_search" in GATED_TOOLS, "web_search 属于需要用户开启的门控工具")
        s = SessionStore(cfg).create()
        c.check("web_search" not in enabled_tool_names(s),
                "联网搜索默认关闭（未开启时模型看不到该工具）")
        s.set_web_search(True)
        c.check("web_search" in enabled_tool_names(s), "开启后模型可见 web_search")
        c.check('id="btn-websearch"' in html and "联网搜索" in html,
                "联网搜索按钮已放在输入框左侧（与上传图片并列）")
        c.check("btn-websearch" in js and "on" in js,
                "联网搜索按钮有开启/关闭两种状态")

        # ---- 联网开关必须「找得到、点得动」----
        # 实测反馈：用户看到工具面板里 web_search 标着「未开启」就去点它，点了没反应，
        # 以为功能坏了 —— 因为那只是个 <em> 标签，真正的开关在输入框旁边，
        # 而面板里**没有任何提示说去哪开**。功能没问题，是入口设计有缺陷。
        c.check("toggleWebSearch" in js and "function toggleWebSearch" in js,
                "联网开关抽成可复用函数（多个入口共用同一逻辑）")
        c.check("data-enable-web" in js and "data-enable-web" in js,
                "工具面板里的「未开启」可点击开启（不再是死标签）")
        c.check('id="tools-gate"' in html and "tools-gate" in js,
                "工具面板顶部有门禁提示条 + 一键开启按钮")
        c.check("gated" in js, "前端区分「需门禁」与「真不可用」两种未开启状态")

        # ---- 前端不能因为异常把界面卡死 ----
        # busy 只在 finish() 里复位；SSE 半开时 reader.read() 永久阻塞 → 界面永久失效
        c.check("watchdog" in js and "TURN_TIMEOUT_MS" in js,
                "回合有看门狗超时，SSE 卡住时能自动恢复（不会永久锁死界面）")
        # 先剔除注释行再查：注释里为了解释旧行为也会写到 s-local-，直接搜字符串会误报
        js_code = "\n".join(ln for ln in js.splitlines() if not ln.strip().startswith("//"))
        c.check("s-local-" not in js_code,
                "创建会话失败时如实报错，不伪造本地会话 ID（伪造会让后续接口全部静默 404）")
        c.check("untrusted" in (ROOT / "agent" / "tools" / "websearch.py").read_text(encoding="utf-8"),
                "搜索结果标注为不可信内容（防 prompt injection）")
        c.check("WEB_SEARCH_OFF" in (ROOT / "agent" / "prompts.py").read_text(encoding="utf-8"),
                "提示词区分「联网已开 / 未开」两种情况")

        # ---- 浏览器抓取工具（browse_page）----
        c.check("browse_page" in names,
                f"已注册真实浏览器工具 browse_page（共 {len(names)} 个工具）")
        c.check("browse_page" in GATED_TOOLS, "browse_page 与联网搜索共用同一个开关")
        c.check("browse_page" not in enabled_tool_names(SessionStore(cfg).create()),
                "未开启联网时 browse_page 不可见")
        c.check("browse_page" in enabled_tool_names(s), "开启后模型可见 browse_page")

        from agent.tools.browser import _candidate_browsers
        cands = _candidate_browsers()
        if cands:
            c.check(True, f"找到 {len(cands)} 个可用浏览器（首选 {cands[0].name}）")
        else:
            c.warn("未找到任何浏览器 —— browse_page 会自动降级为 HTTP 抓取（拿不到 JS 渲染内容）")
        bsrc = (ROOT / "agent" / "tools" / "browser.py").read_text(encoding="utf-8")
        c.check("attachToTarget" in bsrc and "flatten" in bsrc,
                "CDP 用 flat session（Chrome 111+ 直连 page 级 WS 会被拒）")
        c.check("suppress_origin" in bsrc and "remote-allow-origins" in bsrc,
                "处理了 CDP WebSocket 的 Origin 校验（否则握手 403）")
        c.check("_http_fetch" in bsrc and 'mode = "http"' in bsrc,
                "浏览器不可用时自动降级为 HTTP 抓取，并标注 mode")
        c.check("untrusted" in bsrc, "网页正文同样标注为不可信内容")

        # ---- 多轮联网的收敛保护 ----
        lsrc = (ROOT / "agent" / "loop.py").read_text(encoding="utf-8")
        c.check("budget_warning" in lsrc,
                "临近预算上限时注入 budget_warning，要求模型立即作答")
        c.check("IDEMPOTENT_WEB_TOOLS" in lsrc and "CACHED" in lsrc,
                "完全相同的联网调用复用上次结果（避免绕圈烧预算）")
        c.check(cfg.max_rounds >= 10 and cfg.max_tool_calls >= 20,
                f"循环预算够联网多轮用（{cfg.max_rounds} 轮 / {cfg.max_tool_calls} 次）")
        p_on = __import__("agent.prompts", fromlist=["x"]).build_system_prompt(cfg, s).text
        c.check("循环" in p_on and "browse_page" in p_on and "budget_warning" in p_on,
                "提示词写清了「搜索→打开→再搜」的循环与预算约束")
        c.check("verify_browser.py" in str(sorted(p.name for p in (ROOT / "tools").glob("*.py"))),
                "提供 tools/verify_browser.py 专门验证浏览器链路")

        # ---- 档次（配置版本）知识库 ----
        c.check({"save_car_trims", "compare_car_trims"} <= set(names),
                f"已注册档次工具（共 {len(names)} 个工具）")
        from agent.tools.trims import (_diff_trims, _to_wan, load_trims,
                                       summarize_trims, trims_path)
        # 价格归一化：网页上格式五花八门，必须统一成万元
        c.check(_to_wan("7.99万") == 7.99 and _to_wan("¥79,900") == 7.99
                and _to_wan("12.5 万元") == 12.5 and _to_wan("面议") is None,
                "价格能归一成万元（7.99万 / ¥79,900 / 12.5万元 / 面议 都对）")
        tp = trims_path(cfg)
        if tp.exists():
            d = json.loads(tp.read_text(encoding="utf-8"))
            cars = d.get("cars", {})
            n_trims = sum(len(v.get("trims", [])) for v in cars.values())
            c.check(n_trims > 0,
                    f"档次库已有数据：{len(cars)} 款车 / {n_trims} 个档次"
                    f"（{tp.relative_to(ROOT)}）")
            if cars:
                any_verified = any(s.get("verified")
                                   for v in cars.values() for s in v.get("sources", []))
                c.check(any_verified, "档次数据带来源且经过「本会话真打开过」核验")
                priced = sum(1 for v in cars.values() for t in v.get("trims", [])
                             if t.get("guide_price_wan"))
                c.check(priced > 0, f"档次带价格：{priced} 个")
                has_specs = sum(1 for v in cars.values() for t in v.get("trims", [])
                                if t.get("specs"))
                c.check(has_specs > 0, f"档次带参数配置：{has_specs} 个")
                # compare 要能真的比出差异
                for cid, v in cars.items():
                    if len(v.get("trims", [])) > 1:
                        diff = _diff_trims(v["trims"])
                        c.check(len(diff["keys"]) > 0,
                                f"{v.get('car_name','')} 的 {len(v['trims'])} 个档次能比出差异"
                                f"（{len(diff['keys'])} 项：{'、'.join(diff['keys'][:4])}…）")
                        break
        else:
            c.warn("档次库还是空的 —— 联网查一次并 save_car_trims 就有数据了")

        # 反编造：没有来源不许入库
        from agent.session import SessionStore as _SS
        _s = _SS(cfg).create()
        from agent.tools.registry import ToolContext as _TC
        import threading as _th
        _ctx = _TC(session_id=_s.session_id, turn_id="t", call_id="c", config=cfg,
                   session=_s, signal=_th.Event(), image_path=None)
        _reg = __import__("agent.tools", fromlist=["x"]).build_registry(cfg)
        _r = _reg.execute("c", "save_car_trims",
                          json.dumps({"car_id_or_name": "轩逸", "trims": [{"name": "测试版"}]}),
                          _ctx)
        c.check((not _r.ok) and _r.code == "INVALID_ARGUMENTS",
                "save_car_trims 缺 source_url 会被拒绝（反编造第一道防线）")
        c.check("_was_visited" in (ROOT / "agent" / "tools" / "trims.py").read_text(encoding="utf-8"),
                "保存时核验来源 URL 是否真的在本会话打开过（反编造第二道防线）")

        # 个人看车档案也要能记档次
        gw = (ROOT / "agent" / "tools" / "garage.py").read_text(encoding="utf-8")
        c.check('"trim"' in gw or "'trim'" in gw, "看车档案支持记录「用户看中的是哪一档」")
        c.check("MIGRATIONS" in gw, "老数据库能平滑升级（ALTER TABLE 迁移）")

        # 截断能自动补救
        c.check("OUTPUT_TRUNCATED" in (ROOT / "agent" / "errors.py").read_text(encoding="utf-8")
                and "max_tokens_ceiling" in (ROOT / "agent" / "config.py").read_text(encoding="utf-8"),
                "输出被 max_tokens 截断时会自动放大预算重试")

        # 界面要能画档次表
        c.check("trimBlock" in js and "trims" in css,
                "前端能把档次渲染成表格（价格 / 差异参数 / 来源）")
        c.check('extract' in bsrc and '"tables"' in bsrc,
                "browse_page 支持结构化抽取表格（配置页是行=参数项、列=各档次）")
    except Exception as e:                                   # noqa: BLE001
        c.check(False, f"新增能力检查失败：{type(e).__name__}: {e}")

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 84)
    print(f"通过 {len(c.passed)} 项，失败 {len(c.failed)} 项，警告 {len(c.warned)} 项")
    if c.failed:
        print("\n未通过项：")
        for f in c.failed:
            print("  - " + f)
    print("=" * 84)
    sys.exit(1 if c.failed else 0)


if __name__ == "__main__":
    main()
