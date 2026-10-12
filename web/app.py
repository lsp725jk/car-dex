# -*- coding: utf-8 -*-
"""任务三：Web 系统。

一个 Flask 应用，把「视觉模型 + 智能体 + 工具 + 看车档案」串成可交互的页面。

启动：
    python web/app.py                 # 默认 http://127.0.0.1:5000
    python web/app.py --port 8000

配置：
    API Key 通过环境变量 DEEPSEEK_API_KEY 或 agent/.env.local 提供；
    未配置时自动进入**离线替身模式**（页面顶部会显示醒目提示）。

接口一览：
    GET    /                      页面
    POST   /api/session           新建会话
    GET    /api/session/<sid>     读取会话状态
    GET    /api/sessions          会话列表（左侧边栏）
    PATCH  /api/sessions/<sid>    重命名会话
    DELETE /api/sessions/<sid>    删除会话
    POST   /api/sessions/<sid>/web-search   开/关联网搜索（逐会话）
    POST   /api/upload            上传图片
    POST   /api/chat              对话（SSE 流式返回事件）
    GET    /api/tools             工具清单
    GET    /api/settings          读取运行配置（**Key 只返回掩码**）
    POST   /api/settings          保存运行配置
    POST   /api/settings/test     用给定配置试连一次 API
    GET    /api/records           看车档案
    DELETE /api/records/<id>      删除档案记录
    GET    /api/events/<sid>      事件时间线（审计/调试）
    GET    /uploads/<name>        访问已上传图片

安全约定：
    · API Key **只**写进 `agent/.env.local`（已 gitignore）或环境变量，
      绝不写进 `runtime_config.json`、日志、页面或任何返回体；
    · `GET /api/settings` 只返回掩码（形如 `sk-f…fbe5（长度 35）`）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agent.config import (RUNTIME_CONFIG_FILE, RUNTIME_EDITABLE,  # noqa: E402
                          load_config, load_runtime_config, save_api_key, save_runtime_config)
from agent.loop import Agent                  # noqa: E402
from agent.session import SessionStore        # noqa: E402
from agent.tools import GATED_TOOLS, enabled_tool_names  # noqa: E402
from agent.tools.garage import db             # noqa: E402

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_MB = 12

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"),
            static_folder=str(Path(__file__).parent / "static"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

CONFIG = load_config()
STORE = SessionStore(CONFIG)
AGENT = Agent(CONFIG, store=STORE)


def rebuild_agent() -> None:
    """配置变更后重建 Agent（新模型 / 新 Key / 新阈值立刻生效），但保留已加载的会话。"""
    global CONFIG, AGENT
    CONFIG = load_config()
    STORE.rebind(CONFIG)
    AGENT = Agent(CONFIG, store=STORE)


# --------------------------------------------------------------------------- #
def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


def runtime_info() -> dict:
    return {
        "model": CONFIG.model,
        "offline": bool(CONFIG.offline),
        "api_key": CONFIG.masked_key(),
        "vision_models": [p.parent.name for p in CONFIG.checkpoint_paths() if p.exists()],
        "unknown_threshold": CONFIG.unknown_threshold,
        "display_confidence_threshold": CONFIG.display_confidence_threshold,
        "prior_adjust_tau": CONFIG.prior_adjust_tau,
        "max_rounds": CONFIG.max_rounds,
        "tool_count": len(AGENT.registry.names()),
        "web_search_provider": CONFIG.web_search_provider,
    }


def settings_payload() -> dict:
    """给「设置」面板的完整运行配置。

    ⚠️ **绝不返回明文 Key**：只回 `api_key_masked` 和 `api_key_configured`。
    前端提交 Key 时是"只写不读"——留空表示保持原值不变。
    """
    current = {k: getattr(CONFIG, k) for k in RUNTIME_EDITABLE}
    defaults = {k: getattr(type(CONFIG)(), k) for k in RUNTIME_EDITABLE}
    saved = load_runtime_config()
    return {
        "ok": True,
        "config": current,
        "defaults": defaults,
        "overridden": sorted(saved.keys()),
        "api_key_configured": bool(CONFIG.api_key),
        "api_key_masked": CONFIG.masked_key(),
        "api_key_source": ("环境变量 DEEPSEEK_API_KEY"
                           if (os.getenv("DEEPSEEK_API_KEY") or "").strip()
                           else ("agent/.env.local" if CONFIG.api_key else "未配置")),
        "offline": bool(CONFIG.offline),
        "runtime_config_file": str(RUNTIME_CONFIG_FILE.relative_to(ROOT)),
        "env_file": "agent/.env.local",
        "provider_options": ["bing", "tavily", "duckduckgo"],
        "provider_help": {
            "bing": "默认通道，无需 Key（实测本机唯一稳定可用的）",
            "tavily": "需要环境变量 TAVILY_API_KEY，结果质量最好",
            "duckduckgo": "需要能访问境外网络的代理环境",
        },
    }


# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html", info=runtime_info())


@app.route("/api/runtime")
def api_runtime():
    return jsonify(runtime_info())


@app.route("/api/tools")
def api_tools():
    """工具清单。

    带 `session_id` 时会额外标出每个工具**当前是否对该会话可用**
    （目前只有联网搜索会被开关挡住），前端据此把它显示成"未开启"。
    """
    sid = request.args.get("session_id", "").strip()
    session = STORE.get(sid) if sid else None
    allowed = enabled_tool_names(session) if session is not None else set(AGENT.registry.names())
    tools = []
    for t in AGENT.describe_tools():
        t = dict(t)
        t["enabled"] = t["name"] in allowed
        t["gated"] = t["name"] in GATED_TOOLS
        tools.append(t)
    return jsonify({"ok": True, "tools": tools,
                    "enabled_count": sum(1 for t in tools if t["enabled"]),
                    "total_count": len(tools)})


@app.route("/api/session", methods=["POST"])
def api_new_session():
    s = AGENT.store.create()
    return jsonify({"ok": True, "session": s.to_dict(with_messages=True)})


@app.route("/api/session/<sid>")
def api_get_session(sid: str):
    s = AGENT.store.get(sid)
    if s is None:
        # 事件日志不存在 → 视为新会话，用同一个 id 建一个（便于前端刷新后继续）
        s = AGENT.store.create(sid)
        return jsonify({"ok": True, "session": s.to_dict(with_messages=True), "created": True})
    return jsonify({"ok": True, "session": s.to_dict(with_messages=True)})


@app.route("/api/events/<sid>")
def api_events(sid: str):
    s = AGENT.store.get(sid)
    if s is None:
        return jsonify({"ok": False, "error": "会话不存在"}), 404
    return jsonify({"ok": True, "events": s.events(400)})


# --------------------------------------------------------------------------- #
# 会话管理（左侧边栏）
# --------------------------------------------------------------------------- #
@app.route("/api/sessions")
def api_sessions():
    rows = STORE.list_summaries(limit=120)
    rows.sort(key=lambda r: r.get("mtime") or 0, reverse=True)
    for r in rows:
        r.pop("mtime", None)
    return jsonify({"ok": True, "sessions": rows, "count": len(rows)})


@app.route("/api/sessions/<sid>", methods=["PATCH"])
def api_rename_session(sid: str):
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"ok": False, "error": {
            "code": "EMPTY_TITLE", "message": "标题不能为空"}}), 400
    if len(title) > 60:
        return jsonify({"ok": False, "error": {
            "code": "TOO_LONG", "message": "标题最长 60 字"}}), 400
    s = STORE.get(sid)
    if s is None:
        return jsonify({"ok": False, "error": {
            "code": "SESSION_NOT_FOUND", "message": f"会话 {sid} 不存在"}}), 404
    s.set_title(title)
    return jsonify({"ok": True, "session": s.to_dict()})


@app.route("/api/sessions/<sid>", methods=["DELETE"])
def api_delete_session(sid: str):
    existed = STORE.delete(sid)
    if not existed:
        return jsonify({"ok": False, "error": {
            "code": "SESSION_NOT_FOUND", "message": f"会话 {sid} 不存在"}}), 404
    return jsonify({"ok": True, "deleted": sid})


@app.route("/api/sessions/<sid>/web-search", methods=["POST"])
def api_toggle_web_search(sid: str):
    """开/关某个会话的联网搜索。

    联网搜索是"会往外发请求、会引入公网不可信内容"的能力，
    所以做成**逐会话、由用户显式开关**，而不是让模型自己决定何时联网。
    """
    data = request.get_json(silent=True) or {}
    s = STORE.get(sid)
    if s is None:
        return jsonify({"ok": False, "error": {
            "code": "SESSION_NOT_FOUND", "message": f"会话 {sid} 不存在"}}), 404
    enabled = bool(data.get("enabled"))
    s.set_web_search(enabled)
    tools = sorted(AGENT.registry.names()) if enabled else \
        sorted(set(AGENT.registry.names()) - {"web_search"})
    return jsonify({"ok": True, "web_search_enabled": enabled,
                    "provider": CONFIG.web_search_provider, "tools": tools})


# --------------------------------------------------------------------------- #
# 设置（API 配置 + 模型配置）
# --------------------------------------------------------------------------- #
@app.route("/api/settings")
def api_get_settings():
    return jsonify(settings_payload())


@app.route("/api/settings", methods=["POST"])
def api_save_settings():
    """保存设置。

    请求体可以是任意子集：
        { "api_key": "sk-…",        // 可选；留空 = 不改。只写进 agent/.env.local
          "model": "...", "temperature": 0.2, ... }
    """
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": {
            "code": "INVALID_ARGUMENTS", "message": "请求体必须是 JSON 对象"}}), 400

    changed: list[str] = []
    warnings: list[str] = []

    # ---- API Key：只写不读 ----
    raw_key = (data.pop("api_key", "") or "").strip()
    if raw_key:
        if not raw_key.startswith("sk-") or len(raw_key) < 20:
            return jsonify({"ok": False, "error": {
                "code": "INVALID_API_KEY",
                "message": "Key 格式不对（应以 sk- 开头且长度足够）",
                "hint": "去 platform.deepseek.com 复制完整的 Key"}}), 400
        save_api_key(raw_key)
        changed.append("api_key")

    # ---- 被改回默认值的项：从配置文件里删掉，而不是把默认值再写一遍 ----
    reset = data.pop("reset", []) or []
    if not isinstance(reset, list):
        reset = []
    reset = [k for k in reset if k in RUNTIME_EDITABLE]
    for k in reset:
        changed.append(k)

    # ---- 其余可调参数 ----
    for key, value in list(data.items()):
        if key not in RUNTIME_EDITABLE:
            warnings.append(f"忽略未知配置项 {key}")
            continue
        # 空字符串 = 不改（前端把未填的输入框也提交上来了）
        if isinstance(value, str) and not value.strip():
            continue
        changed.append(key)

    saved = save_runtime_config(data, reset=reset)

    # 合法性提醒（不阻断保存，只是提示）
    if saved.get("unknown_threshold") is not None and not 0 <= saved["unknown_threshold"] <= 1:
        warnings.append("拒识阈值 t 应在 0~1 之间")
    if saved.get("max_rounds") and saved["max_rounds"] < 1:
        warnings.append("最大轮数至少为 1")
    if saved.get("temperature") is not None and not 0 <= saved["temperature"] <= 2:
        warnings.append("temperature 建议在 0~2 之间")

    rebuild_agent()

    return jsonify({"ok": True, "changed": sorted(set(changed)), "warnings": warnings,
                    "settings": settings_payload()})


@app.route("/api/settings/test", methods=["POST"])
def api_test_settings():
    """用"当前已保存配置 + 请求体里的临时覆盖"试连一次接口。

    这样用户可以在**保存之前**就确认 Key / 模型名对不对，
    而不是保存完发现不能用再改回去。
    ⚠️ 只回错误码与一句话，**不回显请求体**，避免 Key 出现在响应里。
    """
    from dataclasses import replace

    data = request.get_json(silent=True) or {}
    cand = (data.get("api_key") or "").strip()
    overrides = {k: v for k, v in data.items() if k in RUNTIME_EDITABLE}

    cfg = replace(CONFIG)                      # 浅拷贝，不动全局
    for k, v in overrides.items():
        try:
            setattr(cfg, k, type(getattr(cfg, k))(v))
        except (TypeError, ValueError):
            pass
    if cand:
        cfg.api_key = cand
    cfg.offline = False

    if not cfg.api_key:
        return jsonify({"ok": False, "error": {
            "code": "API_KEY_MISSING", "message": "还没有配置 API Key",
            "hint": "填入 Key 后点「测试连接」"}}), 400

    from agent.llm import DeepSeekClient
    t0 = time.perf_counter()
    try:
        resp = DeepSeekClient(cfg).complete(
            [{"role": "user", "content": "只回复两个字：在线"}])
    except Exception as e:                          # noqa: BLE001
        code = getattr(e, "code", type(e).__name__)
        msg = getattr(e, "message", str(e))
        info = getattr(e, "info", {}) or {}
        return jsonify({"ok": False, "error": {
            "code": code, "message": msg, "hint": info.get("hint", ""),
            "retryable": info.get("retryable", False)}}), 200

    return jsonify({
        "ok": True,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        "model_reported": resp.model,
        "reply": (resp.content or "").strip()[:40],
        "retries": getattr(resp, "retries", 0),
        "tokens": (resp.usage or {}).get("total_tokens"),
    })


@app.route("/api/settings/test-search", methods=["POST"])
def api_test_search():
    """用当前配置的通道跑一次真实联网搜索，让用户知道搜索到底通不通。"""
    from agent.tools.registry import ToolContext
    from agent.tools.websearch import get_provider

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "日产轩逸 官方指导价").strip()[:80]
    try:
        provider = get_provider(CONFIG)
    except Exception as e:                          # noqa: BLE001
        return jsonify({"ok": False, "error": {
            "code": getattr(e, "code", "PROVIDER_ERROR"),
            "message": getattr(e, "message", str(e)),
            "hint": (getattr(e, "hint", "") or "")}}), 200

    t0 = time.perf_counter()
    try:
        results = provider.search(query, 3)
    except Exception as e:                          # noqa: BLE001
        return jsonify({"ok": False, "error": {
            "code": getattr(e, "code", "TOOL_FAILED"),
            "message": getattr(e, "message", str(e)),
            "hint": (getattr(e, "hint", "") or "")}}), 200

    return jsonify({
        "ok": True,
        "provider": provider.name,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        "count": len(results),
        "first": (results[0]["title"][:40] if results else ""),
        "results": results,
    })


# --------------------------------------------------------------------------- #
@app.route("/api/upload", methods=["POST"])
def api_upload():
    sid = request.form.get("session_id", "").strip()
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"ok": False, "error": {"code": "NO_FILE", "message": "没有收到文件"}}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"ok": False, "error": {
            "code": "BAD_FORMAT",
            "message": f"不支持的图片格式 {ext or '(无扩展名)'}",
            "hint": f"请上传 {'/'.join(sorted(ALLOWED_EXT))} 格式的图片",
        }}), 400

    s = AGENT.store.get_or_create(sid or None)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}{ext}"
    dst = CONFIG.upload_dir / name
    f.save(dst)

    # 校验能真正打开（防止改扩展名的假图片）
    try:
        from PIL import Image
        with Image.open(dst) as im:
            w, h = im.size
            im.verify()
    except Exception as e:                              # noqa: BLE001
        dst.unlink(missing_ok=True)
        return jsonify({"ok": False, "error": {
            "code": "IMAGE_UNREADABLE",
            "message": f"图片无法解析：{type(e).__name__}",
            "hint": "请确认这是有效的图片文件",
        }}), 400

    s.image_path = str(dst)
    s.append_event("image_uploaded", {"path": str(dst), "name": name, "size": [w, h]})
    return jsonify({"ok": True, "session_id": s.session_id,
                    "image": {"name": name, "url": f"/uploads/{name}", "size": [w, h]}})


@app.route("/uploads/<path:name>")
def uploaded_file(name: str):
    return send_from_directory(str(CONFIG.upload_dir), name)


# --------------------------------------------------------------------------- #
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    sid = (data.get("session_id") or "").strip()
    text = (data.get("message") or "").strip()

    if not text:
        return jsonify({"ok": False, "error": {
            "code": "EMPTY_INPUT", "message": "请输入内容后再发送"}}), 400
    if len(text) > 2000:
        return jsonify({"ok": False, "error": {
            "code": "TOO_LONG", "message": "单条消息过长（上限 2000 字）"}}), 400

    session = AGENT.store.get_or_create(sid or None)

    def generate():
        yield sse({"type": "session", "session_id": session.session_id})
        queue: list[dict] = []

        def on_event(ev: dict) -> None:
            queue.append(ev)

        # 把 Agent 的同步事件转成生成器输出：
        # 用一个后台线程跑 run_turn，主线程轮询队列并 yield。
        import threading

        result_box: dict = {}

        def worker() -> None:
            try:
                result_box["r"] = AGENT.run_turn(session, text, on_event=on_event)
            except Exception as e:                       # noqa: BLE001
                result_box["exc"] = f"{type(e).__name__}: {e}"
            finally:
                result_box["done"] = True

        th = threading.Thread(target=worker, daemon=True)
        th.start()

        sent = 0
        while not result_box.get("done") or sent < len(queue):
            while sent < len(queue):
                yield sse(queue[sent])
                sent += 1
            if not result_box.get("done"):
                time.sleep(0.05)
        while sent < len(queue):
            yield sse(queue[sent])
            sent += 1

        if "exc" in result_box:
            yield sse({"type": "error", "code": "INTERNAL_ERROR", "message": result_box["exc"]})
        r = result_box.get("r")
        if r is not None:
            payload = r.to_dict()
            payload["session"] = session.to_dict()
            yield sse({"type": "result", **payload})

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------------- #
@app.route("/api/records")
def api_records():
    sid = request.args.get("session_id", "").strip()
    scope = request.args.get("scope", "session")
    limit = min(int(request.args.get("limit", 50)), 200)

    where, params = [], []
    if scope == "session" and sid:
        where.append("session_id=?")
        params.append(sid)
    for key in ("car_id", "brand", "body_group"):
        v = request.args.get(key)
        if v:
            where.append(f"{key}=?")
            params.append(v)
    kw = request.args.get("keyword")
    if kw:
        where.append("(car_name LIKE ? OR brand LIKE ? OR IFNULL(note,'') LIKE ?)")
        params += [f"%{kw}%"] * 3

    sql = "SELECT * FROM car_records"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY datetime(created_at) DESC, id DESC LIMIT ?"
    params.append(limit)

    with db(CONFIG) as conn:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        total = conn.execute("SELECT COUNT(*) n FROM car_records").fetchone()["n"]
        by_group = [dict(r) for r in conn.execute(
            "SELECT body_group, COUNT(*) n FROM car_records GROUP BY body_group").fetchall()]

    for r in rows:
        r["image"] = f"/uploads/{Path(r['image_path']).name}" if r.get("image_path") else None
    return jsonify({"ok": True, "records": rows, "total": total, "by_group": by_group})


@app.route("/api/records/<int:rid>", methods=["DELETE"])
def api_delete_record(rid: int):
    sid = request.args.get("session_id", "").strip()
    with db(CONFIG) as conn:
        row = conn.execute("SELECT * FROM car_records WHERE id=?", (rid,)).fetchone()
        if row is None:
            return jsonify({"ok": False, "error": {"code": "RECORD_NOT_FOUND",
                                                   "message": f"记录 #{rid} 不存在"}}), 404
        if sid and row["session_id"] != sid:
            return jsonify({"ok": False, "error": {"code": "DENIED_BY_POLICY",
                                                   "message": "不能删除其他会话的记录"}}), 403
        conn.execute("DELETE FROM car_records WHERE id=?", (rid,))
    return jsonify({"ok": True, "deleted": rid})


# --------------------------------------------------------------------------- #
@app.errorhandler(413)
def too_large(_e):
    return jsonify({"ok": False, "error": {
        "code": "FILE_TOO_LARGE",
        "message": f"图片超过 {MAX_UPLOAD_MB} MB 限制",
        "hint": "请压缩后再上传",
    }}), 413


@app.errorhandler(500)
def server_error(e):                                     # pragma: no cover
    return jsonify({"ok": False, "error": {
        "code": "SERVER_ERROR", "message": f"服务器内部错误：{e}"}}), 500


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--offline", action="store_true",
                    help="强制使用离线替身（即使已配置 API Key）")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.offline:
        # 模块级 CONFIG 已按「有 Key 就用真模型」装载，这里按需重建
        global CONFIG, AGENT
        CONFIG = load_config(offline=True)
        AGENT = Agent(CONFIG, store=STORE)

    # 启动时清掉上次崩溃 / 强杀留下的孤儿浏览器进程（browse_page 用的）。
    # 靠 --user-data-dir 里的 cardex-browser- 前缀识别，不会碰用户自己开的 Chrome/Edge。
    try:
        from agent.tools.browser import cleanup_orphans
        n_orphans = cleanup_orphans()
    except Exception:                                # noqa: BLE001
        n_orphans = 0

    info = runtime_info()
    print("=" * 78)
    print("识车档案 CarDex — Web 系统")
    print(f"  地址      : http://{args.host}:{args.port}")
    print(f"  模型      : {info['model']}")
    print(f"  运行模式  : {'离线替身（未配置 DEEPSEEK_API_KEY）' if info['offline'] else '真实 DeepSeek API'}")
    print(f"  API Key   : {info['api_key']}")
    print(f"  视觉模型  : {info['vision_models'] or '未找到（识别功能会报错）'}")
    print(f"  工具数    : {info['tool_count']}（联网工具需在页面上开启）")
    print(f"  拒识阈值  : {info['unknown_threshold']}    低置信度提示阈值: {info['display_confidence_threshold']}")
    if n_orphans:
        print(f"  已清理    : {n_orphans} 个上次残留的浏览器进程")
    print("=" * 78)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
