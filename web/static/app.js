/* 识车档案 CarDex — 前端逻辑 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const messages = $("messages");
  const input = $("input");
  const fileInput = $("file");
  const btnSend = $("btn-send");
  const preview = $("preview");
  const previewImg = $("preview-img");
  const previewName = $("preview-name");
  const previewSize = $("preview-size");
  const composerHint = $("composer-hint");
  const btnWebSearch = $("btn-websearch");

  let sessionId = null;
  let uploading = false;
  let busy = false;
  let pendingImage = null;
  let webSearchOn = false;      // 逐会话的联网搜索开关
  let activeReader = null;      // 当前 SSE 流的 reader（超时中断时用它 cancel）

  /* ---------------- 主题 ---------------- */
  const THEME_KEY = "cardex-web-theme";
  const applyTheme = (t) => {
    document.documentElement.setAttribute("data-theme", t);
    $("btn-theme").textContent = t === "dark" ? "☀ 亮色" : "☾ 暗色";
  };
  let savedTheme = null;
  try { savedTheme = localStorage.getItem(THEME_KEY); } catch (e) {}
  applyTheme(savedTheme || (window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  $("btn-theme").onclick = () => {
    const t = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(t);
    try { localStorage.setItem(THEME_KEY, t); } catch (e) {}
  };

  /* ---------------- 工具函数 ---------------- */
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // 极简 Markdown：**粗体**、`代码`、换行
  const md = (s) => esc(s)
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br>");

  const scrollDown = () => { messages.scrollTop = messages.scrollHeight; };

  // 重试期间会往这里写「正在自动重试…」，属于临时提示；本轮结束后必须清掉，
  // 否则会一直挂在输入框下面，看起来像界面卡死了。
  let transientHint = false;
  const setHint = (t, transient) => {
    composerHint.textContent = t || "";
    transientHint = !!transient;
  };

  /* ---------------- 会话 ---------------- */
  async function ensureSession() {
    if (sessionId) return sessionId;
    try {
      const r = await fetch("/api/session", { method: "POST" });
      if (!r.ok) throw new Error(`服务端返回 ${r.status}`);
      const d = await r.json();
      const s = d.session || {};
      if (!s.session_id) throw new Error("响应里没有 session_id");
      sessionId = s.session_id;
      webSearchOn = !!s.web_search_enabled;
    } catch (e) {
      // ⚠️ 以前失败时会**伪造**一个 `s-local-<时间戳>` 的会话 ID 兜底。
      // 那是个大坑：之后所有 `/api/sessions/<假ID>/...` 都会 404，
      // 而且用户**看不到任何错误** —— 表现就是"点了没反应"。
      // 正确做法是如实抛错，让调用方把失败显示出来。
      setHint("创建会话失败：" + e.message + "（请确认后端服务在运行）");
      throw e;
    }
    setChipSession();
    paintWebSearch();
    loadRecords();
    loadSessions();
    return sessionId;
  }

  function setChipSession() {
    const chip = $("chip-session");
    chip.textContent = "会话 " + (sessionId || "—");
    chip.title = sessionId || "";
  }

  /* ==================== 左侧：会话侧边栏 ==================== */
  const RAIL_KEY = "cardex-rail-collapsed";
  let allSessions = [];

  function applyRail(collapsed) {
    document.body.classList.toggle("rail-collapsed", collapsed);
    $("btn-rail").textContent = collapsed ? "☰" : "⟨";
    try { localStorage.setItem(RAIL_KEY, collapsed ? "1" : "0"); } catch (e) {}
  }
  $("btn-rail").onclick = () =>
    applyRail(!document.body.classList.contains("rail-collapsed"));
  let railSaved = null;
  try { railSaved = localStorage.getItem(RAIL_KEY); } catch (e) {}
  applyRail(railSaved === "1");

  const relTime = (iso) => {
    if (!iso) return "";
    const t = new Date(String(iso).replace(" ", "T"));
    if (isNaN(t)) return String(iso).slice(5, 16);
    const s = (Date.now() - t.getTime()) / 1000;
    if (s < 60) return "刚刚";
    if (s < 3600) return Math.floor(s / 60) + " 分钟前";
    if (s < 86400) return Math.floor(s / 3600) + " 小时前";
    if (s < 86400 * 7) return Math.floor(s / 86400) + " 天前";
    return `${t.getMonth() + 1}/${t.getDate()}`;
  };

  async function loadSessions() {
    try {
      const d = await (await fetch("/api/sessions")).json();
      allSessions = d.sessions || [];
    } catch (e) {
      allSessions = [];
    }
    renderSessions();
  }

  function renderSessions() {
    const kw = $("sess-filter").value.trim().toLowerCase();
    // 一张嘴都没张过的空会话（0 轮 / 0 条消息）不进列表，否则侧边栏会被
    // "新会话"刷屏；但当前正在用的那个始终显示，用户才知道自己在哪。
    const rows = allSessions
      .filter((s) => (s.turns || 0) > 0 || (s.message_count || 0) > 0
        || s.session_id === sessionId)
      .filter((s) => !kw
        || (s.title || "").toLowerCase().includes(kw)
        || (s.session_id || "").includes(kw));
    const list = $("sess-list");
    if (!rows.length) {
      list.innerHTML = `<div class="rail-empty">${
        kw ? "没有匹配的会话" : "还没有会话。<br>发一条消息就会自动创建。"}</div>`;
      return;
    }
    list.innerHTML = rows.map((s) => `
      <div class="sess ${s.session_id === sessionId ? "active" : ""}" data-sid="${esc(s.session_id)}">
        <div class="t" title="${esc(s.title || "")}">${esc(s.title || "新会话")}</div>
        <div class="s">
          <span>${relTime(s.updated_at)}</span>
          <span>${s.turns || 0} 轮</span>
          ${s.record_count ? `<span>${s.record_count} 条记录</span>` : ""}
          ${s.web_search_enabled ? `<span class="badge-web">🌐 联网</span>` : ""}
        </div>
        <div class="acts">
          <button class="ren" data-sid="${esc(s.session_id)}" title="重命名">✎</button>
          <button class="del" data-sid="${esc(s.session_id)}" title="删除会话">×</button>
        </div>
      </div>`).join("");

    list.querySelectorAll(".sess").forEach((el) => {
      el.onclick = (e) => {
        if (e.target.closest(".acts")) return;
        openSession(el.dataset.sid);
      };
    });
    list.querySelectorAll(".ren").forEach((b) => {
      b.onclick = async (e) => {
        e.stopPropagation();
        const sid = b.dataset.sid;
        const cur = allSessions.find((x) => x.session_id === sid);
        const name = prompt("会话标题：", cur ? cur.title : "");
        if (name === null) return;
        if (!name.trim()) { setHint("标题不能为空"); return; }
        const r = await fetch(`/api/sessions/${encodeURIComponent(sid)}`,
          { method: "PATCH", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: name.trim() }) });
        const d = await r.json();
        if (!d.ok) { setHint("重命名失败：" + (d.error?.message || "")); return; }
        if (sid === sessionId) setHint("已重命名");
        loadSessions();
      };
    });
    list.querySelectorAll(".del").forEach((b) => {
      b.onclick = async (e) => {
        e.stopPropagation();
        const sid = b.dataset.sid;
        if (!confirm("删除这个会话？对话记录会一并删除，看车档案不受影响。")) return;
        const r = await fetch(`/api/sessions/${encodeURIComponent(sid)}`, { method: "DELETE" });
        const d = await r.json();
        if (!d.ok) { setHint("删除失败：" + (d.error?.message || "")); return; }
        if (sid === sessionId) {
          sessionId = null;
          messages.innerHTML = "";
          addAssistant("会话已删除。发一条消息即可开始新的对话。");
          loadRecords();
        }
        setHint("会话已删除");
        loadSessions();
      };
    });
  }

  // 切到某个历史会话：重放它的事件日志，把对话与工具卡片重新画出来
  async function openSession(sid) {
    if (busy) { setHint("正在生成回复，稍后再切换会话"); return; }
    sessionId = sid;
    setChipSession();
    pendingImage = null;
    preview.hidden = true;
    messages.innerHTML = `<div class="rail-empty" style="padding:24px">正在加载会话…</div>`;
    let evs = [];
    try {
      const d = await (await fetch(`/api/events/${encodeURIComponent(sid)}`)).json();
      evs = d.events || [];
    } catch (e) { /* 忽略 */ }
    messages.innerHTML = "";
    if (!evs.length) {
      addAssistant("这个会话还没有内容。");
    } else {
      renderHistory(evs);
    }
    webSearchOn = evs.some((e) => e.type === "web_search_toggled"
      && (e.payload || {}).enabled === true);
    // 以最后一次开关事件为准
    for (const e of evs) {
      if (e.type === "web_search_toggled") webSearchOn = !!(e.payload || {}).enabled;
    }
    paintWebSearch();
    renderSessions();
    loadRecords();
    setHint(`已切换到会话 ${sid}`);
  }

  function renderHistory(evs) {
    const turns = {};                       // turn_id -> { el, strip, content }
    const ensureTurn = (tid) => {
      if (turns[tid]) return turns[tid];
      const el = startAssistant();
      el.content.innerHTML = "";
      turns[tid] = el;
      return el;
    };
    for (const ev of evs) {
      const p = ev.payload || {};
      const tid = p.turn_id || ev.turn_id || "";
      if (ev.type === "user_message") {
        const img = p.image
          ? "/uploads/" + encodeURIComponent(String(p.image).split(/[\\/]/).pop())
          : null;
        addUser(p.text || "", img);
      } else if (ev.type === "tool_result") {
        const t = ensureTurn(tid);
        t.strip.insertAdjacentHTML("beforeend", historyToolCard(p));
      } else if (ev.type === "assistant_message" && p.final) {
        const t = ensureTurn(tid);
        t.content.innerHTML = md(p.message?.content || p.content || "");
      } else if (ev.type === "error") {
        const t = ensureTurn(tid);
        t.content.insertAdjacentHTML("beforeend",
          `<div class="err"><b>${esc(p.code || "ERROR")}</b>：${esc(p.message || "")}</div>`);
      }
    }
    // 有工具体但没留下正文的轮次，补一句占位，避免出现空气泡
    Object.values(turns).forEach((t) => {
      if (!t.content.innerHTML.trim()) t.content.innerHTML = md("（该轮没有留下回复）");
    });
    scrollDown();
  }

  function historyToolCard(p) {
    const ok = !!p.ok;
    const cls = p.tool === "web_search" ? " search-tool" : "";
    const res = p.result;
    const vision = ok && p.tool === "classify_car" && res && typeof res === "object"
      ? visionBlock(res) : "";
    return `<div class="tool-card ${ok ? "ok" : "err"}${cls}">
        <div class="tool-head">
          <span class="badge ${ok ? "ok" : "err"}">${ok ? "成功" : "失败"}</span>
          <span class="tool-name">${esc(p.tool || "")}</span>
          <span class="tool-sum">${esc(p.code && !ok ? p.code : "")}</span>
          ${p.elapsed_ms != null ? `<span class="tool-ms">${Number(p.elapsed_ms).toFixed(0)} ms</span>` : ""}
        </div>
        <div class="tool-body">${vision}
          <h5>结果</h5><pre>${esc(JSON.stringify(res ?? {}, null, 2))}</pre>
        </div>
      </div>`;
  }

  $("sess-filter").oninput = renderSessions;

  /* ==================== 联网搜索开关 ==================== */
  function paintWebSearch() {
    btnWebSearch.classList.toggle("on", webSearchOn);
    btnWebSearch.title = webSearchOn
      ? "联网搜索已开启：智能体可以调用 web_search 查询公网信息。点击关闭。"
      : "联网搜索已关闭：点击开启后，智能体才能查询公网信息";
    const kv = $("kv-web");
    if (kv) kv.textContent = (webSearchOn ? "开启" : "关闭");
  }

  // 切换联网搜索。抽成独立函数，因为有两个入口：
  // 输入框旁边的 🌐 按钮，**和工具面板里的「未开启 · 点击开启」**
  // （原来只有前者，用户在工具面板看到「未开启」点了没反应，以为坏了）
  async function toggleWebSearch(force) {
    if (busy) { setHint("正在生成回复，稍后再切换"); return; }
    const next = (typeof force === "boolean") ? force : !webSearchOn;
    if (next === webSearchOn) return;             // 已经是目标状态，不用发请求
    try {
      await ensureSession();
      const r = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/web-search`,
        { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: next }) });
      const d = await r.json();
      if (!d.ok) { setHint("切换失败：" + (d.error?.message || "")); return; }
      webSearchOn = !!d.web_search_enabled;
      paintWebSearch();
      loadTools();          // 工具清单里 web_search 的"未开启"标记要跟着变
      setHint(webSearchOn
        ? `联网搜索已开启（通道 ${d.provider}）。现在可以问「轩逸现在多少钱」这类问题。`
        : "联网搜索已关闭。");
      loadSessions();
    } catch (e) {
      setHint("切换失败：" + e.message);
    }
  }
  btnWebSearch.onclick = () => toggleWebSearch();
  paintWebSearch();

  $("btn-new").onclick = async () => {
    sessionId = null;
    pendingImage = null;
    preview.hidden = true;
    messages.innerHTML = "";
    addAssistant("已开启新会话。上传一张车照，或者直接问我问题。");
    await ensureSession();
    renderSessions();
    setHint("新会话已创建");
  };

  /* ---------------- 消息渲染 ---------------- */
  function addUser(text, imageUrl) {
    const el = document.createElement("div");
    el.className = "msg user";
    el.innerHTML = `<div class="avatar">我</div><div class="bubble">
        ${imageUrl ? `<img src="${imageUrl}" alt="已上传图片">` : ""}
        ${text ? `<p>${md(text)}</p>` : ""}
      </div>`;
    messages.appendChild(el);
    scrollDown();
    return el;
  }

  function addAssistant(text) {
    const el = document.createElement("div");
    el.className = "msg assistant";
    el.innerHTML = `<div class="avatar">识</div><div class="bubble">${md(text)}</div>`;
    messages.appendChild(el);
    scrollDown();
    return el;
  }

  function startAssistant() {
    const el = document.createElement("div");
    el.className = "msg assistant";
    el.innerHTML = `<div class="avatar">识</div><div class="bubble">
        <div class="tools-strip"></div>
        <div class="content"><span class="typing"><i></i><i></i><i></i> 思考中…</span></div>
      </div>`;
    messages.appendChild(el);
    scrollDown();
    return {
      el,
      strip: el.querySelector(".tools-strip"),
      content: el.querySelector(".content"),
    };
  }

  function addToolCard(strip, ev) {
    const card = document.createElement("div");
    // 联网搜索单独一种配色，一眼能看出"这一轮真的联网了"
    card.className = "tool-card pending" + (ev.tool === "web_search" ? " search-tool" : "");
    card.dataset.callId = ev.call_id || "";
    card.innerHTML = `
      <div class="tool-head">
        <span class="badge">调用中</span>
        <span class="tool-name">${esc(ev.tool)}</span>
        <span class="tool-sum">${esc(JSON.stringify(ev.arguments || {}))}</span>
      </div>
      <div class="tool-body">
        <h5>参数</h5>
        <pre>${esc(JSON.stringify(ev.arguments || {}, null, 2))}</pre>
      </div>`;
    card.querySelector(".tool-head").onclick = () => card.classList.toggle("open");
    strip.appendChild(card);
    scrollDown();
    return card;
  }

  /* 视觉识别结果的「人话版」摘要：
     任务书 5.5 要求低置信度 / unknown 时有明确反馈，光靠模型的行文不够直观，
     这里把 classify_car 的结构化结论（车型、置信度、Top-K 候选、分支建议）直接画出来。 */
  function visionBlock(res) {
    if (!res || !res.ok) return "";
    const conf = Number(res.confidence || 0);
    const pct = (conf * 100).toFixed(1);
    const unk = !!res.is_unknown;
    const act = (res.decision || {}).action || (unk ? "report_unknown" : "accept");
    const label = {
      accept: "置信度充足，可作为识别结论继续后续操作",
      confirm_with_user: "置信度偏低，建议向用户确认是哪一款，再写入档案",
      report_unknown: "判定不属于本组支持的 50 款车型，不会据此编造具体车型",
    }[act] || act;

    const cands = (res.top_candidates || []);
    const list = cands.map((c) => `
      <li>
        <span class="vc-name">${esc(c.car_name)}</span>
        <span class="vc-bar"><i style="width:${Math.max(1.5, Number(c.confidence) * 100).toFixed(1)}%"></i></span>
        <span class="vc-pct">${(Number(c.confidence) * 100).toFixed(1)}%</span>
      </li>`).join("");

    return `
      <div class="vision ${act}">
        <div class="vision-head">
          <span class="vision-car">${esc(unk ? "未识别（unknown）" : (res.car_name || "—"))}</span>
          <span class="vision-pct">${pct}%</span>
        </div>
        <div class="vision-note">${esc(label)}</div>
        ${list ? `<div class="vision-cands"><b>Top-${cands.length} 候选车型${unk ? "（仅供参考，非结论）" : ""}</b>
          <ul>${list}</ul></div>` : ""}
      </div>`;
  }

  /* 档次（配置版本）表格：价格 + 差异参数 + 来源。
     query_car_info / save_car_trims / compare_car_trims 三个工具共用。 */
  function trimBlock(tool, res) {
    if (!res || !res.ok) return "";
    const trims = res.trims || [];
    const rows = res.table || [];
    const isCompare = tool === "compare_car_trims";
    const isSave = tool === "save_car_trims";

    if (!trims.length && !rows.length) {
      if (tool === "query_car_info" && res.trim_count === 0) {
        return `<div class="trims empty">
            <div class="trims-head">档次 / 价格数据：暂无</div>
            <div class="trims-note">教师数据集不含价格与配置，这部分需要联网查。
              开启「🌐 联网搜索」后我可以去查，并把查到的存进档案库。</div>
          </div>`;
      }
      return "";
    }

    const list = isCompare ? rows : trims;
    const diffKeys = res.differing_specs || [];
    const head = isCompare
      ? `档次对比（${res.compared}/${res.total_trims} 个）`
      : (isSave ? `已存入档案库：${res.saved} 个档次（共 ${res.total_trims} 个）`
                : `档次 / 价格（${res.trim_count} 个）`);

    const body = list.map((t) => {
      const price = t.guide_price_wan != null ? `指导价 ${t.guide_price_wan} 万` : "";
      const dealer = t.dealer_price_wan != null ? ` · 经销商 ${t.dealer_price_wan} 万` : "";
      const same = t.price_text ? ` · ${esc(t.price_text)}` : "";
      // 对比模式下只显示"有差别"的那几个参数，一眼看出贵在哪
      const specs = isCompare
        ? Object.entries(t.differing || {})
        : Object.entries(t.specs || {}).slice(0, 6);
      const specHtml = specs.length
        ? `<div class="trim-specs">${specs.map(([k, v]) =>
            `<span><b>${esc(k)}</b>${esc(v)}</span>`).join("")}</div>` : "";
      const hl = (t.highlights || []).length
        ? `<div class="trim-hl">＋ ${(t.highlights || []).map(esc).join("、")}</div>` : "";
      return `<div class="trim-row">
          <div class="trim-name">${esc(t.trim || t.name || "")}</div>
          <div class="trim-price">${esc(price)}${esc(dealer)}${same}</div>
          ${specHtml}${hl}
        </div>`;
    }).join("");

    const srcs = res.sources || res.trim_sources || [];
    const src = srcs.length
      ? `<div class="trims-src">来源：${srcs.map((s) =>
          `<a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.site || s.url)}</a>${
            s.verified === false ? '<span class="unverified">未核实</span>' : ""}`
        ).join("、")} · 抓取于 ${esc(String(srcs[srcs.length - 1].fetched_at || "").slice(0, 16))}</div>`
      : "";
    const diffHint = isCompare && diffKeys.length
      ? `<div class="trims-note">有差别的参数：${diffKeys.map(esc).join("、")}
         （各档次取值一致的参数已省略）</div>` : "";

    return `<div class="trims ${isCompare ? "compare" : ""}${isSave ? " saved" : ""}">
        <div class="trims-head">${esc(head)}${res.trim_summary ? " · " + esc(res.trim_summary) : ""}</div>
        ${diffHint}
        <div class="trims-body">${body}</div>
        ${src}
      </div>`;
  }

  function completeToolCard(strip, ev) {
    const cards = strip.querySelectorAll(".tool-card");
    let card = null;
    for (const c of cards) {
      if (c.dataset.callId && c.dataset.callId === ev.call_id) { card = c; break; }
    }
    if (!card) card = cards[cards.length - 1];
    if (!card) return;
    const ok = !!ev.ok;
    card.classList.remove("pending");
    card.classList.add(ok ? "ok" : "err");
    card.dataset.callId = "";
    card.querySelector(".badge").className = "badge " + (ok ? "ok" : "err");
    card.querySelector(".badge").textContent = ok ? "成功" : "失败";
    card.querySelector(".tool-sum").textContent = ev.summary || "";
    const head = card.querySelector(".tool-head");
    if (ev.elapsed_ms != null) {
      const ms = document.createElement("span");
      ms.className = "tool-ms";
      ms.textContent = ev.elapsed_ms.toFixed(0) + " ms";
      head.appendChild(ms);
    }
    const body = card.querySelector(".tool-body");
    if (ok && ev.tool === "classify_car") {
      body.insertAdjacentHTML("beforeend", visionBlock(ev.result));
    }
    // 档次/价格类结果画成表格，比让用户看 JSON 直观得多
    if (ok && (ev.tool === "query_car_info" || ev.tool === "save_car_trims"
               || ev.tool === "compare_car_trims")) {
      body.insertAdjacentHTML("beforeend", trimBlock(ev.tool, ev.result));
    }
    body.insertAdjacentHTML("beforeend",
      `<h5>结果</h5><pre>${esc(JSON.stringify(ev.result ?? {}, null, 2))}</pre>`);
    scrollDown();
  }

  /* ---------------- 上传 ---------------- */
  fileInput.onchange = async () => {
    const f = fileInput.files?.[0];
    if (!f) return;
    if (!f.type.startsWith("image/")) { setHint("请选择图片文件"); fileInput.value = ""; return; }
    if (f.size > 12 * 1024 * 1024) { setHint("图片超过 12 MB，请压缩后再上传"); fileInput.value = ""; return; }

    uploading = true;
    setHint("正在上传…");
    const fd = new FormData();
    fd.append("file", f);
    fd.append("session_id", await ensureSession());
    try {
      const r = await fetch("/api/upload", { method: "POST", body: fd });
      const d = await r.json();
      if (!d.ok) {
        setHint("上传失败：" + (d.error?.message || "未知错误"));
        addAssistant("⚠️ 图片上传失败：" + (d.error?.message || "") +
          (d.error?.hint ? "（" + d.error.hint + "）" : ""));
        return;
      }
      pendingImage = d.image;
      previewImg.src = d.image.url;
      previewName.textContent = d.image.name;
      previewSize.textContent = d.image.size[0] + "×" + d.image.size[1] + " · " +
        (f.size / 1024).toFixed(0) + " KB";
      preview.hidden = false;
      setHint("图片已就绪，发送消息时我会调用 classify_car 识别它");
    } catch (e) {
      setHint("上传失败：" + e.message);
    } finally {
      uploading = false;
      fileInput.value = "";
    }
  };

  $("btn-clear-img").onclick = () => {
    pendingImage = null;
    preview.hidden = true;
    setHint("已移除待上传图片");
  };

  /* ---------------- 发送 ---------------- */
  // 这几类错误属于「环境/网络」问题，不是用户输入有毛病：
  // 失败时把他刚才那句话放回输入框，改都不用改，直接再点一次发送。
  const NETWORK_CODES = new Set([
    "NETWORK_DNS_FAILED", "NETWORK_UNREACHABLE", "NETWORK_TLS_FAILED",
    "API_TIMEOUT", "API_RATE_LIMITED", "API_SERVER_ERROR", "API_REQUEST_FAILED",
  ]);

  async function send() {
    if (busy) return;
    const text = input.value.trim();
    if (!text && !pendingImage) { setHint("请输入内容，或先上传一张图片"); return; }
    if (uploading) { setHint("图片还在上传中，稍等"); return; }

    await ensureSession();
    busy = true;
    btnSend.disabled = true;

    // ⚠️ 看门狗：`await reader.read()` 在 SSE 连接**半开**时会**永久阻塞** ——
    // 服务端线程死了但 socket 没关，浏览器就一直等，`busy` 永远是 true，
    // 结果是「发送」和「联网开关」全部点了没反应，而且刷新前无法恢复。
    // 给一个硬上限，到点强制收尾并复位。
    const TURN_TIMEOUT_MS = 240000;      // 4 分钟（联网多轮实测最长 ~60s，留足余量）
    let watchdog = setTimeout(() => {
      if (busy) {
        try { activeReader?.cancel(); } catch (e) { /* 忽略 */ }
        finish(`<div class="err">本轮超过 ${TURN_TIMEOUT_MS / 1000} 秒仍未结束，已强制中断。
          后端可能仍在执行 —— 结果会写进会话日志，刷新页面即可看到。</div>`);
      }
    }, TURN_TIMEOUT_MS);

    input.value = "";
    const imgUrl = pendingImage?.url || null;
    addUser(text || "（上传了一张车辆照片）", imgUrl);
    const imgForTurn = pendingImage;
    pendingImage = null;
    preview.hidden = true;
    setHint("");

    const ui = startAssistant();

    let done = false;
    const finish = (html) => {
      if (done) return;
      done = true;
      clearTimeout(watchdog);              // 正常收尾就撤掉看门狗
      activeReader = null;
      ui.content.innerHTML = html;
      if (transientHint) setHint("");     // 收尾时清掉「正在重试…」这类临时提示
      busy = false;
      btnSend.disabled = false;
      scrollDown();
      loadRecords();
      loadSessions();                     // 会话标题/轮次会变，左侧边栏跟着刷新
      loadTools();
    };

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text || "这是什么车？" }),
      });
      if (!resp.ok || !resp.body) {
        const d = await resp.json().catch(() => ({}));
        finish(`<div class="err">请求失败：${esc(d.error?.message || resp.status)}</div>`);
        return;
      }

      const reader = resp.body.getReader();
      activeReader = reader;                 // 看门狗中断时要用它 cancel
      const decoder = new TextDecoder("utf-8");
      let buf = "";

      while (true) {
        const { value, done: streamDone } = await reader.read();
        if (streamDone) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop();
        for (const part of parts) {
          const line = part.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;
          let ev;
          try { ev = JSON.parse(line.slice(6)); } catch (e) { continue; }

          switch (ev.type) {
            case "session":
              sessionId = ev.session_id;
              $("chip-session").textContent = "会话 " + sessionId;
              break;
            case "turn_start":
              if (ev.turn && ev.turn.image) {
                setHint("本轮已绑定图片：" + ev.turn.image);
              }
              break;
            case "tool_call":
              addToolCard(ui.strip, ev);
              break;
            case "tool_result":
              completeToolCard(ui.strip, ev);
              break;
            case "assistant_message":
              // 中间轮次模型可能先说话，先展示出来
              if (ev.content) {
                ui.content.innerHTML = md(ev.content) +
                  `<div class="typing"><i></i><i></i><i></i></div>`;
              }
              break;
            case "api_retry":
              // 客户端在自动重试（DNS 抖动 / 限流 / 服务端 5xx），让用户知道系统在自救
              setHint(`网络波动（${ev.code}），正在自动重试 ${ev.attempt + 1}/${ev.attempts}，`
                + `${ev.wait_s}s 后重发…`, true);
              break;
            case "api_recovered":
              setHint(`已自动恢复（第 ${ev.attempts} 次尝试成功）`, true);
              break;
            case "error":
              ui.content.insertAdjacentHTML("beforeend",
                `<div class="err"><b>${esc(ev.code || "ERROR")}</b>：${esc(ev.message || "")}${
                  ev.info?.hint ? `<br>建议：${esc(ev.info.hint)}` : ""}</div>`);
              // 网络类错误不是用户输入的问题：把刚才那句话放回输入框，一键重发即可
              if (NETWORK_CODES.has(ev.code) && !input.value) input.value = text;
              break;
            case "result": {
              let html = md(ev.reply || "");
              if (!ev.ok && ev.error) {
                // ev.error 里带着 hint / status / retryable 等补充信息（来自 TurnResult.error_info），
                // 必须把「建议」也渲染出来，否则用户只知道失败了、不知道该改什么。
                html += `<div class="err"><b>${esc(ev.error.code)}</b>：${esc(ev.error.message)}`;
                if (ev.error.hint) html += `<br>建议：${esc(ev.error.hint)}`;
                if (ev.error.attempts > 1) {
                  html += `<br>已自动重试 ${ev.error.attempts - 1} 次仍未成功。`;
                }
                html += `</div>`;
                if (NETWORK_CODES.has(ev.error.code) && !input.value) input.value = text;
              }
              const meta = [];
              meta.push(`${ev.rounds} 轮`);
              meta.push(`${(ev.tool_calls || []).length} 次工具调用`);
              meta.push(`${((ev.elapsed_ms || 0) / 1000).toFixed(1)}s`);
              if (ev.simulated) meta.push("离线替身");
              html += `<p class="hint">${meta.join(" · ")}</p>`;
              finish(html);
              break;
            }
          }
        }
      }
      finish(ui.content.innerHTML || md("（没有收到回复）"));
    } catch (e) {
      finish(`<div class="err">连接中断：${esc(e.message)}</div>`);
    }
  }

  btnSend.onclick = send;
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });

  /* ---------------- 侧栏：工具清单 ---------------- */
  async function loadTools() {
    try {
      const qs = sessionId ? "?session_id=" + encodeURIComponent(sessionId) : "";
      const d = await (await fetch("/api/tools" + qs)).json();
      const list = d.tools || [];
      const on = list.filter((t) => t.enabled).length;
      const allOn = d.total_count === on;
      $("tool-count").textContent = allOn ? on : `${on}/${d.total_count}`;
      $("kv-tools").textContent = allOn ? on : `${on}/${d.total_count}`;

      // 被门禁挡住的工具（联网类）单独提示 —— 光标记「未开启」用户不知道去哪开
      const gatedOff = list.filter((t) => !t.enabled && t.gated);
      const bar = $("tools-gate");
      if (bar) {
        if (gatedOff.length) {
          bar.hidden = false;
          bar.innerHTML =
            `<span>${gatedOff.map((t) => esc(t.name)).join("、")} 需要开启联网后才能用</span>` +
            `<button type="button" class="mini" id="btn-enable-web">开启联网</button>`;
          const b = $("btn-enable-web");
          if (b) b.onclick = () => toggleWebSearch(true);
        } else {
          bar.hidden = true;
          bar.innerHTML = "";
        }
      }

      $("tools").innerHTML = list.map((t) => {
        const props = Object.keys(t.parameters?.properties || {});
        const req = new Set(t.parameters?.required || []);
        const tags = props.slice(0, 6).map((k) =>
          `<span>${esc(k)}${req.has(k) ? "" : "?"}</span>`).join("");
        // 「未开启」做成可点的按钮：用户第一反应就是点它，不该点了没反应
        const offTag = t.enabled
          ? ""
          : (t.gated
              ? ` <button type="button" class="off-tag clickable" data-enable-web="1"
                   title="点击开启联网搜索">未开启 · 点击开启</button>`
              : ` <em class="off-tag">未开启</em>`);
        return `<div class="tool-item ${t.enabled ? "" : "off"}">
            <b>${esc(t.name)}${offTag}</b>
            <p>${esc(t.description)}</p>
            <div class="tags">${tags}${t.requires_image ? "<span>需要图片</span>" : ""}</div>
          </div>`;
      }).join("");

      document.querySelectorAll("#tools [data-enable-web]").forEach((el) => {
        el.onclick = () => toggleWebSearch(true);
      });
    } catch (e) { /* 忽略 */ }
  }

  /* ---------------- 侧栏：看车档案 ---------------- */
  let recTimer = null;
  async function loadRecords() {
    try {
      const kw = $("rec-kw").value.trim();
      const scope = $("rec-scope").value;
      const qs = new URLSearchParams({ session_id: sessionId || "", scope });
      if (kw) qs.set("keyword", kw);
      const d = await (await fetch("/api/records?" + qs)).json();
      $("rec-count").textContent = d.records.length;
      if (!d.records.length) {
        $("records").innerHTML = `<div class="empty">${
          kw ? "没有匹配的记录" : "还没有记录。上传车照后说「记录下来」。"}</div>`;
        return;
      }
      $("records").innerHTML = d.records.map((r) => `
        <div class="rec">
          ${r.image ? `<img src="${esc(r.image)}" alt="">` : `<div class="no-img">无图</div>`}
          <div class="body">
            <div class="name">${esc(r.car_name)}<span class="id">${esc(r.car_id)}</span></div>
            <div class="sub">${esc(r.brand || "")}${r.body_group ? " · " + esc(r.body_group) : ""}
              ${r.confidence != null ? " · 置信度 " + (r.confidence * 100).toFixed(1) + "%" : ""}
              · ${esc(r.created_at)}</div>
            ${r.note ? `<div class="note">备注：${esc(r.note)}</div>` : ""}
          </div>
          <button class="del" data-rid="${r.id}" title="删除">×</button>
        </div>`).join("");

      $("records").querySelectorAll(".del").forEach((b) => {
        b.onclick = async () => {
          const rid = b.dataset.rid;
          if (!confirm("确定从看车档案中删除这条记录？")) return;
          await fetch(`/api/records/${rid}?session_id=${encodeURIComponent(sessionId || "")}`,
            { method: "DELETE" });
          loadRecords();
        };
      });
    } catch (e) { /* 忽略 */ }
  }

  $("rec-kw").addEventListener("input", () => {
    clearTimeout(recTimer);
    recTimer = setTimeout(loadRecords, 250);
  });
  $("rec-scope").addEventListener("change", loadRecords);

  /* ==================== 设置：API 配置 / 模型配置 ==================== */
  // 字段名 ↔ DOM id ↔ 所属页签。新增可调参数时只改这张表。
  const SETTING_FIELDS = [
    ["api_url", "set-api-url"],
    ["connect_timeout", "set-connect-timeout"],
    ["read_timeout", "set-read-timeout"],
    ["max_retries", "set-max-retries"],
    ["retry_backoff", "set-retry-backoff"],
    ["model", "set-model"],
    ["temperature", "set-temperature"],
    ["max_tokens", "set-max-tokens"],
    ["max_rounds", "set-max-rounds"],
    ["max_tool_calls", "set-max-tool-calls"],
    ["top_k", "set-top-k"],
    ["unknown_threshold", "set-unknown-threshold"],
    ["display_confidence_threshold", "set-display-threshold"],
    ["prior_adjust_tau", "set-prior-tau"],
    ["web_search_provider", "set-web-provider"],
  ];
  let settingsCache = null;

  const openSettings = async () => {
    $("settings-mask").hidden = false;
    await loadSettings();
  };
  const closeSettings = () => { $("settings-mask").hidden = true; };

  $("btn-settings").onclick = openSettings;
  $("btn-settings-close").onclick = closeSettings;
  $("settings-mask").onclick = (e) => { if (e.target.id === "settings-mask") closeSettings(); };
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("settings-mask").hidden) closeSettings();
  });

  $("settings-tabs").querySelectorAll(".tab").forEach((tab) => {
    tab.onclick = () => {
      $("settings-tabs").querySelectorAll(".tab").forEach((t) =>
        t.classList.toggle("active", t === tab));
      document.querySelectorAll(".tabpane").forEach((p) =>
        p.classList.toggle("active", p.dataset.pane === tab.dataset.tab));
    };
  });

  function fillSettings(d) {
    settingsCache = d;
    for (const [key, id] of SETTING_FIELDS) {
      const el = $(id);
      if (!el) continue;
      const v = d.config[key];
      el.value = v === null || v === undefined ? "" : v;
      el.classList.remove("dirty");
      el.oninput = () => {
        const changed = String(el.value) !== String(d.defaults[key] ?? "");
        el.classList.toggle("dirty", changed);
        updateChangedHint();
      };
    }
    $("set-api-key").value = "";
    $("set-api-key").placeholder = d.api_key_configured
      ? `留空 = 保持不变（当前 ${d.api_key_masked}）`
      : "还没有配置，粘贴一个 sk- 开头的 Key";
    $("tip-key").innerHTML = d.api_key_configured
      ? `已配置，来源：<b>${esc(d.api_key_source)}</b>。留空即保持不变。`
      : `未配置 —— 当前运行在<b>离线替身模式</b>，文字由脚本化替身生成，不代表模型效果。`;
    $("tip-provider").textContent = (d.provider_help || {})[d.config.web_search_provider] || "";
    $("set-web-provider").onchange = () => {
      $("tip-provider").textContent = (d.provider_help || {})[$("set-web-provider").value] || "";
    };
    $("kv-model").textContent = d.config.model;
    $("kv-key").textContent = d.api_key_masked;
    $("kv-mode").textContent = d.offline ? "离线替身" : "真实 API";
    $("kv-t").textContent = d.config.unknown_threshold;
    $("kv-c").textContent = d.config.display_confidence_threshold;
    $("kv-tau").textContent = d.config.prior_adjust_tau;
    $("settings-note").innerHTML =
      `运行配置保存在 <code>${esc(d.runtime_config_file)}</code>（不含密钥）；` +
      `密钥单独写在 <code>${esc(d.env_file)}</code>，该文件已被 <code>.gitignore</code> 忽略。`;
    updateChangedHint();
  }

  function collectSettings() {
    // 只提交**偏离默认值**的字段，外加一份"已改回默认值"的列表。
    // 这样 runtime_config.json 里只留真正的差量：用户没碰过的项，
    // 以后代码里改了默认值仍然能生效；改回默认值的项会把旧的覆盖删掉。
    const d = settingsCache || { defaults: {} };
    const out = {};
    const reset = [];
    const key = $("set-api-key").value.trim();
    if (key) out.api_key = key;                 // 留空 = 不改
    for (const [k, id] of SETTING_FIELDS) {
      const el = $(id);
      if (!el) continue;
      const raw = el.value.trim();
      if (raw === "") continue;                 // 空值也不改
      if (String(d.defaults[k] ?? "") === raw) reset.push(k);
      else out[k] = raw;
    }
    if (reset.length) out.reset = reset;
    return out;
  }

  function updateChangedHint() {
    const d = settingsCache;
    if (!d) { $("settings-changed").textContent = ""; return; }
    const body = collectSettings();
    delete body.api_key;
    delete body.reset;
    const diffs = Object.entries(body);
    $("settings-changed").textContent = diffs.length
      ? `已改动 ${diffs.length} 项：${diffs.map(([k]) => k).join("、")}`
      : "未改动（保持默认）";
  }

  async function loadSettings() {
    try {
      const d = await (await fetch("/api/settings")).json();
      if (!d.ok) throw new Error("接口返回异常");
      fillSettings(d);
    } catch (e) {
      $("settings-note").innerHTML =
        `<span class="warn-line">读取设置失败：${esc(e.message)}</span>`;
    }
  }

  $("btn-settings-save").onclick = async () => {
    const body = collectSettings();
    if (!Object.keys(body).length) { setHint("没有需要保存的改动"); closeSettings(); return; }
    const btn = $("btn-settings-save");
    btn.disabled = true; btn.textContent = "保存中…";
    try {
      const r = await fetch("/api/settings", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const d = await r.json();
      if (!d.ok) {
        $("settings-note").innerHTML =
          `<span class="warn-line">保存失败 [${esc(d.error?.code || "")}]：${
            esc(d.error?.message || "")}${d.error?.hint ? " —— " + esc(d.error.hint) : ""}</span>`;
        return;
      }
      fillSettings(d.settings);
      await loadTools();
      loadRecords();
      const warn = (d.warnings || []).length
        ? "（" + d.warnings.join("；") + "）" : "";
      setHint(`设置已保存并生效${warn}`);
      closeSettings();
    } catch (e) {
      $("settings-note").innerHTML =
        `<span class="warn-line">保存失败：${esc(e.message)}</span>`;
    } finally {
      btn.disabled = false; btn.textContent = "保存并生效";
    }
  };

  $("btn-settings-reset").onclick = () => {
    if (!settingsCache) return;
    if (!confirm("把界面上改过的参数恢复成默认值？（不影响 agent/.env.local 里的 Key）")) return;
    for (const [k, id] of SETTING_FIELDS) {
      const el = $(id);
      if (el) { el.value = settingsCache.defaults[k] ?? ""; el.classList.remove("dirty"); }
    }
    updateChangedHint();
    setHint("已填入默认值，点「保存并生效」才会真正写盘");
  };

  $("btn-test-api").onclick = async () => {
    const box = $("test-result");
    const btn = $("btn-test-api");
    box.className = "test-result";
    box.textContent = "正在测试…";
    btn.disabled = true;
    try {
      const r = await fetch("/api/settings/test", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectSettings()),
      });
      const d = await r.json();
      if (d.ok) {
        box.className = "test-result ok";
        box.textContent = `✓ 连接成功 · ${d.elapsed_ms} ms · 服务端模型 ${d.model_reported}`
          + ` · tokens ${d.tokens ?? "?"}`
          + (d.retries ? ` · 自动重试 ${d.retries} 次后成功` : "")
          + (d.reply ? ` · 回复「${d.reply}」` : "");
      } else {
        box.className = "test-result err";
        box.textContent = `✗ [${d.error?.code || ""}] ${d.error?.message || ""}`
          + (d.error?.hint ? ` —— ${d.error.hint}` : "");
      }
    } catch (e) {
      box.className = "test-result err";
      box.textContent = "✗ " + e.message;
    } finally {
      btn.disabled = false;
    }
  };

  $("btn-test-search").onclick = async () => {
    const box = $("search-result");
    box.className = "test-result";
    box.textContent = "正在搜索…";
    try {
      // 复用对话链路不方便，这里直接看 /api/tools 里 web_search 是否已注册，
      // 再让后端用它自己的 provider 跑一次真实搜索（借用一次闲聊会话）。
      const r = await fetch("/api/settings/test-search", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: "日产轩逸 2024款 官方指导价" }),
      });
      const d = await r.json();
      if (d.ok) {
        box.className = "test-result ok";
        box.textContent = `✓ ${d.provider} 通道正常 · ${d.elapsed_ms} ms · 返回 ${d.count} 条`
          + (d.first ? ` · 首条「${d.first}」` : "");
      } else {
        box.className = "test-result err";
        box.textContent = `✗ [${d.error?.code || ""}] ${d.error?.message || ""}`
          + (d.error?.hint ? ` —— ${d.error.hint}` : "");
      }
    } catch (e) {
      box.className = "test-result err";
      box.textContent = "✗ " + e.message;
    }
  };

  /* ---------------- 启动 ---------------- */
  (async () => {
    await ensureSession();
    loadTools();
    loadRecords();
    loadSessions();
    loadSettings();
  })();
})();
