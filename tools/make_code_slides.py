# -*- coding: utf-8 -*-
"""生成演示视频用的「代码展示页」HTML。

痛点：演示视频要展示代码，但直接开 IDE 字体小、还要手动滚，录出来很难看。
做法：把要展示的代码片段抽出来，生成一个**自带语法高亮、自动滚动、
底部有台词字幕**的单文件 HTML，录制时切到这个标签页就行。

用法：
    python tools/make_code_slides.py
    → outputs/report/demo/code_slides.html
    浏览器打开，← → 切页，空格暂停/继续自动滚动，F 全屏。
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report" / "demo" / "code_slides.html"

# 要展示的片段：文件、起止锚点（含）、标题、要念的台词、要点标注
SLIDES = [
    dict(
        file="agent/tools/vision.py",
        start="def _handler(args: dict, ctx: ToolContext)",
        end="", max=86,
        title="① classify_car：视觉识别确实由工具真实调用模型",
        say="这是 classify_car 的实现。它不走任何捷径，而是加载集成权重做真实前向推理。",
        points=["真实加载 outputs/checkpoints 下的三模型集成权重",
                "第 151 行 clf.predict_path() 是真正的前向推理调用",
                "模型不可用时抛 MODEL_NOT_LOADED，绝不返回伪造结果"],
    ),
    dict(
        file="src/cardex/inference.py",
        start="    def score_logits(self, images: list[Image.Image]",
        # 这个函数本身只有 29 行，用 end 会在函数结束处截断；改成往下多带一段
        # 上下文（后续的 predict_batch 等方法），让看代码的时间不至于太短。
        end="", max=64,
        title="② 集成打分：三个模型 + 先验校正的唯一入口",
        say="三个模型各自出 logits，先做先验校正，再按权重融合。训练和工具共用这一份实现。",
        points=["先验校正系数 τ 由验证集网格搜索得到（最终 0.2）",
                "训练评估与线上工具调用共用同一份打分逻辑，避免口径漂移"],
    ),
    dict(
        file="agent/loop.py",
        start="    def run_turn(self, session: Session, user_text: str,",
        end="    def ", max=76,
        title="③ 智能体循环：模型决策 → 执行工具 → 结果回传",
        say="每一轮把工具结果回传给模型，由模型决定下一步调哪个工具，循环直到给出最终回答。",
        points=["工具在独立线程执行，超时与异常都被结构化捕获",
                "每步的输入、输出、耗时写入 tool_calls.jsonl，可逐条复核"],
    ),
    dict(
        file="agent/tools/registry.py",
        start="class ToolRegistry:",
        end="", max=72,
        title="④ 工具注册表：12 个工具与其 JSON Schema",
        say="十二个工具集中注册，参数用 JSON Schema 校验，模型只能在这张表里选工具。",
        points=["1 个视觉工具 + 11 个业务工具，共 12 个",
                "非法参数在进入工具前就被拦下，返回 INVALID_ARGUMENT",
                "工具名不在注册表内直接报错，不会静默忽略"],
    ),
    dict(
        file="src/cardex/metrics.py",
        start="def compute_metrics(",
        end="def ", max=78,
        title="⑤ 评分口径：51 类 Macro-F1（含 unknown）",
        say="评分口径严格对齐教师要求：五十一类等权 Macro-F1，unknown 作为第五十一类参与计算。",
        points=["51 类等权，unknown 与 50 个已知类权重完全相同",
                "与教师评测口径一致，本地分数可直接对照线上得分"],
    ),
]


def extract(path: Path, start: str, end: str, max_lines: int = 30) -> tuple[list[str], int]:
    """抽出 [start, end) 之间的代码；end 为空则抽到文件末尾。"""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    s = next((i for i, l in enumerate(lines) if l.startswith(start)), None)
    if s is None:
        return [], 0
    e = len(lines)
    if end:
        for i in range(s + 1, len(lines)):
            if lines[i].startswith(end):
                e = i
                break
    return lines[s:min(e, s + max_lines)], s + 1


KEYWORDS = (r"\b(def|class|return|if|elif|else|for|while|try|except|finally|with|as|"
            r"import|from|in|not|and|or|is|None|True|False|raise|yield|lambda|"
            r"global|nonlocal|pass|break|continue|async|await)\b")


def _enc(i: int) -> str:
    """把序号编码成**纯字母**（前缀 z 保证不会拼成关键字）。

    ⚠️ 踩过的坑：第一版占位符写成 \\x00序号\\x00，序号是数字，
       结果被后面的「数字高亮」正则又匹配了一次（\\x005\\x00 → 中间那个 5 被包了 span），
       还原正则再也匹配不上，整段注释/字符串就变成了一个孤零零的数字。
    """
    s = ""
    while True:
        s = chr(ord("a") + i % 26) + s
        i //= 26
        if i == 0:
            return "z" + s


def highlight(code: str) -> str:
    """极简 Python 高亮：先转义，再按「注释/字符串 → 关键字 → 数字」上色。

    顺序很重要：必须先把注释和字符串换成占位符，否则字符串里的关键字
    也会被染色（例如 "return" 会被当成关键字）。
    """
    out = html.escape(code)
    tokens: list[str] = []

    def stash(m, cls):
        tokens.append(f'<span class="{cls}">{m.group(0)}</span>')
        return f"\ue000{_enc(len(tokens) - 1)}\ue001"

    out = re.sub(r"(#[^\n]*)", lambda m: stash(m, "cm"), out)
    out = re.sub(r"(&quot;&quot;&quot;[\s\S]*?&quot;&quot;&quot;|&#39;&#39;&#39;[\s\S]*?&#39;&#39;&#39;)",
                 lambda m: stash(m, "st"), out)
    out = re.sub(r"(&quot;[^&\n]*?&quot;|&#39;[^&\n]*?&#39;)", lambda m: stash(m, "st"), out)
    out = re.sub(KEYWORDS, r'<span class="kw">\1</span>', out)
    out = re.sub(r"\b(\d+\.?\d*)\b", r'<span class="nm">\1</span>', out)
    out = re.sub("\ue000([a-z]+)\ue001",
                 lambda m: tokens[int(_dec(m.group(1)))], out)
    return out


def _dec(s: str) -> int:
    """_enc 的逆运算。"""
    n = 0
    for ch in s[1:]:
        n = n * 26 + (ord(ch) - ord("a"))
    return n


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    slides = []
    for spec in SLIDES:
        code, ln = extract(ROOT / spec["file"], spec["start"], spec["end"],
                           max_lines=spec.get("max", 30))
        if not code:
            print(f"  [跳过] {spec['file']} 未找到锚点 {spec['start']}")
            continue
        slides.append(dict(
            title=spec["title"], say=spec["say"], points=spec["points"],
            file=spec["file"], start_line=ln,
            html=highlight("\n".join(l.rstrip() for l in code)),
        ))
        print(f"  [片段] {spec['file']}:{ln}  {len(code)} 行")

    data = json.dumps(slides, ensure_ascii=False)
    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>CarDex 演示 · 代码展示页</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#0d1117; color:#e6edf3;
         font:15px/1.7 "Microsoft YaHei",sans-serif; overflow:hidden; }}
  #wrap {{ display:flex; height:100vh; }}
  #side {{ width:290px; flex:none; background:#161b22; padding:22px 20px;
           border-right:1px solid #30363d; display:flex; flex-direction:column; }}
  #title {{ font-size:20px; font-weight:700; color:#79c0ff; line-height:1.45;
            margin-bottom:16px; }}
  #file {{ font:13px/1 monospace; color:#8b949e; margin-bottom:20px;
           padding:8px 10px; background:#0d1117; border-radius:6px; }}
  #points {{ list-style:none; padding:0; margin:0; }}
  #points li {{ position:relative; padding-left:18px; margin-bottom:12px;
                font-size:13.5px; color:#adbac7; }}
  #points li:before {{ content:"▸"; position:absolute; left:0; color:#3fb950; }}
  #code {{ flex:1; overflow:hidden; position:relative; padding:26px 0 0 0; }}
  #inner {{ position:absolute; top:0; left:0; right:0; }}
  table {{ border-collapse:collapse; width:100%; }}
  td {{ vertical-align:top; padding:0; }}
  td.ln {{ width:64px; text-align:right; padding-right:16px; color:#484f58;
           font:14px/1.75 ui-monospace,Consolas,monospace; user-select:none; }}
  td.src {{ font:14px/1.75 ui-monospace,Consolas,"Cascadia Code",monospace;
            white-space:pre; padding-right:24px; }}
  .kw {{ color:#ff7b72; }} .st {{ color:#a5d6ff; }} .cm {{ color:#8b949e; font-style:italic; }}
  .nm {{ color:#79c0ff; }}
  #bar {{ position:fixed; left:0; right:0; bottom:0; background:rgba(12,18,28,.96);
          border-top:3px solid #4F81BD; padding:14px 26px; z-index:9; }}
  #say {{ font-size:19px; font-weight:600; }}
  #hint {{ font-size:13px; color:#8b949e; margin-top:5px; }}
  #prog {{ position:fixed; right:20px; top:16px; z-index:10; font:600 14px monospace;
           color:#7CFC9B; background:rgba(12,18,28,.85); padding:6px 12px;
           border-radius:6px; }}
</style></head><body>
<div id="wrap">
  <div id="side">
    <div id="title"></div>
    <div id="file"></div>
    <ul id="points"></ul>
  </div>
  <div id="code"><div id="inner"><table><tbody id="tb"></tbody></table></div></div>
</div>
<div id="bar"><div id="say"></div>
  <div id="hint">← → 切换片段　空格 暂停/继续滚动　↑↓ 手动滚动　F 全屏　R 重播</div></div>
<div id="prog"></div>
<script>
const SLIDES = {data};
let cur = -1, raf = null, paused = false, y = 0, speed = 0.55, last = 0;

function render(i) {{
  cur = i; const s = SLIDES[i];
  document.getElementById('title').textContent = s.title;
  document.getElementById('file').textContent = s.file + ' : 第 ' + s.start_line + ' 行起';
  document.getElementById('points').innerHTML =
    s.points.map(p => '<li>' + p + '</li>').join('');
  const rows = s.html.split('\\n').map((l, k) =>
    '<tr><td class="ln">' + (s.start_line + k) + '</td><td class="src">' + (l || ' ') + '</td></tr>');
  document.getElementById('tb').innerHTML = rows.join('');
  document.getElementById('say').textContent = s.say;
  y = 0; paused = false;
  document.getElementById('inner').style.transform = 'translateY(0px)';
}}

function tick(ts) {{
  if (!last) last = ts;
  const dt = (ts - last) / 1000; last = ts;
  if (!paused) {{
    const area = document.getElementById('code');
    const inner = document.getElementById('inner');
    const max = Math.max(0, inner.scrollHeight - area.clientHeight + 90);
    y = Math.min(max, y + speed * 60 * dt);
    inner.style.transform = 'translateY(' + (-y) + 'px)';
  }}
  raf = requestAnimationFrame(tick);
}}

addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') render((cur + 1) % SLIDES.length);
  else if (e.key === 'ArrowLeft')  render((cur - 1 + SLIDES.length) % SLIDES.length);
  else if (e.key === ' ') {{ e.preventDefault(); paused = !paused; }}
  else if (e.key === 'ArrowDown') {{ paused = true; y += 90;
        document.getElementById('inner').style.transform = 'translateY(' + (-y) + 'px)'; }}
  else if (e.key === 'ArrowUp') {{ paused = true; y = Math.max(0, y - 90);
        document.getElementById('inner').style.transform = 'translateY(' + (-y) + 'px)'; }}
  else if (e.key.toLowerCase() === 'f') document.documentElement.requestFullscreen();
  else if (e.key.toLowerCase() === 'r') render(cur);
}});

document.getElementById('prog').textContent = SLIDES.length + ' 段代码';
render(0); requestAnimationFrame(tick);
</script></body></html>"""
    OUT.write_text(doc, encoding="utf-8")
    print(f"\n[写出] {OUT}  （{len(slides)} 段代码）")

    # 同时输出一份 JS 数据文件：给「页内浮层」用。
    # 这样录屏时只需要一个浏览器窗口置顶，代码段由 demo_autoplay.js
    # 自己弹浮层展示，不用切标签页/切窗口（切窗口会闪白屏，剪辑也麻烦）。
    data_js = OUT.with_name("code_slides_data.js")
    data_js.write_text(
        "/* 由 tools/make_code_slides.py 生成，请勿手改。*/\n"
        "window.__CODE_SLIDES = " + json.dumps(slides, ensure_ascii=False, indent=1) + ";\n",
        encoding="utf-8")
    print(f"[写出] {data_js}  （供页内浮层加载）")


if __name__ == "__main__":
    main()
