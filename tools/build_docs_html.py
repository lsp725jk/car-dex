# -*- coding: utf-8 -*-
"""把 docs/ 下的 Markdown 文档渲染成 HTML 文档站（单文件版 + 分篇版）。

## 解决的核心问题

Markdown 里的交叉引用写的是 `[项目总览](01_项目总览与快速开始.md)`。
直接把 Markdown 转成 HTML，这类链接会指向 `.md` 文件本身——
在浏览器里点开就变成一堆原始 Markdown 文本，非常难看。

本脚本在**渲染时**把这类链接改写掉（Markdown 源文件保持语义不变）：
  · 单文件版：`xxx.md` → `#docNN`（页面内跳转到对应章节）
  · 分篇版  ：`xxx.md` → `./xxx.html`（跳到同目录下的对应 HTML）

## 产出

1. `docs/识车档案_CarDex_说明文档.html` —— 11 篇合并成一个页面，单文件零依赖
2. `docs/html/*.html`                     —— 每篇文档一个独立页面，互相跳转

## 功能

  · 亮色 / 暗色双主题（跟随系统 + 手动切换 + localStorage 记忆）
  · 侧边栏：当前文档可展开目录、其他文档是跳转链接；实时搜索（/ 聚焦，Esc 清空）
  · 顶部阅读进度条 + 面包屑；篇末上一篇 / 下一篇
  · 表格：圆角卡片、表头吸顶、行悬停、横向滚动提示、宽表自动放宽
  · 代码块：语言标签 + 一键复制；标题悬停锚点可复制链接
  · 移动端抽屉侧边栏；打印样式（隐藏侧栏、h1 分页、表格不跨页）

依赖：markdown、pygments（仅生成时需要）
用法：
    python tools/build_docs_html.py
    python tools/build_docs_html.py --no-split      # 只出单文件版
"""
from __future__ import annotations

import argparse
import html as _html
import re
import sys
from pathlib import Path

try:
    import markdown
    from markdown.extensions.toc import TocExtension
    from pygments.formatters import HtmlFormatter
except ImportError:  # pragma: no cover
    sys.exit("需要先安装依赖：pip install markdown pygments")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
SITE_DIR = DOCS_DIR / "html"

DOC_FILES = [
    "00_文档索引.md",
    "01_项目总览与快速开始.md",
    "02_目录与文件全说明.md",
    "03_配置参数详解.md",
    "04_数据格式与文件约定.md",
    "05_脚本与命令手册.md",
    "06_评估指标与决策规则.md",
    "07_实验记录与结果分析.md",
    "08_常见问题与排错.md",
    "09_任务二三交接说明.md",
    "10_真实API与端到端测试报告.md",
    "11_Web界面与会话管理.md",
    "12_车型档次知识库.md",
    "13_技术栈全解.md",
    "14_GPU加速实战.md",
    "15_微调实战指南.md",
    "16_实训报告.md",
    "17_答辩预设问答.md",
    "18_进阶训练技术.md",
    "19_技术全景与关系图谱.md",
    "22_测试集提交与失分诊断.md",
    "23_源码清单与提交说明.md",
    "任务二三_agent架构调研与设计.md",
    "任务二三_智能体与Web系统说明.md",
    "任务一_数据分析与视觉模型训练.md",
    "任务二三_实训报告章节.md",
]

SHORT_LABELS = {
    "00_文档索引.md": "00 · 文档索引",
    "01_项目总览与快速开始.md": "01 · 项目总览与快速开始",
    "02_目录与文件全说明.md": "02 · 目录与文件全说明",
    "03_配置参数详解.md": "03 · 配置参数详解",
    "04_数据格式与文件约定.md": "04 · 数据格式与文件约定",
    "05_脚本与命令手册.md": "05 · 脚本与命令手册",
    "06_评估指标与决策规则.md": "06 · 评估指标与决策规则",
    "07_实验记录与结果分析.md": "07 · 实验记录与结果分析",
    "08_常见问题与排错.md": "08 · 常见问题与排错",
    "09_任务二三交接说明.md": "09 · 任务二三交接说明",
    "10_真实API与端到端测试报告.md": "10 · 真实API与端到端测试报告",
    "11_Web界面与会话管理.md": "11 · Web界面与会话管理",
    "12_车型档次知识库.md": "12 · 车型档次知识库",
    "13_技术栈全解.md": "13 · 技术栈全解",
    "14_GPU加速实战.md": "14 · GPU 加速实战",
    "15_微调实战指南.md": "15 · 微调实战指南",
    "16_实训报告.md": "★ · 实训报告（8.1 要求）",
    "17_答辩预设问答.md": "★ · 答辩预设问答",
    "18_进阶训练技术.md": "18 · 进阶训练技术",
    "19_技术全景与关系图谱.md": "★ · 技术全景与关系图谱",
    "22_测试集提交与失分诊断.md": "★ · 测试集提交与失分诊断",
    "23_源码清单与提交说明.md": "★ · 源码清单与提交说明",
    "任务二三_agent架构调研与设计.md": "20 · Agent 架构调研与设计",
    "任务二三_智能体与Web系统说明.md": "21 · 智能体与 Web 系统说明",
    "任务一_数据分析与视觉模型训练.md": "★ · 实训报告正文（任务一）",
    "任务二三_实训报告章节.md": "★ · 实训报告（任务二三）",
}

CSS = r"""
:root{
  --accent:#2563eb; --accent-soft:#eff6ff; --accent-line:#bfdbfe;
  --bg:#fff; --bg-soft:#f8fafc; --bg-sink:#f1f5f9;
  --fg:#0f172a; --fg-soft:#334155; --muted:#64748b; --faint:#94a3b8;
  --border:#e2e8f0; --border-strong:#cbd5e1;
  --code-bg:#f8fafc; --code-border:#e2e8f0; --ok:#15803d;
  --shadow-sm:0 1px 2px rgba(15,23,42,.06);
  --shadow:0 6px 22px rgba(15,23,42,.10);
  --radius:12px; --radius-sm:8px;
  --sidebar-w:340px; --content-max:900px;
  --sans:"Segoe UI","Microsoft YaHei UI","Microsoft YaHei","PingFang SC","Hiragino Sans GB",
         "Source Han Sans SC","Noto Sans CJK SC",-apple-system,BlinkMacSystemFont,Roboto,sans-serif;
  --mono:"Cascadia Mono","Cascadia Code",Consolas,"SF Mono",Menlo,Monaco,"Courier New",monospace;
}
html[data-theme="dark"]{
  --accent:#60a5fa; --accent-soft:#172554; --accent-line:#1e3a8a;
  --bg:#0b1220; --bg-soft:#111a2e; --bg-sink:#0f172a;
  --fg:#e6edf7; --fg-soft:#c7d2e3; --muted:#93a3bb; --faint:#6b7c96;
  --border:#21304a; --border-strong:#31425f;
  --code-bg:#0f1830; --code-border:#21304a; --ok:#4ade80;
  --shadow-sm:0 1px 2px rgba(0,0,0,.4); --shadow:0 6px 24px rgba(0,0,0,.5);
}
*,*::before,*::after{box-sizing:border-box}
html{scroll-behavior:smooth;-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);
  font-size:15.5px;line-height:1.82;text-rendering:optimizeLegibility;-webkit-font-smoothing:antialiased}
code,kbd,pre,samp{font-family:var(--mono)}

#app{display:flex;align-items:flex-start}
#sidebar{flex:0 0 var(--sidebar-w);width:var(--sidebar-w);position:sticky;top:0;height:100vh;
  overflow-y:auto;overscroll-behavior:contain;background:var(--bg-soft);
  border-right:1px solid var(--border);padding:20px 14px 80px 20px;scrollbar-width:thin}
#sidebar::-webkit-scrollbar{width:9px}
#sidebar::-webkit-scrollbar-thumb{background:var(--border-strong);border-radius:6px;border:2px solid var(--bg-soft)}
#content{flex:1 1 auto;min-width:0;padding:0 56px 140px}
.wrap{max-width:var(--content-max);margin:0 auto}

.brand{display:flex;gap:11px;align-items:center;margin:2px 0 16px}
.logo{width:38px;height:38px;flex:0 0 38px;border-radius:10px;
  background:linear-gradient(135deg,var(--accent),#7c3aed);color:#fff;display:grid;place-items:center;
  font-weight:800;font-size:14px;box-shadow:var(--shadow-sm);letter-spacing:-.5px;text-decoration:none}
.brand-txt b{display:block;font-size:15px;line-height:1.3}
.brand-txt span{display:block;font-size:11.5px;color:var(--muted);line-height:1.55;margin-top:3px}

.search-box{position:relative;margin-bottom:14px}
.search-box svg{position:absolute;left:11px;top:50%;transform:translateY(-50%);opacity:.45}
#search{width:100%;padding:9px 34px;font-family:var(--sans);font-size:13.5px;color:var(--fg);
  background:var(--bg);border:1px solid var(--border-strong);border-radius:var(--radius-sm);
  transition:border-color .15s,box-shadow .15s}
#search::placeholder{color:var(--faint)}
#search:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
.kbd-hint{position:absolute;right:9px;top:50%;transform:translateY(-50%);font-size:10.5px;
  color:var(--faint);border:1px solid var(--border);border-radius:4px;padding:0 5px;background:var(--bg-soft)}
#search:focus ~ .kbd-hint{opacity:0}

.nav-group{margin-bottom:3px;border-radius:var(--radius-sm)}
.nav-head{display:flex;align-items:center;gap:6px;width:100%;padding:7px 9px;border:0;background:none;
  cursor:pointer;color:var(--fg);font-family:var(--sans);font-size:13.2px;font-weight:600;
  text-align:left;border-radius:var(--radius-sm);line-height:1.45;transition:background .12s;text-decoration:none}
.nav-head:hover{background:var(--bg-sink)}
a.nav-head{padding-left:27px;color:var(--muted);font-weight:500}
a.nav-head:hover{color:var(--accent)}
a.nav-head .doc-arrow{opacity:0;margin-left:auto;font-size:11px;transition:opacity .15s}
a.nav-head:hover .doc-arrow{opacity:1}
.caret{flex:0 0 12px;width:12px;height:12px;opacity:.5;transition:transform .18s}
.nav-group.collapsed .caret{transform:rotate(-90deg)}
.nav-group.active > .nav-head{background:var(--accent-soft);color:var(--accent)}
.nav-list{margin:1px 0 6px 5px;padding-left:9px;border-left:2px solid var(--border);overflow:hidden}
.nav-group.collapsed .nav-list{display:none}
.nav-list a{display:block;padding:3.5px 8px;border-radius:6px;font-size:12.6px;line-height:1.55;
  color:var(--muted);text-decoration:none;transition:background .12s,color .12s}
.nav-list a:hover{background:var(--bg-sink);color:var(--fg)}
.nav-list a.current{color:var(--accent);background:var(--accent-soft);font-weight:600}
.nav-list a.lv4{padding-left:19px;font-size:12.1px}
.side-foot{margin-top:18px;padding-top:12px;border-top:1px solid var(--border);
  font-size:11.5px;color:var(--faint);line-height:1.75}
.side-foot a{color:var(--accent);text-decoration:none}
.hidden{display:none !important}

#progress{position:fixed;top:0;left:0;height:3px;width:0;z-index:60;
  background:linear-gradient(90deg,var(--accent),#7c3aed);transition:width .1s linear}
#topbar{position:sticky;top:0;z-index:50;margin:0 -56px;padding:10px 56px;background:var(--bg);
  border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px;
  font-size:12.5px;color:var(--muted)}
#crumb{flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#crumb b{color:var(--fg);font-weight:600}
.tool{flex:0 0 auto;display:inline-flex;align-items:center;gap:5px;padding:5px 11px;
  border:1px solid var(--border-strong);border-radius:999px;background:var(--bg);color:var(--fg-soft);
  font-size:12px;cursor:pointer;font-family:var(--sans);transition:all .14s;white-space:nowrap;text-decoration:none}
.tool:hover{border-color:var(--accent);color:var(--accent);background:var(--accent-soft)}
#hamburger{display:none}

.doc{padding-top:14px;scroll-margin-top:70px}
.doc + .doc{margin-top:64px;padding-top:52px;border-top:1px solid var(--border)}
h1,h2,h3,h4,h5{line-height:1.35;font-weight:700;scroll-margin-top:74px;position:relative}
h1{font-size:29px;margin:0 0 18px;padding-bottom:14px;border-bottom:2px solid var(--border);letter-spacing:-.4px}
h2{font-size:21.5px;margin:46px 0 16px;padding-left:14px;letter-spacing:-.2px}
h2::before{content:"";position:absolute;left:0;top:.22em;bottom:.22em;width:4px;border-radius:2px;background:var(--accent)}
h3{font-size:17.5px;margin:32px 0 12px}
h4{font-size:15.5px;margin:24px 0 9px;color:var(--fg-soft)}
h5{font-size:14.5px;margin:18px 0 8px;color:var(--muted)}
.anchor{opacity:0;margin-left:8px;color:var(--faint);text-decoration:none;font-weight:400;
  font-size:.78em;transition:opacity .15s}
h1:hover .anchor,h2:hover .anchor,h3:hover .anchor,h4:hover .anchor{opacity:1}
.anchor:hover{color:var(--accent)}

p{margin:13px 0}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid transparent;transition:border-color .14s}
a:hover{border-bottom-color:var(--accent)}
ul,ol{margin:13px 0;padding-left:26px}
li{margin:5px 0}
li::marker{color:var(--faint)}
hr{border:0;height:1px;background:var(--border);margin:34px 0}
blockquote{margin:16px 0;padding:12px 18px;color:var(--fg-soft);background:var(--bg-soft);
  border-left:4px solid var(--accent);border-radius:0 var(--radius-sm) var(--radius-sm) 0}
blockquote p:first-child{margin-top:0}
blockquote p:last-child{margin-bottom:0}

.table-wrap{position:relative;margin:18px 0;border:1px solid var(--border);border-radius:var(--radius);
  background:var(--bg);box-shadow:var(--shadow-sm);overflow:hidden}
.table-scroll{overflow-x:auto;overscroll-behavior-x:contain;scrollbar-width:thin}
.table-scroll::-webkit-scrollbar{height:10px}
.table-scroll::-webkit-scrollbar-thumb{background:var(--border-strong);border-radius:6px;border:3px solid var(--bg)}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:13.6px;line-height:1.65}
thead th{position:sticky;top:0;z-index:2;background:var(--bg-sink);font-weight:700;text-align:left;
  padding:9px 13px;border-bottom:2px solid var(--border-strong);white-space:nowrap}
tbody td{padding:8px 13px;border-bottom:1px solid var(--border);vertical-align:top}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover td{background:var(--accent-soft)}
td code,th code{white-space:nowrap;font-size:12.4px}
.table-hint{position:absolute;right:9px;bottom:7px;font-size:10.5px;color:var(--faint);
  background:var(--bg-soft);padding:1px 7px;border-radius:999px;pointer-events:none;opacity:0;
  transition:opacity .2s;border:1px solid var(--border)}
.table-wrap.scrollable .table-hint{opacity:1}
.table-wrap.scrollable.at-end .table-hint{opacity:0}
.table-wrap.wide{margin-left:-90px;margin-right:-90px}

code{background:var(--code-bg);border:1px solid var(--code-border);border-radius:5px;
  padding:1.5px 5.5px;font-size:13px;word-break:break-word}
pre{margin:18px 0;background:var(--code-bg);border:1px solid var(--code-border);
  border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow-sm)}
pre > code{display:block;background:none;border:0;padding:14px 16px;overflow-x:auto;
  font-size:12.9px;line-height:1.68;white-space:pre}
.codehilite{position:relative;margin:18px 0;background:var(--code-bg);border:1px solid var(--code-border);
  border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow-sm)}
.codehilite pre{margin:0;border:0;box-shadow:none;border-radius:0}
.codehilite pre > code{padding:14px 16px}
.code-bar{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:6px 12px;
  background:var(--bg-sink);border-bottom:1px solid var(--code-border);font-size:11px;
  color:var(--muted);font-family:var(--sans)}
.copy-btn{border:1px solid var(--border-strong);background:var(--bg);color:var(--muted);
  border-radius:6px;padding:2px 9px;font-size:11px;cursor:pointer;font-family:var(--sans);transition:all .14s}
.copy-btn:hover{color:var(--accent);border-color:var(--accent);background:var(--accent-soft)}
.copy-btn.done{color:var(--ok);border-color:var(--ok)}

img{max-width:100%;height:auto;display:block;margin:16px 0;border:1px solid var(--border);
  border-radius:var(--radius);box-shadow:var(--shadow-sm);background:var(--bg-soft)}
p > img{margin:16px auto}

.docnav{display:flex;gap:14px;margin-top:44px;padding-top:22px;border-top:1px solid var(--border)}
.docnav a{flex:1 1 0;border:1px solid var(--border);border-radius:var(--radius);padding:12px 16px;
  background:var(--bg-soft);text-decoration:none;color:var(--fg);transition:all .15s}
.docnav a:hover{border-color:var(--accent);background:var(--accent-soft)}
.docnav .dir{display:block;font-size:11px;color:var(--faint);margin-bottom:3px}
.docnav .ttl{display:block;font-size:13.5px;font-weight:600}
.docnav .next{text-align:right}

#totop{position:fixed;right:26px;bottom:26px;z-index:60;display:none;width:42px;height:42px;
  border-radius:50%;border:1px solid var(--border-strong);background:var(--bg);color:var(--fg-soft);
  cursor:pointer;font-size:16px;box-shadow:var(--shadow);transition:all .16s}
#totop:hover{color:var(--accent);border-color:var(--accent);transform:translateY(-2px)}
#mask{position:fixed;inset:0;background:rgba(15,23,42,.5);z-index:70;display:none}
#mask.on{display:block}

@media print{
  #sidebar,#totop,#topbar,#progress,#mask,.copy-btn,.table-hint,.docnav{display:none !important}
  #content{padding:0}.wrap{max-width:none}
  .doc + .doc{border-top:0;margin-top:0;padding-top:0}
  .doc h1{page-break-before:always}
  .doc:first-of-type h1{page-break-before:avoid}
  table,pre,.codehilite,blockquote,img{page-break-inside:avoid}
  h2,h3{page-break-after:avoid}
  a{color:#000;border:0}
  .table-wrap.wide{margin-left:0;margin-right:0}
}
@media (max-width:1320px){ .table-wrap.wide{margin-left:0;margin-right:0} }
@media (max-width:1180px){ :root{--content-max:100%} #content{padding:0 34px 120px} #topbar{margin:0 -34px;padding:10px 34px} }
@media (max-width:900px){
  :root{--sidebar-w:300px}
  #sidebar{position:fixed;top:0;left:0;bottom:0;height:100%;z-index:80;transform:translateX(-102%);
    transition:transform .24s cubic-bezier(.4,0,.2,1);box-shadow:var(--shadow)}
  #sidebar.open{transform:none}
  #content{padding:0 18px 110px}
  #topbar{margin:0 -18px;padding:10px 18px}
  #hamburger{display:inline-flex}
  h1{font-size:24px} h2{font-size:19.5px} h3{font-size:16.5px}
  .docnav{flex-direction:column}
}
"""

JS = r"""
(function(){
  var doc = document.documentElement, KEY = 'cardex-docs-theme';
  function applyTheme(t){
    doc.setAttribute('data-theme', t);
    var b = document.getElementById('btn-theme');
    if (b) b.textContent = (t === 'dark' ? '\u2600 亮色' : '\u263E 暗色');
  }
  var saved = null; try { saved = localStorage.getItem(KEY); } catch(e){}
  applyTheme(saved || ((window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light'));
  document.getElementById('btn-theme').addEventListener('click', function(){
    var t = doc.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    applyTheme(t); try { localStorage.setItem(KEY, t); } catch(e){}
  });

  /* 表格 */
  Array.prototype.forEach.call(document.querySelectorAll('.doc table'), function(tb){
    if (tb.parentElement && tb.parentElement.classList.contains('table-scroll')) return;
    var wrap = document.createElement('div'); wrap.className = 'table-wrap';
    var scroll = document.createElement('div'); scroll.className = 'table-scroll';
    tb.parentNode.insertBefore(wrap, tb); scroll.appendChild(tb); wrap.appendChild(scroll);
    var hint = document.createElement('div'); hint.className = 'table-hint';
    hint.textContent = '\u2190 \u53ef\u6a2a\u5411\u6eda\u52a8 \u2192'; wrap.appendChild(hint);
    if (tb.querySelectorAll('thead th').length >= 9) wrap.classList.add('wide');
    function upd(){
      var can = scroll.scrollWidth - scroll.clientWidth > 4;
      wrap.classList.toggle('scrollable', can);
      wrap.classList.toggle('at-end', scroll.scrollLeft + scroll.clientWidth >= scroll.scrollWidth - 4);
    }
    scroll.addEventListener('scroll', upd); window.addEventListener('resize', upd); upd();
  });

  /* 代码块 */
  var LANG = {python:'Python', bash:'Shell', sh:'Shell', shell:'Shell', json:'JSON', sql:'SQL',
              yaml:'YAML', js:'JavaScript', javascript:'JavaScript', css:'CSS', html:'HTML'};
  Array.prototype.forEach.call(document.querySelectorAll('.codehilite, pre'), function(block){
    if (block.closest && block.closest('.codehilite') && !block.classList.contains('codehilite')) return;
    if (block.querySelector && block.querySelector('.code-bar')) return;
    var isHl = block.classList.contains('codehilite');
    var pre = isHl ? block.querySelector('pre') : block;
    if (!pre) return;
    var code = pre.querySelector('code') || pre;
    var txt = code.textContent || '', lang = '';
    var m = (code.className || '').match(/language-([\w+-]+)/);
    if (m) lang = m[1];
    if (!lang && isHl){
      if (/^\s*(\$|python |pip |cd |Get-|start |\.\/)/m.test(txt)) lang = 'bash';
      else if (/^\s*(import |from |def |class |print\(|sys\.)/m.test(txt)) lang = 'python';
      else if (/^\s*[{\[]/.test(txt) && /":/.test(txt)) lang = 'json';
      else if (/^\s*(CREATE|SELECT|INSERT|--)/im.test(txt)) lang = 'sql';
      else if (/^\s*<(!DOCTYPE|html|div|section)/i.test(txt)) lang = 'html';
    }
    var bar = document.createElement('div'); bar.className = 'code-bar';
    var nm = document.createElement('span'); nm.textContent = LANG[lang] || (lang || '\u4ee3\u7801');
    var btn = document.createElement('button'); btn.className = 'copy-btn'; btn.type = 'button';
    btn.textContent = '\u590d\u5236';
    function fb(t, cb){
      var ta = document.createElement('textarea'); ta.value = t;
      ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select();
      try { document.execCommand('copy'); cb(); } catch(e){}
      document.body.removeChild(ta);
    }
    btn.addEventListener('click', function(){
      var t = code.innerText;
      var done = function(){ btn.textContent = '\u5df2\u590d\u5236'; btn.classList.add('done');
        setTimeout(function(){ btn.textContent = '\u590d\u5236'; btn.classList.remove('done'); }, 1400); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(t).then(done, function(){ fb(t, done); });
      else fb(t, done);
    });
    bar.appendChild(nm); bar.appendChild(btn);
    if (isHl) block.insertBefore(bar, block.firstChild);
    else {
      var w = document.createElement('div'); w.className = 'codehilite';
      block.parentNode.insertBefore(w, block); w.appendChild(bar); w.appendChild(block);
    }
  });

  /* 标题锚点 */
  Array.prototype.forEach.call(document.querySelectorAll('h2[id],h3[id],h4[id]'), function(h){
    var a = document.createElement('a');
    a.className = 'anchor'; a.href = '#' + h.id; a.textContent = '#'; a.title = '\u590d\u5236\u94fe\u63a5';
    a.addEventListener('click', function(ev){
      if (navigator.clipboard && navigator.clipboard.writeText){
        ev.preventDefault();
        navigator.clipboard.writeText(location.href.split('#')[0] + '#' + h.id);
        try { history.replaceState(null, '', '#' + h.id); } catch(e){}
        a.textContent = '\u2713'; setTimeout(function(){ a.textContent = '#'; }, 1200);
      }
    });
    h.appendChild(a);
  });

  /* 侧边栏 */
  var groups = Array.prototype.slice.call(document.querySelectorAll('.nav-group'));
  Array.prototype.forEach.call(document.querySelectorAll('button.nav-head'), function(h){
    h.addEventListener('click', function(){ h.parentElement.classList.toggle('collapsed'); });
  });
  var search = document.getElementById('search');
  function filter(){
    var t = (search.value || '').trim().toLowerCase();
    groups.forEach(function(g){
      var hit = !t || (g.getAttribute('data-key') || '').indexOf(t) >= 0;
      g.classList.toggle('hidden', !hit);
      if (hit && t) g.classList.remove('collapsed');
    });
  }
  search.addEventListener('input', filter);
  document.addEventListener('keydown', function(e){
    if (e.key === '/' && document.activeElement !== search){ e.preventDefault(); search.focus(); }
    if (e.key === 'Escape' && document.activeElement === search){ search.value = ''; filter(); search.blur(); }
  });

  var heads = Array.prototype.slice.call(document.querySelectorAll('#content h2[id],#content h3[id]'));
  var links = {};
  Array.prototype.forEach.call(document.querySelectorAll('.nav-list a'), function(a){
    links[a.getAttribute('href').slice(1)] = a;
  });
  var cur = null, curDoc = null;
  function sync(){
    if (heads.length){
      var y = window.scrollY + 100, best = null;
      heads.forEach(function(h){ if (h.offsetTop <= y) best = h; });
      if (cur !== best){
        if (cur && links[cur.id]) links[cur.id].classList.remove('current');
        cur = best;
        if (cur && links[cur.id]){
          var l = links[cur.id]; l.classList.add('current');
          var g = l.closest('.nav-group');
          if (g){
            groups.forEach(function(x){ x.classList.remove('active'); });
            g.classList.add('active'); g.classList.remove('collapsed');
            var r = l.getBoundingClientRect(), sr = document.getElementById('sidebar').getBoundingClientRect();
            if (r.top < sr.top + 100 || r.bottom > sr.bottom - 40) l.scrollIntoView({block:'center'});
          }
        }
      }
    }
    var cd = null;
    Array.prototype.forEach.call(document.querySelectorAll('.doc'), function(s){
      if (s.offsetTop <= window.scrollY + 130) cd = s;
    });
    if (cd && cd !== curDoc){
      curDoc = cd;
      var cb = document.getElementById('crumb');
      if (cb) cb.innerHTML = '\u6b63\u5728\u9605\u8bfb \u00b7 <b>' + (cd.getAttribute('data-title') || '') + '</b>';
    }
  }

  var bar = document.getElementById('progress'), totop = document.getElementById('totop');
  var side = document.getElementById('sidebar'), mask = document.getElementById('mask');
  function onScroll(){
    var h = document.documentElement.scrollHeight - window.innerHeight;
    bar.style.width = (h > 0 ? Math.min(100, window.scrollY / h * 100) : 0) + '%';
    totop.style.display = window.scrollY > 700 ? 'block' : 'none';
    sync();
  }
  var ticking = false;
  window.addEventListener('scroll', function(){
    if (!ticking){ ticking = true; requestAnimationFrame(function(){ onScroll(); ticking = false; }); }
  }, {passive:true});
  totop.addEventListener('click', function(){ window.scrollTo({top:0, behavior:'smooth'}); });
  document.getElementById('hamburger').addEventListener('click', function(){
    side.classList.add('open'); mask.classList.add('on');
  });
  mask.addEventListener('click', function(){ side.classList.remove('open'); mask.classList.remove('on'); });
  Array.prototype.forEach.call(document.querySelectorAll('.nav-list a, .docnav a, a.nav-head'), function(a){
    a.addEventListener('click', function(){ side.classList.remove('open'); mask.classList.remove('on'); });
  });

  onScroll(); filter();
})();
"""

MD_LINK_RE = re.compile(r'href="([^"]*?\.md)(#[^"]*)?"')


def slugify(text: str, fallback: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", text.strip().lower()).strip("-")
    return s or fallback


def html_name(md_name: str) -> str:
    return Path(md_name).stem + ".html"


def load_docs() -> list[dict]:
    """读取并渲染全部文档，返回条目列表。"""
    md_kwargs = dict(
        extensions=[
            "tables", "fenced_code", "codehilite", "attr_list", "def_list",
            "sane_lists", "md_in_html",
            TocExtension(slugify=lambda v, s: slugify(v, s), toc_depth="2-4"),
        ],
        extension_configs={"codehilite": {"guess_lang": False, "css_class": "codehilite"}},
    )
    entries: list[dict] = []
    for i, fname in enumerate(DOC_FILES, 1):
        path = DOCS_DIR / fname
        if not path.exists():
            print(f"[警告] 缺少 {fname}，已跳过")
            continue
        src = path.read_text(encoding="utf-8")
        md = markdown.Markdown(**md_kwargs)
        body = md.convert(src)

        doc_id = f"doc{i:02d}"
        prefix = f"{doc_id}-"
        body = re.sub(r'(<h[1-6] id=")([^"]+)(")',
                      lambda m: m.group(1) + prefix + m.group(2) + m.group(3), body)
        body = body.replace('href="#', f'href="#{prefix}')

        title = re.sub(r"^#\s+", "", src.split("\n", 1)[0]).strip()
        toc = [t for t in (md.toc_tokens or []) if t["level"] in (2, 3)]
        entries.append({
            "file": fname, "html": html_name(fname), "id": doc_id, "prefix": prefix,
            "title": title, "label": SHORT_LABELS.get(fname, title),
            "toc": toc, "body": body,
            "key": (fname + " " + title + " " + " ".join(t["name"] for t in toc)).lower(),
        })
    return entries


def rewrite_md_links(body: str, entries: list[dict], mode: str) -> tuple[str, list[str]]:
    """把指向 .md 的链接改写成本站可跳转的目标。mode: 'single' | 'split'"""
    by_file = {e["file"]: e for e in entries}
    by_stem = {Path(e["file"]).stem: e for e in entries}
    unresolved: list[str] = []

    def repl(m: re.Match) -> str:
        raw, _anchor = m.group(1), m.group(2)
        base = Path(raw.replace("\\", "/")).name
        target = by_file.get(base) or by_stem.get(Path(base).stem)
        if target is None:
            unresolved.append(raw)
            return m.group(0)
        if mode == "single":
            return f'href="#{target["id"]}"'
        return f'href="./{target["html"]}"'

    body = MD_LINK_RE.sub(repl, body)

    # 文档里的配图：相对路径统一以 docs/ 为基准。
    # 单文件版就放在 docs/ 内，原样可用；分页版放在 docs/html/ 内，
    # 所有相对路径都要多回退一级（images/x.png → ../images/x.png，
    # ../outputs/x.png → ../../outputs/x.png）。
    if mode == "split":
        body = re.sub(r'(<img[^>]+src=")(?!https?:|data:|/)([^"]+")', r"\1../\2", body)
    return body, unresolved


def sidebar_single(entries: list[dict]) -> str:
    out = []
    for e in entries:
        subs = "".join(
            f'<a class="lv{t["level"]}" href="#{e["prefix"]}{t["id"]}">{_html.escape(t["name"])}</a>'
            for t in e["toc"]
        )
        caret = ('<svg class="caret" viewBox="0 0 16 16" fill="none" stroke="currentColor" '
                 'stroke-width="2"><path d="M4 6l4 4 4-4"/></svg>')
        out.append(
            f'<div class="nav-group" data-key="{_html.escape(e["key"])}">'
            f'<button class="nav-head" type="button">{caret}<span>{_html.escape(e["label"])}</span></button>'
            + (f'<div class="nav-list">{subs}</div>' if subs else "") + "</div>"
        )
    return "".join(out)


def sidebar_split(entries: list[dict], cur_idx: int) -> str:
    out = []
    for j, e in enumerate(entries):
        if j == cur_idx:
            subs = "".join(
                f'<a class="lv{t["level"]}" href="#{e["prefix"]}{t["id"]}">{_html.escape(t["name"])}</a>'
                for t in e["toc"]
            )
            caret = ('<svg class="caret" viewBox="0 0 16 16" fill="none" stroke="currentColor" '
                     'stroke-width="2"><path d="M4 6l4 4 4-4"/></svg>')
            out.append(
                f'<div class="nav-group active" data-key="{_html.escape(e["key"])}">'
                f'<button class="nav-head" type="button">{caret}<span>{_html.escape(e["label"])}</span></button>'
                + (f'<div class="nav-list">{subs}</div>' if subs else "") + "</div>"
            )
        else:
            out.append(
                f'<div class="nav-group" data-key="{_html.escape(e["key"])}">'
                f'<a class="nav-head" href="./{e["html"]}"><span>{_html.escape(e["label"])}</span>'
                f'<span class="doc-arrow">↗</span></a></div>'
            )
    return "".join(out)


def render_page(*, title: str, sidebar: str, body: str, crumb: str, total: int,
                split_mode: bool = False) -> str:
    icon = ('<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" '
            'stroke-width="1.8"><circle cx="7" cy="7" r="4.5"/><path d="M10.5 10.5L14 14"/></svg>')
    pyg_light = HtmlFormatter(style="friendly").get_style_defs(".codehilite")
    pyg_dark = HtmlFormatter(style="monokai").get_style_defs('html[data-theme="dark"] .codehilite')

    home = "./" + html_name(DOC_FILES[0]) if split_mode else "#doc01"
    if split_mode:
        foot = (f'共 {total} 篇 · 分篇版 · '
                '<a href="../识车档案_CarDex_说明文档.html">打开单文件完整版</a><br>'
                '快捷键：<code>/</code> 搜索 · <code>Esc</code> 清空 · <code>Ctrl+P</code> 打印')
    else:
        foot = (f'共 {total} 篇 · 单文件版 · 由 <code>tools/build_docs_html.py</code> 生成<br>'
                '快捷键：<code>/</code> 搜索 · <code>Esc</code> 清空 · <code>Ctrl+P</code> 打印 / 导出 PDF')

    head = (
        "<!DOCTYPE html>\n<html lang=\"zh-CN\" data-theme=\"light\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<meta name=\"generator\" content=\"tools/build_docs_html.py\">\n"
        f"<title>{_html.escape(title)}</title>\n"
        "<style>\n" + CSS + "\n" + pyg_light + "\n" + pyg_dark + "\n</style>\n</head>\n<body>\n"
        '<div id="progress"></div>\n<div id="app">\n<aside id="sidebar">\n'
        f'<a class="brand" href="{home}"><div class="logo">识车</div>'
        '<div class="brand-txt"><b>识车档案 CarDex</b>'
        '<span>视觉智能体综合实训 · 任务一<br>私人车型资料库 / 汽车知识检索 / 对话式看车档案</span>'
        '</div></a>\n'
        '<div class="search-box">' + icon +
        '<input id="search" type="search" placeholder="搜索文档或标题…" autocomplete="off" spellcheck="false">'
        '<span class="kbd-hint">/</span></div>\n' + sidebar +
        f'\n<div class="side-foot">{foot}</div>\n</aside>\n'
    )
    main = (
        '<main id="content">\n<div id="topbar">\n'
        '<button class="tool" id="hamburger" type="button" title="打开目录">☰</button>\n'
        f'<div id="crumb">{crumb}</div>\n'
        '<button class="tool" id="btn-theme" type="button">☾ 暗色</button>\n'
        '</div>\n<div class="wrap">\n' + body + "\n</div>\n</main>\n</div>\n"
        '<div id="mask"></div>\n<button id="totop" type="button" title="回到顶部">↑</button>\n'
        "<script>\n" + JS + "\n</script>\n</body>\n</html>\n"
    )
    return head + main


def build() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="识车档案 CarDex — 说明文档")
    ap.add_argument("--out", default=str(DOCS_DIR / "识车档案_CarDex_说明文档.html"))
    ap.add_argument("--site-dir", default=str(SITE_DIR))
    ap.add_argument("--no-split", action="store_true", help="只生成单文件版，不生成分篇版")
    args = ap.parse_args()

    entries = load_docs()
    if not entries:
        sys.exit("没有找到任何可渲染的文档")

    all_unresolved: list[str] = []

    # ---------------- 1) 单文件版 ----------------
    blocks = []
    for idx, e in enumerate(entries):
        body, un = rewrite_md_links(e["body"], entries, "single")
        all_unresolved += un
        nav = ['<div class="docnav">']
        if idx > 0:
            p = entries[idx - 1]
            nav.append(f'<a class="prev" href="#{p["id"]}"><span class="dir">← 上一篇</span>'
                       f'<span class="ttl">{_html.escape(p["label"])}</span></a>')
        if idx + 1 < len(entries):
            n = entries[idx + 1]
            nav.append(f'<a class="next" href="#{n["id"]}"><span class="dir">下一篇 →</span>'
                       f'<span class="ttl">{_html.escape(n["label"])}</span></a>')
        nav.append("</div>")
        blocks.append(f'<section class="doc" id="{e["id"]}" data-title="{_html.escape(e["title"])}">'
                      + body + "".join(nav) + "</section>")

    page = render_page(
        title=args.title, sidebar=sidebar_single(entries), body="".join(blocks),
        crumb=f"共 {len(entries)} 篇说明文档", total=len(entries),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"[写出] {out}  ({out.stat().st_size / 1024:.0f} KB，单文件版)")

    # ---------------- 2) 分篇版 ----------------
    if args.no_split:
        return
    site = Path(args.site_dir)
    site.mkdir(parents=True, exist_ok=True)

    for idx, e in enumerate(entries):
        body, un = rewrite_md_links(e["body"], entries, "split")
        all_unresolved += un
        nav = ['<div class="docnav">']
        if idx > 0:
            p = entries[idx - 1]
            nav.append(f'<a class="prev" href="./{p["html"]}"><span class="dir">← 上一篇</span>'
                       f'<span class="ttl">{_html.escape(p["label"])}</span></a>')
        if idx + 1 < len(entries):
            n = entries[idx + 1]
            nav.append(f'<a class="next" href="./{n["html"]}"><span class="dir">下一篇 →</span>'
                       f'<span class="ttl">{_html.escape(n["label"])}</span></a>')
        nav.append("</div>")

        single = f'<section class="doc" id="{e["id"]}" data-title="{_html.escape(e["title"])}">' \
                 + body + "".join(nav) + "</section>"
        pg = render_page(
            title=e["title"] + " — 识车档案 CarDex",
            sidebar=sidebar_split(entries, idx), body=single,
            crumb=f"{_html.escape(e['label'])}　（{idx + 1}/{len(entries)}）",
            total=len(entries), split_mode=True,
        )
        (site / e["html"]).write_text(pg, encoding="utf-8")

    # 3) 站点入口：重定向到 00 索引
    entry = entries[0]["html"]
    (site / "index.html").write_text(
        "<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        f'<meta http-equiv="refresh" content="0; url=./{entry}">'
        f"<title>识车档案 CarDex — 说明文档</title></head>"
        f'<body>正在打开 <a href="./{entry}">说明文档</a>…</body></html>',
        encoding="utf-8")

    total_kb = sum((site / e["html"]).stat().st_size for e in entries) / 1024
    print(f"[写出] {site}\\*.html  ({len(entries)} 个页面 + index.html，合计 {total_kb:.0f} KB)")

    if all_unresolved:
        uniq = sorted(set(all_unresolved))
        print("[警告] 以下 .md 链接未在文档集内找到对应文件，保持原样：" + "、".join(uniq))


if __name__ == "__main__":
    build()
