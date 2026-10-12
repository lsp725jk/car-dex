# -*- coding: utf-8 -*-
"""单独改写第 1 段：加入自我介绍环节。

改动范围仅限第 1 段：
  · 台词补上组员、班级、指导教师与分工
  · 画面加一排成员卡片
  · 时长 20s → 32s（台词从 76 字增至 122 字）
其它 14 段一律不动。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "tools" / "make_presentation.py"
src = F.read_text(encoding="utf-8")

OLD = '''    dict(kind="title", dur=20, t="识车档案 CarDex",
         sub="视觉智能体综合实训 · 罗烨嵘 · 焦超",
         bullets=["看懂车照 —— 自训练 51 类视觉模型", "查资料比价格 —— 12 个工具的智能体",
                  "记住每一次看车 —— 可检索的个人档案"],
         say="这是识车档案 CarDex，我们的视觉智能体综合实训项目。它要解决的是一个很具体的问题："
             "普通人在选车、看车的过程中拍了很多车照，这些照片散在相册里，既搜不到，"
             "也记不起当时看到的是哪一款、什么价。CarDex 让用户上传一张车照，由视觉模型识别车型，"
             "再由智能体去查档案、比价格、写进个人看车记录，最终形成一份可检索、可积累的对话式档案。"),'''

NEW = '''    dict(kind="title", dur=32, t="识车档案 CarDex",
         sub="视觉智能体综合实训 · 智能 234 · 指导教师 林俊武",
         members=[("罗烨嵘", "视觉模型训练 · Web 前端", "202312903430"),
                  ("焦超", "智能体 · 后端与测试", "202312903414")],
         bullets=["看懂车照 —— 自训练 51 类视觉模型", "查资料比价格 —— 12 个工具的智能体",
                  "记住每一次看车 —— 可检索的个人档案"],
         say="大家好，我们是智能 234 班的罗烨嵘和焦超，指导老师是林俊武。"
             "我负责视觉模型的训练与 Web 前端，焦超负责智能体与后端。"
             "我们的项目叫识车档案 CarDex——用户上传一张车照，视觉模型识别车型，"
             "再由智能体查档案、比价格、写进个人看车记录，"
             "最终形成一份可检索、可积累的对话式档案。"),'''

if OLD in src:
    src = src.replace(OLD, NEW)
    print("① 第 1 段脚本已改写")
else:
    print("① ⚠️ 未找到第 1 段原文")

# ── 独立动画页：标题段渲染加成员卡片 ──
old_html = """    h = `<div class="center">
      <h1 class="anim d1">${esc(s.t)}</h1>
      <div class="sub anim d2">${esc(s.sub||'')}</div>
      <ul class="bullets anim d3">${(s.bullets||[]).map(b=>`<li>${esc(b)}</li>`).join('')}</ul>
    </div>`;"""
new_html = """    h = `<div class="center">
      <h1 class="anim d1">${esc(s.t)}</h1>
      <div class="sub anim d2">${esc(s.sub||'')}</div>
      ${(s.members||[]).length ? `<div class="members anim d3">${s.members.map(m=>
        `<div class="member"><div class="mn">${esc(m[0])}</div>
         <div class="mr">${esc(m[1])}</div><div class="mi">${esc(m[2])}</div></div>`).join('')}</div>` : ''}
      <ul class="bullets anim d4">${(s.bullets||[]).map(b=>`<li>${esc(b)}</li>`).join('')}</ul>
    </div>`;"""
if old_html in src:
    src = src.replace(old_html, new_html)
    print("② 独立页标题段已加成员卡片")
else:
    print("② ⚠️ 未找到独立页标题段渲染")

# ── 共用浮层：同上 ──
old_ov = """      h = '<div class="center"><h1 class="anim d1">' + esc(s.t) + '</h1>' +
          '<div class="sub anim d2">' + esc(s.sub) + '</div>' +
          '<ul class="bullets anim d3">' + (s.bullets||[]).map(function(b){
             return '<li>' + esc(b) + '</li>'; }).join('') + '</ul></div>';"""
new_ov = """      var mem = (s.members||[]).length
        ? '<div class="members anim d3">' + s.members.map(function (m) {
            return '<div class="member"><div class="mn">' + esc(m[0]) +
                   '</div><div class="mr">' + esc(m[1]) +
                   '</div><div class="mi">' + esc(m[2]) + '</div></div>';
          }).join('') + '</div>'
        : '';
      h = '<div class="center"><h1 class="anim d1">' + esc(s.t) + '</h1>' +
          '<div class="sub anim d2">' + esc(s.sub) + '</div>' + mem +
          '<ul class="bullets anim d4">' + (s.bullets||[]).map(function(b){
             return '<li>' + esc(b) + '</li>'; }).join('') + '</ul></div>';"""
if old_ov in src:
    src = src.replace(old_ov, new_ov)
    print("③ 共用浮层标题段已加成员卡片")
else:
    print("③ ⚠️ 未找到浮层标题段渲染")

# ── 两处 CSS 都要加 .members / .member 样式 ──
CSS_HTML = """  .members { display:flex; gap:22px; margin-top:30px; }
  .member { background:#121a26; border:1px solid #23303f; border-radius:12px;
            padding:18px 30px; min-width:250px; }
  .member .mn { font-size:23px; font-weight:700; color:#e8eef6; letter-spacing:3px; }
  .member .mr { color:#4F9BFF; font-size:15px; margin-top:9px; }
  .member .mi { color:#7d8b9c; font-size:13px; margin-top:5px;
                font-family:Consolas,monospace; }
"""
src = src.replace("  .bullets { list-style:none; margin-top:34px; }",
                  CSS_HTML + "  .bullets { list-style:none; margin-top:34px; }")

CSS_OV = """  #__ov .members { display:flex; gap:22px; margin-top:28px; }
  #__ov .member { background:#121a26; border:1px solid #23303f; border-radius:12px;
                  padding:17px 28px; min-width:240px; }
  #__ov .member .mn { font-size:22px; font-weight:700; color:#e8eef6;
                      letter-spacing:3px; }
  #__ov .member .mr { color:#4F9BFF; font-size:15px; margin-top:9px; }
  #__ov .member .mi { color:#7d8b9c; font-size:13px; margin-top:5px;
                      font-family:Consolas,monospace; }
"""
src = src.replace("  #__ov .bullets { list-style:none; margin-top:32px; }",
                  CSS_OV + "  #__ov .bullets { list-style:none; margin-top:32px; }")

F.write_text(src, encoding="utf-8")
print("④ 成员卡片样式已写入两处 CSS")
