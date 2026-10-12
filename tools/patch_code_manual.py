# -*- coding: utf-8 -*-
"""给演示浮层打补丁：让代码块支持滚轮 / 方向键 / 空格的人工控制。

原实现的问题：
    代码块用的是 `transform: translateY()` 做自动滚动，容器本身 `overflow:hidden`，
    所以**在物理上不可能被人工滚动** —— 滚轮和方向键完全无效。
改造：
    ① 容器改成真实可滚动区（overflow-y:auto），自动滚动驱动 scrollTop；
    ② 滚轮天然生效；方向键 / PgUp / PgDn 手动步进；
    ③ 空格：代码未到底时向下翻一屏，到底后再按才切下一段；
    ④ 任何人工操作都会**自动暂停自动播放**并在右上角显示「手动」，
       按 A 恢复自动；这样人工和自动不会互相打架。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "tools" / "make_presentation.py"
src = F.read_text(encoding="utf-8")

# ── ① CSS：两处（独立页 + 共用浮层）都要把 codebox 改成真实滚动区 ──
src = src.replace(
    """  .codebox { flex:1; background:#0d1117; border:1px solid #26313d; border-radius:10px;
             overflow:hidden; position:relative; }
  .codebox .inner { position:absolute; top:0; left:0; right:0; }""",
    """  .codebox { flex:1; background:#0d1117; border:1px solid #26313d; border-radius:10px;
             overflow-y:auto; overflow-x:hidden; position:relative;
             scrollbar-width:thin; scrollbar-color:#3c4756 #0d1117; }
  .codebox::-webkit-scrollbar { width:10px; }
  .codebox::-webkit-scrollbar-thumb { background:#3c4756; border-radius:5px; }
  .codebox .inner { position:relative; }""")
src = src.replace(
    """  #__ov .codebox { flex:1; background:#0d1117; border:1px solid #26313d;
                   border-radius:10px; overflow:hidden; position:relative; }
  #__ov .codebox .inner { position:absolute; top:0; left:0; right:0; }""",
    """  #__ov .codebox { flex:1; background:#0d1117; border:1px solid #26313d;
                   border-radius:10px; overflow-y:auto; overflow-x:hidden;
                   position:relative; scrollbar-width:thin;
                   scrollbar-color:#3c4756 #0d1117; }
  #__ov .codebox::-webkit-scrollbar { width:10px; }
  #__ov .codebox::-webkit-scrollbar-thumb { background:#3c4756; border-radius:5px; }
  #__ov .codebox .inner { position:relative; }""")

# ── ② 自动滚动改驱动 scrollTop，并加手动接管 ──
old_scroll = """  function scrollCode(s) {
    var inner = stage.querySelector('.codebox .inner');
    var box   = stage.querySelector('.codebox');
    if (!inner || !box) return;
    var y = 0, secs = Math.max(6, (s.dur || 20) - 4);
    clearInterval(timer);
    timer = setInterval(function () {
      var max = Math.max(1, inner.scrollHeight - box.clientHeight + 60);
      if (y >= max || ov.style.display === 'none') return clearInterval(timer);
      y = Math.min(max, y + max / (secs * 60));
      inner.style.transform = 'translateY(' + (-y) + 'px)';
    }, 16);
  }"""

new_scroll = """  var autoScroll = true;          // 自动播放开关
  var manualHint = null;          // 「手动」角标

  function setManual(on) {
    autoScroll = !on;
    if (!manualHint) {
      manualHint = document.createElement('div');
      manualHint.style.cssText = 'position:absolute;right:14px;top:12px;z-index:5;' +
        'font:600 12px/1 "Microsoft YaHei",sans-serif;padding:6px 11px;border-radius:6px;' +
        'background:rgba(12,18,28,.9);color:#F79646;pointer-events:none';
      ov.appendChild(manualHint);
    }
    manualHint.textContent = on ? '手动 · 按 A 恢复自动' : '自动';
    manualHint.style.color = on ? '#F79646' : '#7CFC9B';
    clearTimeout(manualHint._t);
    if (!on) manualHint._t = setTimeout(function () { manualHint.textContent = ''; }, 2200);
  }

  function scrollCode(s) {
    var box = stage.querySelector('.codebox');
    if (!box) return;
    var secs = Math.max(6, (s.dur || 20) - 4);
    clearInterval(timer);
    if (!autoScroll) return;
    timer = setInterval(function () {
      if (!autoScroll || ov.style.display === 'none') return clearInterval(timer);
      var max = Math.max(1, box.scrollHeight - box.clientHeight);
      if (box.scrollTop >= max - 1) return clearInterval(timer);
      box.scrollTop = Math.min(max, box.scrollTop + max / (secs * 60));
    }, 16);
  }

  /** 供键盘/空格调用：返回 true 表示「已消费」（还在滚代码，不该翻页） */
  function codeStep(ratio) {
    var box = stage.querySelector('.codebox');
    if (!box) return false;
    var max = Math.max(0, box.scrollHeight - box.clientHeight);
    if (box.scrollTop >= max - 2) return false;         // 已到底 → 交给翻页
    setManual(true);
    box.scrollTop = Math.min(max,
      box.scrollTop + Math.max(60, box.clientHeight * (ratio || 0.9)));
    return true;
  }"""

if old_scroll in src:
    src = src.replace(old_scroll, new_scroll)
    print("① 自动滚动 → scrollTop 改造完成")
else:
    print("① ⚠️ 未找到 scrollCode，跳过")

# ── ③ 事件绑定：滚轮天然生效；方向键/空格接管 ──
src = src.replace(
    """  window.__DemoOverlay = api;""",
    """  // 人工操作立刻接管，避免和自动滚动打架
  ov.addEventListener('wheel', function (e) {
    if (e.target && e.target.closest && e.target.closest('.codebox')) setManual(true);
  }, { passive: true, capture: true });
  ov.addEventListener('mousedown', function (e) {
    var b = stage.querySelector('.codebox');
    if (b && b.contains(e.target)) setManual(true);
  }, true);

  api.setManual = setManual;
  api.codeStep = codeStep;
  api.isAuto = function () { return autoScroll; };
  window.__DemoOverlay = api;""")

# ── ④ 键盘：方向键/空格在代码段内优先滚代码 ──
old_key = """addEventListener('keydown', e => {
  const k = e.key;
  if (k === ' ') { e.preventDefault(); go(cur + 1); }"""
new_key = """addEventListener('keydown', e => {
  const k = e.key;
  const ovApi = window.__DemoOverlay;
  // 代码段：方向键 / 空格 / PgUp / PgDn 先用来滚代码，滚到底才翻页
  if (ovApi && SLIDES[cur] && SLIDES[cur].kind === 'code') {
    if (k === 'ArrowDown' || k === 'PageDown') {
      if (ovApi.codeStep(0.9)) { e.preventDefault(); return; }
    } else if (k === 'ArrowUp' || k === 'PageUp') {
      const box = document.querySelector('#__ov .codebox');
      if (box && box.scrollTop > 2) {
        ovApi.setManual(true);
        box.scrollTop = Math.max(0, box.scrollTop - box.clientHeight * 0.9);
        e.preventDefault(); return;
      }
    } else if (k === ' ') {
      if (ovApi.codeStep(0.9)) { e.preventDefault(); return; }
    } else if (k.toLowerCase() === 'a') {
      ovApi.setManual(!ovApi.isAuto()); e.preventDefault(); return;
    }
  }
  if (k === ' ') { e.preventDefault(); go(cur + 1); }"""
src = src.replace(old_key, new_key)

F.write_text(src, encoding="utf-8")
print("② 已写入浮层的手动控制与事件绑定")
print("③ 已写入键盘接管逻辑")
