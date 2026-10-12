# -*- coding: utf-8 -*-
"""同步改造独立动画页的滚动实现，并统一两边的键盘行为。

背景：独立页 (HTML) 与共用浮层 (OVERLAY_TPL) 各有一份滚动代码，
前者叫 startCodeScroll、后者叫 scrollCode。上一步只改了浮层，
而 CSS 是两处一起改的 —— 若不把 HTML 那份也改成 scrollTop 驱动，
独立页的代码块会因为 .inner 从 absolute 变成 relative 而完全不动。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "tools" / "make_presentation.py"
src = F.read_text(encoding="utf-8")

OLD = """function startCodeScroll(s) {
  const inner = stage.querySelector('.codebox .inner');
  const box   = stage.querySelector('.codebox');
  if (!inner || !box) return;
  let y = 0;
  const secs = Math.max(6, (s.dur||20) - 4);        // 在念完前滚到底
  clearInterval(timer);
  timer = setInterval(() => {
    const max = Math.max(1, inner.scrollHeight - box.clientHeight + 60);
    if (y >= max) return clearInterval(timer);
    y = Math.min(max, y + max / (secs * 60));
    inner.style.transform = `translateY(${-y}px)`;
  }, 16);
}"""

NEW = """let autoScroll = true, manualHint = null;

function setManual(on) {
  autoScroll = !on;
  if (!manualHint) {
    manualHint = document.createElement('div');
    manualHint.style.cssText = 'position:fixed;right:22px;top:56px;z-index:30;' +
      'font:600 12px/1 "Microsoft YaHei",sans-serif;padding:6px 11px;border-radius:6px;' +
      'background:rgba(12,18,28,.9);color:#F79646;pointer-events:none';
    document.body.appendChild(manualHint);
  }
  manualHint.textContent = on ? '手动 · 按 A 恢复自动' : '';
  manualHint.style.color = on ? '#F79646' : '#7CFC9B';
}

function startCodeScroll(s) {
  const box = stage.querySelector('.codebox');
  if (!box) return;
  clearInterval(timer);
  if (!autoScroll) return;
  const secs = Math.max(6, (s.dur||20) - 4);        // 在念完前滚到底
  timer = setInterval(() => {
    if (!autoScroll) return clearInterval(timer);
    const max = Math.max(1, box.scrollHeight - box.clientHeight);
    if (box.scrollTop >= max - 1) return clearInterval(timer);
    box.scrollTop = Math.min(max, box.scrollTop + max / (secs * 60));
  }, 16);
}

/** 返回 true 表示已消费（仍在滚代码，不该翻页） */
function codeStep(ratio) {
  const box = stage.querySelector('.codebox');
  if (!box) return false;
  const max = Math.max(0, box.scrollHeight - box.clientHeight);
  if (box.scrollTop >= max - 2) return false;
  setManual(true);
  box.scrollTop = Math.min(max, box.scrollTop + Math.max(60, box.clientHeight * (ratio||0.9)));
  return true;
}"""

if OLD in src:
    src = src.replace(OLD, NEW)
    print("① 独立页滚动实现已改为 scrollTop 驱动")
else:
    print("① ⚠️ 未找到 startCodeScroll 原文")

# 独立页也加滚轮/点击接管
src = src.replace(
    """go(0);
</script></body></html>\"\"\"""",
    """// 人工操作立刻接管自动滚动
stage.addEventListener('wheel', e => {
  const b = stage.querySelector('.codebox');
  if (b && b.contains(e.target)) setManual(true);
}, { passive: true, capture: true });
stage.addEventListener('mousedown', e => {
  const b = stage.querySelector('.codebox');
  if (b && b.contains(e.target)) setManual(true);
}, true);

go(0);
</script></body></html>\"\"\"""")

F.write_text(src, encoding="utf-8")
print("② 独立页滚轮接管已写入")
