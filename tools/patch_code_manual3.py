# -*- coding: utf-8 -*-
"""把代码块的键盘控制补进共用浮层。

上一步只给独立动画页 (HTML) 挂了键盘处理，而浏览器自动演示加载的是
demo_overlay.js —— 浮层里根本没有 keydown 监听，所以方向键/空格在
浏览器演示里完全无效（实测 scrollTop 恒为 0）。

这里在浮层内部补一份，并用 **capture + stopPropagation** 抢在演示主循环
的 waitSpace 之前处理：代码没滚到底时空格只滚代码，滚到底了才放行去翻页。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "tools" / "make_presentation.py"
src = F.read_text(encoding="utf-8")

ANCHOR = """  api.setManual = setManual;
  api.codeStep = codeStep;
  api.isAuto = function () { return autoScroll; };"""

ADD = """  api.setManual = setManual;
  api.codeStep = codeStep;
  api.isAuto = function () { return autoScroll; };

  // 浮层内部的键盘控制（独立页另有一份；两处都要有，因为
  // 浏览器自动演示加载的是本模块，拿不到独立页的监听器）。
  // 用 capture 抢在演示主循环前面处理，并用 stopPropagation 阻止它翻页。
  addEventListener('keydown', function (e) {
    if (ov.style.display === 'none') return;
    var s = SLIDES[api.current];
    if (!s || s.kind !== 'code') return;
    var k = e.key, box = stage.querySelector('.codebox');
    if (k === ' ' || k === 'ArrowDown' || k === 'PageDown') {
      if (api.codeStep(0.9)) { e.preventDefault(); e.stopPropagation(); }
    } else if (k === 'ArrowUp' || k === 'PageUp') {
      if (box && box.scrollTop > 2) {
        setManual(true);
        box.scrollTop = Math.max(0, box.scrollTop - box.clientHeight * 0.9);
        e.preventDefault(); e.stopPropagation();
      }
    } else if (k === 'Home') {
      if (box) { setManual(true); box.scrollTop = 0; e.preventDefault(); e.stopPropagation(); }
    } else if (k === 'End') {
      if (box) { setManual(true); box.scrollTop = box.scrollHeight;
                 e.preventDefault(); e.stopPropagation(); }
    } else if (k.toLowerCase() === 'a') {
      setManual(autoScroll); e.preventDefault(); e.stopPropagation();
    }
  }, true);"""

if ANCHOR in src:
    src = src.replace(ANCHOR, ADD)
    F.write_text(src, encoding="utf-8")
    print("✅ 已把键盘控制补进共用浮层")
else:
    print("⚠️ 未找到锚点，未修改")
