# -*- coding: utf-8 -*-
"""粗略检查前端渲染函数里的插值是否都过了 esc()/md()（XSS 面）。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path("web/static/app.js")
lines = SRC.read_text(encoding="utf-8").splitlines()

# 找出所有顶层/内层 function 定义
funcs = [(i, m.group(1)) for i, l in enumerate(lines)
         if (m := re.search(r"\bfunction\s+(\w+)\s*\(", l))]

print(f"共 {len(funcs)} 个函数\n")
risky_total = 0
for idx, (start, name) in enumerate(funcs):
    # 函数体 = 到下一个函数定义（粗略但够用）
    end = funcs[idx + 1][0] if idx + 1 < len(funcs) else len(lines)
    body = lines[start:end]
    bad = []
    for off, l in enumerate(body):
        s = l.strip()
        if s.startswith("//") or s.startswith("*"):
            continue
        if "${" not in s:
            continue
        # 有插值，且没走 esc()/md()/数字格式化
        if "esc(" in s or "md(" in s:
            continue
        # 允许纯数字/枚举类插值
        if re.search(r"\$\{\s*[\w.]+\s*\}\s*$", s) and re.search(
                r"(count|length|id|index|pct|ms|s|n|turns|rounds)\b", s):
            continue
        bad.append((start + off + 1, s))
    if bad:
        risky_total += len(bad)
        print(f"[{name}]  L{start + 1}")
        for ln, s in bad[:8]:
            print(f"    L{ln}: {s[:120]}")
        print()

print(f"=== 疑似未转义的插值：{risky_total} 处（需人工判断，数字类的可忽略）===")
