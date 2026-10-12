# -*- coding: utf-8 -*-
"""列出「静默吞异常」的位置（except Exception: pass），人工判断哪些是隐患。"""
from __future__ import annotations

import ast
from pathlib import Path

rows = []
for p in sorted(Path(".").rglob("*.py")):
    if any(x in p.parts for x in ("__pycache__", "outputs", ".venv")):
        continue
    try:
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except Exception:
        continue
    lines = src.splitlines()
    for n in ast.walk(tree):
        if not isinstance(n, ast.ExceptHandler):
            continue
        tp = ast.unparse(n.type) if n.type else "(bare)"
        if tp not in ("Exception", "(bare)"):
            continue
        body = n.body
        # 只有 pass / continue / 赋值给未使用变量
        silent = all(isinstance(b, (ast.Pass, ast.Continue)) for b in body)
        if not silent:
            continue
        where = lines[n.lineno - 1].strip() if n.lineno <= len(lines) else ""
        rows.append((str(p), n.lineno, tp, where))

print(f"静默吞异常：{len(rows)} 处\n")
by_file: dict[str, list] = {}
for f, ln, tp, where in rows:
    by_file.setdefault(f, []).append((ln, tp, where))

# 按风险排序：核心链路优先
PRIORITY = ("agent/llm.py", "agent/loop.py", "agent/session.py", "web/app.py",
            "agent/tools/registry.py", "agent/tools/trims.py", "agent/tools/garage.py",
            "agent/tools/browser.py", "agent/tools/vision.py", "agent/tools/profile.py")
order = sorted(by_file, key=lambda f: (PRIORITY.index(f) if f in PRIORITY else 99, f))
for f in order:
    print(f"── {f}")
    for ln, tp, where in by_file[f]:
        print(f"     L{ln:<5} {where[:100]}")
    print()
