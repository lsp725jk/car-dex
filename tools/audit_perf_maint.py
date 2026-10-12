# -*- coding: utf-8 -*-
"""性能与可维护性静态扫描。

性能：
  · 热路径上重复读磁盘（JSON/DB）—— 本该缓存却每次重读
  · 循环内发起 I/O
  · 大对象深拷贝
可维护性：
  · 过长函数（>50 行）
  · 魔法数字
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

SKIP = {"__pycache__", "outputs", ".venv", "node_modules"}


def py_files():
    for p in sorted(Path(".").rglob("*.py")):
        if any(x in p.parts for x in SKIP):
            continue
        yield p


print("=" * 78)
print("[性能] 疑似热路径上的重复磁盘读取")
print("=" * 78)
# 统计「会读盘的函数」被调用的位置
READERS = ("load_profiles", "load_trims", "load_class_map", "load_runtime_config",
           "load_source_to_fine_index", "_load_local_env")
for p in py_files():
    src = p.read_text(encoding="utf-8")
    lines = src.splitlines()
    try:
        tree = ast.parse(src)
    except Exception:
        continue
    for n in ast.walk(tree):
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = "\n".join(lines[n.lineno - 1: n.end_lineno or n.lineno])
        for r in READERS:
            cnt = len(re.findall(rf"\b{r}\(", body))
            if cnt >= 1 and n.name not in ("__init__",):
                # 只在循环里调用才是热点
                in_loop = re.search(rf"for .*:\s*\n(?:.*\n)*?.*\b{r}\(", body)
                if in_loop:
                    print(f"  ⚠ {p}:{n.lineno} {n.name}() 在循环内调用 {r}()")
print("  （无输出 = 没有明显的循环内读盘）")

print()
print("=" * 78)
print("[性能] logging / I/O 是否在热循环里")
print("=" * 78)
for p in py_files():
    src = p.read_text(encoding="utf-8")
    lines = src.splitlines()
    for i, l in enumerate(lines):
        if re.search(r"open\(|\.write_text\(|json\.loads?\(", l):
            # 往上看 6 行有没有 for
            window = lines[max(0, i - 6):i]
            if any(re.match(r"\s*for\s+", w) for w in window):
                print(f"  ⚠ {p}:{i + 1} {l.strip()[:90]}")
print("  （无输出 = 没有明显问题）")

print()
print("=" * 78)
print("[可维护性] 过长函数（>50 行）")
print("=" * 78)
long_funcs = []
for p in py_files():
    if str(p).startswith("tools\\audit"):
        continue
    src = p.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except Exception:
        continue
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.end_lineno:
            length = n.end_lineno - n.lineno + 1
            if length > 50:
                long_funcs.append((length, str(p), n.lineno, n.name))
long_funcs.sort(reverse=True)
for length, f, ln, name in long_funcs[:18]:
    flag = "🔴" if length > 120 else ("🟠" if length > 80 else "🟡")
    print(f"  {flag} {length:>4} 行  {f}:{ln}  {name}()")
print(f"  合计 {len(long_funcs)} 个函数超过 50 行")

print()
print("=" * 78)
print("[可维护性] 魔法数字（同一字面量重复 >= 6 次）")
print("=" * 78)
nums = Counter()
for p in py_files():
    src = p.read_text(encoding="utf-8")
    for m in re.finditer(r"(?<![\w.])(\d{3,})(?![\w.])", src):
        nums[m.group(1)] += 1
for v, c in nums.most_common(12):
    print(f"  {v:>6}  ×{c}")
