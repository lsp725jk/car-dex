# -*- coding: utf-8 -*-
"""侦察：画图库、中文字体、智能体可用性。"""
from __future__ import annotations

import importlib
import sys

print("=== 画图相关库 ===")
for m in ("matplotlib", "graphviz", "networkx", "PIL", "pydot", "numpy"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12} OK   {getattr(mod, '__version__', '')}")
    except Exception:
        print(f"  {m:12} 缺失")

print("\n=== 中文字体 ===")
try:
    import matplotlib.font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    for c in ("SimHei", "Microsoft YaHei", "SimSun", "KaiTi", "FangSong",
              "Noto Sans CJK SC", "Source Han Sans SC", "DengXian"):
        print(f"  {c:20} {'有' if c in names else '无'}")
except Exception as e:
    print("  matplotlib 不可用：", e)

print("\n=== graphviz 可执行文件 ===")
import shutil
print("  dot:", shutil.which("dot") or "无")

print("\n=== Python ===")
print(" ", sys.version.split()[0], sys.executable)
