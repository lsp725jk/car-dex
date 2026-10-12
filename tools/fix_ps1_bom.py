# -*- coding: utf-8 -*-
"""给 .ps1 补 UTF-8 BOM。

⚠️ 为什么需要：Windows PowerShell 5.1 读取**无 BOM** 的 .ps1 时会按系统 ANSI
   代码页解码，中文注释与字符串全部变成乱码，进而引发
   「字符串缺少终止符」这类看不懂的语法错误。
   这是本项目已记录的环境坑（docs/10 的 BUG-4），但写新脚本时极容易再犯。

用法：
    python tools/fix_ps1_bom.py tools/demo_window.ps1 [更多文件...]
"""
from __future__ import annotations

import sys
from pathlib import Path

BOM = b"\xef\xbb\xbf"


def fix(p: Path) -> str:
    raw = p.read_bytes()
    if raw.startswith(BOM):
        return f"  {p.name}：已有 BOM，跳过"
    # 若原本是 UTF-16 则不动（PowerShell 也能读）
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return f"  {p.name}：UTF-16，跳过"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return f"  {p.name}：不是合法 UTF-8，跳过（需人工确认编码）"
    p.write_bytes(BOM + text.encode("utf-8"))
    return f"  {p.name}：已补 BOM"


def main() -> None:
    args = sys.argv[1:]
    if not args:
        root = Path(__file__).resolve().parents[1]
        args = [str(p) for p in root.rglob("*.ps1")]
    print("处理 .ps1 文件：")
    for a in args:
        p = Path(a)
        if p.is_file():
            print(fix(p))
        else:
            print(f"  {a}：不存在")


if __name__ == "__main__":
    main()
