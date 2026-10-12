# -*- coding: utf-8 -*-
"""抓取真实桌面截图（所有显示器），用于确认「用户屏幕上到底显示的是什么」。

为什么需要它：dsh 的 browser_screenshot 抓的是**自动化浏览器实例的视口**，
那个窗口通常是不可见/最小化的，和用户屏幕上看到的完全是两回事。
排查「我看到的和你看到的不一样」这类问题时，必须以真实桌面为准。

用法：
    python tools/grab_screen.py                    # 抓全部屏幕，存到默认路径
    python tools/grab_screen.py -o shot.png        # 指定输出
    python tools/grab_screen.py --window "CarDex"  # 只抓标题匹配的窗口
"""
from __future__ import annotations

import argparse
import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def virtual_screen():
    """返回整个虚拟桌面（含多显示器）的 (left, top, width, height)。"""
    user32 = ctypes.WinDLL("user32")
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        user32.SetProcessDPIAware()
    SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
    SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79
    return (user32.GetSystemMetrics(SM_XVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_YVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))


def window_rect(title_pattern: str):
    """找到标题匹配的窗口并返回其矩形（用于只抓某个窗口）。"""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    import re
    found = []

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if not n:
            return True
        buf = ctypes.create_unicode_buffer(n + 2)
        user32.GetWindowTextW(hwnd, buf, n + 2)
        if re.search(title_pattern, buf.value, re.I):
            r = RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(r))
            w, h = r.right - r.left, r.bottom - r.top
            if r.left > -30000 and w > 200 and h > 200:
                found.append((r.left, r.top, w, h, buf.value))
        return True

    user32.EnumWindows(EnumProc(cb), 0)
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--window", default=None, help="只抓标题匹配的窗口")
    a = ap.parse_args()

    from PIL import ImageGrab

    if a.window:
        cands = window_rect(a.window)
        if not cands:
            print(f"[!] 没找到可见的窗口匹配 «{a.window}»")
            print("    用 --list-windows 看有哪些，或者不加 --window 抓整个桌面。")
            sys.exit(1)
        x, y, w, h, title = max(cands, key=lambda t: t[2] * t[3])
        bbox = (x, y, x + w, y + h)
        print(f"抓取窗口：{title}  ({x},{y}) {w}x{h}")
    else:
        x, y, w, h = virtual_screen()
        bbox = (x, y, x + w, y + h)
        print(f"抓取整个桌面：({x},{y}) {w}x{h}")

    img = ImageGrab.grab(bbox=bbox, all_screens=True)
    out = Path(a.out) if a.out else ROOT / "outputs" / "report" / "demo" / "desktop.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"[写出] {out}  ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
