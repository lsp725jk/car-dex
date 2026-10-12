# -*- coding: utf-8 -*-
"""CarDex 录屏辅助 —— 浏览器窗口置顶 / 固定位置与尺寸。

为什么用 Python 而不是 PowerShell：
    同一个功能先用 .ps1 写了一版，踩了两个坑 ——
      ① 编辑工具保存时会去掉 UTF-8 BOM，PowerShell 5.1 随即按 ANSI 解码，
         C# 代码块里的中文变乱码，报一串「意外的字符 / 常量中有换行符」；
      ② 默认执行策略禁止运行未签名脚本，必须额外加 -ExecutionPolicy Bypass。
    改用 ctypes 后两个问题都不存在，输出也不再受控制台代码页影响。

用法：
    python tools/demo_window.py                     # 置顶 + 定位 (0,0) 1280x800
    python tools/demo_window.py --list              # 列出候选窗口
    python tools/demo_window.py --diag              # 列出匹配窗口及坐标（排查用）
    python tools/demo_window.py --borderless        # 去标题栏
    python tools/demo_window.py --restore           # 取消置顶并还原
"""
from __future__ import annotations

import argparse
import ctypes
import re
import sys
import time
from ctypes import wintypes

class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


user32 = ctypes.WinDLL("user32", use_last_error=True)

# ⚠️ 必须显式声明 argtypes：否则 ctypes 按默认的 32 位 c_int 传参，
#    而 HWND_TOPMOST(-1) 在 64 位下需要作为 0xFFFFFFFFFFFFFFFF 传给
#    HWND 形参，传成 0x00000000FFFFFFFF 会被当成无效句柄 ——
#    SetWindowPos 静默失败（返回 0），但窗口位置尺寸又确实改了，
#    于是出现「位置对了、置顶没生效」这种半成功状态，极难排查。
user32.SetWindowPos.argtypes = [
    wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_uint]
user32.SetWindowPos.restype = wintypes.BOOL
user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowLongW.restype = ctypes.c_long
user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
user32.SetWindowLongW.restype = ctypes.c_long
# GetWindowRect 刻意**不声明 argtypes**：声明成 POINTER(RECT) 后
#    ctypes.byref() 会报 "expected LP_RECT instance"。
#    不声明时 byref 走默认转换，工作正常。
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = [wintypes.HWND, ctypes.c_wchar_p, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int

GWL_STYLE, GWL_EXSTYLE = -16, -20
WS_CAPTION = 0x00C00000
WS_EX_TOPMOST = 0x00000008
HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
SWP_SHOWWINDOW, SWP_FRAMECHANGED = 0x0040, 0x0020

BROWSER_RE = re.compile(r"Edge|Chrome|Firefox|Brave|Opera|Chromium", re.I)


class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


def _titles() -> list[tuple[int, str, RECT]]:
    """枚举所有可见的顶层窗口 → (句柄, 标题, 矩形)。"""
    out: list[tuple[int, str, RECT]] = []
    EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n:
            buf = ctypes.create_unicode_buffer(n + 2)
            user32.GetWindowTextW(hwnd, buf, n + 2)
            r = RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(r))
            out.append((hwnd, buf.value, r))
        return True

    user32.EnumWindows(EnumProc(cb), 0)
    return out


def pick(pattern: str, diag: bool = False):
    """在所有标题匹配的窗口里挑出真正该置顶的那一个。

    踩过的两个坑：
      ① EnumWindows 按 z 序返回，直接取第一个匹配会抓错窗口 ——
         标题含「识车档案」的除了浏览器，还有 WPS 打开的实训报告。
      ② 更隐蔽：**自动化驱动的浏览器窗口通常是「最小化」状态**
         （Playwright/CDP 不需要窗口可见就能渲染），坐标是 (-32000,-32000)、
         尺寸 237x39。之前把这种窗口"置顶"当然毫无视觉效果。
         所以这里先在正常窗口里挑；一个都没有时，退而恢复浏览器窗口。

    返回 (选中项, 诊断行, 是否需要先恢复)。
    """
    rx = re.compile(pattern)
    rows, normal, minimized = [], None, None
    best_score = -1
    for hwnd, title, r in _titles():
        if not rx.search(title):
            continue
        w, h = r.right - r.left, r.bottom - r.top
        is_min = r.left <= -30000 or r.top <= -30000
        tiny = (not is_min) and (w < 400 or h < 300)
        is_browser = bool(BROWSER_RE.search(title))
        why = ""
        if is_min:
            why = "最小化"
        elif tiny:
            why = f"尺寸过小({w}x{h})"
        if diag:
            rows.append(f"0x{hwnd:X}  ({r.left},{r.top}) {w}x{h}  "
                        f"{'[浏览器]' if is_browser else '[其他]  '}  "
                        f"{title[:52]}  {why}")
        if is_min:
            # 记录一个「最小化的浏览器窗口」作为兜底
            if is_browser and minimized is None:
                minimized = (hwnd, title, r)
            continue
        if tiny:
            continue
        score = w * h // 1000 + (1_000_000 if is_browser else 0)
        if score > best_score:
            normal, best_score = (hwnd, title, r), score
    if normal is not None:
        return normal, rows, False
    return minimized, rows, True


def state(hwnd: int) -> str:
    r = RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 2)
    user32.GetWindowTextW(hwnd, buf, n + 2)
    ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    st = user32.GetWindowLongW(hwnd, GWL_STYLE)
    top = "是 (OK)" if ex & WS_EX_TOPMOST else "否 (未生效!)"
    return (f"    标题   : {buf.value}\n"
            f"    位置   : ({r.left}, {r.top})\n"
            f"    尺寸   : {r.right - r.left} x {r.bottom - r.top}\n"
            f"    置顶   : {top}\n"
            f"    标题栏 : {'有' if st & WS_CAPTION else '无（已去边框）'}\n"
            f"    句柄   : 0x{hwnd:X}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="识车档案|CarDex")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--x", type=int, default=0)
    ap.add_argument("--y", type=int, default=0)
    ap.add_argument("--borderless", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--diag", action="store_true")
    a = ap.parse_args()

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass

    if a.list:
        print("当前可见窗口：")
        for _, t, _r in sorted(_titles(), key=lambda x: x[1]):
            if t.strip():
                print("   " + t)
        return

    if a.diag:
        _best, rows, _need = pick(a.title, diag=True)
        print(f"标题匹配 «{a.title}» 的窗口：")
        for r in rows:
            print("   " + r)
        if not rows:
            print("   （无）")
        print("\n判据：坐标<=-30000 视为最小化；宽<400 或 高<300 视为过小；浏览器标题优先。")
        return

    best, _rows, need_restore = pick(a.title)
    if best is None:
        print(f"[!] 没找到可用的窗口（标题需匹配 «{a.title}»）。")
        print("    先跑  python tools/demo_window.py --diag  看看实际有哪些窗口。")
        sys.exit(1)

    hwnd, title, _r = best
    st = user32.GetWindowLongW(hwnd, GWL_STYLE)

    # 自动化驱动的浏览器窗口通常是「最小化」的 —— 不先恢复，置顶了也看不见。
    if need_restore:
        SW_RESTORE = 9
        user32.ShowWindow(hwnd, SW_RESTORE)
        time.sleep(0.5)
        print("[提示] 找到的浏览器窗口原本是最小化的，已先恢复显示。")

    if a.restore:
        user32.SetWindowLongW(hwnd, GWL_STYLE, st | WS_CAPTION)
        user32.SetWindowPos(hwnd, HWND_NOTOPMOST, a.x, a.y, a.width, a.height,
                            SWP_SHOWWINDOW | SWP_FRAMECHANGED)
        print("已取消置顶并还原标题栏。回读确认：")
        print(state(hwnd))
        return

    flags = SWP_SHOWWINDOW
    if a.borderless:
        user32.SetWindowLongW(hwnd, GWL_STYLE, st & ~WS_CAPTION)
        flags |= SWP_FRAMECHANGED

    ok = user32.SetWindowPos(hwnd, HWND_TOPMOST, a.x, a.y, a.width, a.height, flags)
    user32.SetForegroundWindow(hwnd)

    print("窗口已置顶。回读确认（这才是真正生效的证据）：")
    print(state(hwnd))
    if not ok:
        err = ctypes.get_last_error()
        print(f"\n[!] SetWindowPos 返回失败（Win32 错误码 {err}）。")
        print("    常见原因：目标窗口属于更高完整性级别的进程，普通权限改不动。")
        print("    可尝试：以管理员身份重跑本脚本。")
    print("\n录制结束后用  python tools/demo_window.py --restore  还原。")


if __name__ == "__main__":
    main()
