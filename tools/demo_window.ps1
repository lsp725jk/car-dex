# ============================================================================
# CarDex 录屏辅助 —— 把浏览器窗口置顶 / 固定位置与尺寸
#
# 为什么要它：录全屏时只要有别的窗口盖上来，整段素材就废了。
# 浏览器自己没法把自己置顶（安全限制），必须由操作系统来做。
#
# 用法（PowerShell 5.1 或 7 均可）：
#   .\tools\demo_window.ps1                       # 置顶 + 定位到 (0,0) 1280x800
#   .\tools\demo_window.ps1 -Width 1600 -Height 900 -X 100 -Y 0
#   .\tools\demo_window.ps1 -Borderless           # 去掉标题栏，画面更干净
#   .\tools\demo_window.ps1 -List                 # 只列出候选窗口，不改动
#   .\tools\demo_window.ps1 -Restore              # 取消置顶并还原边框
#
# 注意：脚本用 Add-Type 调 Win32 API，必须在 FullLanguage 模式下运行。
# ============================================================================
[CmdletBinding()]
param(
    [string] $TitleMatch = '识车档案|CarDex',
    [int]    $Width  = 1280,
    [int]    $Height = 800,
    [int]    $X = 0,
    [int]    $Y = 0,
    [switch] $Borderless,
    [switch] $Restore,
    [switch] $List
)

Add-Type -AssemblyName System.Windows.Forms

Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public class Win32Demo {
    public delegate bool EnumProc(IntPtr hWnd, IntPtr lParam);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb, IntPtr p);
    [DllImport("user32.dll")] public static extern int  GetWindowTextLength(IntPtr h);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)]
    public static extern int  GetWindowText(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll")] public static extern bool SetWindowPos(
        IntPtr h, IntPtr after, int x, int y, int cx, int cy, uint flags);
    [DllImport("user32.dll")] public static extern int  GetWindowLong(IntPtr h, int i);
    [DllImport("user32.dll")] public static extern int  SetWindowLong(IntPtr h, int i, int v);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left, Top, Right, Bottom; }

    /// 回读窗口实况。只打印「已置顶」而不回读，等于没有验证 ——
    /// SetWindowPos 可能因权限或句柄失效静默失败。
    ///
    /// ⚠️ 本方法刻意**只用 ASCII**：C# 代码块里有中文时，只要 .ps1 丢了 BOM，
    ///    PowerShell 5.1 就会按 ANSI 解码，中文变乱码进而引发一串
    ///    「意外的字符 / 常量中有换行符」的编译错误。中文标签改到
    ///    PowerShell 侧拼接，这样无论文件编码如何都能跑。
    public static string Title(IntPtr h) {
        int n = GetWindowTextLength(h);
        var sb = new StringBuilder(n + 2);
        GetWindowText(h, sb, sb.Capacity);
        return sb.ToString();
    }
    public static int[] Rect(IntPtr h) {
        RECT r; GetWindowRect(h, out r);
        return new int[] { r.Left, r.Top, r.Right - r.Left, r.Bottom - r.Top };
    }
    public static bool IsTopmost(IntPtr h) { return (GetWindowLong(h, -20) & 0x8) != 0; }
    public static bool HasCaption(IntPtr h) { return (GetWindowLong(h, -16) & 0xC00000) != 0; }
    public static long Handle(IntPtr h) { return h.ToInt64(); }

    public static IntPtr Find(string pattern) { return FindBest(pattern); }

    /// 在所有标题匹配的窗口里挑出**真正该置顶的那一个**。
    ///
    /// 这里踩过一个很隐蔽的坑：EnumWindows 按 z 序返回，
    /// 直接取第一个匹配会抓到别的窗口。实测中标题含「识车档案」的
    /// 除了浏览器，还有 WPS 打开的《识车档案CarDex_实训报告_v2.docx》，
    /// 而最小化的 WPS 窗口位置是 (-32000,-32000)、尺寸 237x39 ——
    /// 于是脚本一直"成功"地把一个看不见的窗口置了顶，用户当然看不到效果。
    ///
    /// 判据：排除最小化 / 排除过小窗口 / 浏览器标题优先 / 面积大的优先。
    public static IntPtr FindBest(string pattern) {
        IntPtr best = IntPtr.Zero;
        int bestScore = -1;
        var rx = new System.Text.RegularExpressions.Regex(pattern);
        var browser = new System.Text.RegularExpressions.Regex(
            "Edge|Chrome|Firefox|Brave|Opera|Chromium");
        EnumWindows(delegate(IntPtr h, IntPtr l) {
            if (!IsWindowVisible(h)) return true;
            int n = GetWindowTextLength(h);
            if (n == 0) return true;
            var sb = new StringBuilder(n + 2);
            GetWindowText(h, sb, sb.Capacity);
            string t = sb.ToString();
            if (!rx.IsMatch(t)) return true;

            RECT r; GetWindowRect(h, out r);
            int w = r.Right - r.Left, ht = r.Bottom - r.Top;
            if (r.Left <= -30000 || r.Top <= -30000) return true;  // 最小化
            if (w < 400 || ht < 300) return true;                  // 太小
            int score = w * ht / 1000;
            if (browser.IsMatch(t)) score += 1000000;              // 浏览器优先
            if (score > bestScore) { bestScore = score; best = h; }
            return true;
        }, IntPtr.Zero);
        return best;
    }

    public static string[] Titles() {
        var list = new System.Collections.Generic.List<string>();
        EnumWindows(delegate(IntPtr h, IntPtr l) {
            if (!IsWindowVisible(h)) return true;
            int n = GetWindowTextLength(h);
            if (n == 0) return true;
            var sb = new StringBuilder(n + 2);
            GetWindowText(h, sb, sb.Capacity);
            list.Add(sb.ToString());
            return true;
        }, IntPtr.Zero);
        return list.ToArray();
    }
}
'@

[void][Win32Demo]::SetProcessDPIAware()

if ($List) {
    Write-Host '当前可见窗口：' -ForegroundColor Cyan
    [Win32Demo]::Titles() | Where-Object { $_.Trim() } | Sort-Object -Unique |
        ForEach-Object { Write-Host "  $_" }
    return
}

$hwnd = [Win32Demo]::Find($TitleMatch)
if ($hwnd -eq [IntPtr]::Zero) {
    Write-Warning "没找到标题匹配 «$TitleMatch» 的窗口。请确认浏览器已打开演示页面；"
    Write-Warning "或者先跑 .\tools\demo_window.ps1 -List 看看实际标题。"
    return
}

$GWL_STYLE   = -16
$WS_CAPTION  = 0x00C00000
$HWND_TOPMOST   = [IntPtr](-1)
$HWND_NOTOPMOST = [IntPtr](-2)
$SWP_SHOWWINDOW = 0x0040
$SWP_FRAMECHANGED = 0x0020

$style = [Win32Demo]::GetWindowLong($hwnd, $GWL_STYLE)

function Show-WindowState([IntPtr]$h, [string]$label) {
    $r = [Win32Demo]::Rect($h)
    $top = if ([Win32Demo]::IsTopmost($h)) { '是 (OK)' } else { '否 (未生效!)' }
    $cap = if ([Win32Demo]::HasCaption($h)) { '有' } else { '无（已去边框）' }
    Write-Host "  $label"
    Write-Host "    标题   : $([Win32Demo]::Title($h))"
    Write-Host "    位置   : ($($r[0]), $($r[1]))"
    Write-Host "    尺寸   : $($r[2]) x $($r[3])"
    Write-Host "    置顶   : $top"
    Write-Host "    标题栏 : $cap"
    Write-Host ("    句柄   : 0x{0:X}" -f [Win32Demo]::Handle($h))
}

if ($Restore) {
    [void][Win32Demo]::SetWindowLong($hwnd, $GWL_STYLE, $style -bor $WS_CAPTION)
    [void][Win32Demo]::SetWindowPos($hwnd, $HWND_NOTOPMOST, $X, $Y, $Width, $Height,
        $SWP_SHOWWINDOW -bor $SWP_FRAMECHANGED)
    Write-Host '已取消置顶并还原标题栏。回读确认：' -ForegroundColor Yellow
    Show-WindowState $hwnd '当前实况'
    return
}

if ($Borderless) {
    # 去掉标题栏。⚠️ 去掉后窗口就拖不动了，务必用 -Restore 还原。
    [void][Win32Demo]::SetWindowLong($hwnd, $GWL_STYLE, $style -band (-bnot $WS_CAPTION))
    $flags = $SWP_SHOWWINDOW -bor $SWP_FRAMECHANGED
} else {
    $flags = $SWP_SHOWWINDOW
}

[void][Win32Demo]::SetWindowPos($hwnd, $HWND_TOPMOST, $X, $Y, $Width, $Height, $flags)
[void][Win32Demo]::SetForegroundWindow($hwnd)

Write-Host '窗口已置顶。回读确认（这才是真正生效的证据）：' -ForegroundColor Green
Show-WindowState $hwnd '当前实况'
Write-Host ''
Write-Host '录制结束后用  .\tools\demo_window.ps1 -Restore  还原。' -ForegroundColor DarkGray
