#Requires -Version 5.1
<#
.SYNOPSIS
    识车档案 CarDex —— 一键启动脚本

.DESCRIPTION
    做四件事，然后把系统跑起来：
      1. 找到可用的 Python 解释器（虚拟环境 > Anaconda > PATH）
      2. 确认 API Key 已配置（没有就当场输入一次，写进被 gitignore 忽略的 agent/.env.local）
      3. 跑一遍环境自检（依赖 / 交付文件 / 模型权重 / 工具装配）
      4. 启动 Web 系统或命令行智能体

    密钥**只会**写进 agent/.env.local 或环境变量，脚本本身不含任何 Key，
    控制台只显示掩码（如 sk-a…9f2c）。

.PARAMETER Mode
    web   —— 启动任务三的 Web 系统（默认）
    cli   —— 启动任务二的命令行智能体
    check —— 只做环境自检，不启动

.PARAMETER Port
    Web 端口，默认 5000；被占用时自动顺延。

.PARAMETER Offline
    强制离线替身模式（不调用真实模型，用于无网/无 Key 演示工具链）。

.PARAMETER DeepCheck
    自检时额外做「真实 API 调用」与「真实视觉推理」两项实测。

.PARAMETER Install
    启动前执行 pip install -r requirements.txt。

.PARAMETER Image / .PARAMETER Question
    cli 模式下：先指定一张图片 / 只问一次就退出。

.EXAMPLE
    .\start.ps1
    双击「启动识车档案.bat」等价于此。

.EXAMPLE
    .\start.ps1 -Mode check -DeepCheck
    只做全量自检（含真实 API 与视觉推理实测）。

.EXAMPLE
    .\start.ps1 -Port 8000 -NoBrowser
    换端口启动且不自动打开浏览器。
#>
[CmdletBinding()]
param(
    [ValidateSet('web', 'cli', 'check')]
    [string]$Mode = 'web',

    [string]$BindHost = '127.0.0.1',

    [int]$Port = 5000,

    [switch]$Offline,
    [switch]$DeepCheck,
    [switch]$NoCheck,
    [switch]$NoBrowser,
    [switch]$Install,

    [string]$Image,
    [string]$Question
)

# 注意：这里**不能**用 'Stop'。脚本要反复调用 python.exe，而 Windows PowerShell 5.1
# 会把外部程序写到 stderr 的每一行都当成错误记录，一旦设成 Stop，
# Python 的一条普通警告就会把整个启动流程打断。统一用 Continue，
# 关键步骤靠显式的 $LASTEXITCODE / -ErrorAction Stop 判断。
$ErrorActionPreference = 'Continue'

# ---- 控制台按 UTF-8 输出，保证中文与框线字符正常显示 ----
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch { }
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

if (-not $PSScriptRoot) { $PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path }
$Root = $PSScriptRoot
Set-Location $Root

# --------------------------------------------------------------------------- #
# 输出小工具
# --------------------------------------------------------------------------- #
function Write-Head($t) { Write-Host "`n--- $t" -ForegroundColor Cyan }
function Write-Ok($t) { Write-Host "  [OK] $t" -ForegroundColor Green }
function Write-Msg($t) { Write-Host "  [--] $t" -ForegroundColor Gray }
function Write-Note($t) { Write-Host "  [!]  $t" -ForegroundColor Yellow }
function Write-Bad($t) { Write-Host "  [X]  $t" -ForegroundColor Red }

function Write-Banner {
    Write-Host ''
    Write-Host '============================================================' -ForegroundColor DarkCyan
    Write-Host '   识车档案 CarDex  ·  视觉智能体综合实训' -ForegroundColor White
    Write-Host '   任务一 视觉分类模型  |  任务二 智能体工具调用  |  任务三 Web 系统' -ForegroundColor DarkGray
    Write-Host '============================================================' -ForegroundColor DarkCyan
}

# --------------------------------------------------------------------------- #
# 1. 找 Python
# --------------------------------------------------------------------------- #
function Get-PythonExe {
    $cands = New-Object System.Collections.Generic.List[string]

    $venv = Join-Path $Root '.venv\Scripts\python.exe'
    if (Test-Path $venv) { $cands.Add($venv) }

    foreach ($p in @(
            'C:\ProgramData\anaconda3\python.exe',
            (Join-Path $env:USERPROFILE 'anaconda3\python.exe'),
            (Join-Path $env:USERPROFILE 'miniconda3\python.exe'),
            (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'),
            (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
            (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'),
            (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310\python.exe')
        )) {
        if ($p -and (Test-Path $p)) { $cands.Add($p) }
    }

    $cmd = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { $cands.Add($cmd.Source) }

    $seen = @{}
    $fallback = $null
    foreach ($c in $cands) {
        if ($seen.ContainsKey($c)) { continue }
        $seen[$c] = $true
        if (-not $fallback) { $fallback = $c }
        # 首选「依赖齐全」的解释器。
        # 传给外部程序的 Python 代码一律用「PowerShell 双引号 + Python 单引号」：
        # PowerShell 5.1 拼装命令行时会吃掉 PS 单引号串里的双引号。
        & $c -c 'import torch, flask, requests, PIL' 2>$null
        if ($LASTEXITCODE -eq 0) { return $c }
    }
    return $fallback
}

# --------------------------------------------------------------------------- #
# 2. API Key
# --------------------------------------------------------------------------- #
function Read-DotEnv {
    $f = Join-Path $Root 'agent\.env.local'
    $out = @{}
    if (Test-Path $f) {
        foreach ($line in (Get-Content -LiteralPath $f -Encoding UTF8)) {
            $s = $line.Trim()
            if (-not $s -or $s.StartsWith('#') -or -not $s.Contains('=')) { continue }
            $i = $s.IndexOf('=')
            $out[$s.Substring(0, $i).Trim()] = $s.Substring($i + 1).Trim().Trim('"').Trim("'")
        }
    }
    return $out
}

function Save-DotEnvKey([string]$Key) {
    $f = Join-Path $Root 'agent\.env.local'
    $kept = New-Object System.Collections.Generic.List[string]
    if (Test-Path $f) {
        foreach ($line in (Get-Content -LiteralPath $f -Encoding UTF8)) {
            $s = $line.Trim()
            if ($s -match '^\s*DEEPSEEK_API_KEY\s*=') { continue }   # 旧值丢掉
            $kept.Add($line)
        }
    } else {
        $kept.Add('# 识车档案 CarDex —— 本地密钥配置（已被 .gitignore 忽略，不随作业提交）')
        $kept.Add('# 真实 Key 只允许出现在本文件或环境变量中，绝不写进代码 / 报告 / 截图。')
        $kept.Add('')
    }
    $kept.Add("DEEPSEEK_API_KEY=$Key")
    # 无 BOM 的 UTF-8，避免读取端把 BOM 当成密钥首字符
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($f, ($kept -join [Environment]::NewLine) + [Environment]::NewLine, $utf8)
}

function Resolve-ApiKey {
    $envKey = ($env:DEEPSEEK_API_KEY | ForEach-Object { "$_".Trim() })
    if ($envKey) {
        Write-Ok "API Key 来自环境变量 DEEPSEEK_API_KEY（$($envKey.Substring(0,4))...$($envKey.Substring($envKey.Length-4))）"
        return $true
    }
    $local = Read-DotEnv
    if ($local['DEEPSEEK_API_KEY']) {
        $k = $local['DEEPSEEK_API_KEY']
        Write-Ok "API Key 来自 agent/.env.local（$($k.Substring(0,4))...$($k.Substring($k.Length-4))）"
        return $true
    }

    Write-Note '未检测到 DeepSeek API Key。'
    Write-Msg  '直接回车 = 用离线替身模式演示（工具链可跑通，文字由脚本化替身生成，不代表模型效果）'
    Write-Msg  '申请地址：https://platform.deepseek.com/'
    $typed = Read-Host '  请粘贴 API Key'

    if (-not $typed -or -not $typed.Trim()) {
        $script:ForceOffline = $true
        Write-Note '使用离线替身模式。'
        return $false
    }
    $k = $typed.Trim()
    if (-not $k.StartsWith('sk-') -or $k.Length -lt 20) {
        Write-Bad '格式不像 DeepSeek Key（应以 sk- 开头）。已改用离线替身模式。'
        $script:ForceOffline = $true
        return $false
    }
    Save-DotEnvKey $k
    $env:DEEPSEEK_API_KEY = $k
    Write-Ok "已写入 agent/.env.local（已 gitignore，不会随作业提交）：$($k.Substring(0,4))...$($k.Substring($k.Length-4))"
    return $true
}

# --------------------------------------------------------------------------- #
# 3. 端口
# --------------------------------------------------------------------------- #
function Test-PortBusy([int]$p) {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $iar = $client.BeginConnect('127.0.0.1', $p, $null, $null)
        $reached = $iar.AsyncWaitHandle.WaitOne(400, $false)
        $busy = $false
        if ($reached -and $client.Connected) { $busy = $true }
        $client.Close()
        return $busy
    } catch { return $false }
}

function Find-FreePort([int]$start) {
    for ($p = $start; $p -lt ($start + 20); $p++) {
        if (-not (Test-PortBusy $p)) { return $p }
    }
    return $start
}

# =========================================================================== #
#  main
# =========================================================================== #
Write-Banner

# ---- Python ----
Write-Head 'Python 解释器'
$Python = Get-PythonExe
if (-not $Python) {
    Write-Bad '找不到任何 Python 解释器。'
    Write-Msg '请安装 Python 3.10+（推荐 Anaconda），或在项目根目录创建 .venv。'
    Write-Msg '下载：https://www.python.org/downloads/'
    exit 1
}
$pyVer = (& $Python -c "import sys;print('%d.%d.%d'%sys.version_info[:3])" 2>$null)
if (-not $pyVer) { $pyVer = '未知版本' }
Write-Ok "使用 $Python （Python $pyVer）"

# ---- 依赖 ----
if ($Install) {
    Write-Head '安装依赖'
    & $Python -m pip install -r (Join-Path $Root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { Write-Bad 'pip install 失败，请检查网络或改用国内镜像源。'; exit 1 }
    Write-Ok '依赖安装完成'
}

# ---- Key ----
Write-Head 'API Key'
$script:ForceOffline = $false
$hasKey = Resolve-ApiKey

# ---- 自检 ----
if (-not $NoCheck) {
    Write-Head '环境自检'
    $scArgs = @((Join-Path $Root 'tools\selfcheck.py'))
    if ($DeepCheck -or $Mode -eq 'check') { $scArgs += '--all' }
    & $Python @scArgs
    $sc = $LASTEXITCODE
    if ($sc -ne 0) {
        Write-Bad '自检未通过。按上面的 [FAIL] 提示修复后重试；'
        Write-Msg  '若只是缺依赖，可用：  .\start.ps1 -Install'
        exit 1
    }
}

if ($Mode -eq 'check') {
    Write-Host ''
    Write-Ok '自检完成（-Mode check，不启动服务）'
    exit 0
}

$useOffline = $Offline -or $script:ForceOffline

# ---- 启动 ----
if ($Mode -eq 'cli') {
    Write-Head '启动命令行智能体'
    $cliArgs = @('-m', 'agent.cli')
    if ($useOffline) { $cliArgs += '--offline' }
    if ($Image) { $cliArgs += @('--image', $Image) }
    if ($Question) { $cliArgs += @('--once', $Question) }
    Write-Msg ("$Python " + ($cliArgs -join ' '))
    & $Python @cliArgs
    exit $LASTEXITCODE
}

# web
Write-Head '启动 Web 系统'
$port = Find-FreePort $Port
if ($port -ne $Port) {
    Write-Note "端口 $Port 已被占用，自动改用 $port"
}
$url = "http://${BindHost}:${port}/"

$webArgs = @((Join-Path $Root 'web\app.py'), '--host', $BindHost, '--port', "$port")
if ($useOffline) { $webArgs += '--offline' }

Write-Host ''
Write-Host '============================================================' -ForegroundColor DarkCyan
Write-Host "   访问地址 : $url" -ForegroundColor White
Write-Host "   运行模式 : $(if ($useOffline) { '离线替身（不代表模型效果）' } else { '真实 DeepSeek API' })" -ForegroundColor White
Write-Host '   停止服务 : 在本窗口按 Ctrl+C' -ForegroundColor DarkGray
Write-Host '============================================================' -ForegroundColor DarkCyan
Write-Host ''

if (-not $NoBrowser) {
    # 延迟 3 秒再开浏览器，避免抢在服务起来之前访问
    try {
        $cmdLine = "timeout /t 3 /nobreak >nul & start `"`" `"$url`""
        Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', $cmdLine -WindowStyle Hidden -ErrorAction Stop
    } catch {
        Write-Note "未能自动打开浏览器，请手动访问 $url"
    }
}

& $Python @webArgs
