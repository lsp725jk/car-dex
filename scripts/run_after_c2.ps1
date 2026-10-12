# 任务一 - C2（ResNet50@224）训练完成后的自动收尾
#
# 等待 outputs/runs/improved_resnet50_224/summary.json 出现（最长 220 分钟），
# 然后依次执行：评估 → 决策标定 → 候选模型/集成比选 → 重生成 predictions.csv
#              → 汇总实验对比 → 用真值核算 → 提交前自查
#
# 用法：pwsh -File scripts/run_after_c2.ps1

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$py = "C:\ProgramData\anaconda3\python.exe"

$c2Summary = Join-Path $root "outputs\runs\improved_resnet50_224\summary.json"
$deadline = (Get-Date).AddMinutes(220)

Write-Output "[runner] 等待 C2 训练完成：$c2Summary"
while (-not (Test-Path $c2Summary)) {
    if ((Get-Date) -gt $deadline) {
        Write-Output "[runner] 等待超时（220 分钟），C2 尚未完成，改为只对已有模型收尾。"
        break
    }
    Start-Sleep -Seconds 60
}
if (Test-Path $c2Summary) {
    Write-Output "[runner] C2 已完成，开始收尾。"
} else {
    Write-Output "[runner] 未获得 C2 结果。"
}

# 1) C2 评估
if (Test-Path $c2Summary) {
    Write-Output "[runner] === 1/6 评估 improved_resnet50_224 ==="
    & $py src\evaluate.py --checkpoint outputs\checkpoints\improved_resnet50_224\best.pt --tta-hflip
    Write-Output "[runner] === 2/6 决策标定 improved_resnet50_224 ==="
    & $py src\calibrate.py --run improved_resnet50_224 --tta-hflip
}

# 2) 候选比选（含集成）
Write-Output "[runner] === 3/6 候选模型与集成比选 ==="
$cands = @()
foreach ($r in @("improved_resnet18_224", "improved_resnet50_224")) {
    if (Test-Path (Join-Path $root "outputs\checkpoints\$r\best.pt")) { $cands += $r }
}
$ens = @()
if ($cands.Count -ge 2) { $ens += ($cands -join "+") }

if ($cands.Count -gt 0) {
    if ($ens.Count -gt 0) {
        & $py src\finalize.py --candidates @cands --ensembles @ens
    } else {
        & $py src\finalize.py --candidates @cands
    }
}

# 3) 汇总与校验
Write-Output "[runner] === 4/6 汇总实验对比 ==="
& $py src\make_reports.py
Write-Output "[runner] === 5/6 用真值核算 predictions.csv ==="
& $py src\score_predictions.py
Write-Output "[runner] === 6/6 提交前自查 ==="
& $py src\verify_submission.py

Write-Output "[runner] 收尾完成。"
