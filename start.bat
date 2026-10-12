@echo off
rem ============================================================
rem  CarDex - double-click launcher
rem  Opens the Web system by default. Any arguments are passed
rem  straight through to start.ps1, e.g.:
rem      start.bat -Mode check -DeepCheck
rem      start.bat -Port 8000 -NoBrowser
rem      start.bat -Mode cli
rem ============================================================
chcp 65001 >nul 2>nul
setlocal
cd /d "%~dp0"

set "PSEXE="
where pwsh >nul 2>nul && set "PSEXE=pwsh"
if not defined PSEXE if exist "%ProgramFiles%\PowerShell\7\pwsh.exe" set "PSEXE=%ProgramFiles%\PowerShell\7\pwsh.exe"
if not defined PSEXE if exist "%LOCALAPPDATA%\Microsoft\WindowsApps\pwsh.exe" set "PSEXE=%LOCALAPPDATA%\Microsoft\WindowsApps\pwsh.exe"
if not defined PSEXE set "PSEXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

"%PSEXE%" -NoProfile -NoLogo -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" echo [exit code %RC%]
pause
endlocal
