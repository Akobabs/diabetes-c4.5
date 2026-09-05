@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows_runner.ps1" -Mode Robustness %*
exit /b %errorlevel%
