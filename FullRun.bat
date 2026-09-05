@echo off
setlocal
rem Full research rebuild: no smoke mode and no cached parameter search.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows_runner.ps1" -Mode FullRun %*
exit /b %errorlevel%
