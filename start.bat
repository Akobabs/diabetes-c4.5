@echo off
setlocal
rem Resolve the project from this file, including paths containing spaces.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows_runner.ps1" -Mode Start %*
exit /b %errorlevel%
