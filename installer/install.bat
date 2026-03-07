@echo off
title theSpaceDB Installer

:: ── elevate to admin if needed ───────────────────────────────────────
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo  Requesting admin rights — click Yes to continue...
    powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

:: ── run the PowerShell installer ────────────────────────────────────
:: %~dp0     = folder of this .bat  (i.e. installer\)
:: %~dp0..   = parent folder        (i.e. D:\SpaceDB\)
powershell.exe -NoProfile -ExecutionPolicy Bypass ^
    -File "%~dp0install.ps1" ^
    -SourceDir "%~dp0.." ^
    -InstallDir "%LOCALAPPDATA%\theSpaceDB"
