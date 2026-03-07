@echo off
setlocal EnableDelayedExpansion
title theSpaceDB Installer

:: ── resolve full paths BEFORE elevation ─────────────────────────────
:: %~dp0 = installer\ folder, %~dp0.. = D:\SpaceDB\
set "INSTALLER_DIR=%~dp0"
set "SOURCE_DIR=%~dp0.."
set "INSTALL_DIR=%LOCALAPPDATA%\theSpaceDB"
set "PS_SCRIPT=%~dp0install.ps1"

:: ── check if already admin ───────────────────────────────────────────
net session >nul 2>&1
if %errorLevel% equ 0 goto :RUN

:: ── not admin — re-launch elevated with full hardcoded paths ─────────
echo  Requesting administrator rights...
powershell.exe -NoProfile -Command ^
    "Start-Process cmd.exe -ArgumentList '/c \"%~f0\"' -Verb RunAs -Wait"
exit /b

:RUN
:: ── run PowerShell installer ─────────────────────────────────────────
powershell.exe -NoProfile -ExecutionPolicy Bypass ^
    -File "%PS_SCRIPT%" ^
    -SourceDir "%SOURCE_DIR%" ^
    -InstallDir "%INSTALL_DIR%"

:: ── keep window open if PowerShell exits with error ──────────────────
if %errorLevel% neq 0 (
    echo.
    echo  [ERROR] Installer exited with code %errorLevel%
    echo  Check the messages above for details.
    pause
)
