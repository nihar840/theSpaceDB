@echo off
:: ═══════════════════════════════════════════════════════
::  theSpaceDB Installer
::  Double-click this file to install.
:: ═══════════════════════════════════════════════════════

:: Check for admin rights — elevate if needed
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:: Run the PowerShell installer
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
