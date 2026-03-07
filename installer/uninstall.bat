@echo off
:: theSpaceDB Uninstaller

net session >nul 2>&1
if %errorLevel% neq 0 (
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
    "$dir = '$env:LOCALAPPDATA\theSpaceDB'; ^
     if (Test-Path $dir) { ^
        Remove-Item $dir -Recurse -Force; ^
        $p = [Environment]::GetEnvironmentVariable('PATH','User'); ^
        $p = ($p -split ';' | Where-Object { $_ -notlike '*theSpaceDB*' }) -join ';'; ^
        [Environment]::SetEnvironmentVariable('PATH', $p, 'User'); ^
        $d = [Environment]::GetFolderPath('Desktop') + '\spacesh.lnk'; ^
        if (Test-Path $d) { Remove-Item $d -Force }; ^
        Write-Host 'theSpaceDB uninstalled.' -ForegroundColor Green ^
     } else { Write-Host 'theSpaceDB not found.' -ForegroundColor Yellow }; ^
     Read-Host 'Press ENTER to close'"
