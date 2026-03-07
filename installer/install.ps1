# ═══════════════════════════════════════════════════════════════════
#  theSpaceDB Installer  v0.1.0
#  Blocks in free space. We chase infinity.
# ═══════════════════════════════════════════════════════════════════

param(
    [string]$InstallDir = "$env:LOCALAPPDATA\theSpaceDB",
    [switch]$Unattended = $false
)

$ErrorActionPreference = "Stop"

# ── colours ─────────────────────────────────────────────────────────
function Write-Cyan   { param($m) Write-Host "  $m" -ForegroundColor Cyan    }
function Write-Green  { param($m) Write-Host "  $m" -ForegroundColor Green   }
function Write-Yellow { param($m) Write-Host "  $m" -ForegroundColor Yellow  }
function Write-Red    { param($m) Write-Host "  $m" -ForegroundColor Red     }
function Write-Grey   { param($m) Write-Host "  $m" -ForegroundColor DarkGray}

function Write-Step { param($n, $total, $msg)
    Write-Host ""
    Write-Host "  [$n/$total] " -ForegroundColor DarkGray -NoNewline
    Write-Host $msg -ForegroundColor White
}

# ── banner ───────────────────────────────────────────────────────────
Clear-Host
Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║                                                      ║" -ForegroundColor Cyan
Write-Host "  ║   ∞  theSpaceDB  Installer  v0.1.0                  ║" -ForegroundColor Cyan
Write-Host "  ║      Blocks in free space. We chase infinity.        ║" -ForegroundColor Cyan
Write-Host "  ║                                                      ║" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── confirm install dir ──────────────────────────────────────────────
if (-not $Unattended) {
    Write-Yellow "Install location:  $InstallDir"
    Write-Host ""
    $ans = Read-Host "  Press ENTER to install here, or type a new path"
    if ($ans -ne "") { $InstallDir = $ans.Trim() }
}

# ── STEP 1 — check Python ────────────────────────────────────────────
Write-Step 1 5 "Checking Python 3.11+"

$python = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3\.(\d+)") {
            $minor = [int]$Matches[1]
            if ($minor -ge 11) { $python = $cmd; break }
        }
    } catch {}
}

if (-not $python) {
    Write-Yellow "  Python 3.11+ not found. Attempting to install via winget..."
    try {
        winget install -e --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
        $python = "python"
        Write-Green "  Python installed."
    } catch {
        Write-Red "Could not auto-install Python."
        Write-Red "Please install Python 3.11+ from https://python.org and re-run."
        Read-Host "Press ENTER to exit"
        exit 1
    }
} else {
    $ver = & $python --version 2>&1
    Write-Green "Found: $ver  →  $python"
}

# ── STEP 2 — create install directory ───────────────────────────────
Write-Step 2 5 "Creating install directory"

New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
Write-Green "Directory: $InstallDir"

# ── STEP 3 — create virtual environment ─────────────────────────────
Write-Step 3 5 "Creating virtual environment"

$venv = Join-Path $InstallDir ".venv"
& $python -m venv $venv
$pip    = Join-Path $venv "Scripts\pip.exe"
$pyexe  = Join-Path $venv "Scripts\python.exe"
Write-Green "Virtual environment ready."

# ── STEP 4 — install theSpaceDB ─────────────────────────────────────
Write-Step 4 5 "Installing theSpaceDB + dependencies"

Write-Grey "  Installing numpy, sentence-transformers, scikit-learn..."
& $pip install --quiet --upgrade pip
& $pip install --quiet numpy scikit-learn

Write-Grey "  Installing sentence-transformers (may take a moment)..."
& $pip install --quiet sentence-transformers

Write-Grey "  Installing theSpaceDB from GitHub..."
& $pip install --quiet "git+https://github.com/nihar840/theSpaceDB.git"

Write-Green "Installation complete."

# ── STEP 5 — create launcher + PATH entry ───────────────────────────
Write-Step 5 5 "Creating spacesh launcher + adding to PATH"

# spacesh.bat launcher
$launcherPath = Join-Path $InstallDir "spacesh.bat"
$launcherContent = @"
@echo off
"$pyexe" -m spacedb.shell %*
"@
Set-Content -Path $launcherPath -Value $launcherContent -Encoding ASCII
Write-Green "Launcher: $launcherPath"

# Add InstallDir to user PATH (if not already there)
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($userPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("PATH", "$userPath;$InstallDir", "User")
    Write-Green "Added to PATH."
} else {
    Write-Grey "Already in PATH."
}

# Desktop shortcut
$desktopPath = [System.IO.Path]::Combine([Environment]::GetFolderPath("Desktop"), "spacesh.lnk")
$wsh    = New-Object -ComObject WScript.Shell
$link   = $wsh.CreateShortcut($desktopPath)
$link.TargetPath       = "cmd.exe"
$link.Arguments        = "/k `"$launcherPath`""
$link.WorkingDirectory = $env:USERPROFILE
$link.Description      = "theSpaceDB Shell - spacesh"
$link.IconLocation     = "cmd.exe"
$link.Save()
Write-Green "Desktop shortcut created."

# ── done ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "  ║                                                      ║" -ForegroundColor Green
Write-Host "  ║   ✓  theSpaceDB installed successfully!              ║" -ForegroundColor Green
Write-Host "  ║                                                      ║" -ForegroundColor Green
Write-Host "  ║   Run anywhere:   spacesh <data-folder>              ║" -ForegroundColor Green
Write-Host "  ║   Desktop:        double-click 'spacesh' shortcut    ║" -ForegroundColor Green
Write-Host "  ║                                                      ║" -ForegroundColor Green
Write-Host "  ║   We chase infinity.  ∞                              ║" -ForegroundColor Green
Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

if (-not $Unattended) {
    Read-Host "  Press ENTER to close"
}
