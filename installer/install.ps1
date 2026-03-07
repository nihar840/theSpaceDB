# theSpaceDB Installer - Cool Edition
# Run via install.bat only

param(
    [string]$SourceDir  = (Split-Path $PSScriptRoot -Parent),
    [string]$InstallDir = "$env:LOCALAPPDATA\theSpaceDB",
    [switch]$Unattended
)

$ErrorActionPreference = "Stop"

# Top-level crash guard - window never silently closes
trap {
    Write-Host ""
    Write-Host "  +------------------------------------------+" -ForegroundColor Red
    Write-Host "  |  INSTALLER CRASHED - Details below:      |" -ForegroundColor Red
    Write-Host "  +------------------------------------------+" -ForegroundColor Red
    Write-Host ""
    Write-Host "  ERROR: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "  SourceDir : $SourceDir"  -ForegroundColor DarkGray
    Write-Host "  InstallDir: $InstallDir" -ForegroundColor DarkGray
    Write-Host ""
    Read-Host "  Press ENTER to close"
    exit 1
}

$Host.UI.RawUI.WindowTitle = "theSpaceDB Installer"

# ── helpers ──────────────────────────────────────────────────────────

function Ln  { Write-Host "" }

function C($color, $text) {
    Write-Host $text -ForegroundColor $color -NoNewline
}

function ProgressBar($percent, $width = 44) {
    $filled = [int]($percent / 100 * $width)
    $empty  = $width - $filled
    $bar    = ("=" * $filled) + ("-" * $empty)
    Write-Host "`r  [$bar] $percent%" -NoNewline
}

function FakeProgress($from, $to, $ms = 600) {
    $steps = [Math]::Max(1, $to - $from)
    $delay = [int]($ms / $steps)
    for ($i = $from; $i -le $to; $i++) {
        ProgressBar $i
        Start-Sleep -Milliseconds $delay
    }
}

function Step($n, $msg) {
    Ln
    C Yellow "  >> Step $n "; C White $msg
    Ln
}

function OK($msg)   { C Green  "  [DONE] $msg";  Ln }
function WARN($msg) { C Yellow "  [WARN] $msg";  Ln }
function ERR($msg)  { C Red    "  [FAIL] $msg";  Ln }
function INFO($msg) { C DarkGray "         $msg"; Ln }

function RunCmd($cmd, $argList, $desc) {
    Ln
    $result = & $cmd @argList 2>&1
    $code   = $LASTEXITCODE
    if ($code -ne 0) {
        Ln
        ERR "$desc failed (exit code: $code)"
        Ln
        $result | Select-Object -Last 10 | ForEach-Object {
            Write-Host "     $_" -ForegroundColor DarkGray
        }
        Ln
        Read-Host "  Press ENTER to close"
        exit 1
    }
}

$jokes = @(
    "Convincing electrons to behave...",
    "Downloading more RAM... just kidding.",
    "Teaching blocks to float in space...",
    "Negotiating with numpy. It is complicated.",
    "Installing the soul of the DB (W matrix)...",
    "Asking Python to not break things for once...",
    "Manifesting intelligence from pure chaos...",
    "Making sure the installer does not judge you...",
    "Almost there. Or are we? Infinity is relative.",
    "Your RAM signed a waiver. We are good."
)

function Joke {
    $j = $jokes[(Get-Random -Maximum $jokes.Count)]
    INFO $j
}

# ═════════════════════════════════════════════════════════════════════
# BANNER
# ═════════════════════════════════════════════════════════════════════
Clear-Host
Ln
C Cyan "  +============================================================+"; Ln
C Cyan "  |                                                            |"; Ln
C Cyan "  |    [inf]  theSpaceDB  v0.1.0  Installer                   |"; Ln
C Cyan "  |           Blocks in free space. We chase infinity.        |"; Ln
C Cyan "  |                                                            |"; Ln
C Cyan "  +============================================================+"; Ln
Ln
C White  "  Welcome, human."; Ln
INFO "You are about to install a database that thinks."
INFO "Blocks drift. Clusters emerge. Personalities form."
INFO "Your RAM may never be the same again."
Ln

if (-not $Unattended) {
    C DarkGray "  Install to : "; C Cyan $InstallDir; Ln
    C DarkGray "  Source     : "; C Cyan $SourceDir;  Ln
    Ln
    C White "  Press ENTER to begin (or Ctrl+C to escape reality)  "
    Read-Host | Out-Null
}

Ln
C Cyan "  ------------------------------------------------------------"; Ln

# ═════════════════════════════════════════════════════════════════════
# STEP 1 - Python
# ═════════════════════════════════════════════════════════════════════
Step "1/5" "Checking Python 3.11+"
Joke

$python = $null
foreach ($cmd in @("python", "python3")) {
    try {
        $v = & $cmd --version 2>&1
        if ("$v" -match "Python 3\.(\d+)" -and [int]$Matches[1] -ge 11) {
            $python = $cmd
            break
        }
    } catch {}
}

# Also try py launcher
if (-not $python) {
    try {
        $v = & py -3 --version 2>&1
        if ("$v" -match "Python 3\.(\d+)" -and [int]$Matches[1] -ge 11) {
            $python = "py"
        }
    } catch {}
}

if ($python) {
    $ver = & $python --version 2>&1
    OK "$ver found. Python is cooperating today."
    FakeProgress 0 20 400
} else {
    WARN "Python 3.11+ not found. Trying winget install..."
    try {
        winget install -e --id Python.Python.3.11 --silent `
            --accept-package-agreements --accept-source-agreements | Out-Null
        $python = "python"
        OK "Python installed via winget."
    } catch {
        ERR "Could not auto-install Python."
        INFO "Please install Python 3.11+ from https://python.org"
        INFO "Check 'Add Python to PATH' during install, then re-run."
        Read-Host "`n  Press ENTER to exit"
        exit 1
    }
}

Ln

# ═════════════════════════════════════════════════════════════════════
# STEP 2 - Directory
# ═════════════════════════════════════════════════════════════════════
Step "2/5" "Preparing the space"
FakeProgress 20 35 300
Ln

New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
OK "Directory ready: $InstallDir"
INFO "This is where your blocks will live. Treat it well."
Ln

# ═════════════════════════════════════════════════════════════════════
# STEP 3 - Virtual environment
# ═════════════════════════════════════════════════════════════════════
Step "3/5" "Creating virtual environment"
Joke
FakeProgress 35 50 400
Ln

$venv  = Join-Path $InstallDir ".venv"
$pip   = Join-Path $venv "Scripts\pip.exe"
$pyexe = Join-Path $venv "Scripts\python.exe"

if ($python -eq "py") {
    RunCmd "py" @("-3", "-m", "venv", $venv) "Virtual environment creation"
} else {
    RunCmd $python @("-m", "venv", $venv) "Virtual environment creation"
}

OK "Virtual environment ready. A clean mind deserves a clean env."
Ln

# ═════════════════════════════════════════════════════════════════════
# STEP 4 - Dependencies
# ═════════════════════════════════════════════════════════════════════
Step "4/5" "Installing dependencies"
INFO "This may take 2-3 minutes. Go drink some water. Seriously."
Ln

FakeProgress 50 55 300
Ln
INFO "Upgrading pip (it has trust issues with old versions)..."
RunCmd $pyexe @("-m", "pip", "install", "--quiet", "--upgrade", "pip", "setuptools", "wheel") "pip upgrade"
OK "pip upgraded."

FakeProgress 55 63 300
Ln
INFO "Installing numpy (the math guy)..."
RunCmd $pyexe @("-m", "pip", "install", "--quiet", "numpy>=1.26") "numpy install"
OK "numpy installed."

FakeProgress 63 70 300
Ln
INFO "Installing scikit-learn (the clustering wizard)..."
RunCmd $pyexe @("-m", "pip", "install", "--quiet", "scikit-learn>=1.4") "scikit-learn install"
OK "scikit-learn installed."

FakeProgress 70 83 500
Ln
INFO "Installing sentence-transformers (the brain that reads text)..."
INFO "This one is chunky. Be patient."
RunCmd $pyexe @("-m", "pip", "install", "--quiet", "sentence-transformers>=2.7") "sentence-transformers install"
OK "sentence-transformers installed. Your DB now understands human."

FakeProgress 83 93 400
Ln
INFO "Installing theSpaceDB from: $SourceDir"
RunCmd $pyexe @("-m", "pip", "install", "--quiet", $SourceDir) "theSpaceDB install"
OK "theSpaceDB installed. The space is ready."
Ln

# ═════════════════════════════════════════════════════════════════════
# STEP 5 - Launcher + PATH
# ═════════════════════════════════════════════════════════════════════
Step "5/5" "Creating spacesh launcher"
FakeProgress 93 99 300
Ln

# spacesh.bat launcher
$launcherPath = Join-Path $InstallDir "spacesh.bat"
$launcherContent = "@echo off`r`n`"$pyexe`" -m spacedb.shell %*"
[System.IO.File]::WriteAllText($launcherPath, $launcherContent, [System.Text.Encoding]::ASCII)
OK "Launcher: $launcherPath"

# Add to user PATH
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($userPath -notlike "*$InstallDir*") {
    $newPath = $userPath + ";" + $InstallDir
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    OK "Added spacesh to PATH. Restart your terminal after install."
} else {
    OK "Already in PATH."
}

# Desktop shortcut
try {
    $desk = [Environment]::GetFolderPath("Desktop")
    $lnk  = Join-Path $desk "spacesh.lnk"
    $wsh  = New-Object -ComObject WScript.Shell
    $link = $wsh.CreateShortcut($lnk)
    $link.TargetPath       = "cmd.exe"
    $link.Arguments        = "/k spacesh"
    $link.WorkingDirectory = $env:USERPROFILE
    $link.Description      = "theSpaceDB Shell"
    $link.Save()
    OK "Desktop shortcut created."
} catch {
    WARN "Could not create desktop shortcut (not critical)."
}

FakeProgress 99 100 200
Ln; Ln

# ═════════════════════════════════════════════════════════════════════
# DONE
# ═════════════════════════════════════════════════════════════════════
C Green "  +============================================================+"; Ln
C Green "  |                                                            |"; Ln
C Green "  |  DONE!  theSpaceDB installed successfully.                |"; Ln
C Green "  |                                                            |"; Ln
C Green "  |  1. Open a NEW terminal (important!)                      |"; Ln
C Green "  |  2. Type:  spacesh C:\my_mind                             |"; Ln
C Green "  |     Or double-click 'spacesh' on your Desktop.            |"; Ln
C Green "  |                                                            |"; Ln
C Green "  |  Nothing fixed. Everything evolving.  [inf]               |"; Ln
C Green "  |                                                            |"; Ln
C Green "  +============================================================+"; Ln
Ln

if (-not $Unattended) {
    Read-Host "  Press ENTER to close"
}
