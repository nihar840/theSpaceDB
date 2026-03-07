# theSpaceDB Installer — Cool Edition
# Run via install.bat (do not run directly)

param(
    [string]$SourceDir   = (Split-Path $PSScriptRoot -Parent),
    [string]$InstallDir  = "$env:LOCALAPPDATA\theSpaceDB",
    [switch]$Unattended
)

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "theSpaceDB Installer"

# ── helpers ──────────────────────────────────────────────────────────

function C($color, $text) {
    Write-Host $text -ForegroundColor $color -NoNewline
}

function Ln { Write-Host "" }

function Banner {
    Clear-Host
    Ln
    C Cyan    "  ╔══════════════════════════════════════════════════════════╗"; Ln
    C Cyan    "  ║                                                          ║"; Ln
    C Cyan    "  ║  "; C Yellow "∞"; C White "  theSpaceDB  v0.1.0  Installer"; C Cyan "                        ║"; Ln
    C Cyan    "  ║  "; C DarkGray "   Blocks in free space. We chase infinity."; C Cyan "          ║"; Ln
    C Cyan    "  ║                                                          ║"; Ln
    C Cyan    "  ╚══════════════════════════════════════════════════════════╝"; Ln
    Ln
}

function ProgressBar($percent, $width = 40) {
    $filled = [int]($percent / 100 * $width)
    $empty  = $width - $filled
    $bar    = ("█" * $filled) + ("░" * $empty)
    Write-Host "`r  [" -NoNewline
    Write-Host $bar -ForegroundColor Cyan -NoNewline
    Write-Host "]  $percent%" -NoNewline
}

function Step($icon, $msg) {
    Ln
    C Yellow "  $icon "; C White $msg; Ln
}

function OK($msg)   { C Green  "    ✔  $msg"; Ln }
function WARN($msg) { C Yellow "    ⚠  $msg"; Ln }
function ERR($msg)  { C Red    "    ✘  $msg"; Ln }
function INFO($msg) { C DarkGray "       $msg"; Ln }

function FakeProgress($from, $to, $ms = 600) {
    $steps = $to - $from
    $delay = [int]($ms / $steps)
    for ($i = $from; $i -le $to; $i++) {
        ProgressBar $i
        Start-Sleep -Milliseconds $delay
    }
}

function RunCmd($cmd, $args, $desc) {
    $result = & $cmd @args 2>&1
    if ($LASTEXITCODE -ne 0) {
        ERR "$desc failed."
        INFO ($result | Select-Object -Last 5 | Out-String)
        Read-Host "`n  Press ENTER to exit"
        exit 1
    }
}

# ── funny loading lines ───────────────────────────────────────────────
$loadingLines = @(
    "Convincing electrons to behave...",
    "Downloading more RAM... just kidding.",
    "Teaching blocks to float in space...",
    "Negotiating with numpy. It's complicated.",
    "Installing the soul of the database (W matrix)...",
    "Spinning up cognitive drift engines...",
    "Asking Python to not break things...",
    "Manifesting intelligence from chaos...",
    "Making sure the installer doesn't judge you...",
    "Almost there. Or are we? Infinity is relative."
)

function FunnyLine {
    $line = $loadingLines[(Get-Random -Maximum $loadingLines.Count)]
    C DarkGray "       $line"; Ln
}

# ═════════════════════════════════════════════════════════════════════
Banner

# ── welcome ───────────────────────────────────────────────────────────
C White "  Welcome, human."; Ln
C DarkGray "  You are about to install a database that thinks."; Ln
C DarkGray "  Blocks drift. Clusters emerge. Personalities form."; Ln
C DarkGray "  Your RAM may never be the same again."; Ln
Ln

if (-not $Unattended) {
    C DarkGray "  Install location: "; C Cyan $InstallDir; Ln
    C DarkGray "  Source:           "; C Cyan $SourceDir;  Ln
    Ln
    C White "  Press ENTER to begin (or Ctrl+C to escape reality)  "
    Read-Host | Out-Null
}

Ln
C Cyan "  ──────────────────────────────────────────────────────────"; Ln
Ln

# ── STEP 1: Python check ─────────────────────────────────────────────
Step "🐍" "Step 1/5 — Checking Python 3.11+"
FunnyLine

$python = $null
foreach ($cmd in @("python", "python3", "py -3.11", "py")) {
    try {
        $v = & $cmd.Split()[0] ($cmd.Split() | Select-Object -Skip 1) --version 2>&1
        if ($v -match "Python 3\.(\d+)" -and [int]$Matches[1] -ge 11) {
            $python = $cmd.Split()[0]
            break
        }
    } catch {}
}

if ($python) {
    $ver = & $python --version 2>&1
    OK "$ver found. Python is cooperating today."
    FakeProgress 0 20 400
} else {
    WARN "Python 3.11+ not found. Attempting winget install..."
    FunnyLine
    try {
        winget install -e --id Python.Python.3.11 --silent `
            --accept-package-agreements --accept-source-agreements | Out-Null
        $python = "python"
        OK "Python installed via winget. We believe in second chances."
    } catch {
        ERR "Could not auto-install Python."
        INFO "Please install Python 3.11+ from https://python.org"
        INFO "Make sure to check 'Add Python to PATH' during install."
        Read-Host "`n  Press ENTER to exit"
        exit 1
    }
}

Ln

# ── STEP 2: Create directory ─────────────────────────────────────────
Step "📁" "Step 2/5 — Preparing the space"
FakeProgress 20 35 300
Ln

New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
OK "Directory created: $InstallDir"
INFO "This is where your blocks will live. Treat it well."

Ln

# ── STEP 3: Virtual environment ──────────────────────────────────────
Step "🔮" "Step 3/5 — Creating virtual environment"
FunnyLine
FakeProgress 35 50 400
Ln

$venv   = Join-Path $InstallDir ".venv"
$pip    = Join-Path $venv "Scripts\pip.exe"
$pyexe  = Join-Path $venv "Scripts\python.exe"

RunCmd $python @("-m", "venv", $venv) "Virtual environment creation"
OK "Virtual environment ready. A clean mind deserves a clean env."

Ln

# ── STEP 4: Install dependencies ────────────────────────────────────
Step "📦" "Step 4/5 — Installing dependencies"
INFO "This might take 2-3 minutes. Go drink some water. Seriously."
Ln

FakeProgress 50 55 200
Ln
INFO "Upgrading pip (it has trust issues with old versions)..."
& $pip install --quiet --upgrade pip setuptools wheel 2>&1 | Out-Null
OK "pip upgraded."

FakeProgress 55 62 300
Ln
INFO "Installing numpy (the math guy)..."
& $pip install --quiet "numpy>=1.26" 2>&1 | Out-Null
OK "numpy installed."

FakeProgress 62 68 300
Ln
INFO "Installing scikit-learn (the clustering wizard)..."
& $pip install --quiet "scikit-learn>=1.4" 2>&1 | Out-Null
OK "scikit-learn installed."

FakeProgress 68 80 400
Ln
INFO "Installing sentence-transformers (the brain that reads text)..."
INFO "This one is chunky. 400MB of pure language understanding."
& $pip install --quiet "sentence-transformers>=2.7" 2>&1 | Out-Null
OK "sentence-transformers installed. Your DB now speaks human."

FakeProgress 80 90 400
Ln
INFO "Installing theSpaceDB from: $SourceDir"
RunCmd $pip @("install", "--quiet", $SourceDir) "theSpaceDB installation"
OK "theSpaceDB installed. The space is ready."

Ln

# ── STEP 5: Launcher + PATH ──────────────────────────────────────────
Step "🚀" "Step 5/5 — Creating spacesh launcher"
FakeProgress 90 98 300
Ln

# spacesh.bat launcher
$launcherPath = Join-Path $InstallDir "spacesh.bat"
Set-Content -Path $launcherPath -Encoding ASCII -Value "@echo off`r`n`"$pyexe`" -m spacedb.shell %*"
OK "Launcher created: spacesh.bat"

# Add to user PATH
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($userPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("PATH", "$userPath;$InstallDir", "User")
    OK "Added to PATH. You can now type 'spacesh' from anywhere."
} else {
    OK "Already in PATH. The space remembers you."
}

# Desktop shortcut
try {
    $desk   = [Environment]::GetFolderPath("Desktop")
    $lnk    = Join-Path $desk "spacesh.lnk"
    $wsh    = New-Object -ComObject WScript.Shell
    $link   = $wsh.CreateShortcut($lnk)
    $link.TargetPath       = "cmd.exe"
    $link.Arguments        = "/k spacesh"
    $link.WorkingDirectory = $env:USERPROFILE
    $link.Description      = "theSpaceDB Shell"
    $link.Save()
    OK "Desktop shortcut created. Double-click to enter the space."
} catch {
    WARN "Could not create desktop shortcut (not critical)."
}

FakeProgress 98 100 200
Ln; Ln

# ── DONE ─────────────────────────────────────────────────────────────
C Cyan    "  ╔══════════════════════════════════════════════════════════╗"; Ln
C Cyan    "  ║                                                          ║"; Ln
C Cyan    "  ║  "; C Green "✔  Installation complete!"; C Cyan "                              ║"; Ln
C Cyan    "  ║                                                          ║"; Ln
C Cyan    "  ║  "; C White "  Open a NEW terminal and type:"; C Cyan "                      ║"; Ln
C Cyan    "  ║                                                          ║"; Ln
C Cyan    "  ║  "; C Yellow "    spacesh C:\my_mind"; C Cyan "                                    ║"; Ln
C Cyan    "  ║                                                          ║"; Ln
C Cyan    "  ║  "; C DarkGray "  Or double-click 'spacesh' on your Desktop."; C Cyan "          ║"; Ln
C Cyan    "  ║                                                          ║"; Ln
C Cyan    "  ║  "; C DarkGray "  Nothing fixed. Everything evolving.  "; C Yellow "∞"; C Cyan "              ║"; Ln
C Cyan    "  ║                                                          ║"; Ln
C Cyan    "  ╚══════════════════════════════════════════════════════════╝"; Ln
Ln

if (-not $Unattended) {
    C DarkGray "  Press ENTER to close this window and open your mind.  "
    Read-Host | Out-Null
}
