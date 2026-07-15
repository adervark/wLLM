<#
.SYNOPSIS
    wLLM robust installer for Windows 10/11.

.DESCRIPTION
    Reproducible, idempotent, hardware-aware installation built on uv + uv.lock.

    By default this performs a locked install (uv sync --locked) so every machine
    gets the exact versions pinned in uv.lock, including torch 2.x +cu128. The
    CUDA wheels bundle their own CUDA runtime and import fine on CPU-only machines
    (cuda.is_available() simply returns False), so the locked path is safe whether
    or not an NVIDIA GPU is present.

.PARAMETER Clean
    Remove the existing .venv and build artifacts before installing. Without this
    flag the installer is idempotent and repairs/updates the existing environment.

.PARAMETER Cpu
    Install the lighter CPU-only torch wheels instead of the locked cu128 wheels.
    Use on machines with no NVIDIA GPU where the smaller download is preferred.
    This path is not pinned by uv.lock.

.PARAMETER NoPath
    Skip adding the venv Scripts directory to the user PATH.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install.ps1
    powershell -ExecutionPolicy Bypass -File install.ps1 -Clean
    powershell -ExecutionPolicy Bypass -File install.ps1 -Cpu
#>

[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Cpu,
    [switch]$NoPath,
    [switch]$SkipPreflight
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# Always operate relative to this script, not the caller's CWD.
$ProjectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Set-Location -LiteralPath $ProjectRoot

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
function Write-Step    ([string]$Step, [string]$Message) { Write-Host "`n[$Step] $Message" -ForegroundColor Cyan }
function Write-Success  ([string]$Message) { Write-Host "[ OK    ] $Message" -ForegroundColor Green }
function Write-Info     ([string]$Message) { Write-Host "[ INFO  ] $Message" -ForegroundColor Gray }
function Write-Warn     ([string]$Message) { Write-Host "[ WARN  ] $Message" -ForegroundColor Yellow }
function Fail           ([string]$Message) {
    Write-Host "[ ERROR ] $Message" -ForegroundColor Red
    Wait-IfInteractive
    exit 1
}

# Run a native command and abort with a clear message if it returns non-zero.
function Invoke-Checked ([string]$What, [scriptblock]$Action) {
    Write-Info $What
    & $Action
    if ($LASTEXITCODE -ne 0) {
        Fail "$What`n         Command exited with code $LASTEXITCODE."
    }
}

function Wait-IfInteractive {
    # ReadKey throws when the script is piped via stdin (install.bat); guard it.
    if ([Environment]::UserInteractive -and $Host.Name -eq 'ConsoleHost') {
        try {
            Write-Host "`nPress any key to close..."
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        } catch { }
    }
}

# Catch-all: with $ErrorActionPreference = "Stop", any unhandled error would
# otherwise terminate and close the window before the user can read it. Surface
# the error and pause so a double-clicked launch never just "flashes and dies".
trap {
    Write-Host "`n[ FATAL ] $($_.Exception.Message)" -ForegroundColor Red
    if ($_.ScriptStackTrace) { Write-Host $_.ScriptStackTrace -ForegroundColor DarkGray }
    Wait-IfInteractive
    exit 1
}

Write-Host "`n===================================================" -ForegroundColor Blue
Write-Host "             wLLM Installer" -ForegroundColor Blue
Write-Host "===================================================" -ForegroundColor Blue

# ---------------------------------------------------------------------------
# 0. Preflight: verify the prerequisites the installer cannot provision.
#    Fails fast with actionable messages BEFORE any large downloads begin.
# ---------------------------------------------------------------------------
function Test-TcpReachable ([string]$HostName, [int]$Port = 443, [int]$TimeoutMs = 5000) {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $async = $client.BeginConnect($HostName, $Port, $null, $null)
        $ok = $async.AsyncWaitHandle.WaitOne($TimeoutMs, $false)
        if ($ok -and $client.Connected) { $client.EndConnect($async); return $true }
        return $false
    } catch {
        return $false
    } finally {
        if ($client) { $client.Close() }
    }
}

if ($SkipPreflight) {
    Write-Warn "Preflight checks skipped (-SkipPreflight)."
} else {
    Write-Step "0/6" "Preflight checks..."

    # --- OS architecture: cu128 wheels and uv builds are Windows x64 only. ---
    if (-not [Environment]::Is64BitOperatingSystem) {
        Fail "64-bit Windows is required. This machine reports a 32-bit OS."
    }
    Write-Success "64-bit Windows confirmed."

    # --- PowerShell version. ---
    if ($PSVersionTable.PSVersion.Major -lt 5) {
        Fail "Windows PowerShell 5.1 or newer is required (found $($PSVersionTable.PSVersion))."
    }

    # --- Free disk space on the target drive (cu128 torch stack is large). ---
    $minFreeGB = if ($Cpu) { 5 } else { 10 }
    try {
        $qualifier = (Split-Path -Qualifier $ProjectRoot).TrimEnd(':')
        $freeGB = [math]::Round((Get-PSDrive -Name $qualifier -ErrorAction Stop).Free / 1GB, 1)
        if ($freeGB -lt $minFreeGB) {
            Fail "Only ${freeGB} GB free on drive ${qualifier}:. Need at least ${minFreeGB} GB for the install."
        }
        Write-Success "Disk space OK (${freeGB} GB free on ${qualifier}:)."
    } catch {
        Write-Warn "Could not determine free disk space; need ~${minFreeGB} GB. Continuing."
    }

    # --- Network reachability for the hosts the install pulls from. ---
    $endpoints = @(
        @{ Name = "astral.sh";              Host = "astral.sh" },
        @{ Name = "PyPI";                   Host = "pypi.org" },
        @{ Name = "python.org wheels";      Host = "files.pythonhosted.org" }
    )
    if (-not $Cpu) {
        $endpoints += @{ Name = "PyTorch CUDA index"; Host = "download.pytorch.org" }
    } else {
        $endpoints += @{ Name = "PyTorch CPU index";  Host = "download.pytorch.org" }
    }
    $unreachable = @()
    foreach ($e in $endpoints) {
        if (Test-TcpReachable $e.Host) {
            Write-Info "Reachable: $($e.Name) ($($e.Host):443)"
        } else {
            $unreachable += $e
        }
    }
    if ($unreachable.Count -gt 0) {
        Write-Warn "Could not reach over HTTPS (port 443):"
        foreach ($e in $unreachable) { Write-Warn "  - $($e.Name) [$($e.Host)]" }
        Write-Warn "A proxy/firewall may still allow uv to connect. The install will fail loudly if it cannot."
    } else {
        Write-Success "Network reachability OK."
    }

    # --- Visual C++ runtime: PyTorch wheels link against vcruntime140*.dll.
    #     The installer cannot provision this; warn so a bare machine knows. ---
    $sys32 = Join-Path $env:SystemRoot "System32"
    $vcDlls = @("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")
    $missingVc = $vcDlls | Where-Object { -not (Test-Path (Join-Path $sys32 $_)) }
    if ($missingVc.Count -gt 0) {
        Write-Warn "Microsoft Visual C++ Redistributable appears to be missing ($($missingVc -join ', '))."
        Write-Warn "PyTorch may fail to import. Install 'VC++ Redistributable 2015-2022 x64':"
        Write-Warn "  https://aka.ms/vs/17/release/vc_redist.x64.exe"
    } else {
        Write-Success "Visual C++ runtime present."
    }

    Write-Success "Preflight complete."
}

# ---------------------------------------------------------------------------
# 1. Locate or bootstrap uv
# ---------------------------------------------------------------------------
Write-Step "1/6" "Ensuring 'uv' is available..."

function Find-Uv {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @(
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "uv\bin\uv.exe"),
        (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe")
    )) {
        if (Test-Path $p) {
            $env:Path = "$(Split-Path $p);$env:Path"
            return $p
        }
    }
    return $null
}

$uv = Find-Uv
if (-not $uv) {
    Write-Info "'uv' not found. Bootstrapping from astral.sh..."
    try {
        Invoke-RestMethod -Uri "https://astral.sh/uv/install.ps1" | Invoke-Expression
    } catch {
        Fail "Failed to download/run the uv installer: $($_.Exception.Message)"
    }
    $uv = Find-Uv
    if (-not $uv) {
        Fail "uv was installed but could not be located on PATH. Open a new terminal and re-run."
    }
}
$uvVersion = (& uv --version) 2>$null
Write-Success "uv ready: $uvVersion ($uv)"

# ---------------------------------------------------------------------------
# 2. Optional clean
# ---------------------------------------------------------------------------
Write-Step "2/6" "Preparing environment..."
if ($Clean) {
    foreach ($artifact in @(".venv", "winllm.egg-info", ".pytest_cache", "build", "dist")) {
        $path = Join-Path $ProjectRoot $artifact
        if (Test-Path $path) {
            Write-Info "Removing $artifact..."
            Remove-Item -LiteralPath $path -Recurse -Force
        }
    }
    Write-Success "Clean complete."
} else {
    Write-Info "Idempotent mode (use -Clean for a fresh rebuild)."
}

# ---------------------------------------------------------------------------
# 3. Provision Python 3.12
# ---------------------------------------------------------------------------
Write-Step "3/6" "Provisioning Python 3.12..."
Invoke-Checked "uv python install 3.12" { uv python install 3.12 }
Write-Success "Python 3.12 available."

# ---------------------------------------------------------------------------
# 4. Hardware probe (informational; drives CPU vs CUDA messaging)
# ---------------------------------------------------------------------------
Write-Step "4/6" "Detecting hardware..."
$hasNvidia = $false
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    try {
        $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
        if ($LASTEXITCODE -eq 0 -and $gpuName) {
            $hasNvidia = $true
            Write-Success "NVIDIA GPU detected: $($gpuName.Trim())"
        }
    } catch { }
}
if (-not $hasNvidia) {
    Write-Warn "No NVIDIA GPU detected via nvidia-smi."
    if (-not $Cpu) {
        Write-Info "Proceeding with the locked CUDA build anyway (it runs on CPU; re-run with -Cpu for smaller wheels)."
    }
}

# ---------------------------------------------------------------------------
# 5. Install dependencies
# ---------------------------------------------------------------------------
Write-Step "5/6" "Installing dependencies (this can take several minutes on first run)..."

if ($Cpu) {
    Write-Info "CPU-only install path selected."
    Invoke-Checked "Creating virtual environment" { uv venv .venv --python 3.12 }
    Invoke-Checked "Installing CPU torch stack" {
        uv pip install --python .venv torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    }
    Invoke-Checked "Installing winllm (editable)" {
        uv pip install --python .venv -e . --link-mode=copy
    }
} else {
    # Reproducible, locked install. Falls back to a fresh resolve if the lock
    # has drifted from pyproject.toml so a stale lock never hard-blocks setup.
    Write-Info "Locked install from uv.lock (reproducible)."
    uv sync --locked
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "uv.lock is out of date or sync --locked failed; re-resolving against pyproject.toml..."
        Invoke-Checked "uv sync (re-resolve)" { uv sync }
    }
}
Write-Success "Dependencies installed."

# ---------------------------------------------------------------------------
# 6. PATH configuration + verification
# ---------------------------------------------------------------------------
Write-Step "6/6" "Configuring PATH and verifying install..."

$venvScripts = Join-Path $ProjectRoot ".venv\Scripts"
$venvPython  = Join-Path $venvScripts "python.exe"
if (-not (Test-Path $venvPython)) {
    Fail "Expected interpreter not found at $venvPython. Installation did not complete."
}

if (-not $NoPath) {
    $absScripts = (Get-Item $venvScripts).FullName
    $userPath = [Environment]::GetEnvironmentVariable("PATH", [EnvironmentVariableTarget]::User)
    if (-not $userPath) { $userPath = "" }
    $already = ($userPath -split ';') | Where-Object { $_.TrimEnd('\') -ieq $absScripts.TrimEnd('\') }
    if (-not $already) {
        $newPath = if ($userPath.TrimEnd(';')) { "$($userPath.TrimEnd(';'));$absScripts" } else { $absScripts }
        [Environment]::SetEnvironmentVariable("PATH", $newPath, [EnvironmentVariableTarget]::User)
        $env:Path = "$absScripts;$env:Path"
        Write-Success "Added wLLM to your User PATH."
    } else {
        Write-Info "wLLM is already on your User PATH."
    }
} else {
    Write-Info "Skipping PATH update (-NoPath)."
}

# Self-check: import the package, report the torch/CUDA state, exercise the CLI.
# The probe is written to a temp file rather than passed via -c, because
# Windows PowerShell mangles inner quotes when handing a multi-line string to
# a native executable.
Write-Info "Running post-install self-check..."
$probe = @'
import sys
try:
    import winllm
    import torch
    print("  winllm import : OK")
    print(f"  torch version : {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU           : {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  SELF-CHECK FAILED: {e}", file=sys.stderr)
    sys.exit(1)
'@
$probeFile = Join-Path ([System.IO.Path]::GetTempPath()) "winllm_selfcheck_$PID.py"
Set-Content -LiteralPath $probeFile -Value $probe -Encoding UTF8
try {
    & $venvPython $probeFile
    if ($LASTEXITCODE -ne 0) {
        Fail "Post-install self-check failed. The environment is not usable yet."
    }
} finally {
    Remove-Item -LiteralPath $probeFile -Force -ErrorAction SilentlyContinue
}

# Confirm the console entry point resolves and runs.
$winllmExe = Join-Path $venvScripts "winllm.exe"
if (Test-Path $winllmExe) {
    & $winllmExe --help *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "'winllm --help' did not run cleanly. Check it manually."
    } else {
        Write-Success "CLI entry point verified."
    }
} else {
    Write-Warn "winllm.exe was not found in $venvScripts."
}

Write-Host "`n===================================================" -ForegroundColor Blue
Write-Success "wLLM installed successfully!"
Write-Host "===================================================" -ForegroundColor Blue
Write-Host "`nOpen a NEW terminal, then try:"
Write-Host "  winllm detect" -ForegroundColor Yellow
Write-Host "  winllm chat --model microsoft/Phi-3-mini-4k-instruct" -ForegroundColor Yellow

Wait-IfInteractive
