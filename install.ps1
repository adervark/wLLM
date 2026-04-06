# install.ps1 - wLLM Modern Installer
# Optimized for Windows 10/11 zero-dependency bootstrapping.

$ErrorActionPreference = "Stop"

# Enforce TLS 1.2 for modern CDN downloads (Astral/uv/PyTorch)
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Write-Step ([string]$Step, [string]$Message) {
    Write-Host "`n[$Step] $Message" -ForegroundColor Cyan
}

function Write-Success ([string]$Message) {
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Info ([string]$Message) {
    Write-Host "[INFO] $Message" -ForegroundColor Gray
}

function Write-Error-Custom ([string]$Message) {
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

Write-Host "`n===================================================" -ForegroundColor Blue
Write-Host "             wLLM Modern Installer" -ForegroundColor Blue
Write-Host "===================================================" -ForegroundColor Blue

# 1. Clean up old artifacts
Write-Step "1/5" "Cleaning up old installation artifacts..."
@(".venv", "winllm.egg-info", ".pytest_cache", "build", "dist") | ForEach-Object {
    if (Test-Path $_) {
        Write-Info "Removing $_..."
        Remove-Item -Path $_ -Recurse -Force
    }
}
Write-Success "Cleanup complete."

# 2. Bootstrap uv
Write-Step "2/5" "Ensuring 'uv' is installed..."
$uvPath = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvPath) {
    Write-Info "'uv' not found. Bootstrapping via official installer..."
    Invoke-RestMethod -Uri "https://astral.sh/uv/install.ps1" | Invoke-Expression
    
    # Refresh PATH for current session
    $potentialPaths = @(
        (Join-Path $env:USERPROFILE ".cargo\bin"),
        (Join-Path $env:APPDATA "Roaming\uv\bin"),
        (Join-Path $env:LOCALAPPDATA "bin"),
        (Join-Path $env:USERPROFILE ".local\bin")
    )
    
    foreach ($path in $potentialPaths) {
        if (Test-Path $path) {
            Write-Info "Found uv at $path. Updating current session path..."
            $env:Path = "$path;$env:Path"
        }
    }
    
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Error-Custom "uv was installed but could not be located in the current session PATH."
        exit 1
    }
    Write-Success "uv bootstrapped successfully."
} else {
    Write-Info "uv detected at $($uvPath.Source)."
}

# 3. Provision Python 3.12
Write-Step "3/5" "Provisioning Python 3.12 environment..."
Write-Info "Running: uv python install 3.12"
uv python install 3.12
if ($LASTEXITCODE -ne 0) {
    Write-Error-Custom "Failed to install Python 3.12 via uv."
    exit 1
}

Write-Info "Creating virtual environment (.venv)..."
uv venv .venv --python 3.12
Write-Success "Environment created."

# 4. Install Dependencies
Write-Step "4/5" "Installing dependencies (PyTorch with CUDA 12.4)..."
Write-Info "This may take a few minutes for the initial download."

# Use the venv's python to run uv pip to ensure it targets the correct env
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install --upgrade -e . --link-mode=copy --extra-index-url https://download.pytorch.org/whl/cu124

if ($LASTEXITCODE -ne 0) {
    Write-Error-Custom "Dependency installation failed."
    exit 1
}
Write-Success "Dependencies installed."

# 5. System PATH Configuration
Write-Step "5/5" "Configuring System PATH..."
$absVenvPath = (Get-Item ".venv\Scripts").FullName
$userPath = [Environment]::GetEnvironmentVariable("PATH", [EnvironmentVariableTarget]::User)

if ($userPath -notmatch [regex]::Escape($absVenvPath)) {
    [Environment]::SetEnvironmentVariable("PATH", "$userPath;$absVenvPath", [EnvironmentVariableTarget]::User)
    Write-Success "Successfully added wLLM binary path to your User PATH."
} else {
    Write-Info "wLLM is already in your User PATH."
}

# Final Verification
Write-Host "`n===================================================" -ForegroundColor Blue
Write-Success "wLLM installed successfully!"
Write-Host "===================================================" -ForegroundColor Blue
Write-Host "`nYou can now use wLLM from ANY terminal window exactly like Ollama!"
Write-Host "Try running: " -NoNewline; Write-Host "winllm chat --model microsoft/Phi-3-mini-4k-instruct" -ForegroundColor Yellow
Write-Host "`n[HARDWARE VERIFICATION]"
& ".venv\Scripts\python.exe" -c "import torch; print(f'  PyTorch Version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

Write-Host "`nPress any key to close..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
