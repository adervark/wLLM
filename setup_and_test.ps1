# setup_and_test.ps1 - wLLM Dev Setup & Test Runner
# Optimized for high-performance development workflows.

$ErrorActionPreference = "Stop"

# Enforce TLS 1.2 for modern CDN downloads (Astral/uv/PyTorch)
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Write-Info ([string]$Message) {
    Write-Host "[INFO] $Message" -ForegroundColor Gray
}

function Write-Success ([string]$Message) {
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

Write-Host "`nEnvironment Cleanup..." -ForegroundColor Cyan
@(".venv", "winllm.egg-info", ".pytest_cache") | ForEach-Object {
    if (Test-Path $_) {
        Remove-Item -Path $_ -Recurse -Force
    }
}
Write-Success "Cleanup complete."

# 1. Dependency Resolution
Write-Host "`nResolving Toolchain..." -ForegroundColor Cyan
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] 'uv' not found. Please run install.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Info "Creating dev environment (.venv)..."
uv venv .venv --python 3.12
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install --upgrade -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu124

Write-Success "=== PIP INSTALL DONE ==="

# 2. Test Execution
Write-Host "`nExecuting wLLM Test Suite..." -ForegroundColor Cyan
uv run pytest tests/ -v

Write-Success "`n=== TESTS DONE ==="
