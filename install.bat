@echo off
setlocal

echo ===================================================
echo             wLLM Installation Script
echo ===================================================
echo.

:: Check if uv is installed
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] 'uv' is not installed or not in PATH.
    echo Please install it first using one of the following commands:
    echo:
    echo PowerShell:
    echo   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo:
    echo Or via pip:
    echo   pip install uv
    echo.
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment (.venv)...
if not exist .venv (
    uv venv .venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
echo [2/4] Installing PyTorch with CUDA 12.4 support...
echo This may take a few minutes as PyTorch is a large download.
call .venv\Scripts\activate.bat
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu124

echo.
echo [3/4] Installing wLLM and its dependencies...
uv pip install -e . --extra-index-url https://download.pytorch.org/whl/cu124

echo.
echo [4/4] Adding wLLM to system PATH...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$userPath = [Environment]::GetEnvironmentVariable('PATH', 'User'); $wllmPath = Join-Path -Path $PWD -ChildPath '.venv\Scripts'; if ($userPath -notmatch [regex]::Escape($wllmPath)) { [Environment]::SetEnvironmentVariable('PATH', $userPath + ';' + $wllmPath, 'User'); Write-Host 'Successfully added wLLM to your User PATH.' } else { Write-Host 'wLLM is already in your PATH.' }"

echo.
echo ===================================================
echo [SUCCESS] wLLM installed successfully!
echo ===================================================
echo.
echo You can now use wLLM from ANY terminal window exactly like Ollama!
echo Please completely restart your terminal for the PATH changes to take effect.
echo.
echo Try running:
echo     winllm chat --model "microsoft/Phi-3-mini-4k-instruct" --quantization 4bit
echo.
pause
