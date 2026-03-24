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

echo [1/3] Creating virtual environment (.venv)...
if not exist .venv (
    uv venv .venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
echo [2/3] Installing PyTorch with CUDA 12.4 support...
echo This may take a few minutes as PyTorch is a large download.
call .venv\Scripts\activate.bat
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu124

echo.
echo [3/3] Installing wLLM and its dependencies...
uv pip install -e . --extra-index-url https://download.pytorch.org/whl/cu124

echo.
echo ===================================================
echo [SUCCESS] wLLM installed successfully!
echo ===================================================
echo.
echo To start using wLLM, activate the virtual environment:
echo     .venv\Scripts\activate
echo.
echo Then try running:
echo     winllm chat --model "microsoft/Phi-3-mini-4k-instruct" --quantization 4bit
echo.
pause
