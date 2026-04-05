@echo off
setlocal

echo Cleaning up environment...
if exist .venv rmdir /s /q .venv
if exist winllm.egg-info rmdir /s /q winllm.egg-info
if exist .pytest_cache rmdir /s /q .pytest_cache
echo Cleanup complete.

:: Check if uv is installed
where uv >nul 2>nul
if errorlevel 1 (
    echo [INFO] 'uv' not found. Falling back to standard Python/pip.
    set USE_UV=0
) else (
    echo [INFO] 'uv' detected. Using uv for faster testing.
    set USE_UV=1
)

set VIRTUAL_ENV=
if %USE_UV%==1 (
    uv venv
    call .venv\Scripts\activate.bat
    uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    uv pip install --upgrade -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu124
    echo === PIP INSTALL DONE ===
    uv run pytest tests/ -v
) else (
    python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    python -m pip install --upgrade -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu124
    echo === PIP INSTALL DONE ===
    pytest tests/ -v
)
echo === TESTS DONE ===
