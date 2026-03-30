@echo off
echo Cleaning up environment...
if exist .venv rmdir /s /q .venv
if exist winllm.egg-info rmdir /s /q winllm.egg-info
if exist .pytest_cache rmdir /s /q .pytest_cache
echo Cleanup complete.

set VIRTUAL_ENV=
uv venv
call .venv\Scripts\activate.bat
uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install --upgrade -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu124
echo === PIP INSTALL DONE ===
uv run pytest tests/ -v
echo === TESTS DONE ===
