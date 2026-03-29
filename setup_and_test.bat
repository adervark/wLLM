@echo off
set VIRTUAL_ENV=
if not exist .venv uv venv
call .venv\Scripts\activate.bat
uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
uv pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu124
echo === PIP INSTALL DONE ===
uv run pytest tests/ -v
echo === TESTS DONE ===
