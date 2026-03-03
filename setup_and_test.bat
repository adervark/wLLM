@echo off
set VIRTUAL_ENV=
if not exist .venv uv venv
call .venv\Scripts\activate.bat
uv pip install -e .[dev]
echo === PIP INSTALL DONE ===
uv run pytest tests/ -v
echo === TESTS DONE ===
