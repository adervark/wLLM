@echo off
call S:\Code\vLLM\.venv\Scripts\activate.bat
pip install -e .[dev]
echo === PIP INSTALL DONE ===
python -m pytest tests/ -v
echo === TESTS DONE ===
