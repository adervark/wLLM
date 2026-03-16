@echo off
call .venv\Scripts\activate.bat
set WINLLM_MAX_BATCH_SIZE=128
set WINLLM_QUANTIZATION=none
python -m winllm detect
