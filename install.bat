@echo off
setlocal

echo ===================================================
echo             wLLM Installation Script
echo ===================================================
echo.

:: Check if uv is installed
where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo [INFO] 'uv' detected.
    set USE_UV=1
) else (
    :: Check if python is installed
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        echo [INFO] 'uv' not found, but 'python' detected.
        set USE_UV=0
    ) else (
        echo [WARN] Neither 'uv' nor 'python' were found on your system.
        echo [INFO] Bootstrapping 'uv' (the modern Python toolchain) to continue...
        echo.
        
        :: Install uv via PowerShell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        if errorlevel 1 (
            echo [ERROR] Failed to install 'uv' via PowerShell. Please install Python or uv manually.
            goto error
        )
        
        :: Add uv to current session PATH
        :: The installer usually puts it in %USERPROFILE%\.local\bin or %APPDATA%\revup\bin
        :: but the most common for the ps1 installer on Windows is %USERPROFILE%\.cargo\bin or %APPDATA%\Roaming\uv\bin
        if exist "%USERPROFILE%\.cargo\bin\uv.exe" set "PATH=%PATH%;%USERPROFILE%\.cargo\bin"
        if exist "%APPDATA%\Roaming\uv\bin\uv.exe" set "PATH=%PATH%;%APPDATA%\Roaming\uv\bin"
        if exist "%USERPROFILE%\.local\bin\uv.exe" set "PATH=%PATH%;%USERPROFILE%\.local\bin"
        
        :: Verify uv is now available
        where uv >nul 2>nul
        if errorlevel 1 (
            echo [ERROR] 'uv' was installed but could not be located in the current session PATH.
            echo Please restart your terminal and run this script again.
            goto error
        )
        
        echo [SUCCESS] 'uv' bootstrapped successfully.
        set USE_UV=1
    )
)

:: If using uv, ensure Python 3.12 is installed
if %USE_UV%==1 (
    echo [INFO] Ensuring Python 3.12 is available via uv...
    uv python install 3.12
    if errorlevel 1 (
        echo [ERROR] Failed to install Python 3.12 via 'uv'.
        goto error
    )
)

echo [1/5] Cleaning up old installation artifacts...
if exist ".venv" echo Deleting old virtual environment...
if exist ".venv" rmdir /s /q ".venv"
if exist "winllm.egg-info" rmdir /s /q "winllm.egg-info"
if exist ".pytest_cache" rmdir /s /q ".pytest_cache"
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
echo Cleanup complete.

echo.
echo [2/5] Creating fresh virtual environment (.venv)...
if %USE_UV%==1 (
    :: PyTorch does not currently support Python 3.14+ wheels yet, so we explicitly pin to 3.12
    uv venv .venv --python 3.12
) else (
    python -m venv .venv
)
if errorlevel 1 goto error

echo.
echo [3/5] Installing PyTorch with CUDA 12.4 support...
echo This may take a few minutes as PyTorch is a large download.
if %USE_UV%==1 (
    call .venv\Scripts\activate.bat
    uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
) else (
    ".venv\Scripts\python" -m pip install --upgrade pip
    ".venv\Scripts\python" -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
)
if errorlevel 1 goto error

echo.
echo [4/5] Installing wLLM and its dependencies...
if %USE_UV%==1 (
    uv pip install --upgrade -e . --link-mode=copy --extra-index-url https://download.pytorch.org/whl/cu124
) else (
    ".venv\Scripts\python" -m pip install --upgrade -e . --extra-index-url https://download.pytorch.org/whl/cu124
)
if errorlevel 1 goto error

echo.
echo [5/5] Adding wLLM to system PATH...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$userPath = [Environment]::GetEnvironmentVariable('PATH', 'User'); $wllmPath = '%~dp0.venv\Scripts'; if ($userPath -notmatch [regex]::Escape($wllmPath)) { [Environment]::SetEnvironmentVariable('PATH', $userPath + ';' + $wllmPath, 'User'); Write-Host \"Successfully added wLLM ($wllmPath) to your User PATH.\" } else { Write-Host 'wLLM is already in your PATH.' }"

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
echo.
echo [HARDWARE VERIFICATION]
".venv\Scripts\python" -c "import torch; print(f'  PyTorch Version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
if errorlevel 1 echo [ERROR] Hardware verification failed. Check if NVIDIA drivers are installed.

pause
goto :eof

:error
echo.
echo [ERROR] Installation failed at the current step.
pause
exit /b 1

:eof
