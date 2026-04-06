@echo off
setlocal
:: Ensure PowerShell is available
where powershell >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] PowerShell is required for installation.
    pause
    exit /b 1
)

:: Launch the modern PowerShell installer via script injection
:: (Bypasses Group Policy restrictions that block .ps1 file execution)
type "%~dp0install.ps1" | powershell -ExecutionPolicy Bypass -NoProfile -Command -
if %errorlevel% neq 0 (
    echo [ERROR] Installation failed.
    pause
    exit /b 1
)

goto :eof
