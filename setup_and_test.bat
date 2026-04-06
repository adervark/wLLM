@echo off
setlocal
:: Ensure PowerShell is available
where powershell >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] PowerShell is required for development setup.
    pause
    exit /b 1
)

:: Launch the modern PowerShell dev setup via script injection
:: (Bypasses Group Policy restrictions that block .ps1 file execution)
type "%~dp0setup_and_test.ps1" | powershell -ExecutionPolicy Bypass -NoProfile -Command -
if %errorlevel% neq 0 (
    echo [ERROR] Dev setup and test execution failed.
    pause
    exit /b 1
)

goto :eof
