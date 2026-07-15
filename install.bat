@echo off
setlocal
:: wLLM installer launcher. Forwards any args (e.g. -Clean, -Cpu) to install.ps1.

where powershell >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] PowerShell is required for installation.
    pause
    exit /b 1
)

:: Preferred path: execute the script file directly so parameters pass through.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
if %errorlevel% equ 0 goto :done

:: Fallback for environments where Group Policy blocks .ps1 file execution:
:: inject the script over stdin. (Parameters cannot be forwarded in this mode.)
echo [INFO] Direct execution failed; retrying via stdin injection...
type "%~dp0install.ps1" | powershell -NoProfile -ExecutionPolicy Bypass -Command -
if %errorlevel% neq 0 (
    echo [ERROR] Installation failed.
    pause
    exit /b 1
)

:done
endlocal
