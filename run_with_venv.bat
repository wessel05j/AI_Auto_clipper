@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
    echo Usage: run_with_venv.bat script.py [args...]
    exit /b 1
)

set "TARGET_SCRIPT=%~1"
shift

python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH.
    echo Install Python from https://www.python.org/downloads/
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment.
        exit /b 1
    )
)

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment.
    exit /b 1
)

echo Preparing dependencies...
python setup_env.py --torch auto
if errorlevel 1 (
    echo Error: Environment setup failed.
    call venv\Scripts\deactivate.bat >nul 2>&1
    exit /b 1
)

python "%TARGET_SCRIPT%" %*
set "EXIT_CODE=%ERRORLEVEL%"

call venv\Scripts\deactivate.bat >nul 2>&1
exit /b %EXIT_CODE%
