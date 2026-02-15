@echo off
setlocal
cd /d "%~dp0\.."

python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH.
    echo Install Python 3.10+ from https://www.python.org/downloads/
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

echo Preparing runtime dependencies...
python setup_env.py --torch auto
if errorlevel 1 (
    echo Error: Dependency setup failed.
    call venv\Scripts\deactivate.bat >nul 2>&1
    exit /b 1
)

python main.py %*
set "EXIT_CODE=%ERRORLEVEL%"

call venv\Scripts\deactivate.bat >nul 2>&1
exit /b %EXIT_CODE%

