@echo off
REM AI Auto Clipper Launcher (Ollama + Python venv)
cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Optional: Check Ollama installation
ollama --version >nul 2>&1
if errorlevel 1 (
    echo Warning: Ollama is not installed or not in PATH.
    echo Download Ollama for Windows: https://ollama.com/download
)

REM Create venv if missing
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate venv
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install requirements
echo Installing required packages...
pip install -q --upgrade pip
pip install -q -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install requirements
    pause
    exit /b 1
)

REM Run main.py
echo Starting AI Auto Clipper...
python main.py

REM Deactivate venv
call venv\Scripts\deactivate.bat

exit /b %errorlevel%