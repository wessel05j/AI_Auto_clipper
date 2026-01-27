@echo off
REM AI Auto Clipper Settings/Setup Launcher (Ollama + Python venv)
cd /d "%~dp0"
REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)
pip install --upgrade pip
REM Run settings.py
echo Starting AI Auto Clipper Settings...
python settings.py
REM Deactivate venv
call venv\Scripts\deactivate.bat
exit /b %errorlevel%