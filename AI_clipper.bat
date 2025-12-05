@echo off
setlocal

:: ===================================================================
:: CHECK IF THIS IS THE PYTHON PHASE
:: ===================================================================
if "%1"=="--python" goto python_phase


:: ===================================================================
:: NORMAL START: Launch LM Studio
:: ===================================================================
start "" "C:\Users\ejwes\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\LM Studio.lnk"

:: ===================================================================
:: OPEN ONE NEW TERMINAL FOR PYTHON
:: ===================================================================
start "" cmd /k "%~f0 --python"
exit /b



:python_phase
echo [Python phase started]

:: ================================
:: 1. Python check
:: ================================
echo Checking for Python 3.13...
py -3.13 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.13 is not installed.
    pause
    exit /b
)

:: ================================
:: 2. Create venv
:: ================================
set VENV_DIR=venv

if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    py -3.13 -m venv "%VENV_DIR%"
)

:: ================================
:: 3. Activate venv
:: ================================
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

:: ================================
:: 4. Install dependencies
:: ================================
if exist requirements.txt (
    echo Installing dependencies...
    pip install --upgrade pip
    pip install -r requirements.txt
)

:: ================================
:: 5. Run Python script
:: ================================
echo Running main.py...
python main.py

echo ------------------------------
echo Python script finished.
pause
exit /b
