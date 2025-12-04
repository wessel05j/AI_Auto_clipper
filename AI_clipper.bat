@echo off
setlocal

:: ================================
:: 1. Launch external app (optional)
:: ================================
set APP_PATH="C:\Users\ejwes\AppData\Roaming\Microsoft\Windows\Start Menu\Programs"
echo Launching external app...
start "" %APP_PATH%

:: ================================
:: 2. Python 3.13 check
:: ================================
echo Checking for Python 3.13...
py -3.13 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.13 is not installed or not registered with the 'py' launcher.
    echo Please install Python 3.13 and enable "Add to PATH" + "Add to py launcher".
    pause
    exit /b
)

:: ================================
:: 3. Create venv (Python 3.13)
:: ================================
set VENV_DIR=venv

if not exist "%VENV_DIR%" (
    echo Creating virtual environment using Python 3.13...
    py -3.13 -m venv "%VENV_DIR%"
)

:: ================================
:: 4. Activate venv
:: ================================
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

:: ================================
:: 5. Install dependencies
:: ================================
if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    pip install --upgrade pip
    pip install -r requirements.txt
) else (
    echo WARNING: requirements.txt not found!
)

:: ================================
:: 6. Launch Python script
:: ================================
echo Running main.py...
python main.py

echo ------------------------------
echo Script finished. Press any key.
pause