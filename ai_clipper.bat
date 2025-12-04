@echo off
setlocal

REM Change to script directory (project root)
cd /d "%~dp0"

REM === Create / reuse virtual environment ===

set "VENV_DIR=.venv"

if not exist "%VENV_DIR%\Scripts\python.exe" (
	echo Creating virtual environment in "%VENV_DIR%" ...
	py -3 -m venv "%VENV_DIR%"
	if errorlevel 1 (
		echo Failed to create virtual environment. Make sure Python is installed and the "py" launcher is available.
		pause
		goto :EOF
	)
)

echo Installing / updating dependencies from requirements.txt ...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
	echo Failed to install Python dependencies.
	pause
	goto :EOF
)

REM === Run main.py ===

echo Starting main.py ...
"%VENV_DIR%\Scripts\python.exe" main.py

echo.
echo main.py has finished. Press any key to exit.
pause >nul

endlocal

