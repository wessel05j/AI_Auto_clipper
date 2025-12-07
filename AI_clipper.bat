@echo off
setlocal enabledelayedexpansion

:: ===================================================================
:: Configuration: File to store the LM Studio path
:: ===================================================================
set "CONFIG_FILE=lm_studio_path.cfg"
set "LM_STUDIO_PATH="
set "DEFAULT_PATH=C:\Users\ejwes\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\LM Studio.lnk"

:: ===================================================================
:: CHECK IF THIS IS THE PYTHON PHASE
:: ===================================================================
if "%1"=="--python" goto python_phase

:: ===================================================================
:: NORMAL START: PATH DETECTION BLOCK
:: ===================================================================

:: 1. Attempt to read path from config file
if exist "%CONFIG_FILE%" (
    :: Read the path, stripping quotes, and set it to LM_STUDIO_PATH
    set "TEMP_PATH="
    for /f "usebackq tokens=*" %%i in ("%CONFIG_FILE%") do (
        set "TEMP_PATH=%%i"
    )
    call set "LM_STUDIO_PATH=%%TEMP_PATH:\"=%%" 

    if exist "!LM_STUDIO_PATH!" (
        echo LM Studio path loaded from config: !LM_STUDIO_PATH!
        goto launch
    ) else (
        echo ERROR: Configured path is invalid. Proceeding to manual entry.
    )
)

:: 2. If config failed, start path input loop (guaranteed to save valid path)
:path_check_loop
    echo.
    echo --- PATH SETUP REQUIRED ---
    echo Please enter the **FULL, CORRECT PATH** to "LM Studio.lnk" or "LM Studio.exe".
    echo (e.g., C:\Users\ejwes\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\LM Studio.lnk)
    echo.
    
    set /p "NEW_PATH=Enter path: "

    :: --- STRIP QUOTATIONS ---
    set "LM_STUDIO_PATH=!NEW_PATH:"=!"
    
    :: Re-check the path
    if not exist "!LM_STUDIO_PATH!" (
        echo.
        echo The path "!LM_STUDIO_PATH!" is INCORRECT. Please try again.
        goto path_check_loop
    ) else (
        :: --- SAVE THE PATH ---
        echo Correct path accepted: !LM_STUDIO_PATH!
        echo Saving path to "%CONFIG_FILE%"...
        echo !LM_STUDIO_PATH!>"%CONFIG_FILE%"
        echo Path saved successfully.
        goto launch
    )


:: ===================================================================
:: LAUNCH PHASE (IDENTICAL TO YOUR WORKING SCRIPT)
:: ===================================================================
:launch
echo Launching LM Studio...

:: This line MUST work, as it is nearly identical to your original working line.
start "" "!LM_STUDIO_PATH!"

:: OPEN ONE NEW TERMINAL FOR PYTHON
echo Launching Python phase...
start "" cmd /k "%~f0 --python"
exit /b


:: --- PYTHON PHASE ---
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