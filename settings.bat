@echo off
cd /d "%~dp0"
call run_with_venv.bat settings.py %*
exit /b %ERRORLEVEL%
