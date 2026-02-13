@echo off
cd /d "%~dp0"
call run_with_venv.bat main.py %*
exit /b %ERRORLEVEL%
