@echo off
cd /d "%~dp0"
call run_with_venv.bat fetch_yt_links.py %*
exit /b %ERRORLEVEL%
