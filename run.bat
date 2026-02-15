@echo off
cd /d "%~dp0"
call launcher\run.bat %*
exit /b %ERRORLEVEL%

