@echo off
REM ==============================================================================
REM PHYTOSCAN · BOTANICAL PATHOLOGY WORKSTATION (v4.2)
REM Native Windows Launcher (Command Prompt & PowerShell)
REM ==============================================================================

echo ========================================================================
echo PHYTOSCAN · BOTANICAL PATHOLOGY WORKSTATION (v4.2)
echo Initializing Desktop Interface...
echo ========================================================================

REM Check local virtual environment first
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0plantguard_app.py" %*
    goto :EOF
)

if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" "%~dp0plantguard_app.py" %*
    goto :EOF
)

REM Check PATH python
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    python "%~dp0plantguard_app.py" %*
    goto :EOF
)

REM Check Python Launcher 'py'
where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    py "%~dp0plantguard_app.py" %*
    goto :EOF
)

echo [ERROR] Python executable not detected in system PATH.
echo Please install Python 3.10+ from python.org and execute:
echo   pip install -r "%~dp0requirements.txt"
pause
