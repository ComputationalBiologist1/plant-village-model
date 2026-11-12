@echo off
REM ==============================================================================
REM PHYTOSCAN · BOTANICAL PATHOLOGY WORKSTATION (v4.2)
REM Root Windows Launcher -> Delegates to Part 2 Workstation
REM ==============================================================================

if exist "%~dp0Part_2_Model_Testing_and_GUI\launch.bat" (
    call "%~dp0Part_2_Model_Testing_and_GUI\launch.bat" %*
) else if exist "%~dp0Compatable_run\launch.bat" (
    call "%~dp0Compatable_run\launch.bat" %*
) else (
    echo [ERROR] Part_2_Model_Testing_and_GUI\launch.bat not found.
    pause
)
