@echo off
chcp 65001 >nul
setlocal
rem ============================================================
rem  Standing Recognition - quick run script
rem  Usage: run.bat <input video> [output video]
rem  Example: run.bat "C:\videos\class.mp4"
rem ============================================================
set "PY="
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not defined PY for /f "delims=" %%i in ('where python 2^>nul') do if not defined PY set "PY=%%i"
if not defined PY (
    echo [ERROR] Python not found. Install Python 3.10+ or edit the PY variable in this script.
    pause
    exit /b 1
)
if "%~1"=="" (
    echo Usage: run.bat ^<input video^> [output video]
    echo Example: run.bat "C:\videos\class.mp4"
    pause
    exit /b 1
)
set "SRC=%~1"
set "OUT=%~2"
if "%OUT%"=="" set "OUT=output\%~n1_standing.mp4"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo [info] Python : %PY%
echo [info] Input  : %SRC%
echo [info] Output : %OUT%
echo.
"%PY%" "%~dp0stand_detector.py" --source "%SRC%" --output "%OUT%" --csv "%OUT%.frames.csv" --json "%OUT%.summary.json"
echo.
pause
