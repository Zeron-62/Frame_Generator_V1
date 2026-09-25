@echo off
setlocal EnableExtensions
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found in PATH.
  echo Install Python 3.11 or newer and make sure "python" works in Command Prompt.
  pause
  exit /b 1
)

if not exist .venv (
  echo Creating FrameForge virtual environment...
  python -m venv .venv
  if errorlevel 1 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
  )
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Failed to install dependencies.
  pause
  exit /b 1
)

echo.
echo FrameForge v4 dependencies installed.
echo ComfyUI Desktop path is not required.
echo Start ComfyUI Desktop, then run run.bat.
pause
