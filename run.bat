@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo FrameForge is not installed yet. Run install.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python app.py
