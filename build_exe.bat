@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo [ERROR] venv\Scripts\python.exe was not found.
  echo Create the environment and install the project first.
  exit /b 1
)

"venv\Scripts\python.exe" -m pip install -e ".[build]"
if errorlevel 1 exit /b 1

"venv\Scripts\python.exe" scripts\build_desktop_exe.py %*
endlocal
