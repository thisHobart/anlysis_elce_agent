@echo off
setlocal
cd /d "%~dp0"

set "BUILD_PYTHON=venv\Scripts\python.exe"
if not exist "%BUILD_PYTHON%" set "BUILD_PYTHON=.venv\Scripts\python.exe"
if not exist "%BUILD_PYTHON%" (
  echo [ERROR] venv or .venv Python was not found.
  echo Create the environment and install the project first.
  exit /b 1
)

"%BUILD_PYTHON%" -m pip install -e ".[build]"
if errorlevel 1 exit /b 1

"%BUILD_PYTHON%" scripts\build_desktop_exe.py %*
set "BUILD_EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %BUILD_EXIT_CODE%
