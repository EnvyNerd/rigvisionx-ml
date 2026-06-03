@echo off
setlocal
cd /d "%~dp0"

echo TerraEnergy AI one-click (Local Python)

where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Install Python 3.10+ and ensure it is on PATH.
  pause
  exit /b 1
)

set PYTHONPATH=%CD%\src

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 (
    echo Failed to create venv.
    pause
    exit /b 1
  )

  echo Installing dependencies...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  ".venv\Scripts\python.exe" -m pip install -e .
)

if not exist "data\generated\single_rig.csv" (
  echo Generating demo data...
  ".venv\Scripts\python.exe" -m rigvisionx.data_generator --scenario single --n-days 30
)

echo Starting API and dashboard...
start "TerraEnergy AI API" cmd /k "\"%CD%\\.venv\\Scripts\\python.exe\" -m uvicorn rigvisionx.serving.api:app --host 0.0.0.0 --port 8080"
timeout /t 2 >nul
start "TerraEnergy AI Dashboard" cmd /k "\"%CD%\\.venv\\Scripts\\python.exe\" -m rigvisionx.serving.dashboard"
start "" "http://localhost:8050"

echo Done. API: http://localhost:8080  Dashboard: http://localhost:8050
pause

