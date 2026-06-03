@echo off
setlocal
cd /d "%~dp0"

echo TerraEnergy AI one-click (Docker)

where docker >nul 2>nul
if errorlevel 1 (
  echo Docker not found. Install Docker Desktop, then rerun.
  pause
  exit /b 1
)

set DC=docker compose
docker compose version >nul 2>nul
if errorlevel 1 set DC=docker-compose

echo Starting services...
%DC% up -d --build
if errorlevel 1 (
  echo Failed to start Docker Compose.
  pause
  exit /b 1
)

if not exist "data\generated\single_rig.csv" (
  echo Generating demo data...
  %DC% run --rm terraenergy-ai-ml python -m rigvisionx.data_generator --scenario single --n-days 30
)

echo Opening dashboard...
start "" "http://localhost:8050"
echo Done. API: http://localhost:8080  Dashboard: http://localhost:8050
pause

