@echo off
setlocal
cd /d "%~dp0"

echo TerraEnergy AI one-click (Demo 7-day trial)

set RIGVISIONX_DEMO_MODE=true
set RIGVISIONX_DEMO_DAYS=7
set RIGVISIONX_DEMO_STATE_PATH=%CD%\data\demo_state.json
set RIGVISIONX_DB_PATH=%CD%\data\demo\rigvisionx_demo.db
set MODEL_DIR=%CD%\models\demo

call one-click-local.bat

