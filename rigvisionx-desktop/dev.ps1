param(
  [string]$ApiBase = "http://localhost:5173",
  [switch]$UseDockerApi,
  [string]$DockerApiUrl = "http://127.0.0.1:8080"
)

$root = $PSScriptRoot
$backendDir = Join-Path $root "backend"
$uiDir = Join-Path $root "desktop\ui"
$desktopDir = Join-Path $root "desktop"
$venvPython = Join-Path $root "..\.venv\Scripts\python.exe"
$pythonCmd = if (Test-Path $venvPython) { $venvPython } else { "python" }

if (-not $UseDockerApi) {
  Write-Host "Starting backend..."
  $prevDisableAuth = $env:RIGVISIONX_DISABLE_AUTH
  $env:RIGVISIONX_DISABLE_AUTH = "1"
  Start-Process -WorkingDirectory $backendDir -FilePath $pythonCmd -ArgumentList "-m app.main" | Out-Null
  if ($null -ne $prevDisableAuth) {
    $env:RIGVISIONX_DISABLE_AUTH = $prevDisableAuth
  } else {
    Remove-Item Env:RIGVISIONX_DISABLE_AUTH -ErrorAction SilentlyContinue
  }
} else {
  Write-Host "Using Docker API at $DockerApiUrl (skipping local backend startup)."
}

Write-Host "Starting Vite UI..."
$prevViteApiBase = $env:VITE_API_BASE
if ($UseDockerApi) {
  $env:VITE_API_BASE = $DockerApiUrl
} else {
  Remove-Item Env:VITE_API_BASE -ErrorAction SilentlyContinue
}
Start-Process -WorkingDirectory $uiDir -FilePath "cmd.exe" -ArgumentList "/c", "npm run dev" | Out-Null
if ($null -ne $prevViteApiBase) {
  $env:VITE_API_BASE = $prevViteApiBase
} else {
  Remove-Item Env:VITE_API_BASE -ErrorAction SilentlyContinue
}

Write-Host "Starting Electron..."
$prevElectronRunAsNode = $env:ELECTRON_RUN_AS_NODE
$prevDevServerUrl = $env:VITE_DEV_SERVER_URL
Remove-Item Env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue
$env:VITE_DEV_SERVER_URL = $ApiBase
Start-Process -WorkingDirectory $desktopDir -FilePath "cmd.exe" -ArgumentList "/c", "npm run dev" | Out-Null
if ($null -ne $prevElectronRunAsNode) {
  $env:ELECTRON_RUN_AS_NODE = $prevElectronRunAsNode
} else {
  Remove-Item Env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue
}
if ($null -ne $prevDevServerUrl) {
  $env:VITE_DEV_SERVER_URL = $prevDevServerUrl
} else {
  Remove-Item Env:VITE_DEV_SERVER_URL -ErrorAction SilentlyContinue
}

Write-Host "TerraEnergy AI Desktop dev environment launched."

