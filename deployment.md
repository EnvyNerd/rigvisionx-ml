# TerraEnergy AI Deployment Options

- Local Python run: `pip install -e .` + `uvicorn rigvisionx.serving.api:app` + `python -m rigvisionx.serving.dashboard`
- One-click Windows scripts: `one-click-local.bat` (local) and `one-click-docker.bat` (Docker)
- Docker Compose: `docker compose up -d --build` for API + dashboard
- Demo mode deployments: `one-click-demo.bat` or `docker compose -f docker-compose.yml -f docker-compose.demo.yml up -d --build`
- Container base: `python:3.11-slim` (override via `PYTHON_BASE_IMAGE`)
- Ports: API `8080`, Dashboard `8050` (configurable via env vars)

