# TerraEnergy AI

A machine learning platform for predictive maintenance, anomaly detection, and energy optimization in industrial equipment.

## System Overview

- Data ingestion and preprocessing for industrial time-series data
- Feature engineering and windowed model inputs
- Models for failure risk, remaining useful life (RUL), and anomaly detection
- AEOS energy deviation utilities
- REST API (FastAPI) + interactive dashboard (Dash)
- Role-based authentication (bcrypt + JWT) with admin/viewer access control
- SQLite persistence for users, training jobs, and dataset metadata
- EDA suite with distributions, correlations, outliers, and scatter analysis

## Project Structure

```
rigvisionx-ml/
  configs/              # Configuration files
  data/                 # Data directory
    raw/                # Raw input data
    processed/          # Processed data
    features/           # Feature sets
  notebooks/            # Jupyter notebooks for exploration
  src/rigvisionx/       # Main source code
    ingest/             # Data ingestion adapters
    aeos/               # Energy analysis and optimization
    features/           # Feature engineering
    models/             # ML models (failure_risk, rul, anomaly)
    decision_engine/    # Decision logic and risk fusion
    fintech/            # Financial and cost analysis
    serving/            # API serving
    utils/              # Utilities and helpers
  tests/                # Test suite
```

## Requirements

- Python 3.10+
- Optional: Docker + Docker Compose

## Installation

```bash
pip install -e .
```

## Quick Start (Local)

1) Generate demo data:
```bash
python -m rigvisionx.data_generator --scenario single --n-days 30
```

2) Start the API:
```bash
uvicorn rigvisionx.serving.api:app --host 0.0.0.0 --port 8080
```

3) Start the dashboard:
```bash
python -m rigvisionx.serving.dashboard
```

## One-Click Launch (Windows)

Local Python (no Docker):

```bash
one-click-local.bat
```

- Creates `.venv` if missing, installs dependencies, generates demo data, then starts API + dashboard.
- Opens the dashboard at `http://localhost:8050` (API at `http://localhost:8080`).

Docker (recommended for the easiest setup):

```bash
one-click-docker.bat
```

- Requires Docker Desktop.
- Builds and runs API + dashboard containers, generates demo data if missing, then opens the dashboard.

## Demo Mode (7-day trial)

Use demo mode to ship a duplicate/demo system to clients. The first time the API starts in demo mode,
it records a start timestamp and allows access for 7 days. After the trial ends, protected endpoints
return HTTP 402 until a license is provided.

Quick launch (Windows demo profile):

```bash
one-click-demo.bat
```

The demo profile runs with a separate SQLite database under `data/demo/` and a
separate model directory under `models/demo/`.

Docker demo profile:

```bash
docker compose -f docker-compose.yml -f docker-compose.demo.yml up -d --build
```

Demo environment variables:

- `RIGVISIONX_DEMO_MODE` (true/false)
- `RIGVISIONX_DEMO_DAYS` (default `7`)
- `RIGVISIONX_DEMO_STATE_PATH` (default `data/demo_state.json`)
- `RIGVISIONX_LICENSE_KEY` (set any non-empty value to unlock)
- `RIGVISIONX_LICENSE_FILE` (optional path to json/text with `license_key` or `licensed=true`)
- `RIGVISIONX_LICENSED` (true to force licensed)

Check status:

```
GET /license/status
```

### Auth & Roles (Local)

- First registered user becomes **admin**.
- Additional users default to **viewer** unless explicitly listed as admins.

Set admins via environment variable:

```bash
export RIGVISIONX_ADMIN_USERS="admin,witschi"
```

Register and login:

```bash
curl -X POST http://localhost:8080/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"secret123"}'

curl -X POST http://localhost:8080/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"secret123"}'
```

Use the returned `access_token` as:

```
Authorization: Bearer <token>
```

## Synthetic Data Generator

Generate synthetic oil-rig datasets for demos and testing:

```bash
python -m rigvisionx.data_generator --scenario single --n-days 30
python -m rigvisionx.data_generator --scenario fleet --n-rigs 10 --n-days 90
```

Outputs are written to `data/generated/` by default.

## API (FastAPI)

Run locally with Docker:

```bash
docker-compose up -d --build
```

By default, the API is available at:

- `http://localhost:8080`

### Auth Endpoints

- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`

Admin-only endpoints:

- `POST /train`
- `POST /train/async`
- `GET /train/status/{job_id}`
- `POST /fintech/report`
- `POST /fintech/report/pdf`
- `POST /reports/system/pdf`

### Endpoints

- `GET /`
  - Response:
    ```json
    {"message":"Welcome to TerraEnergy AI Engine API","status":"running"}
    ```
- `GET /health`
  - Response:
    ```json
    {"status":"ok"}
    ```
- `POST /aeos/energy-deviation`
  - Request:
    ```json
    {"actual": 120.5, "baseline": 100.0, "threshold_pct": 15.0}
    ```
  - Response:
    ```json
    {"deviation_pct": 20.5, "anomaly": true}
    ```
- `POST /aeos/energy-deviation/batch`
  - Request:
    ```json
    {
      "items": [
        {"actual": 120.5, "baseline": 100.0, "threshold_pct": 15.0},
        {"actual": 90.0, "baseline": 100.0}
      ]
    }
    ```
  - Response:
    ```json
    {
      "results": [
        {"deviation_pct": 20.5, "anomaly": true},
        {"deviation_pct": -10.0, "anomaly": null}
      ]
    }
    ```
- `POST /train` (multipart/form-data)
  - Fields:
    - `training_type`: `failure_risk`, `rul`, `anomaly`, `all`
    - `save_models`: `true` or `false`
    - `output_dir`: directory for saved models (optional)
    - `window_size`: feature window size (optional)
    - `data_path`: server-side CSV path (optional)
    - `file`: CSV upload (optional)
  - Example:
    ```bash
    curl -X POST http://localhost:8080/train ^
      -F "training_type=all" ^
      -F "save_models=true" ^
      -F "file=@data/generated/single_rig.csv"
    ```
- `POST /train/async` (multipart/form-data)
  - Same fields as `/train`, returns `job_id` immediately.
- `GET /train/status/{job_id}`
  - Returns progress, status, and result when finished.
- `POST /failure-risk/predict` (multipart/form-data)
  - Fields:
    - `window_size`: feature window size (optional)
    - `step`: window step size (optional)
    - `threshold`: risk threshold (optional)
    - `data_path`: server-side CSV path (optional)
    - `file`: CSV upload (optional)
  - Response includes risk scores per window timestamp.
- `POST /decision/risk`
  - Request:
    ```json
    {
      "equipment_id": "RIG_001",
      "failure_risk": 0.82,
      "anomaly_score": 2.4,
      "rul": 18,
      "max_rul": 100
    }
    ```
  - Response includes `risk_score`, `risk_level`, and recommended actions.
- `POST /fintech/report`
  - Request:
    ```json
    {
      "periods": [
        {"period": "2024-01", "maintenance_cost": 1200, "failure_cost": 0, "savings": 400}
      ]
    }
    ```
- `POST /fintech/report/pdf`
  - Request:
    ```json
    {
      "periods": [
        {"period": "2024-01", "total_cost": 1200}
      ]
    }
    ```
- `POST /reports/system/pdf`
  - Request:
    ```json
    {
      "dataset": "<pandas split-orient json>",
      "rig_id": "RIG_001",
      "metrics": ["power_consumption", "temperature"],
      "table_metric": "power_consumption",
      "anomaly_metrics": ["vibration", "temperature"],
      "anomaly_threshold": 3.0,
      "risk_metrics": ["power_consumption", "pressure"],
      "risk_window_size": 12,
      "risk_threshold": 0.7
    }
    ```
- `POST /fintech/forecast`
  - Request:
    ```json
    {
      "history": [{"period": "2024-01", "total_cost": 1200}],
      "periods": 6
    }
    ```
- `POST /fintech/roi`
  - Request:
    ```json
    {"implementation_cost": 50000, "annual_savings": 25000, "years": 3}
    ```
- `POST /fintech/compare`
  - Request:
    ```json
    {"preventive_cost": 8000, "failure_cost": 60000, "failure_probability": 0.2}
    ```
- `POST /fintech/cost-savings`
  - Request:
    ```json
    {"baseline_failures": 12, "actual_failures": 6, "preventive_costs": 10000}
    ```

### Notes

- `0.0.0.0` is a bind address. Use `http://localhost:8080` in your browser.
- `baseline` must be non-zero. The API returns 400 otherwise.
- Invalid or missing fields return 422 with FastAPI validation details.

## Dashboard (Plotly Dash)

Run locally:

```bash
python -m rigvisionx.serving.dashboard
```

Open:

- `http://localhost:8050`

You can also run it via Docker Compose (service: `terraenergy-ai-dashboard`).
Set `DASH_DATA_DIR` to point at your generated CSVs if needed.
To enable the Train panel, set `TRAIN_API_URL` (default: `http://localhost:8080/train/async`).
To enable the Failure Risk model panel, set `RISK_API_URL` (default: `http://localhost:8080/failure-risk/predict`).
To enable login, set `AUTH_API_URL` (default: `http://localhost:8080/auth`).

## Data Format (CSV)

The system expects a time-series CSV. Recommended columns:

- `timestamp` (ISO-8601 or parseable datetime)
- `rig_id` (string identifier per rig)
- Sensor metrics (numeric): `power_consumption`, `temperature`, `vibration`, `pressure`,
  `flow_rate`, `bearing_temperature`

Optional columns used by some views:

- `failure_indicator` or `failure_flag` (0/1)
- `estimated_rul` (numeric)

The synthetic generator emits a compatible schema.

## Model Artifacts

Training saves model files and metadata to the output directory (default `models/`):

- `{model_name}_model.pkl` (e.g., `failure_risk_model.pkl`)
- `baseline.pkl`
- `training_summary.yaml`

The failure risk inference endpoint loads `failure_risk_model.pkl` from `MODEL_DIR`.

## Environment Variables

API:

- `MODEL_DIR` (default `models/`)
- `RIGVISIONX_DB_PATH` (default `data/rigvisionx.db`)
- `RIGVISIONX_JWT_SECRET` (JWT signing secret)
- `RIGVISIONX_JWT_EXPIRE_MINUTES` (default `480`)
- `RIGVISIONX_ADMIN_USERS` (comma-separated admin usernames)

Dashboard:

- `DASH_DATA_DIR` (default `data/generated`)
- `DASH_HOST` (default `0.0.0.0`)
- `DASH_PORT` (default `8050`)
- `AUTH_API_URL` (default `http://localhost:8080/auth`)
- `TRAIN_API_URL` (default `http://localhost:8080/train/async`)
- `TRAIN_STATUS_URL` (defaults to `TRAIN_API_URL` with `/status`)
- `RISK_API_URL` (default `http://localhost:8080/failure-risk/predict`)
- `DECISION_API_URL` (default `http://localhost:8080/decision/risk`)
- `FINTECH_REPORT_URL` (default `http://localhost:8080/fintech/report`)
- `FINTECH_REPORT_PDF_URL` (default `http://localhost:8080/fintech/report/pdf`)
- `FINTECH_FORECAST_URL` (default `http://localhost:8080/fintech/forecast`)
- `FINTECH_ROI_URL` (default `http://localhost:8080/fintech/roi`)
- `FINTECH_COMPARE_URL` (default `http://localhost:8080/fintech/compare`)
- `FINTECH_SAVINGS_URL` (default `http://localhost:8080/fintech/cost-savings`)
- `SYSTEM_REPORT_PDF_URL` (default `http://localhost:8080/reports/system/pdf`)
- `DASH_TRADEMARK_TEXT`, `DASH_PRIVACY_LABEL`, `DASH_PRIVACY_URL`

Demo & Licensing:

- `RIGVISIONX_DEMO_MODE` (enable 7-day demo enforcement)
- `RIGVISIONX_DEMO_DAYS` (demo length in days)
- `RIGVISIONX_DEMO_STATE_PATH` (path to demo state file)
- `RIGVISIONX_LICENSE_KEY` (non-empty value disables demo enforcement)
- `RIGVISIONX_LICENSE_FILE` (optional path to license json/text)
- `RIGVISIONX_LICENSED` (force licensed mode)

## Docker Notes

When running with Docker Compose, the dashboard must talk to the API service name:

```
AUTH_API_URL=http://terraenergy-ai-ml:8080/auth
TRAIN_API_URL=http://terraenergy-ai-ml:8080/train/async
TRAIN_STATUS_URL=http://terraenergy-ai-ml:8080/train/status
RISK_API_URL=http://terraenergy-ai-ml:8080/failure-risk/predict
DECISION_API_URL=http://terraenergy-ai-ml:8080/decision/risk
FINTECH_REPORT_URL=http://terraenergy-ai-ml:8080/fintech/report
FINTECH_REPORT_PDF_URL=http://terraenergy-ai-ml:8080/fintech/report/pdf
FINTECH_FORECAST_URL=http://terraenergy-ai-ml:8080/fintech/forecast
FINTECH_ROI_URL=http://terraenergy-ai-ml:8080/fintech/roi
FINTECH_COMPARE_URL=http://terraenergy-ai-ml:8080/fintech/compare
FINTECH_SAVINGS_URL=http://terraenergy-ai-ml:8080/fintech/cost-savings
SYSTEM_REPORT_PDF_URL=http://terraenergy-ai-ml:8080/reports/system/pdf
```

Also set on the API container:

```
RIGVISIONX_DB_PATH=/app/data/rigvisionx.db
RIGVISIONX_JWT_SECRET=<your-secret>
RIGVISIONX_ADMIN_USERS=admin
```

## Java Example (Swing Training Panel)

Sample panel and client code is in `examples/java/TrainingPanel.java`.
It uses Java 11 `HttpClient` and sends `multipart/form-data` to the `/train` API.

## Configuration

Configuration files are stored in the `configs/` directory:
- `base.yaml` - Base configuration
- `model_failure.yaml` - Failure risk model config
- `model_rul.yaml` - RUL (Remaining Useful Life) model config
- `model_anomaly.yaml` - Anomaly detection model config

## Troubleshooting

- API returns 404 for model inference: train a model first and ensure
  `MODEL_DIR` points to the directory that contains `failure_risk_model.pkl`.
- Dashboard shows no data: confirm `DASH_DATA_DIR` or upload a CSV.
- Training fails with file errors: the API only accepts `.csv` uploads.

