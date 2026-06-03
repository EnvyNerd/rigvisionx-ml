import base64
import binascii
import io
import json
import mimetypes
import os
import uuid
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import Dash, Input, Output, State, callback, dcc, html, no_update, dash_table
from plotly.subplots import make_subplots

DATA_DIR = Path(os.environ.get("DASH_DATA_DIR", "data/generated"))
AUTH_API_BASE = os.environ.get("AUTH_API_URL", "http://localhost:8080/auth").rstrip("/")
AUTH_REGISTER_URL = f"{AUTH_API_BASE}/register"
AUTH_LOGIN_URL = f"{AUTH_API_BASE}/login"
SMOOTH_LINE_SHAPE = "spline"
SMOOTH_LINE_SMOOTHING = 1.2
LANG_DEFAULT = os.environ.get("DASH_LANGUAGE", "en").lower()
LABELS = {
    "en": {
        "app_title": "TerraEnergy AI",
        "app_tagline": "Where AI meets Energy intelligence.",
        "app_subtitle": "AI-Based Predictive Maintenance & Energy Optimization",
        "sign_in_title": "Sign in to TerraEnergy AI",
        "sign_in_subtitle": "Register a new account or sign in to continue.",
        "sign_in": "Sign In",
        "register_hint": "Register new user \u2193",
        "hide_register": "Hide registration \u2191",
        "register_title": "Register As New User",
        "register": "Register",
        "dataset": "Dataset",
        "upload_dataset": "Upload Dataset",
        "upload_hint": "Drag and drop or click",
        "rig": "Rig",
        "metrics": "Metrics",
        "metric_readings": "Metric Readings",
        "table_metric": "Table Metric",
        "eda_title": "Exploratory Data Analysis (EDA)",
        "eda_subtitle": "Quickly inspect distributions, missing values, and correlations.",
        "distribution_metric": "Distribution Metric",
        "correlation_metrics": "Correlation Metrics",
        "summary_stats": "Summary Stats",
        "missing_values": "Missing Values",
        "outliers": "Outliers",
        "outlier_metric": "Outlier Metric",
        "outlier_method": "Outlier Method",
        "z_threshold": "Z Threshold",
        "iqr_multiplier": "IQR Multiplier",
        "scatter_x_metric": "Scatter X Metric",
        "scatter_y_metric": "Scatter Y Metric",
        "anomaly_alerts": "Anomaly Alerts",
        "anomaly_metrics": "Anomaly Metrics",
        "z_score_threshold": "Z-Score Threshold",
        "failure_risk": "Failure Risk",
        "risk_metrics": "Risk Metrics",
        "use_trained_model": "Use Trained Model",
        "rolling_window": "Rolling Window",
        "risk_threshold": "Risk Threshold",
        "model_inference": "Model inference",
        "decision_engine": "Decision Engine",
        "use_failure_model": "Use Failure Risk Model",
        "max_rul": "Max RUL",
        "run_decision": "Run Decision",
        "download_system_report": "Download System Report",
        "fintech_insights": "FinTech Insights",
        "run_fintech": "Run FinTech Analysis",
        "download_fintech_pdf": "Download FinTech PDF",
        "ml_control_center": "ML Control Center",
        "training_type": "Training Type",
        "window_size": "Window Size",
        "output_dir": "Output Dir",
        "data_path": "Data Path (optional)",
        "attach_csv": "Attach CSV",
        "options": "Options",
        "save_models": "Save models on server",
        "train": "Train",
        "admin_banner": "Viewer mode: training and PDF reports require an admin account.",
        "language": "Language",
    },
    "ms": {
        "app_title": "TerraEnergy AI",
        "app_tagline": "Di mana AI bertemu kecerdasan tenaga.",
        "app_subtitle": "Penyelenggaraan Ramalan Berasaskan AI & Pengoptimuman Tenaga",
        "sign_in_title": "Log masuk ke TerraEnergy AI",
        "sign_in_subtitle": "Daftar akaun baharu atau log masuk untuk meneruskan.",
        "sign_in": "Log Masuk",
        "register_hint": "Daftar pengguna baharu \u2193",
        "hide_register": "Sembunyi pendaftaran \u2191",
        "register_title": "Daftar Sebagai Pengguna Baharu",
        "register": "Daftar",
        "dataset": "Set Data",
        "upload_dataset": "Muat Naik Set Data",
        "upload_hint": "Seret dan lepas atau klik",
        "rig": "Pelantar",
        "metrics": "Metrik",
        "metric_readings": "Bacaan Metrik",
        "table_metric": "Metrik Jadual",
        "eda_title": "Analisis Data Eksploratori (EDA)",
        "eda_subtitle": "Semak taburan, nilai hilang, dan korelasi dengan cepat.",
        "distribution_metric": "Metrik Taburan",
        "correlation_metrics": "Metrik Korelasi",
        "summary_stats": "Statistik Ringkas",
        "missing_values": "Nilai Hilang",
        "outliers": "Outlier",
        "outlier_metric": "Metrik Outlier",
        "outlier_method": "Kaedah Outlier",
        "z_threshold": "Ambang Z",
        "iqr_multiplier": "Pengganda IQR",
        "scatter_x_metric": "Metrik Serakan X",
        "scatter_y_metric": "Metrik Serakan Y",
        "anomaly_alerts": "Amaran Anomali",
        "anomaly_metrics": "Metrik Anomali",
        "z_score_threshold": "Ambang Z-Score",
        "failure_risk": "Risiko Kegagalan",
        "risk_metrics": "Metrik Risiko",
        "use_trained_model": "Guna Model Terlatih",
        "rolling_window": "Tetingkap Gelongsor",
        "risk_threshold": "Ambang Risiko",
        "model_inference": "Inferens model",
        "decision_engine": "Enjin Keputusan",
        "use_failure_model": "Guna Model Risiko Kegagalan",
        "max_rul": "RUL Maksimum",
        "run_decision": "Jalankan Keputusan",
        "download_system_report": "Muat Turun Laporan Sistem",
        "fintech_insights": "Wawasan FinTech",
        "run_fintech": "Jalankan Analisis FinTech",
        "download_fintech_pdf": "Muat Turun PDF FinTech",
        "ml_control_center": "Pusat Kawalan ML",
        "training_type": "Jenis Latihan",
        "window_size": "Saiz Tetingkap",
        "output_dir": "Direktori Output",
        "data_path": "Laluan Data (pilihan)",
        "attach_csv": "Lampirkan CSV",
        "options": "Pilihan",
        "save_models": "Simpan model di pelayan",
        "train": "Latih",
        "admin_banner": "Mod penonton: latihan dan laporan PDF memerlukan akaun admin.",
        "language": "Bahasa",
    },
}
AUTH_CONTAINER_STYLE = {
    "display": "flex",
    "justifyContent": "center",
    "alignItems": "center",
    "minHeight": "100vh",
    "position": "fixed",
    "inset": 0,
    "padding": "24px",
    "background": "rgba(250, 250, 250, 0.96)",
    "zIndex": "999",
}
AUTH_HIDDEN_STYLE = {"display": "none"}
LOGOUT_VISIBLE_STYLE = {
    "display": "inline-block",
    "padding": "6px 10px",
    "borderRadius": "6px",
    "border": "1px solid #ccc",
    "background": "#fff",
    "cursor": "pointer",
}
LOGOUT_HIDDEN_STYLE = {"display": "none"}
DEFAULT_METRICS = ["power_consumption", "temperature", "vibration"]
DEFAULT_ANOMALY_METRICS = ["power_consumption", "vibration", "temperature"]
DEFAULT_RISK_METRICS = ["power_consumption", "temperature", "vibration", "pressure"]
POWER_METRIC = "power_consumption"
ATTENTION_METRICS = [
    "power_consumption",
    "temperature",
    "vibration",
    "bearing_temperature",
    "pressure",
]
ATTENTION_ACTIONS = {
    "power_consumption": "Review load profile and shut down idle equipment.",
    "temperature": "Inspect cooling systems and verify airflow.",
    "bearing_temperature": "Inspect bearings and lubrication.",
    "vibration": "Check alignment, balance, and mounting.",
    "pressure": "Inspect pressure controls and valve integrity.",
}
BENCHMARKS = {
    "temperature": {
        "label": "Temperature",
        "unit": "deg C",
        "normal": "40-80",
        "warning": "80-90",
        "critical": ">90",
    },
    "bearing_temperature": {
        "label": "Bearing Temperature",
        "unit": "deg C",
        "normal": "40-85",
        "warning": "85-95",
        "critical": ">95",
    },
    "flow_rate": {
        "label": "Flow Rate",
        "unit": "m3/h",
        "normal": "50-80",
        "warning": "40-50 or 80-90",
        "critical": "<40 or >90",
    },
}
TRAIN_API_URL = os.environ.get("TRAIN_API_URL", "http://localhost:8080/train/async")
RISK_API_URL = os.environ.get("RISK_API_URL", "http://localhost:8080/failure-risk/predict")
DECISION_API_URL = os.environ.get("DECISION_API_URL", "http://localhost:8080/decision/risk")
FINTECH_REPORT_URL = os.environ.get("FINTECH_REPORT_URL", "http://localhost:8080/fintech/report")
FINTECH_REPORT_PDF_URL = os.environ.get(
    "FINTECH_REPORT_PDF_URL", "http://localhost:8080/fintech/report/pdf"
)
SYSTEM_REPORT_PDF_URL = os.environ.get(
    "SYSTEM_REPORT_PDF_URL", "http://localhost:8080/reports/system/pdf"
)
FINTECH_FORECAST_URL = os.environ.get("FINTECH_FORECAST_URL", "http://localhost:8080/fintech/forecast")
FINTECH_ROI_URL = os.environ.get("FINTECH_ROI_URL", "http://localhost:8080/fintech/roi")
FINTECH_COMPARE_URL = os.environ.get("FINTECH_COMPARE_URL", "http://localhost:8080/fintech/compare")
FINTECH_SAVINGS_URL = os.environ.get("FINTECH_SAVINGS_URL", "http://localhost:8080/fintech/cost-savings")
TRAIN_STATUS_URL = os.environ.get("TRAIN_STATUS_URL")
TRADEMARK_TEXT = os.environ.get(
    "DASH_TRADEMARK_TEXT",
    "TerraEnergy AI 2026.",
)
PRIVACY_POLICY_LABEL = os.environ.get("DASH_PRIVACY_LABEL", "Privacy & Policy")
PRIVACY_POLICY_URL = os.environ.get(
    "DASH_PRIVACY_URL",
    "https://terraenergy-ai.vercel.app/",
)
if not TRAIN_STATUS_URL:
    if TRAIN_API_URL.endswith("/train/async"):
        TRAIN_STATUS_URL = TRAIN_API_URL[: -len("/async")] + "/status"
    else:
        TRAIN_STATUS_URL = TRAIN_API_URL.rstrip("/") + "/status"


def _normalize_username(username: str | None) -> str:
    return (username or "").strip()


def _validate_credentials(username: str, password: str) -> str | None:
    if not username:
        return "Username is required."
    if len(username) < 3:
        return "Username must be at least 3 characters."
    if not password:
        return "Password is required."
    if len(password) < 6:
        return "Password must be at least 6 characters."
    return None


def _session_token(session_data: dict | None) -> str | None:
    if not isinstance(session_data, dict):
        return None
    token = session_data.get("token")
    return token if isinstance(token, str) and token else None


def _session_role(session_data: dict | None) -> str | None:
    if not isinstance(session_data, dict):
        return None
    role = session_data.get("role")
    return role if isinstance(role, str) and role else None


def _is_admin(session_data: dict | None) -> bool:
    return _session_role(session_data) == "admin"


def _extract_error(exc: urllib.error.HTTPError) -> str:
    try:
        body_text = exc.read().decode("utf-8")
    except Exception:
        return f"{exc.code}"
    try:
        payload = json.loads(body_text)
        detail = payload.get("detail")
        if detail:
            return f"{exc.code}: {detail}"
    except json.JSONDecodeError:
        pass
    return f"{exc.code}: {body_text}"


def _auth_request(url: str, payload: dict) -> tuple[dict | None, str | None]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw), None
    except urllib.error.HTTPError as exc:
        return None, _extract_error(exc)
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _build_session(payload: dict, fallback_username: str) -> tuple[dict | None, str]:
    token = payload.get("access_token")
    role = payload.get("role", "viewer")
    username = payload.get("username", fallback_username)
    if not isinstance(token, str) or not token:
        return None, "Authentication succeeded but no token was returned."
    if not isinstance(role, str):
        role = "viewer"
    if not isinstance(username, str):
        username = fallback_username
    session = {
        "username": username,
        "role": role,
        "token": token,
        "authenticated_at": datetime.now(timezone.utc).isoformat(),
    }
    return session, f"Signed in as {username} ({role})."


def _auth_register(username: str, password: str) -> tuple[dict | None, str, bool]:
    error = _validate_credentials(username, password)
    if error:
        return None, error, False
    response, auth_error = _auth_request(
        AUTH_REGISTER_URL,
        {"username": username, "password": password},
    )
    if auth_error or response is None:
        return None, auth_error or "Registration failed.", False
    session, message = _build_session(response, username)
    if session is None:
        return None, message, False
    return session, message, True


def _auth_login(username: str, password: str) -> tuple[dict | None, str, bool]:
    if not username or not password:
        return None, "Enter both username and password.", False
    response, auth_error = _auth_request(
        AUTH_LOGIN_URL,
        {"username": username, "password": password},
    )
    if auth_error or response is None:
        return None, auth_error or "Sign-in failed.", False
    session, message = _build_session(response, username)
    if session is None:
        return None, message, False
    return session, message, True


def _list_datasets() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted([path.name for path in DATA_DIR.glob("*.csv")])


def _load_dataset(filename: str) -> tuple[pd.DataFrame | None, str | None]:
    if not filename:
        return None, "No dataset selected."
    path = DATA_DIR / filename
    if not path.exists():
        return None, f"Dataset not found: {path}"
    try:
        data = pd.read_csv(path)
    except Exception as exc:
        return None, f"Failed to read dataset: {exc}"
    if data.empty:
        return None, "Dataset is empty."
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    return data, None


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14},
            }
        ],
    )
    return fig


def _apply_smoothing(fig: go.Figure) -> go.Figure:
    fig.update_traces(
        line_shape=SMOOTH_LINE_SHAPE,
        line={"smoothing": SMOOTH_LINE_SMOOTHING},
        selector={"type": "scatter"},
    )
    return fig


def _t(lang: str | None, key: str) -> str:
    lang_key = lang if lang in LABELS else LANG_DEFAULT
    return LABELS.get(lang_key, LABELS["en"]).get(key, key)


def _card(label: str, value: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"fontSize": "12px", "color": "#666"}),
            html.Div(value, style={"fontSize": "18px", "fontWeight": "600"}),
        ],
        style={
            "border": "1px solid #ddd",
            "borderRadius": "6px",
            "padding": "12px",
            "minWidth": "180px",
            "background": "#fff",
        },
    )


def _format_value(value: float | None, unit: str | None = None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    formatted = f"{value:,.{decimals}f}"
    return f"{formatted} {unit}" if unit else formatted


def _metric_label(metric: str) -> str:
    if metric == POWER_METRIC:
        return "Power Consumption (kW)"
    if metric == "temperature":
        return "Temperature (deg C)"
    if metric == "bearing_temperature":
        return "Bearing Temperature (deg C)"
    if metric == "flow_rate":
        return "Flow Rate (m3/h)"
    if metric == "pressure":
        return "Pressure (psi)"
    if metric == "vibration":
        return "Vibration (mm/s)"
    return metric.replace("_", " ").title()


def _benchmark_card(metric: str | None) -> html.Div:
    if not metric or metric not in BENCHMARKS:
        return html.Div(
            "No benchmark defined for the selected metric.",
            style={"fontSize": "12px", "color": "#666"},
        )

    info = BENCHMARKS[metric]
    unit = info["unit"]
    normal = f"{info['normal']} {unit}"
    warning = f"{info['warning']} {unit}"
    critical = f"{info['critical']} {unit}"

    return html.Div(
        [
            html.Div(
                f"{info['label']} Benchmarks",
                style={"fontSize": "12px", "color": "#555", "fontWeight": "600"},
            ),
            html.Div(f"Normal: {normal}", style={"fontSize": "12px"}),
            html.Div(f"Warning: {warning}", style={"fontSize": "12px"}),
            html.Div(f"Critical: {critical}", style={"fontSize": "12px"}),
        ],
        style={
            "border": "1px solid #e2e2e2",
            "borderRadius": "6px",
            "padding": "10px 12px",
            "background": "#fafafa",
            "display": "inline-block",
        },
    )


def _compute_energy_kwh(data: pd.DataFrame, power_col: str) -> float | None:
    if power_col not in data.columns or data.empty:
        return None
    power = pd.to_numeric(data[power_col], errors="coerce")
    if power.dropna().empty:
        return None
    if "timestamp" not in data.columns:
        return float(power.fillna(0).sum())

    frame = pd.DataFrame(
        {"time": pd.to_datetime(data["timestamp"], errors="coerce"), "power": power}
    ).dropna()
    if frame.empty:
        return None
    frame = frame.sort_values("time")
    if len(frame) < 2:
        return float(frame["power"].sum())

    dt_hours = frame["time"].diff().dt.total_seconds().div(3600)
    avg_power = (frame["power"] + frame["power"].shift(1)) / 2.0
    mask = dt_hours > 0
    energy = (avg_power[mask] * dt_hours[mask]).sum()
    if pd.isna(energy) or energy <= 0:
        return float(frame["power"].sum())
    return float(energy)


def _recommend_action(metric: str) -> str:
    return ATTENTION_ACTIONS.get(
        metric, "Inspect equipment and verify sensor readings."
    )


def _attention_card(problem: str, action: str) -> html.Div:
    return html.Div(
        [
            html.Div(
                "Attention",
                style={"fontSize": "12px", "color": "#9a4f00", "fontWeight": "600"},
            ),
            html.Div(
                problem,
                style={"fontSize": "16px", "fontWeight": "600", "color": "#7a2e0b"},
            ),
            html.Div(
                f"Action: {action}",
                style={"fontSize": "12px", "color": "#5a2b0a", "marginTop": "6px"},
            ),
        ],
        style={
            "border": "1px solid #f0c38b",
            "borderRadius": "6px",
            "padding": "12px",
            "minWidth": "220px",
            "background": "#fff7ec",
        },
    )


def _find_top_anomaly(
    data: pd.DataFrame, metrics: list[str], threshold: float
) -> dict | None:
    top: dict | None = None
    for metric in metrics:
        if metric not in data.columns:
            continue
        values = pd.to_numeric(data[metric], errors="coerce")
        mean = values.mean()
        std = values.std()
        if pd.isna(std) or std == 0:
            continue
        z_scores = (values - mean) / std
        abs_z = z_scores.abs()
        max_z = abs_z.max()
        if pd.isna(max_z):
            continue
        if top is None or max_z > top["z_score"]:
            idx = abs_z.idxmax()
            top = {
                "metric": metric,
                "value": float(values.loc[idx]),
                "z_score": float(max_z),
            }
    if top and top["z_score"] >= threshold:
        return top
    return None

def _decode_upload(contents: str | None) -> bytes | None:
    if not contents:
        return None
    if "," not in contents:
        return None
    encoded = contents.split(",", 1)[1]
    try:
        return base64.b64decode(encoded)
    except (binascii.Error, ValueError):
        return None


def _read_dataset_json(data_json: str) -> pd.DataFrame:
    return pd.read_json(io.StringIO(data_json), orient="split")

def _load_uploaded_dataset(
    contents: str | None, filename: str | None
) -> tuple[pd.DataFrame | None, str | None]:
    if not contents:
        return None, "No upload provided."
    decoded = _decode_upload(contents)
    if decoded is None:
        return None, "Unable to decode upload."
    try:
        data = pd.read_csv(io.BytesIO(decoded))
    except Exception as exc:
        name = filename or "uploaded file"
        return None, f"Failed to parse {name}: {exc}"
    if data.empty:
        return None, "Uploaded dataset is empty."
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    return data, None


def _encode_multipart(fields: dict, file_payload: tuple[str, bytes, str] | None) -> tuple[bytes, str]:
    boundary = f"----TerraEnergyAIBoundary{uuid.uuid4().hex}"
    body = bytearray()

    def _write(text: str) -> None:
        body.extend(text.encode("utf-8"))

    for name, value in fields.items():
        if value is None or value == "":
            continue
        _write(f"--{boundary}\r\n")
        _write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n')
        _write(f"{value}\r\n")

    if file_payload:
        filename, file_bytes, content_type = file_payload
        _write(f"--{boundary}\r\n")
        _write(
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        )
        _write(f"Content-Type: {content_type}\r\n\r\n")
        body.extend(file_bytes)
        _write("\r\n")

    _write(f"--{boundary}--\r\n")
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _post_json(
    url: str,
    payload: dict,
    timeout: int = 30,
    token: str | None = None,
) -> tuple[dict | None, str | None]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw), None
    except urllib.error.HTTPError as exc:
        return None, _extract_error(exc)
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _post_pdf(
    url: str,
    payload: dict,
    timeout: int = 30,
    token: str | None = None,
) -> tuple[bytes | None, str | None]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), None
    except urllib.error.HTTPError as exc:
        return None, _extract_error(exc)
    except urllib.error.URLError as exc:
        return None, str(exc)


def _parse_history_costs(history_costs: str | None) -> list[float]:
    values: list[float] = []
    if not history_costs:
        return values
    for token in history_costs.replace("\n", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _format_job_payload(payload: dict) -> str:
    lines = [
        f"Job: {payload.get('job_id', 'n/a')}",
        f"Status: {payload.get('status', 'unknown')}",
        f"Training Type: {payload.get('training_type', 'n/a')}",
        f"Data Source: {payload.get('data_source', 'n/a')}",
    ]
    progress = payload.get("progress", [])
    if progress:
        lines.append("")
        lines.append("Progress:")
        for step in progress:
            detail = f" ({step.get('detail')})" if step.get("detail") else ""
            lines.append(f"- {step.get('step')}: {step.get('status')}{detail}")
    if payload.get("error"):
        lines.append("")
        lines.append(f"Error: {payload.get('error')}")
    if payload.get("result"):
        lines.append("")
        lines.append("Result:")
        lines.append(json.dumps(payload["result"], indent=2))
    return "\n".join(lines)


def _metric_columns(data: pd.DataFrame) -> list[str]:
    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    excluded = {
        "failure_indicator",
        "failure_flag",
        "estimated_rul",
        "hour",
        "day",
        "day_of_week",
    }
    return [col for col in numeric_cols if col not in excluded]


def _eda_summary_rows(data: pd.DataFrame, metrics: list[str]) -> list[dict]:
    if data.empty or not metrics:
        return []
    frame = data[metrics].apply(pd.to_numeric, errors="coerce")
    desc = frame.describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["missing"] = frame.isna().sum()
    desc["missing_pct"] = (desc["missing"] / len(frame) * 100).round(2)
    desc = desc.reset_index().rename(columns={"index": "metric"})
    columns = [
        "metric",
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
        "missing",
        "missing_pct",
    ]
    available = [col for col in columns if col in desc.columns]
    return desc[available].round(4).to_dict(orient="records")


def _eda_missing_rows(data: pd.DataFrame, metrics: list[str]) -> list[dict]:
    if data.empty or not metrics:
        return []
    frame = data[metrics]
    missing = frame.isna().sum()
    pct = (missing / len(frame) * 100).round(2)
    rows = [
        {
            "metric": metric,
            "missing": int(missing.loc[metric]),
            "missing_pct": float(pct.loc[metric]),
        }
        for metric in metrics
    ]
    rows.sort(key=lambda row: row["missing"], reverse=True)
    return rows


def _eda_distribution_figure(
    data: pd.DataFrame,
    metric: str | None,
    outlier_metric: str | None,
    outlier_mask: pd.Series | None,
) -> go.Figure:
    if not metric or metric not in data.columns:
        return _empty_figure("Select a metric for distribution analysis.")
    series = pd.to_numeric(data[metric], errors="coerce")
    valid = series.dropna()
    if valid.empty:
        return _empty_figure("No numeric values available for this metric.")
    if series.empty:
        return _empty_figure("No numeric values available for this metric.")
    fig = px.histogram(
        x=valid,
        nbins=40,
        labels={"x": metric, "y": "Count"},
        title=f"Distribution of {metric}",
    )
    if outlier_mask is not None and metric == outlier_metric:
        mask = outlier_mask.reindex(series.index).fillna(False).astype(bool)
        outlier_values = series.loc[mask].dropna()
        if not outlier_values.empty:
            fig.add_trace(
                go.Scatter(
                    x=outlier_values,
                    y=np.zeros(len(outlier_values)),
                    mode="markers",
                    name="Outliers",
                    marker={"color": "#d62728", "size": 9, "symbol": "x"},
                    hovertemplate="Outlier: %{x:.4f}<extra></extra>",
                )
            )
            fig.update_yaxes(rangemode="tozero")
    fig.update_layout(margin={"t": 48, "l": 40, "r": 20, "b": 40})
    return fig


def _eda_correlation_figure(
    data: pd.DataFrame, metrics: list[str], rig_value: str | None
) -> tuple[go.Figure, str]:
    if not metrics:
        return _empty_figure("Select metrics to compute correlations."), ""
    frame = data
    if rig_value and rig_value != "ALL" and "rig_id" in frame.columns:
        frame = frame[frame["rig_id"] == rig_value]
    numeric = frame[metrics].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr()
    if corr.empty or len(corr.columns) < 2:
        return _empty_figure("Need at least two numeric metrics."), ""
    fig = go.Figure(
        data=
        [
            go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                colorbar={"title": "Corr"},
                hovertemplate="x=%{x}<br>y=%{y}<br>corr=%{z:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Correlation Heatmap",
        margin={"t": 48, "l": 60, "r": 20, "b": 60},
    )
    note = ""
    if len(numeric) < 10:
        note = "Correlation is based on a small sample."
    return fig, note


def _eda_outlier_mask_and_score(
    data: pd.DataFrame,
    metric: str,
    method: str,
    z_threshold: float,
    iqr_k: float,
) -> tuple[pd.Series | None, pd.Series | None, str | None]:
    if metric not in data.columns:
        return None, None, "Select a valid metric for outlier detection."

    series = pd.to_numeric(data[metric], errors="coerce")
    valid = series.dropna()
    if len(valid) < 4:
        return None, None, "Not enough numeric values to detect outliers."

    method_key = (method or "zscore").lower()
    score = pd.Series(np.nan, index=series.index, dtype="float64")
    mask = pd.Series(False, index=series.index)
    note = ""

    if method_key == "iqr":
        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            return None, None, "IQR is zero; outlier detection is not meaningful."
        lower = q1 - iqr_k * iqr
        upper = q3 + iqr_k * iqr
        mask = (series < lower) | (series > upper)
        score = ((series - valid.median()).abs() / iqr).astype("float64")
        note = f"IQR bounds: [{lower:,.3f}, {upper:,.3f}]"
    else:
        mean = valid.mean()
        std = valid.std()
        if pd.isna(std) or std == 0:
            return None, None, "Standard deviation is zero; z-score outliers are undefined."
        z_scores = (series - mean) / std
        score = z_scores.abs().astype("float64")
        mask = score >= float(z_threshold)
        note = f"Z-score threshold: {float(z_threshold):.2f}"

    return mask.astype(bool), score.astype("float64"), note


def _eda_outlier_rows(
    data: pd.DataFrame,
    metric: str,
    method: str,
    z_threshold: float,
    iqr_k: float,
) -> tuple[list[dict], str, pd.Series | None]:
    mask, score, note = _eda_outlier_mask_and_score(
        data, metric, method, z_threshold, iqr_k
    )
    if mask is None or score is None or note is None:
        return [], note or "Outlier detection failed.", None

    outliers = data.loc[mask].copy()
    if outliers.empty:
        return [], f"No outliers detected. {note}", mask

    outliers["outlier_score"] = score.loc[outliers.index].round(3)
    outliers[metric] = pd.to_numeric(outliers[metric], errors="coerce").round(4)

    rows: list[dict] = []
    for idx, row in outliers.iterrows():
        record: dict[str, object] = {
            "index": str(idx),
            "metric": metric,
            "value": row.get(metric),
            "outlier_score": row.get("outlier_score"),
        }
        if "timestamp" in outliers.columns:
            ts_value = row.get("timestamp")
            record["timestamp"] = (
                ts_value.isoformat() if hasattr(ts_value, "isoformat") else str(ts_value)
            )
        if "rig_id" in outliers.columns:
            record["rig_id"] = str(row.get("rig_id"))
        rows.append(record)

    rows.sort(key=lambda item: float(item.get("outlier_score") or 0), reverse=True)
    capped_rows = rows[:200]
    summary = f"Detected {len(rows)} outliers (showing up to {len(capped_rows)}). {note}"
    return capped_rows, summary, mask


def _eda_scatter_figure(
    data: pd.DataFrame,
    x_metric: str | None,
    y_metric: str | None,
    outlier_mask: pd.Series | None,
    outlier_metric: str | None,
) -> tuple[go.Figure, str]:
    if not x_metric or not y_metric:
        return _empty_figure("Select two metrics to explore their relationship."), ""
    if x_metric == y_metric:
        return _empty_figure("Choose two different metrics."), ""
    if x_metric not in data.columns or y_metric not in data.columns:
        return _empty_figure("Selected metrics are not available in the dataset."), ""

    frame = data[[x_metric, y_metric]].apply(pd.to_numeric, errors="coerce")
    if outlier_mask is not None:
        mask = outlier_mask.reindex(frame.index).fillna(False).astype(bool)
        frame["outlier_label"] = np.where(mask, "Outlier", "Normal")
    else:
        frame["outlier_label"] = "Normal"
    frame = frame.dropna(subset=[x_metric, y_metric])
    if frame.empty:
        return _empty_figure("No paired numeric values available."), ""

    fig = px.scatter(
        frame,
        x=x_metric,
        y=y_metric,
        color="outlier_label",
        color_discrete_map={"Normal": "#1f77b4", "Outlier": "#d62728"},
        opacity=0.65,
        title=f"{x_metric} vs {y_metric}",
    )
    for trace in fig.data:
        if trace.name == "Outlier":
            trace.update(marker={"size": 10, "symbol": "diamond"})
        else:
            trace.update(marker={"size": 7})

    note = ""
    if len(frame) >= 2:
        x_values = frame[x_metric].to_numpy(dtype="float64")
        y_values = frame[y_metric].to_numpy(dtype="float64")
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x_line = np.linspace(float(x_values.min()), float(x_values.max()), 60)
        y_line = slope * x_line + intercept
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Trendline",
                line={"color": "#d62728", "width": 2},
            )
        )
        corr = frame[x_metric].corr(frame[y_metric])
        if not pd.isna(corr):
            note = f"Pearson correlation: {corr:.3f}"
    if outlier_metric:
        suffix = f"Outliers are based on {outlier_metric}."
        note = f"{note} {suffix}".strip()

    _apply_smoothing(fig)
    fig.update_layout(margin={"t": 48, "l": 40, "r": 20, "b": 40})
    return fig, note


dataset_files = _list_datasets()

external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&display=swap"
]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "TerraEnergy AI-Based Predictive Maintenance & Energy Optimization"

app.layout = html.Div(
    [
        dcc.Store(id="auth-session", storage_type="session"),
        dcc.Store(id="register-visible", data=False),
        dcc.Store(id="lang-store", data=LANG_DEFAULT, storage_type="local"),
        html.Div(
            id="auth-overlay",
            style=AUTH_CONTAINER_STYLE,
            children=[
                html.Div(
                    [
                        html.H2(id="label-sign-in-title", style={"marginBottom": "4px"}),
                        html.Div(
                            id="label-sign-in-subtitle",
                            style={"color": "#666", "marginBottom": "16px"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3(id="label-sign-in", style={"marginTop": "0"}),
                                        dcc.Input(
                                            id="login-username",
                                            type="text",
                                            placeholder="Username",
                                            style={
                                                "width": "100%",
                                                "padding": "8px",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        dcc.Input(
                                            id="login-password",
                                            type="password",
                                            placeholder="Password",
                                            style={
                                                "width": "100%",
                                                "padding": "8px",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        html.Button(
                                            html.Span(id="label-sign-in-button"),
                                            id="login-button",
                                            n_clicks=0,
                                            style={
                                                "width": "100%",
                                                "padding": "8px 10px",
                                                "borderRadius": "6px",
                                                "border": "1px solid #1a73e8",
                                                "background": "#1a73e8",
                                                "color": "#fff",
                                                "cursor": "pointer",
                                            },
                                        ),
                                        html.Button(
                                            html.Span(id="label-register-toggle"),
                                            id="register-toggle",
                                            n_clicks=0,
                                            style={
                                                "marginTop": "8px",
                                                "padding": "0",
                                                "border": "none",
                                                "background": "none",
                                                "color": "#1a73e8",
                                                "cursor": "pointer",
                                                "textAlign": "left",
                                                "fontSize": "12px",
                                            },
                                        ),
                                        html.Div(
                                            id="login-message",
                                            style={"marginTop": "8px", "fontSize": "12px"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    id="register-section",
                                    style={"display": "none"},
                                    children=[
                                        html.Hr(style={"margin": "4px 0 0 0"}),
                                        html.Div(
                                            [
                                                html.H3(id="label-register-title", style={"marginTop": "8px"}),
                                                dcc.Input(
                                                    id="register-username",
                                                    type="text",
                                                    placeholder="Username",
                                                    style={
                                                        "width": "100%",
                                                        "padding": "8px",
                                                        "marginBottom": "8px",
                                                    },
                                                ),
                                                dcc.Input(
                                                    id="register-password",
                                                    type="password",
                                                    placeholder="Password",
                                                    style={
                                                        "width": "100%",
                                                        "padding": "8px",
                                                        "marginBottom": "8px",
                                                    },
                                                ),
                                                dcc.Input(
                                                    id="register-confirm",
                                                    type="password",
                                                    placeholder="Confirm password",
                                                    style={
                                                        "width": "100%",
                                                        "padding": "8px",
                                                        "marginBottom": "8px",
                                                    },
                                                ),
                                                html.Button(
                                                    html.Span(id="label-register-button"),
                                                    id="register-button",
                                                    n_clicks=0,
                                                    style={
                                                        "width": "100%",
                                                        "padding": "8px 10px",
                                                        "borderRadius": "6px",
                                                        "border": "1px solid #0b8043",
                                                        "background": "#0b8043",
                                                        "color": "#fff",
                                                        "cursor": "pointer",
                                                    },
                                                ),
                                                html.Div(
                                                    id="register-message",
                                                    style={
                                                        "marginTop": "8px",
                                                        "fontSize": "12px",
                                                    },
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "12px",
                                "maxWidth": "460px",
                                "margin": "0 auto",
                            },
                        ),
                    ],
                    style={
                        "border": "1px solid #e2e2e2",
                        "borderRadius": "10px",
                        "padding": "20px",
                        "background": "#fff",
                        "boxShadow": "0 8px 24px rgba(0,0,0,0.06)",
                        "maxWidth": "520px",
                        "width": "100%",
                    },
                )
            ],
        ),
        html.Div(
            [
                html.H1(id="label-app-title", style={"marginBottom": "4px"}),
                html.Div(
                    id="label-app-tagline",
                    style={"color": "#666", "marginBottom": "16px"},
                ),
                html.Div(
                    id="label-app-subtitle",
                    style={"color": "#666", "marginBottom": "12px"},
                ),
                html.Div(
                    [
                        html.Span(
                            id="auth-user-label",
                            style={"fontSize": "12px", "color": "#444"},
                        ),
                        html.Button(
                            "Logout",
                            id="logout-button",
                            n_clicks=0,
                            style=LOGOUT_HIDDEN_STYLE,
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "8px",
                        "alignItems": "center",
                        "marginTop": "8px",
                    },
                ),
                html.Div(id="admin-banner", style={"marginTop": "10px"}),
                html.Div(
                    [
                        html.Span(id="label-language", style={"fontSize": "12px", "color": "#555"}),
                        dcc.RadioItems(
                            id="lang-toggle",
                            options=[
                                {"label": "EN", "value": "en"},
                                {"label": "MS", "value": "ms"},
                            ],
                            value=LANG_DEFAULT,
                            inline=True,
                            style={"marginLeft": "8px"},
                        ),
                    ],
                    style={"marginTop": "8px"},
                ),
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(id="label-dataset"),
                        dcc.Dropdown(
                            id="dataset-dropdown",
                            options=[
                                {"label": name, "value": name}
                                for name in dataset_files
                            ],
                            value=dataset_files[0] if dataset_files else None,
                            clearable=False,
                        ),
                    ],
                    style={"minWidth": "280px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Label(id="label-upload-dataset"),
                        dcc.Upload(
                            id="dataset-upload",
                            children=html.Div(id="label-upload-hint"),
                            multiple=False,
                            style={
                                "width": "100%",
                                "padding": "5px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "textAlign": "center",
                                "background": "#fafafa",
                            },
                        ),
                        html.Div(id="dataset-upload-name", style={"marginTop": "6px"}),
                    ],
                    style={"minWidth": "220px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Label(id="label-rig"),
                        dcc.Dropdown(
                            id="rig-dropdown",
                            options=[{"label": "ALL", "value": "ALL"}],
                            value="ALL",
                            clearable=False,
                        ),
                    ],
                    style={"minWidth": "200px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Label(id="label-metrics"),
                        dcc.Dropdown(
                            id="metric-dropdown",
                            options=[],
                            value=[],
                            multi=True,
                        ),
                    ],
                    style={"minWidth": "320px", "flex": "2"},
                ),
            ],
            style={
                "display": "flex",
                "gap": "16px",
                "flexWrap": "wrap",
                "marginBottom": "12px",
            },
        ),
        html.Div(id="dataset-warning", style={"color": "#b00", "marginBottom": "8px"}),
        dcc.Store(id="dataset-store"),
        dcc.Store(id="train-job-id"),
        dcc.Interval(id="train-interval", interval=2000, disabled=True),
        html.Div(
            id="summary-cards",
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
        ),
        html.Div(
            [
                dcc.Graph(id="sensor-graph"),
                dcc.Graph(id="status-graph"),
            ],
            style={"marginTop": "16px"},
        ),
        html.H2(id="label-metric-readings", style={"marginTop": "24px"}),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(id="label-table-metric"),
                        dcc.Dropdown(
                            id="table-metric-dropdown",
                            options=[],
                            value=None,
                            clearable=False,
                        ),
                    ],
                    style={"minWidth": "260px", "flex": "1"},
                ),
            ],
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
        ),
        dash_table.DataTable(
            id="metric-table",
            columns=[],
            data=[],
            page_size=12,
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": "12px"},
            style_header={"fontWeight": "600", "backgroundColor": "#f4f4f4"},
        ),
        html.Div(id="metric-benchmark", style={"marginTop": "8px"}),
        html.H2(id="label-eda-title", style={"marginTop": "24px"}),
        html.Div(
            id="label-eda-subtitle",
            style={"color": "#666", "marginBottom": "8px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(id="label-distribution-metric"),
                        dcc.Dropdown(
                            id="eda-metric-dropdown",
                            options=[],
                            value=None,
                            clearable=False,
                        ),
                    ],
                    style={"minWidth": "240px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Label(id="label-correlation-metrics"),
                        dcc.Dropdown(
                            id="eda-corr-metric-dropdown",
                            options=[],
                            value=[],
                            multi=True,
                        ),
                    ],
                    style={"minWidth": "320px", "flex": "2"},
                ),
            ],
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(id="label-outlier-metric"),
                        dcc.Dropdown(
                            id="eda-outlier-metric-dropdown",
                            options=[],
                            value=None,
                            clearable=False,
                        ),
                    ],
                    style={"minWidth": "220px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Label(id="label-outlier-method"),
                        dcc.Dropdown(
                            id="eda-outlier-method-dropdown",
                            options=[
                                {"label": "Z-score", "value": "zscore"},
                                {"label": "IQR", "value": "iqr"},
                            ],
                            value="zscore",
                            clearable=False,
                        ),
                    ],
                    style={"minWidth": "160px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Label(id="label-z-threshold-short"),
                        dcc.Input(
                            id="eda-outlier-z-threshold",
                            type="number",
                            min=0.5,
                            step=0.1,
                            value=3.0,
                            style={"width": "100%"},
                        ),
                    ],
                    style={"minWidth": "140px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Label(id="label-iqr-multiplier"),
                        dcc.Input(
                            id="eda-outlier-iqr-k",
                            type="number",
                            min=0.5,
                            step=0.1,
                            value=1.5,
                            style={"width": "100%"},
                        ),
                    ],
                    style={"minWidth": "140px", "flex": "1"},
                ),
            ],
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginTop": "4px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(id="label-scatter-x"),
                        dcc.Dropdown(
                            id="eda-scatter-x-dropdown",
                            options=[],
                            value=None,
                            clearable=False,
                        ),
                    ],
                    style={"minWidth": "240px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Label(id="label-scatter-y"),
                        dcc.Dropdown(
                            id="eda-scatter-y-dropdown",
                            options=[],
                            value=None,
                            clearable=False,
                        ),
                    ],
                    style={"minWidth": "240px", "flex": "1"},
                ),
            ],
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginTop": "4px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H4(html.Span(id="label-summary-stats"), style={"marginBottom": "6px"}),
                        dash_table.DataTable(
                            id="eda-summary-table",
                            columns=[],
                            data=[],
                            page_size=8,
                            style_table={"overflowX": "auto"},
                            style_cell={"fontSize": "12px"},
                            style_header={
                                "fontWeight": "600",
                                "backgroundColor": "#f4f4f4",
                            },
                        ),
                    ],
                    style={"flex": "2", "minWidth": "360px"},
                ),
                html.Div(
                    [
                        html.H4(html.Span(id="label-missing-values"), style={"marginBottom": "6px"}),
                        dash_table.DataTable(
                            id="eda-missing-table",
                            columns=[],
                            data=[],
                            page_size=8,
                            style_table={"overflowX": "auto"},
                            style_cell={"fontSize": "12px"},
                            style_header={
                                "fontWeight": "600",
                                "backgroundColor": "#f4f4f4",
                            },
                        ),
                    ],
                    style={"flex": "1", "minWidth": "260px"},
                ),
                html.Div(
                    [
                        html.H4(html.Span(id="label-outliers"), style={"marginBottom": "6px"}),
                        dash_table.DataTable(
                            id="eda-outlier-table",
                            columns=[],
                            data=[],
                            page_size=8,
                            style_table={"overflowX": "auto"},
                            style_cell={"fontSize": "12px"},
                            style_header={
                                "fontWeight": "600",
                                "backgroundColor": "#f4f4f4",
                            },
                        ),
                        html.Div(
                            id="eda-outlier-note",
                            style={
                                "marginTop": "6px",
                                "color": "#8a6d3b",
                                "fontSize": "12px",
                            },
                        ),
                    ],
                    style={"flex": "1.4", "minWidth": "300px"},
                ),
            ],
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginTop": "8px"},
        ),
        html.Div(
            [
                dcc.Graph(id="eda-dist-graph"),
                dcc.Graph(id="eda-corr-graph"),
                dcc.Graph(id="eda-scatter-graph"),
            ],
            style={"marginTop": "12px"},
        ),
        html.Div(
            id="eda-corr-note",
            style={"marginTop": "6px", "color": "#8a6d3b", "fontSize": "12px"},
        ),
        html.Div(
            id="eda-scatter-note",
            style={"marginTop": "4px", "color": "#8a6d3b", "fontSize": "12px"},
        ),
        html.H2(id="label-anomaly-alerts", style={"marginTop": "24px"}),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(id="label-anomaly-metrics"),
                        dcc.Dropdown(
                            id="anomaly-metric-dropdown",
                            options=[],
                            value=[],
                            multi=True,
                        ),
                    ],
                    style={"minWidth": "260px", "flex": "2"},
                ),
                html.Div(
                    [
                        html.Label(id="label-z-threshold"),
                        dcc.Input(
                            id="anomaly-threshold",
                            type="number",
                            min=0.1,
                            step=0.1,
                            value=3.0,
                            style={"width": "100%"},
                        ),
                    ],
                    style={"minWidth": "180px", "flex": "1"},
                ),
            ],
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
        ),
        html.Div(id="anomaly-banner", style={"marginTop": "12px", "fontWeight": "600"}),
        dash_table.DataTable(
            id="anomaly-table",
            columns=[],
            data=[],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": "12px"},
            style_header={"fontWeight": "600", "backgroundColor": "#f4f4f4"},
        ),
        html.H2(id="label-failure-risk", style={"marginTop": "24px"}),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(id="label-risk-metrics"),
                        dcc.Dropdown(
                            id="risk-metric-dropdown",
                            options=[],
                            value=[],
                            multi=True,
                        ),
                    ],
                    style={"minWidth": "260px", "flex": "2"},
                ),
                html.Div(
                    [
                        html.Label(id="label-use-trained-model"),
                        dcc.Checklist(
                            id="risk-use-model",
                            options=[{"label": _t(LANG_DEFAULT, "model_inference"), "value": "use"}],
                            value=["use"],
                        ),
                    ],
                    style={"minWidth": "180px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Label(id="label-rolling-window"),
                        dcc.Input(
                            id="risk-window-size",
                            type="number",
                            min=1,
                            step=1,
                            value=12,
                            style={"width": "100%"},
                        ),
                    ],
                    style={"minWidth": "160px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Label(id="label-risk-threshold"),
                        dcc.Input(
                            id="risk-threshold",
                            type="number",
                            min=0.1,
                            step=0.05,
                            value=0.7,
                            style={"width": "100%"},
                        ),
                    ],
                    style={"minWidth": "160px", "flex": "1"},
                ),
            ],
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
        ),
        html.Div(id="risk-banner", style={"marginTop": "12px", "fontWeight": "600"}),
        dcc.Graph(id="risk-graph"),
        dash_table.DataTable(
            id="risk-table",
            columns=[],
            data=[],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": "12px"},
            style_header={"fontWeight": "600", "backgroundColor": "#f4f4f4"},
        ),
        html.H2(id="label-decision-engine", style={"marginTop": "24px"}),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(id="label-use-failure-model"),
                        dcc.Checklist(
                            id="decision-use-model",
                            options=[{"label": _t(LANG_DEFAULT, "model_inference"), "value": "use"}],
                            value=["use"],
                        ),
                    ],
                    style={
                        "minWidth": "220px",
                        "flex": "1",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "6px",
                    },
                ),
                html.Div(
                    [
                        html.Label(id="label-max-rul"),
                        dcc.Input(
                            id="decision-max-rul",
                            type="number",
                            min=1,
                            step=1,
                            value=100,
                            style={"width": "100%", "height": "38px"},
                        ),
                    ],
                    style={
                        "minWidth": "160px",
                        "flex": "1",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "6px",
                    },
                ),
                html.Div(
                    [
                        html.Label(" "),
                        html.Button(
                            html.Span(id="label-run-decision"),
                            id="decision-run",
                            n_clicks=0,
                            style={"width": "100%", "height": "38px"},
                        ),
                    ],
                    style={
                        "minWidth": "160px",
                        "flex": "1",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "6px",
                    },
                ),
            ],
            style={
                "display": "flex",
                "gap": "16px",
                "flexWrap": "wrap",
                "alignItems": "flex-end",
            },
        ),
        html.Div(id="decision-output", style={"marginTop": "12px", "fontWeight": "600"}),
        html.Button(
            html.Span(id="label-download-system-report"),
            id="system-report-download",
            n_clicks=0,
            disabled=False,
            style={"marginTop": "12px"},
        ),
        dcc.Download(id="system-report-file"),
        html.H2(id="label-fintech-insights", style={"marginTop": "24px"}),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Implementation Cost"),
                        dcc.Input(
                            id="fintech-impl-cost",
                            type="number",
                            min=0,
                            step=1000,
                            value=50000,
                            style={"width": "100%", "height": "38px"},
                        ),
                        html.Label("Annual Savings"),
                        dcc.Input(
                            id="fintech-annual-savings",
                            type="number",
                            min=0,
                            step=1000,
                            value=25000,
                            style={"width": "100%", "height": "38px"},
                        ),
                        html.Label("Years"),
                        dcc.Input(
                            id="fintech-years",
                            type="number",
                            min=1,
                            step=1,
                            value=3,
                            style={"width": "100%", "height": "38px"},
                        ),
                        html.Label("Monthly Savings"),
                        dcc.Input(
                            id="fintech-monthly-savings",
                            type="number",
                            min=0,
                            step=500,
                            value=2000,
                            style={"width": "100%", "height": "38px"},
                        ),
                    ],
                    style={
                        "minWidth": "220px",
                        "flex": "1",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "6px",
                    },
                ),
                html.Div(
                    [
                        html.Label("Preventive Cost"),
                        dcc.Input(
                            id="fintech-preventive-cost",
                            type="number",
                            min=0,
                            step=500,
                            value=8000,
                            style={"width": "100%", "height": "38px"},
                        ),
                        html.Label("Failure Cost"),
                        dcc.Input(
                            id="fintech-failure-cost",
                            type="number",
                            min=0,
                            step=1000,
                            value=60000,
                            style={"width": "100%", "height": "38px"},
                        ),
                        html.Label("Failure Probability"),
                        dcc.Input(
                            id="fintech-failure-prob",
                            type="number",
                            min=0,
                            max=1,
                            step=0.05,
                            value=0.2,
                            style={"width": "100%", "height": "38px"},
                        ),
                    ],
                    style={
                        "minWidth": "220px",
                        "flex": "1",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "6px",
                    },
                ),
                html.Div(
                    [
                        html.Label("Baseline Failures"),
                        dcc.Input(
                            id="fintech-baseline-failures",
                            type="number",
                            min=0,
                            step=1,
                            value=12,
                            style={"width": "100%", "height": "38px"},
                        ),
                        html.Label("Actual Failures"),
                        dcc.Input(
                            id="fintech-actual-failures",
                            type="number",
                            min=0,
                            step=1,
                            value=6,
                            style={"width": "100%", "height": "38px"},
                        ),
                        html.Label("Preventive Costs"),
                        dcc.Input(
                            id="fintech-preventive-costs",
                            type="number",
                            min=0,
                            step=500,
                            value=10000,
                            style={"width": "100%", "height": "38px"},
                        ),
                    ],
                    style={
                        "minWidth": "220px",
                        "flex": "1",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "6px",
                    },
                ),
                html.Div(
                    [
                        html.Label("Historical Costs (comma-separated)"),
                        dcc.Textarea(
                            id="fintech-history-costs",
                            value="12000, 13500, 12800, 14200",
                            style={"width": "100%", "height": "96px"},
                        ),
                        html.Label("Forecast Periods"),
                        dcc.Input(
                            id="fintech-forecast-periods",
                            type="number",
                            min=1,
                            step=1,
                            value=6,
                            style={"width": "100%", "height": "38px"},
                        ),
                    ],
                    style={
                        "minWidth": "260px",
                        "flex": "2",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "6px",
                    },
                ),
            ],
            style={
                "display": "flex",
                "gap": "16px",
                "flexWrap": "wrap",
                "alignItems": "flex-end",
            },
        ),
        html.Div(
            [
                html.Button(
                    html.Span(id="label-run-fintech"),
                    id="fintech-run",
                    n_clicks=0,
                ),
                html.Button(
                    html.Span(id="label-download-fintech-pdf"),
                    id="fintech-download",
                    n_clicks=0,
                    disabled=False,
                ),
                dcc.Download(id="fintech-pdf-download"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginTop": "12px"},
        ),
        html.Div(id="fintech-summary", style={"marginTop": "12px", "fontWeight": "600"}),
        dash_table.DataTable(
            id="fintech-forecast-table",
            columns=[],
            data=[],
            page_size=6,
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": "12px"},
            style_header={"fontWeight": "600", "backgroundColor": "#f4f4f4"},
        ),
        html.H2(id="label-ml-control-center", style={"marginTop": "24px"}),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(id="label-training-type"),
                        dcc.Dropdown(
                            id="train-type",
                            options=[
                                {"label": "Failure Risk", "value": "failure_risk"},
                                {"label": "RUL", "value": "rul"},
                                {"label": "Anomaly", "value": "anomaly"},
                                {"label": "All", "value": "all"},
                            ],
                            value="all",
                            clearable=False,
                        ),
                    ],
                    style={"minWidth": "200px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Label(id="label-window-size"),
                        dcc.Input(
                            id="train-window-size",
                            type="number",
                            min=1,
                            step=1,
                            value=24,
                            style={"width": "100%"},
                        ),
                    ],
                    style={"minWidth": "160px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Label(id="label-output-dir"),
                        dcc.Input(
                            id="train-output-dir",
                            value="models/",
                            style={"width": "100%"},
                        ),
                    ],
                    style={"minWidth": "200px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Label(id="label-data-path"),
                        dcc.Input(
                            id="train-data-path",
                            placeholder="data/processed/training_data.csv",
                            style={"width": "100%"},
                        ),
                    ],
                    style={"minWidth": "260px", "flex": "2"},
                ),
            ],
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(id="label-attach-csv"),
                        dcc.Upload(
                            id="train-upload",
                            children=html.Div(id="label-attach-csv-hint"),
                            multiple=False,
                            style={
                                "width": "100%",
                                "padding": "5px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "textAlign": "center",
                                "background": "#fafafa",
                            },
                        ),
                        html.Div(id="train-upload-name", style={"marginTop": "6px"}),
                    ],
                    style={"minWidth": "320px"},
                ),
                html.Div(
                    [
                        html.Label(id="label-options"),
                        dcc.Checklist(
                            id="train-save-models",
                            options=[{"label": _t(LANG_DEFAULT, "save_models"), "value": "save"}],
                            value=["save"],
                        ),
                        html.Button(
                            html.Span(id="label-train"),
                            id="train-button",
                            n_clicks=0,
                            disabled=False,
                            style={"marginTop": "8px"},
                        ),
                        html.Div(id="train-status", style={"marginTop": "8px"}),
                    ],
                    style={"minWidth": "220px"},
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))",
                "gap": "16px",
                "alignItems": "start",
                "marginTop": "12px",
            },
        ),
        dcc.Loading(
            html.Pre(
                id="train-output",
                style={
                    "background": "#111",
                    "color": "#eee",
                    "padding": "12px",
                    "borderRadius": "6px",
                    "whiteSpace": "pre-wrap",
                },
            ),
            type="default",
        ),
        html.Hr(style={"marginTop": "32px"}),
        html.Footer(
            [
                html.Div(
                    f"(c) {datetime.now(timezone.utc).year} TerraEnergy AI. All rights reserved.",
                    style={"color": "#666", "fontSize": "12px"},
                ),
                html.Div(
                    [
                        html.Span(
                            TRADEMARK_TEXT,
                            style={"color": "#666", "fontSize": "12px"},
                        ),
                        html.Span(" | ", style={"color": "#999", "margin": "0 6px"}),
                        html.A(
                            PRIVACY_POLICY_LABEL,
                            href=PRIVACY_POLICY_URL,
                            target="_blank",
                            rel="noopener",
                            style={"color": "#1a73e8", "fontSize": "12px"},
                        ),
                    ]
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "gap": "12px",
                "flexWrap": "wrap",
            },
        ),
    ],
    style={
        "maxWidth": "1200px",
        "margin": "0 auto",
        "padding": "24px",
        "fontFamily": "'Space Grotesk', sans-serif",
    },
)


@callback(
    Output("auth-session", "data"),
    Output("login-message", "children"),
    Output("register-message", "children"),
    Output("register-visible", "data"),
    Input("login-button", "n_clicks"),
    Input("register-button", "n_clicks"),
    Input("logout-button", "n_clicks"),
    Input("register-toggle", "n_clicks"),
    State("login-username", "value"),
    State("login-password", "value"),
    State("register-username", "value"),
    State("register-password", "value"),
    State("register-confirm", "value"),
    State("auth-session", "data"),
    State("register-visible", "data"),
    prevent_initial_call=True,
)
def handle_auth_actions(
    _login_clicks: int,
    _register_clicks: int,
    _logout_clicks: int,
    _register_toggle_clicks: int,
    login_username: str | None,
    login_password: str | None,
    register_username: str | None,
    register_password: str | None,
    register_confirm: str | None,
    session_data: dict | None,
    register_visible: bool | None,
):
    trigger = dash.callback_context.triggered_id

    if trigger == "logout-button":
        return None, "", "", False

    if trigger == "register-toggle":
        return no_update, no_update, no_update, not bool(register_visible)

    if trigger == "login-button":
        username = _normalize_username(login_username)
        password = login_password or ""
        if not username or not password:
            message = html.Span(
                "Enter both username and password.",
                style={"color": "#b00"},
            )
            return no_update, message, "", no_update
        session, text, success = _auth_login(username, password)
        message = html.Span(text, style={"color": "#0b8043" if success else "#b00"})
        if not success:
            return no_update, message, "", no_update
        return session, message, "", False

    if trigger == "register-button":
        username = _normalize_username(register_username)
        password = register_password or ""
        confirm = register_confirm or ""
        if password != confirm:
            return (
                no_update,
                "",
                html.Span("Passwords do not match.", style={"color": "#b00"}),
                no_update,
            )
        session, text, success = _auth_register(username, password)
        message = html.Span(text, style={"color": "#0b8043" if success else "#b00"})
        if not success:
            return no_update, "", message, no_update
        return session, "", message, False

    return session_data, no_update, no_update, no_update


@callback(
    Output("register-section", "style"),
    Output("register-toggle", "children"),
    Input("register-visible", "data"),
    Input("lang-store", "data"),
)
def toggle_register_section(visible: bool | None, lang: str | None):
    if visible:
        return {"display": "block"}, _t(lang, "hide_register")
    return {"display": "none"}, _t(lang, "register_hint")


@callback(
    Output("auth-overlay", "style"),
    Output("auth-user-label", "children"),
    Output("logout-button", "style"),
    Input("auth-session", "data"),
)
def toggle_auth_overlay(session_data: dict | None):
    username = None
    role = None
    if isinstance(session_data, dict):
        username = _normalize_username(session_data.get("username"))
        role = _session_role(session_data)
    if username:
        role_label = role or "viewer"
        return AUTH_HIDDEN_STYLE, f"Signed in as {username} ({role_label})", LOGOUT_VISIBLE_STYLE
    return AUTH_CONTAINER_STYLE, "", LOGOUT_HIDDEN_STYLE


@callback(
    Output("admin-banner", "children"),
    Output("train-button", "disabled"),
    Output("system-report-download", "disabled"),
    Output("fintech-download", "disabled"),
    Input("auth-session", "data"),
    Input("lang-store", "data"),
)
def enforce_admin_role(session_data: dict | None, lang: str | None):
    if _is_admin(session_data):
        return "", False, False, False
    banner = html.Div(
        _t(lang, "admin_banner"),
        style={
            "padding": "8px 10px",
            "border": "1px solid #f5c2c7",
            "background": "#f8d7da",
            "color": "#842029",
            "borderRadius": "6px",
            "fontSize": "12px",
            "maxWidth": "720px",
        },
    )
    return banner, True, True, True


@callback(
    Output("lang-store", "data"),
    Input("lang-toggle", "value"),
)
def update_language_store(value: str | None):
    return value or LANG_DEFAULT


@callback(
    Output("label-sign-in-title", "children"),
    Output("label-sign-in-subtitle", "children"),
    Output("label-sign-in", "children"),
    Output("label-sign-in-button", "children"),
    Output("label-register-title", "children"),
    Output("label-register-button", "children"),
    Output("label-app-title", "children"),
    Output("label-app-tagline", "children"),
    Output("label-app-subtitle", "children"),
    Output("label-language", "children"),
    Output("label-dataset", "children"),
    Output("label-upload-dataset", "children"),
    Output("label-upload-hint", "children"),
    Output("label-rig", "children"),
    Output("label-metrics", "children"),
    Output("label-metric-readings", "children"),
    Output("label-table-metric", "children"),
    Output("label-eda-title", "children"),
    Output("label-eda-subtitle", "children"),
    Output("label-distribution-metric", "children"),
    Output("label-correlation-metrics", "children"),
    Output("label-summary-stats", "children"),
    Output("label-missing-values", "children"),
    Output("label-outliers", "children"),
    Output("label-outlier-metric", "children"),
    Output("label-outlier-method", "children"),
    Output("label-z-threshold-short", "children"),
    Output("label-iqr-multiplier", "children"),
    Output("label-scatter-x", "children"),
    Output("label-scatter-y", "children"),
    Output("label-anomaly-alerts", "children"),
    Output("label-anomaly-metrics", "children"),
    Output("label-z-threshold", "children"),
    Output("label-failure-risk", "children"),
    Output("label-risk-metrics", "children"),
    Output("label-use-trained-model", "children"),
    Output("label-rolling-window", "children"),
    Output("label-risk-threshold", "children"),
    Output("label-decision-engine", "children"),
    Output("label-use-failure-model", "children"),
    Output("label-max-rul", "children"),
    Output("label-run-decision", "children"),
    Output("label-download-system-report", "children"),
    Output("label-fintech-insights", "children"),
    Output("label-run-fintech", "children"),
    Output("label-download-fintech-pdf", "children"),
    Output("label-ml-control-center", "children"),
    Output("label-training-type", "children"),
    Output("label-window-size", "children"),
    Output("label-output-dir", "children"),
    Output("label-data-path", "children"),
    Output("label-attach-csv", "children"),
    Output("label-attach-csv-hint", "children"),
    Output("label-options", "children"),
    Output("label-train", "children"),
    Output("train-save-models", "options"),
    Output("risk-use-model", "options"),
    Output("decision-use-model", "options"),
    Input("lang-store", "data"),
)
def update_language_labels(lang: str | None):
    return (
        _t(lang, "sign_in_title"),
        _t(lang, "sign_in_subtitle"),
        _t(lang, "sign_in"),
        _t(lang, "sign_in"),
        _t(lang, "register_title"),
        _t(lang, "register"),
        _t(lang, "app_title"),
        _t(lang, "app_tagline"),
        _t(lang, "app_subtitle"),
        _t(lang, "language"),
        _t(lang, "dataset"),
        _t(lang, "upload_dataset"),
        _t(lang, "upload_hint"),
        _t(lang, "rig"),
        _t(lang, "metrics"),
        _t(lang, "metric_readings"),
        _t(lang, "table_metric"),
        _t(lang, "eda_title"),
        _t(lang, "eda_subtitle"),
        _t(lang, "distribution_metric"),
        _t(lang, "correlation_metrics"),
        _t(lang, "summary_stats"),
        _t(lang, "missing_values"),
        _t(lang, "outliers"),
        _t(lang, "outlier_metric"),
        _t(lang, "outlier_method"),
        _t(lang, "z_threshold"),
        _t(lang, "iqr_multiplier"),
        _t(lang, "scatter_x_metric"),
        _t(lang, "scatter_y_metric"),
        _t(lang, "anomaly_alerts"),
        _t(lang, "anomaly_metrics"),
        _t(lang, "z_score_threshold"),
        _t(lang, "failure_risk"),
        _t(lang, "risk_metrics"),
        _t(lang, "use_trained_model"),
        _t(lang, "rolling_window"),
        _t(lang, "risk_threshold"),
        _t(lang, "decision_engine"),
        _t(lang, "use_failure_model"),
        _t(lang, "max_rul"),
        _t(lang, "run_decision"),
        _t(lang, "download_system_report"),
        _t(lang, "fintech_insights"),
        _t(lang, "run_fintech"),
        _t(lang, "download_fintech_pdf"),
        _t(lang, "ml_control_center"),
        _t(lang, "training_type"),
        _t(lang, "window_size"),
        _t(lang, "output_dir"),
        _t(lang, "data_path"),
        _t(lang, "attach_csv"),
        _t(lang, "upload_hint"),
        _t(lang, "options"),
        _t(lang, "train"),
        [{"label": _t(lang, "save_models"), "value": "save"}],
        [{"label": _t(lang, "model_inference"), "value": "use"}],
        [{"label": _t(lang, "model_inference"), "value": "use"}],
    )


@callback(
    Output("dataset-store", "data"),
    Output("rig-dropdown", "options"),
    Output("rig-dropdown", "value"),
    Output("metric-dropdown", "options"),
    Output("metric-dropdown", "value"),
    Output("table-metric-dropdown", "options"),
    Output("table-metric-dropdown", "value"),
    Output("anomaly-metric-dropdown", "options"),
    Output("anomaly-metric-dropdown", "value"),
    Output("risk-metric-dropdown", "options"),
    Output("risk-metric-dropdown", "value"),
    Output("eda-metric-dropdown", "options"),
    Output("eda-metric-dropdown", "value"),
    Output("eda-corr-metric-dropdown", "options"),
    Output("eda-corr-metric-dropdown", "value"),
    Output("eda-outlier-metric-dropdown", "options"),
    Output("eda-outlier-metric-dropdown", "value"),
    Output("eda-scatter-x-dropdown", "options"),
    Output("eda-scatter-x-dropdown", "value"),
    Output("eda-scatter-y-dropdown", "options"),
    Output("eda-scatter-y-dropdown", "value"),
    Output("dataset-warning", "children"),
    Output("dataset-upload-name", "children"),
    Input("dataset-dropdown", "value"),
    Input("dataset-upload", "contents"),
    State("dataset-upload", "filename"),
)
def load_dataset_for_ui(
    dataset_name: str, upload_contents: str | None, upload_filename: str | None
):
    trigger = dash.callback_context.triggered_id
    upload_label = upload_filename or "No upload"
    if upload_contents and trigger == "dataset-upload":
        data, error = _load_uploaded_dataset(upload_contents, upload_filename)
        status = f"Using uploaded dataset: {upload_filename}" if not error else error
    else:
        data, error = _load_dataset(dataset_name)
        status = "" if not error else error

    if status:
        status_node = html.Span(
            status,
            style={"color": "#b00" if error else "#2b6a3f"},
        )
    else:
        status_node = ""

    if error:
        return (
            None,
            [{"label": "ALL", "value": "ALL"}],
            "ALL",
            [],
            [],
            [],
            None,
            [],
            [],
            [],
            [],
            [],
            None,
            [],
            [],
            [],
            None,
            [],
            None,
            [],
            None,
            status_node,
            upload_label,
        )

    rig_options = [{"label": "ALL", "value": "ALL"}]
    if "rig_id" in data.columns:
        rig_options += [
            {"label": rig, "value": rig}
            for rig in sorted(data["rig_id"].unique().tolist())
        ]

    metric_cols = _metric_columns(data)
    metric_options = [{"label": col, "value": col} for col in metric_cols]

    default_metrics = [col for col in DEFAULT_METRICS if col in metric_cols]
    if not default_metrics and metric_cols:
        default_metrics = metric_cols[:3]

    table_options = metric_options
    default_table_metric = None
    if POWER_METRIC in metric_cols:
        default_table_metric = POWER_METRIC
    elif metric_cols:
        default_table_metric = metric_cols[0]

    anomaly_options = [{"label": col, "value": col} for col in metric_cols]
    default_anomaly_metrics = [
        col for col in DEFAULT_ANOMALY_METRICS if col in metric_cols
    ]
    if not default_anomaly_metrics and metric_cols:
        default_anomaly_metrics = metric_cols[:2]

    risk_options = [{"label": col, "value": col} for col in metric_cols]
    default_risk_metrics = [col for col in DEFAULT_RISK_METRICS if col in metric_cols]
    if not default_risk_metrics and metric_cols:
        default_risk_metrics = metric_cols[:2]

    eda_options = metric_options
    default_eda_metric = None
    if POWER_METRIC in metric_cols:
        default_eda_metric = POWER_METRIC
    elif metric_cols:
        default_eda_metric = metric_cols[0]
    default_corr_metrics = metric_cols[:6] if metric_cols else []

    outlier_options = metric_options
    default_outlier_metric = default_eda_metric

    scatter_options = metric_options
    default_scatter_x = default_eda_metric
    default_scatter_y = metric_cols[1] if len(metric_cols) > 1 else default_scatter_x
    if default_scatter_y == default_scatter_x and len(metric_cols) > 1:
        default_scatter_y = metric_cols[1]

    return (
        data.to_json(date_format="iso", orient="split"),
        rig_options,
        "ALL",
        metric_options,
        default_metrics,
        table_options,
        default_table_metric,
        anomaly_options,
        default_anomaly_metrics,
        risk_options,
        default_risk_metrics,
        eda_options,
        default_eda_metric,
        eda_options,
        default_corr_metrics,
        outlier_options,
        default_outlier_metric,
        scatter_options,
        default_scatter_x,
        scatter_options,
        default_scatter_y,
        status_node,
        upload_label,
    )


@callback(
    Output("summary-cards", "children"),
    Output("sensor-graph", "figure"),
    Output("status-graph", "figure"),
    Input("dataset-store", "data"),
    Input("rig-dropdown", "value"),
    Input("metric-dropdown", "value"),
)
def update_dashboard(data_json: str, rig_value: str, metrics: list[str]):
    if not data_json:
        empty_fig = _empty_figure("No data loaded")
        return [], empty_fig, empty_fig

    data = _read_dataset_json(data_json)
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

    if rig_value and rig_value != "ALL" and "rig_id" in data.columns:
        data = data[data["rig_id"] == rig_value]

    rows = len(data)
    rigs = str(data["rig_id"].nunique()) if "rig_id" in data.columns else "n/a"
    date_range = "n/a"
    if "timestamp" in data.columns and not data.empty:
        date_range = f"{data['timestamp'].min()} to {data['timestamp'].max()}"

    failure_col = None
    if "failure_indicator" in data.columns:
        failure_col = "failure_indicator"
    elif "failure_flag" in data.columns:
        failure_col = "failure_flag"

    failure_total = None
    failure_count = "n/a"
    if failure_col:
        failures = pd.to_numeric(data[failure_col], errors="coerce").fillna(0)
        failure_total = int(failures.sum()) if rows else 0
        failure_count = f"{failure_total}"

    power_col = POWER_METRIC if POWER_METRIC in data.columns else None
    energy_used = _compute_energy_kwh(data, power_col) if power_col else None
    avg_power = None
    if power_col:
        avg_power = pd.to_numeric(data[power_col], errors="coerce").mean()

    cards = [
        _card("Rows", f"{rows:,}"),
        _card("Rigs", rigs),
    ]
    if power_col:
        cards.append(_card("Energy Used", _format_value(energy_used, "kWh")))
        cards.append(_card("Avg Power", _format_value(avg_power, "kW")))
    cards.extend(
        [
        _card("Failures", failure_count),
        _card("Date Range", date_range),
        ]
    )

    attention_card = None
    if failure_total is not None and failure_total > 0:
        attention_card = _attention_card(
            f"{failure_total} failure events detected",
            "Inspect affected rigs, review fault logs, and schedule maintenance.",
        )
    else:
        anomaly_metrics = [metric for metric in ATTENTION_METRICS if metric in data.columns]
        top_anomaly = _find_top_anomaly(data, anomaly_metrics, threshold=3.0)
        if top_anomaly:
            metric_label = top_anomaly["metric"].replace("_", " ").title()
            attention_card = _attention_card(
                f"Anomaly in {metric_label} (z={top_anomaly['z_score']:.1f})",
                _recommend_action(top_anomaly["metric"]),
            )
    if attention_card:
        cards.insert(0, attention_card)

    x_values = data["timestamp"] if "timestamp" in data.columns else data.index

    if not metrics:
        sensor_fig = _empty_figure("Select one or more metrics")
    else:
        sensor_fig = px.line(data, x=x_values, y=metrics)
        sensor_fig.update_layout(legend_title_text="Metrics", margin={"t": 40})
        _apply_smoothing(sensor_fig)

    status_fig = make_subplots(specs=[[{"secondary_y": True}]])
    has_status = False
    if failure_col:
        status_fig.add_trace(
            go.Scatter(x=x_values, y=data[failure_col], name="Failure"),
            secondary_y=False,
        )
        has_status = True

    if "estimated_rul" in data.columns:
        status_fig.add_trace(
            go.Scatter(x=x_values, y=data["estimated_rul"], name="Estimated RUL"),
            secondary_y=True,
        )
        has_status = True

    if not has_status:
        status_fig = _empty_figure("No failure or RUL data available")
    else:
        status_fig.update_layout(legend_title_text="Status", margin={"t": 40})
        status_fig.update_yaxes(title_text="Failure", secondary_y=False)
        status_fig.update_yaxes(title_text="Estimated RUL", secondary_y=True)
        _apply_smoothing(status_fig)

    return cards, sensor_fig, status_fig


@callback(
    Output("metric-table", "data"),
    Output("metric-table", "columns"),
    Output("metric-benchmark", "children"),
    Input("dataset-store", "data"),
    Input("rig-dropdown", "value"),
    Input("table-metric-dropdown", "value"),
)
def update_metric_table(
    data_json: str, rig_value: str, table_metric: str | None
):
    if not data_json:
        return [], [], _benchmark_card(table_metric)

    data = _read_dataset_json(data_json)
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

    if rig_value and rig_value != "ALL" and "rig_id" in data.columns:
        data = data[data["rig_id"] == rig_value]

    if not table_metric or table_metric not in data.columns:
        return [], [], _benchmark_card(table_metric)

    values = pd.to_numeric(data[table_metric], errors="coerce").round(3)
    table_frame = pd.DataFrame()
    columns: list[dict] = []

    if "timestamp" in data.columns:
        table_frame["timestamp"] = data["timestamp"].apply(
            lambda value: value.isoformat() if hasattr(value, "isoformat") else str(value)
        )
        columns.append({"name": "timestamp", "id": "timestamp"})
    else:
        table_frame["index"] = data.index.astype(str)
        columns.append({"name": "index", "id": "index"})

    if "rig_id" in data.columns:
        table_frame["rig_id"] = data["rig_id"].astype(str)
        columns.append({"name": "rig_id", "id": "rig_id"})

    table_frame[table_metric] = values
    columns.append({"name": _metric_label(table_metric), "id": table_metric})

    return table_frame.to_dict("records"), columns, _benchmark_card(table_metric)


@callback(
    Output("eda-summary-table", "data"),
    Output("eda-summary-table", "columns"),
    Output("eda-missing-table", "data"),
    Output("eda-missing-table", "columns"),
    Output("eda-outlier-table", "data"),
    Output("eda-outlier-table", "columns"),
    Output("eda-outlier-note", "children"),
    Output("eda-dist-graph", "figure"),
    Output("eda-corr-graph", "figure"),
    Output("eda-corr-note", "children"),
    Output("eda-scatter-graph", "figure"),
    Output("eda-scatter-note", "children"),
    Input("dataset-store", "data"),
    Input("rig-dropdown", "value"),
    Input("eda-metric-dropdown", "value"),
    Input("eda-corr-metric-dropdown", "value"),
    Input("eda-outlier-metric-dropdown", "value"),
    Input("eda-outlier-method-dropdown", "value"),
    Input("eda-outlier-z-threshold", "value"),
    Input("eda-outlier-iqr-k", "value"),
    Input("eda-scatter-x-dropdown", "value"),
    Input("eda-scatter-y-dropdown", "value"),
)
def update_eda_section(
    data_json: str,
    rig_value: str,
    eda_metric: str | None,
    corr_metrics: list[str] | None,
    outlier_metric: str | None,
    outlier_method: str | None,
    outlier_z_threshold: float | None,
    outlier_iqr_k: float | None,
    scatter_x: str | None,
    scatter_y: str | None,
):
    if not data_json:
        empty_fig = _empty_figure("No data loaded")
        return [], [], [], [], [], [], "", empty_fig, empty_fig, "", empty_fig, ""

    data = _read_dataset_json(data_json)
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    if rig_value and rig_value != "ALL" and "rig_id" in data.columns:
        data = data[data["rig_id"] == rig_value]

    metric_cols = _metric_columns(data)
    if not metric_cols:
        empty_fig = _empty_figure("No numeric metrics available")
        return [], [], [], [], [], [], "", empty_fig, empty_fig, "", empty_fig, ""

    selected_metric = eda_metric if eda_metric in metric_cols else metric_cols[0]
    selected_corr_metrics = [m for m in (corr_metrics or []) if m in metric_cols]
    if len(selected_corr_metrics) < 2:
        selected_corr_metrics = metric_cols[: min(6, len(metric_cols))]

    summary_rows = _eda_summary_rows(data, selected_corr_metrics)
    missing_rows = _eda_missing_rows(data, selected_corr_metrics)

    summary_columns = (
        [{"name": key, "id": key} for key in summary_rows[0].keys()]
        if summary_rows
        else []
    )
    missing_columns = (
        [{"name": key, "id": key} for key in missing_rows[0].keys()]
        if missing_rows
        else []
    )

    selected_outlier_metric = (
        outlier_metric if outlier_metric in metric_cols else selected_metric
    )
    z_threshold_value = float(outlier_z_threshold or 3.0)
    iqr_k_value = float(outlier_iqr_k or 1.5)
    outlier_rows, outlier_note, outlier_mask = _eda_outlier_rows(
        data,
        selected_outlier_metric,
        outlier_method or "zscore",
        z_threshold_value,
        iqr_k_value,
    )
    outlier_columns = (
        [{"name": key, "id": key} for key in outlier_rows[0].keys()]
        if outlier_rows
        else []
    )

    dist_fig = _eda_distribution_figure(
        data,
        selected_metric,
        selected_outlier_metric,
        outlier_mask,
    )
    corr_fig, corr_note = _eda_correlation_figure(data, selected_corr_metrics, rig_value)
    scatter_x_metric = scatter_x if scatter_x in metric_cols else selected_metric
    if scatter_y in metric_cols:
        scatter_y_metric = scatter_y
    elif len(metric_cols) > 1:
        scatter_y_metric = metric_cols[1]
    else:
        scatter_y_metric = scatter_x_metric
    if scatter_y_metric == scatter_x_metric and len(metric_cols) > 1:
        scatter_y_metric = metric_cols[1] if metric_cols[1] != scatter_x_metric else metric_cols[0]
    scatter_fig, scatter_note = _eda_scatter_figure(
        data,
        scatter_x_metric,
        scatter_y_metric,
        outlier_mask,
        selected_outlier_metric,
    )

    return (
        summary_rows,
        summary_columns,
        missing_rows,
        missing_columns,
        outlier_rows,
        outlier_columns,
        outlier_note,
        dist_fig,
        corr_fig,
        corr_note,
        scatter_fig,
        scatter_note,
    )


@callback(
    Output("anomaly-banner", "children"),
    Output("anomaly-table", "data"),
    Output("anomaly-table", "columns"),
    Input("dataset-store", "data"),
    Input("rig-dropdown", "value"),
    Input("anomaly-metric-dropdown", "value"),
    Input("anomaly-threshold", "value"),
)
def update_anomaly_alerts(
    data_json: str,
    rig_value: str,
    anomaly_metrics: list[str],
    threshold: float | None,
):
    if not data_json:
        return "No data loaded.", [], []

    data = _read_dataset_json(data_json)
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

    if rig_value and rig_value != "ALL" and "rig_id" in data.columns:
        data = data[data["rig_id"] == rig_value]

    if not anomaly_metrics:
        return "Select anomaly metrics to scan.", [], []

    threshold_value = float(threshold or 3.0)
    anomalies: list[dict] = []

    for metric in anomaly_metrics:
        if metric not in data.columns:
            continue
        values = pd.to_numeric(data[metric], errors="coerce")
        mean = values.mean()
        std = values.std()
        if pd.isna(std) or std == 0:
            continue
        z_scores = (values - mean) / std
        mask = z_scores.abs() >= threshold_value
        if not mask.any():
            continue
        for idx in data[mask].index:
            row = data.loc[idx]
            anomalies.append(
                {
                    "timestamp": row["timestamp"].isoformat()
                    if "timestamp" in data.columns
                    else str(idx),
                    "rig_id": row["rig_id"] if "rig_id" in data.columns else "n/a",
                    "metric": metric,
                    "value": float(values.loc[idx]),
                    "z_score": float(z_scores.loc[idx]),
                }
            )

    anomalies.sort(key=lambda item: abs(item["z_score"]), reverse=True)
    anomalies = anomalies[:200]

    banner = f"Anomalies detected: {len(anomalies)}"
    if not anomalies:
        banner = "No anomalies detected with the current threshold."

    columns = [
        {"name": "timestamp", "id": "timestamp"},
        {"name": "rig_id", "id": "rig_id"},
        {"name": "metric", "id": "metric"},
        {"name": "value", "id": "value"},
        {"name": "z_score", "id": "z_score"},
    ]

    return banner, anomalies, columns


@callback(
    Output("risk-banner", "children"),
    Output("risk-graph", "figure"),
    Output("risk-table", "data"),
    Output("risk-table", "columns"),
    Input("dataset-store", "data"),
    Input("rig-dropdown", "value"),
    Input("risk-metric-dropdown", "value"),
    Input("risk-window-size", "value"),
    Input("risk-threshold", "value"),
    Input("risk-use-model", "value"),
)
def update_failure_risk(
    data_json: str,
    rig_value: str,
    risk_metrics: list[str],
    window_size: int | None,
    threshold: float | None,
    use_model_flags: list[str] | None,
):
    if not data_json:
        empty_fig = _empty_figure("No data loaded")
        return "No data loaded.", empty_fig, [], []

    data = _read_dataset_json(data_json)
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

    if rig_value and rig_value != "ALL" and "rig_id" in data.columns:
        data = data[data["rig_id"] == rig_value]

    if data.empty:
        empty_fig = _empty_figure("No data after filtering")
        return "No data after filtering.", empty_fig, [], []

    if use_model_flags and "use" in use_model_flags:
        csv_bytes = data.to_csv(index=False).encode("utf-8")
        fields = {
            "window_size": str(int(window_size) if window_size else 24),
            "threshold": str(float(threshold) if threshold is not None else 0.7),
        }
        file_payload = ("dataset.csv", csv_bytes, "text/csv")
        body, content_type = _encode_multipart(fields, file_payload)
        request = urllib.request.Request(
            RISK_API_URL,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                raw = response.read().decode("utf-8")
                payload = json.loads(raw)
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8")
            empty_fig = _empty_figure("Model prediction failed")
            return f"Model error ({exc.code}): {body_text}", empty_fig, [], []
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            empty_fig = _empty_figure("Model prediction failed")
            return f"Model error: {exc}", empty_fig, [], []

        predictions = payload.get("predictions", [])
        threshold_value = float(payload.get("threshold", threshold or 0.7))

        if not predictions:
            empty_fig = _empty_figure("No predictions returned")
            return "No predictions returned by model.", empty_fig, [], []

        risk_scores = [item.get("risk_score", 0.0) for item in predictions]
        timestamps = [item.get("timestamp") for item in predictions]
        rig_ids = [item.get("rig_id") for item in predictions]
        x_values = timestamps if any(timestamps) else list(range(len(predictions)))

        risk_fig = go.Figure()
        risk_fig.add_trace(
            go.Scatter(x=x_values, y=risk_scores, name="Risk Score", mode="lines")
        )
        risk_fig.add_trace(
            go.Scatter(
                x=x_values,
                y=[threshold_value] * len(risk_scores),
                name="Threshold",
                mode="lines",
                line={"dash": "dash"},
            )
        )
        risk_fig.update_layout(margin={"t": 40}, yaxis={"title": "Risk Score"})
        _apply_smoothing(risk_fig)

        high_risk_rows = [
            item for item in predictions if item.get("risk_score", 0.0) >= threshold_value
        ]
        banner = f"High failure risk points: {len(high_risk_rows)}"
        if not high_risk_rows:
            banner = "No high failure risk detected at the current threshold."

        table_rows = []
        for item in high_risk_rows[:200]:
            table_rows.append(
                {
                    "timestamp": item.get("timestamp", ""),
                    "rig_id": item.get("rig_id", "n/a"),
                    "risk_score": float(item.get("risk_score", 0.0)),
                }
            )

        columns = [
            {"name": "timestamp", "id": "timestamp"},
            {"name": "rig_id", "id": "rig_id"},
            {"name": "risk_score", "id": "risk_score"},
        ]

        return banner, risk_fig, table_rows, columns

    if not risk_metrics:
        empty_fig = _empty_figure("Select risk metrics to compute risk")
        return "Select risk metrics to compute risk.", empty_fig, [], []

    z_frame = pd.DataFrame(index=data.index)
    for metric in risk_metrics:
        if metric not in data.columns:
            continue
        values = pd.to_numeric(data[metric], errors="coerce")
        if "rig_id" in data.columns:
            means = values.groupby(data["rig_id"]).transform("mean")
            stds = values.groupby(data["rig_id"]).transform("std")
            z_scores = (values - means) / stds.replace(0, pd.NA)
        else:
            std = values.std()
            z_scores = (values - values.mean()) / (std if std else pd.NA)
        z_frame[metric] = z_scores

    if z_frame.empty:
        empty_fig = _empty_figure("No valid risk metrics found")
        return "No valid risk metrics found.", empty_fig, [], []

    risk_raw = z_frame.abs().mean(axis=1)
    risk_score = risk_raw / (risk_raw + 1)
    risk_score = risk_score.fillna(0.0)

    window = int(window_size) if window_size and window_size > 1 else 1
    risk_smoothed = risk_score.rolling(window=window, min_periods=1).mean()

    threshold_value = float(threshold or 0.7)
    high_risk = risk_smoothed >= threshold_value
    count = int(high_risk.sum())

    banner = f"High failure risk points: {count}"
    if count == 0:
        banner = "No high failure risk detected at the current threshold."

    x_values = data["timestamp"] if "timestamp" in data.columns else data.index
    risk_fig = go.Figure()
    risk_fig.add_trace(
        go.Scatter(x=x_values, y=risk_smoothed, name="Risk Score", mode="lines")
    )
    risk_fig.add_trace(
        go.Scatter(
            x=x_values,
            y=[threshold_value] * len(data),
            name="Threshold",
            mode="lines",
            line={"dash": "dash"},
        )
    )
    risk_fig.update_layout(margin={"t": 40}, yaxis={"title": "Risk Score"})
    _apply_smoothing(risk_fig)

    table_rows: list[dict] = []
    if count:
        for idx in data[high_risk].index:
            row = data.loc[idx]
            table_rows.append(
                {
                    "timestamp": row["timestamp"].isoformat()
                    if "timestamp" in data.columns
                    else str(idx),
                    "rig_id": row["rig_id"] if "rig_id" in data.columns else "n/a",
                    "risk_score": float(risk_smoothed.loc[idx]),
                }
            )

    table_rows.sort(key=lambda item: item["risk_score"], reverse=True)
    table_rows = table_rows[:200]

    columns = [
        {"name": "timestamp", "id": "timestamp"},
        {"name": "rig_id", "id": "rig_id"},
        {"name": "risk_score", "id": "risk_score"},
    ]

    return banner, risk_fig, table_rows, columns


@callback(
    Output("decision-output", "children"),
    Input("decision-run", "n_clicks"),
    State("dataset-store", "data"),
    State("rig-dropdown", "value"),
    State("anomaly-metric-dropdown", "value"),
    State("anomaly-threshold", "value"),
    State("decision-use-model", "value"),
    State("risk-window-size", "value"),
    State("decision-max-rul", "value"),
    State("auth-session", "data"),
    prevent_initial_call=True,
)
def evaluate_decision_engine(
    _clicks: int,
    data_json: str,
    rig_value: str,
    anomaly_metrics: list[str],
    threshold: float | None,
    use_model_flags: list[str] | None,
    window_size: int | None,
    max_rul: float | None,
    auth_session: dict | None,
):
    if not data_json:
        return "No data loaded."

    token = _session_token(auth_session)

    data = _read_dataset_json(data_json)
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

    if rig_value and rig_value != "ALL" and "rig_id" in data.columns:
        data = data[data["rig_id"] == rig_value]

    if data.empty:
        return "No data after filtering."

    metrics = anomaly_metrics or [m for m in DEFAULT_ANOMALY_METRICS if m in data.columns]
    threshold_value = float(threshold or 3.0)
    top_anomaly = _find_top_anomaly(data, metrics, threshold_value) if metrics else None
    anomaly_score = float(top_anomaly["z_score"]) if top_anomaly else 0.0

    failure_risk = None
    if use_model_flags and "use" in use_model_flags:
        csv_bytes = data.to_csv(index=False).encode("utf-8")
        fields = {
            "window_size": str(int(window_size) if window_size else 24),
            "threshold": "0.7",
        }
        file_payload = ("dataset.csv", csv_bytes, "text/csv")
        body, content_type = _encode_multipart(fields, file_payload)
        headers = {"Content-Type": content_type}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            RISK_API_URL,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                raw = response.read().decode("utf-8")
                payload = json.loads(raw)
            predictions = payload.get("predictions", [])
            scores = [item.get("risk_score", 0.0) for item in predictions]
            if scores:
                failure_risk = float(max(scores))
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8")
            return f"Decision engine failed: risk model error ({exc.code}): {body_text}"
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            return f"Decision engine failed: risk model error: {exc}"

    if failure_risk is None:
        failure_col = None
        if "failure_indicator" in data.columns:
            failure_col = "failure_indicator"
        elif "failure_flag" in data.columns:
            failure_col = "failure_flag"
        if failure_col:
            failures = pd.to_numeric(data[failure_col], errors="coerce").fillna(0)
            failure_risk = float(min(1.0, failures.mean()))

    rul_value = None
    if "estimated_rul" in data.columns:
        rul_series = pd.to_numeric(data["estimated_rul"], errors="coerce").dropna()
        if not rul_series.empty:
            rul_value = float(rul_series.min())

    payload = {
        "equipment_id": rig_value if rig_value and rig_value != "ALL" else "FLEET",
        "failure_risk": failure_risk,
        "anomaly_score": anomaly_score,
        "rul": rul_value,
        "max_rul": float(max_rul) if max_rul else None,
    }
    response, error = _post_json(DECISION_API_URL, payload, token=token)
    if error:
        return f"Decision engine error: {error}"

    risk_score = response.get("risk_score", 0.0)
    risk_level = response.get("risk_level", "n/a")
    actions = response.get("actions", [])
    components = response.get("components", {})

    component_text = ", ".join(
        f"{key}: {value:.2f}" for key, value in components.items()
    ) if components else "n/a"

    return html.Div(
        [
            html.Div(f"Risk Level: {risk_level} (score {risk_score:.2f})"),
            html.Div(f"Components: {component_text}"),
            html.Div("Recommended Actions:"),
            html.Ul([html.Li(action) for action in actions]) if actions else html.Div("None"),
        ]
    )


@callback(
    Output("fintech-summary", "children"),
    Output("fintech-forecast-table", "data"),
    Output("fintech-forecast-table", "columns"),
    Input("fintech-run", "n_clicks"),
    State("fintech-impl-cost", "value"),
    State("fintech-annual-savings", "value"),
    State("fintech-years", "value"),
    State("fintech-monthly-savings", "value"),
    State("fintech-preventive-cost", "value"),
    State("fintech-failure-cost", "value"),
    State("fintech-failure-prob", "value"),
    State("fintech-baseline-failures", "value"),
    State("fintech-actual-failures", "value"),
    State("fintech-preventive-costs", "value"),
    State("fintech-history-costs", "value"),
    State("fintech-forecast-periods", "value"),
    State("auth-session", "data"),
    prevent_initial_call=True,
)
def run_fintech_analysis(
    _clicks: int,
    implementation_cost: float | None,
    annual_savings: float | None,
    years: int | None,
    monthly_savings: float | None,
    preventive_cost: float | None,
    failure_cost: float | None,
    failure_probability: float | None,
    baseline_failures: int | None,
    actual_failures: int | None,
    preventive_costs: float | None,
    history_costs: str | None,
    forecast_periods: int | None,
    auth_session: dict | None,
):
    summary_items: list = []
    token = _session_token(auth_session)
    admin = _is_admin(auth_session)

    roi_payload = {
        "implementation_cost": float(implementation_cost or 0),
        "annual_savings": float(annual_savings or 0),
        "years": int(years or 1),
        "monthly_savings": float(monthly_savings)
        if monthly_savings is not None
        else None,
    }
    roi_response, roi_error = _post_json(FINTECH_ROI_URL, roi_payload, token=token)
    if roi_error:
        summary_items.append(html.Div(f"ROI error: {roi_error}"))
    else:
        roi_pct = roi_response.get("roi_pct")
        if roi_pct is not None:
            summary_items.append(html.Div(f"ROI: {roi_pct:.1f}%"))
        if "break_even_months" in roi_response:
            summary_items.append(
                html.Div(f"Break-even: {roi_response['break_even_months']:.1f} months")
            )

    compare_payload = {
        "preventive_cost": float(preventive_cost or 0),
        "failure_cost": float(failure_cost or 0),
        "failure_probability": float(failure_probability or 0),
    }
    compare_response, compare_error = _post_json(
        FINTECH_COMPARE_URL, compare_payload, token=token
    )
    if compare_error:
        summary_items.append(html.Div(f"Comparison error: {compare_error}"))
    else:
        recommendation = compare_response.get("recommendation", "n/a")
        expected = compare_response.get("expected_failure_cost")
        summary_items.append(
            html.Div(
                f"Recommendation: {recommendation} "
                f"(expected failure cost {expected:,.0f})"
                if expected is not None
                else f"Recommendation: {recommendation}"
            )
        )

    savings_payload = {
        "baseline_failures": int(baseline_failures or 0),
        "actual_failures": int(actual_failures or 0),
        "preventive_costs": float(preventive_costs or 0),
    }
    savings_response, savings_error = _post_json(
        FINTECH_SAVINGS_URL, savings_payload, token=token
    )
    if savings_error:
        summary_items.append(html.Div(f"Savings error: {savings_error}"))
    else:
        savings_value = savings_response.get("savings")
        if savings_value is not None:
            summary_items.append(html.Div(f"Cost savings: {savings_value:,.0f}"))

    history_values = _parse_history_costs(history_costs)

    forecast_rows: list[dict] = []
    if history_values:
        history_payload = [
            {"period": idx + 1, "total_cost": value}
            for idx, value in enumerate(history_values)
        ]
        forecast_payload = {
            "history": history_payload,
            "periods": int(forecast_periods or 1),
        }
        forecast_response, forecast_error = _post_json(
            FINTECH_FORECAST_URL, forecast_payload, token=token
        )
        if forecast_error:
            summary_items.append(html.Div(f"Forecast error: {forecast_error}"))
        else:
            forecast_rows = forecast_response.get("forecast", [])

        if admin:
            report_response, report_error = _post_json(
                FINTECH_REPORT_URL, {"periods": history_payload}, token=token
            )
            if not report_error:
                totals = report_response.get("totals", {})
                net_cost = totals.get("net_cost")
                if net_cost is not None:
                    summary_items.append(html.Div(f"Net cost (history): {net_cost:,.0f}"))
            else:
                summary_items.append(html.Div(f"Report error: {report_error}"))
        else:
            summary_items.append(
                html.Div("Admin role required for report totals and PDFs.")
            )

    columns = []
    if forecast_rows:
        columns = [{"name": key, "id": key} for key in forecast_rows[0].keys()]

    if not summary_items:
        summary_items = [html.Div("Provide inputs to compute FinTech metrics.")]

    return html.Div(summary_items), forecast_rows, columns


@callback(
    Output("fintech-pdf-download", "data"),
    Input("fintech-download", "n_clicks"),
    State("fintech-history-costs", "value"),
    State("auth-session", "data"),
    prevent_initial_call=True,
)
def download_fintech_pdf(
    _clicks: int,
    history_costs: str | None,
    auth_session: dict | None,
):
    if not _is_admin(auth_session):
        return no_update
    token = _session_token(auth_session)
    history_values = _parse_history_costs(history_costs)
    history_payload = [
        {"period": idx + 1, "total_cost": value}
        for idx, value in enumerate(history_values)
    ]
    pdf_bytes, error = _post_pdf(
        FINTECH_REPORT_PDF_URL, {"periods": history_payload}, token=token
    )
    if error or not pdf_bytes:
        return no_update
    return dcc.send_bytes(pdf_bytes, "rigvisionx-fintech-report.pdf")


@callback(
    Output("system-report-file", "data"),
    Input("system-report-download", "n_clicks"),
    State("dataset-store", "data"),
    State("rig-dropdown", "value"),
    State("metric-dropdown", "value"),
    State("table-metric-dropdown", "value"),
    State("anomaly-metric-dropdown", "value"),
    State("anomaly-threshold", "value"),
    State("risk-metric-dropdown", "value"),
    State("risk-window-size", "value"),
    State("risk-threshold", "value"),
    State("auth-session", "data"),
    prevent_initial_call=True,
)
def download_system_report(
    _clicks: int,
    data_json: str,
    rig_value: str,
    metrics: list[str],
    table_metric: str | None,
    anomaly_metrics: list[str],
    anomaly_threshold: float | None,
    risk_metrics: list[str],
    risk_window_size: int | None,
    risk_threshold: float | None,
    auth_session: dict | None,
):
    if not _is_admin(auth_session):
        return no_update
    token = _session_token(auth_session)
    if not data_json:
        return no_update

    payload = {
        "dataset": data_json,
        "rig_id": rig_value,
        "metrics": metrics or [],
        "table_metric": table_metric,
        "anomaly_metrics": anomaly_metrics or [],
        "anomaly_threshold": float(anomaly_threshold or 3.0),
        "risk_metrics": risk_metrics or [],
        "risk_window_size": int(risk_window_size or 12),
        "risk_threshold": float(risk_threshold or 0.7),
        "title": "TerraEnergy AI System Report",
    }
    pdf_bytes, error = _post_pdf(SYSTEM_REPORT_PDF_URL, payload, token=token)
    if error or not pdf_bytes:
        return no_update
    return dcc.send_bytes(pdf_bytes, "rigvisionx-system-report.pdf")


@callback(
    Output("train-status", "children"),
    Output("train-output", "children"),
    Output("train-upload-name", "children"),
    Output("train-job-id", "data"),
    Output("train-interval", "disabled"),
    Input("train-button", "n_clicks"),
    Input("train-interval", "n_intervals"),
    State("train-job-id", "data"),
    State("train-upload", "contents"),
    State("train-upload", "filename"),
    State("train-type", "value"),
    State("train-save-models", "value"),
    State("train-window-size", "value"),
    State("train-output-dir", "value"),
    State("train-data-path", "value"),
    State("auth-session", "data"),
    prevent_initial_call=True,
)
def handle_training(
    _clicks: int,
    _interval: int,
    job_id: str | None,
    contents: str | None,
    filename: str | None,
    training_type: str,
    save_models_values: list[str] | None,
    window_size: int | None,
    output_dir: str | None,
    data_path: str | None,
    auth_session: dict | None,
):
    trigger = dash.callback_context.triggered_id
    upload_name = filename or "No file selected"
    token = _session_token(auth_session)
    if not _is_admin(auth_session) or not token:
        return (
            "Access denied",
            "Admin role required for training.",
            upload_name,
            None,
            True,
        )

    if trigger == "train-button":
        file_bytes = _decode_upload(contents)
        if contents and file_bytes is None:
            return "Upload failed", "Unable to decode uploaded file.", upload_name, None, True

        if not file_bytes and not data_path:
            return "Missing data", "Attach a CSV or provide a data path.", upload_name, None, True

        fields = {
            "training_type": training_type or "all",
            "save_models": "true" if save_models_values else "false",
            "output_dir": output_dir or "models/",
            "window_size": str(window_size) if window_size else "",
            "data_path": data_path or "",
        }

        file_payload = None
        if file_bytes is not None and filename:
            content_type = mimetypes.guess_type(filename)[0] or "text/csv"
            file_payload = (filename, file_bytes, content_type)

        body, content_type = _encode_multipart(fields, file_payload)
        headers = {"Content-Type": content_type, "Authorization": f"Bearer {token}"}
        request = urllib.request.Request(
            TRAIN_API_URL,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
                payload = json.loads(raw)
                started_job = payload.get("job_id")
                status_message = f"Training started (job_id: {started_job})"
                output = _format_job_payload(payload)
                return status_message, output, upload_name, started_job, False
        except urllib.error.HTTPError as exc:
            return f"Training failed ({exc.code})", _extract_error(exc), upload_name, None, True
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            return "Training failed", str(exc), upload_name, None, True

    if trigger == "train-interval":
        if not job_id:
            return no_update, no_update, upload_name, None, True

        status_url = f"{TRAIN_STATUS_URL}/{job_id}"
        request = urllib.request.Request(
            status_url,
            method="GET",
            headers={"Authorization": f"Bearer {token}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
                payload = json.loads(raw)
                status = payload.get("status", "running")
                output = _format_job_payload(payload)
                if status in {"completed", "failed"}:
                    label = "Training completed" if status == "completed" else "Training failed"
                    return label, output, upload_name, job_id, True
                return "Training in progress...", output, upload_name, job_id, False
        except urllib.error.HTTPError as exc:
            return f"Status failed ({exc.code})", _extract_error(exc), upload_name, job_id, True
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            return "Status failed", str(exc), upload_name, job_id, True

    return no_update, no_update, upload_name, job_id, True


def main() -> None:
    host = os.environ.get("DASH_HOST", "0.0.0.0")
    port = int(os.environ.get("DASH_PORT", "8050"))
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()

