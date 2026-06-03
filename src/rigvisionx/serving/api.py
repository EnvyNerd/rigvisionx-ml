from datetime import datetime, timezone
import io
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Optional
import shutil
import threading
import uuid

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, Form, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import jwt

from rigvisionx.aeos.energy_deviation import calculate_deviation
from rigvisionx.decision_engine.risk_fusion import calculate_overall_risk
from rigvisionx.explainability import SHAPExplainer
from rigvisionx.fintech.reporting import (
    forecast_costs,
    generate_financial_report,
    generate_financial_report_pdf,
    calculate_cost_savings,
)
from rigvisionx.fintech.roi import calculate_roi, calculate_break_even_period, compare_scenarios
from rigvisionx.train_pipeline import TrainingPipeline
from rigvisionx.reporting.system_report import generate_system_report_pdf
from rigvisionx.serving.db import (
    User,
    TrainJobRecord,
    count_users,
    create_dataset_record,
    create_train_job,
    create_user,
    decode_progress,
    decode_result,
    get_session,
    get_train_job,
    get_user_by_username,
    init_db,
    list_datasets,
    update_train_job_progress,
    finalize_train_job,
    update_user_login,
)
from rigvisionx.serving.demo import get_demo_status
from rigvisionx.serving.security import create_access_token, decode_access_token, hash_password, verify_password

app = FastAPI(title="TerraEnergy AI Engine")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
ADMIN_USERS = {
    item.strip()
    for item in os.environ.get("RIGVISIONX_ADMIN_USERS", "").split(",")
    if item.strip()
}
auth_scheme = HTTPBearer(auto_error=False)
DEMO_ALLOWLIST = {"/", "/health", "/license/status"}

class EnergyDeviationRequest(BaseModel):
    actual: float = Field(..., description="Observed value.")
    baseline: float = Field(..., description="Baseline value; must be non-zero.")
    threshold_pct: Optional[float] = Field(
        default=None, description="Optional threshold percentage for anomaly flag."
    )


class EnergyDeviationResponse(BaseModel):
    deviation_pct: float = Field(..., description="Percent deviation from baseline.")
    anomaly: Optional[bool] = Field(
        default=None, description="Present when threshold_pct is provided."
    )

class EnergyDeviationBatchRequest(BaseModel):
    items: list[EnergyDeviationRequest] = Field(
        ..., description="List of energy deviation calculations."
    )


class EnergyDeviationBatchResponse(BaseModel):
    results: list[EnergyDeviationResponse] = Field(
        ..., description="Results aligned to the input items."
    )

class TrainStep(BaseModel):
    step: str
    status: str
    detail: Optional[str] = None


class TrainResponse(BaseModel):
    status: str
    training_type: str
    data_source: str
    data_shape: Optional[list[int]] = None
    features_shape: Optional[list[int]] = None
    models_trained: list[str] = Field(default_factory=list)
    evaluation_results: Optional[dict] = None
    output_dir: Optional[str] = None
    progress: list[TrainStep] = Field(default_factory=list)

class TrainJobStatus(BaseModel):
    job_id: str
    status: str
    training_type: str
    data_source: str
    progress: list[TrainStep] = Field(default_factory=list)
    result: Optional[TrainResponse] = None
    error: Optional[str] = None


class FailureRiskPrediction(BaseModel):
    timestamp: Optional[str] = None
    rig_id: Optional[str] = None
    risk_score: float
    high_risk: bool


class FailureRiskResponse(BaseModel):
    status: str
    threshold: float
    predictions: list[FailureRiskPrediction]

class DecisionRiskRequest(BaseModel):
    equipment_id: str = Field(..., description="Rig or equipment identifier.")
    failure_risk: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Failure risk probability."
    )
    anomaly_score: Optional[float] = Field(
        default=None, description="Anomaly score (z-score or 0-1)."
    )
    rul: Optional[float] = Field(default=None, description="Remaining useful life.")
    weights: Optional[dict[str, float]] = Field(
        default=None, description="Weights for risk fusion."
    )
    max_rul: Optional[float] = Field(
        default=None, gt=0, description="Upper bound for RUL normalization."
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Optional metadata to attach."
    )


class FintechReportRequest(BaseModel):
    periods: list[dict[str, Any]] = Field(
        default_factory=list, description="List of period cost entries."
    )


class FintechForecastRequest(BaseModel):
    history: list[dict[str, Any]] = Field(
        default_factory=list, description="Historical cost entries."
    )
    periods: int = Field(default=12, ge=1, description="Forecast horizon.")


class FintechRoiRequest(BaseModel):
    implementation_cost: float = Field(..., gt=0, description="Implementation cost.")
    annual_savings: float = Field(..., ge=0, description="Annual savings.")
    years: int = Field(default=3, ge=1, description="Time horizon in years.")
    monthly_savings: Optional[float] = Field(
        default=None, ge=0, description="Optional monthly savings."
    )


class FintechCompareRequest(BaseModel):
    preventive_cost: float = Field(..., ge=0)
    failure_cost: float = Field(..., ge=0)
    failure_probability: float = Field(..., ge=0, le=1)


class FintechSavingsRequest(BaseModel):
    baseline_failures: int = Field(..., ge=0)
    preventive_costs: float = Field(..., ge=0)
    actual_failures: int = Field(..., ge=0)

class SystemReportRequest(BaseModel):
    dataset: str = Field(..., description="Dataset JSON (pandas split orient).")
    rig_id: Optional[str] = Field(default=None)
    metrics: list[str] = Field(default_factory=list)
    table_metric: Optional[str] = Field(default=None)
    anomaly_metrics: list[str] = Field(default_factory=list)
    anomaly_threshold: float = Field(default=3.0, ge=0.1)
    risk_metrics: list[str] = Field(default_factory=list)
    risk_window_size: int = Field(default=12, ge=1)
    risk_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    title: Optional[str] = Field(default=None)
_DB_LOCK = threading.Lock()


class AuthRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=120)
    password: str = Field(..., min_length=6, max_length=256)


class AuthLoginRequest(BaseModel):
    username: str
    password: str


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str


class AuthUserResponse(BaseModel):
    username: str
    role: str
    created_at: datetime
    last_login_at: datetime | None


class DatasetMetaResponse(BaseModel):
    dataset_id: str
    filename: str
    path: str
    source: str
    content_type: str | None
    size_bytes: int | None
    uploaded_by: str | None
    uploaded_at: datetime


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_username(username: str | None) -> str:
    return (username or "").strip()


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.middleware("http")
async def _demo_middleware(request: Request, call_next):
    status_payload = get_demo_status()
    if status_payload.get("demo_mode") and not status_payload.get("licensed"):
        if request.url.path not in DEMO_ALLOWLIST and status_payload.get("expired"):
            return JSONResponse(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                content={
                    "detail": "Demo period expired. Please contact TerraEnergy AI to purchase a license.",
                    "status": "demo_expired",
                    "expires_utc": status_payload.get("expires_utc"),
                    "days_remaining": status_payload.get("days_remaining", 0),
                },
            )
    return await call_next(request)


def _get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(auth_scheme),
) -> User:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    token = credentials.credentials
    try:
        payload = decode_access_token(token)
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from exc

    username = _normalize_username(payload.get("sub"))
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    with get_session() as db:
        user = get_user_by_username(db, username)
        if user is None or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive",
            )
        return user


def require_role(required_role: str) -> Callable[[User], User]:
    def _dependency(user: User = Depends(_get_current_user)) -> User:
        if user.role == "admin":
            return user
        if user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {required_role}",
            )
        return user

    return _dependency


def _role_for_new_user(username: str, existing_user_count: int) -> str:
    if username in ADMIN_USERS:
        return "admin"
    if existing_user_count == 0:
        return "admin"
    return "viewer"


def _issue_token(user: User) -> AuthTokenResponse:
    token = create_access_token(subject=user.username, role=user.role)
    return AuthTokenResponse(
        access_token=token,
        username=user.username,
        role=user.role,
    )


@app.get("/")
def root():
    return {"message": "Welcome to TerraEnergy AI Engine API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/license/status")
def license_status():
    return get_demo_status()


@app.post("/auth/register", response_model=AuthTokenResponse)
def auth_register(payload: AuthRegisterRequest) -> AuthTokenResponse:
    username = _normalize_username(payload.username)
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    with get_session() as db:
        existing = get_user_by_username(db, username)
        if existing is not None:
            raise HTTPException(status_code=409, detail="Username already exists")
        user_count = count_users(db)
        role = _role_for_new_user(username, user_count)
        password_hash = hash_password(payload.password)
        user = create_user(db, username=username, password_hash=password_hash, role=role)
        update_user_login(db, user)
        return _issue_token(user)


@app.post("/auth/login", response_model=AuthTokenResponse)
def auth_login(payload: AuthLoginRequest) -> AuthTokenResponse:
    username = _normalize_username(payload.username)
    with get_session() as db:
        user = get_user_by_username(db, username)
        if user is None or not user.is_active:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        if not verify_password(payload.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        update_user_login(db, user)
        return _issue_token(user)


@app.get("/auth/me", response_model=AuthUserResponse)
def auth_me(user: User = Depends(_get_current_user)) -> AuthUserResponse:
    return AuthUserResponse(
        username=user.username,
        role=user.role,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
    )


@app.get("/datasets", response_model=list[DatasetMetaResponse])
def datasets(user: User = Depends(_get_current_user)) -> list[DatasetMetaResponse]:
    with get_session() as db:
        records = list_datasets(db)
    return [
        DatasetMetaResponse(
            dataset_id=item.dataset_id,
            filename=item.filename,
            path=item.path,
            source=item.source,
            content_type=item.content_type,
            size_bytes=item.size_bytes,
            uploaded_by=item.uploaded_by,
            uploaded_at=item.uploaded_at,
        )
        for item in records
    ]


def _save_upload(
    file: UploadFile,
    *,
    uploaded_by: str | None,
    source: str = "upload",
) -> Path:
    safe_name = Path(file.filename or "dataset.csv").name.replace(" ", "_")
    timestamp = _utcnow().strftime("%Y%m%d_%H%M%S")
    dest = Path("data/uploads") / f"{timestamp}_{safe_name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    size_bytes = dest.stat().st_size if dest.exists() else None
    dataset_id = uuid.uuid4().hex
    try:
        with get_session() as db:
            create_dataset_record(
                db,
                dataset_id=dataset_id,
                filename=safe_name,
                path=str(dest),
                source=source,
                content_type=file.content_type,
                size_bytes=size_bytes,
                uploaded_by=uploaded_by,
            )
    except Exception:
        # Do not fail the request if metadata persistence fails.
        pass
    return dest


def _record_dataset_path(path: str, *, uploaded_by: str | None, source: str) -> None:
    dataset_path = Path(path)
    size_bytes = dataset_path.stat().st_size if dataset_path.exists() else None
    dataset_id = uuid.uuid4().hex
    try:
        with get_session() as db:
            create_dataset_record(
                db,
                dataset_id=dataset_id,
                filename=dataset_path.name,
                path=str(dataset_path),
                source=source,
                content_type=None,
                size_bytes=size_bytes,
                uploaded_by=uploaded_by,
            )
    except Exception:
        pass


def _load_failure_risk_model() -> object:
    model_path = MODEL_DIR / "failure_risk_model.pkl"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Failure risk model not found")
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def _predict_failure_risk_scores(model: object, features: pd.DataFrame) -> list[float]:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(features))
        if proba.ndim == 1:
            return proba.astype(float).tolist()
        if proba.shape[1] == 1:
            # Single-class model: map to deterministic 0/1 risk scores.
            positive_class = None
            if hasattr(model, "classes_") and len(model.classes_) == 1:
                positive_class = model.classes_[0]
            if positive_class in (1, "1", True, "true", "True"):
                return [1.0] * proba.shape[0]
            return [0.0] * proba.shape[0]
        if hasattr(model, "classes_") and 1 in list(model.classes_):
            idx = list(model.classes_).index(1)
            return proba[:, idx].astype(float).tolist()
        return proba[:, 1].astype(float).tolist()

    raw = np.asarray(model.predict(features), dtype=float)
    return raw.tolist()

def _execute_training(
    *,
    training_type: str,
    data_source: str,
    save_models: bool,
    output_dir: str,
    window_size: int,
    progress_sink: Optional[list[TrainStep]] = None,
    on_progress: Optional[Callable[[list[TrainStep]], None]] = None,
) -> TrainResponse:
    pipeline = TrainingPipeline(config_path="configs/base.yaml")
    progress: list[TrainStep] = progress_sink if progress_sink is not None else []

    def _mark(step: str, status: str, detail: Optional[str] = None) -> None:
        progress.append(TrainStep(step=step, status=status, detail=detail))
        if on_progress is not None:
            on_progress(progress)

    _mark("load_data", "in_progress")
    pipeline.load_data(data_source)
    _mark("load_data", "completed")

    _mark("aeos_preprocessing", "in_progress")
    pipeline.aeos_preprocessing()
    _mark("aeos_preprocessing", "completed")

    _mark("feature_engineering", "in_progress")
    pipeline.feature_engineering(window_size=window_size)
    _mark("feature_engineering", "completed")

    _mark("train_test_split", "in_progress")
    X_train, X_test, y_train, y_test = pipeline.prepare_data_splits()
    _mark("train_test_split", "completed")

    if training_type in {"failure_risk", "all"}:
        _mark("train_failure_risk", "in_progress")
        result = pipeline.train_failure_risk_model(X_train, y_train)
        if result:
            _mark("train_failure_risk", "completed")
        else:
            _mark("train_failure_risk", "skipped", "No labels provided")
    else:
        _mark("train_failure_risk", "skipped", "Not requested")

    if training_type in {"rul", "all"}:
        _mark("train_rul", "in_progress")
        rul_result = pipeline.train_rul_model(X_train, y_train)
        if rul_result:
            _mark("train_rul", "completed")
        else:
            _mark("train_rul", "skipped", "No training data")
    else:
        _mark("train_rul", "skipped", "Not requested")

    if training_type in {"anomaly", "all"}:
        _mark("train_anomaly", "in_progress")
        anomaly_result = pipeline.train_anomaly_model(X_train)
        if anomaly_result:
            _mark("train_anomaly", "completed")
        else:
            _mark("train_anomaly", "skipped", "No training data")
    else:
        _mark("train_anomaly", "skipped", "Not requested")

    _mark("evaluate_models", "in_progress")
    pipeline.evaluate_models(X_test, y_test)
    _mark("evaluate_models", "completed")

    if save_models:
        _mark("save_models", "in_progress")
        pipeline.save_models(output_dir)
        pipeline.save_results(output_dir)
        _mark("save_models", "completed")
    else:
        _mark("save_models", "skipped", "save_models=false")

    return TrainResponse(
        status="success",
        training_type=training_type,
        data_source=data_source,
        data_shape=list(pipeline.data.shape) if pipeline.data is not None else None,
        features_shape=list(pipeline.features.shape) if pipeline.features is not None else None,
        models_trained=list(pipeline.models.keys()),
        evaluation_results=pipeline.results,
        output_dir=output_dir if save_models else None,
        progress=progress,
    )


def _steps_to_dicts(steps: list[TrainStep]) -> list[dict[str, Any]]:
    return [step.dict() for step in steps]


def _steps_from_dicts(items: list[dict[str, Any]]) -> list[TrainStep]:
    steps: list[TrainStep] = []
    for item in items:
        try:
            steps.append(TrainStep(**item))
        except TypeError:
            continue
    return steps


def _train_job_to_status(record: TrainJobRecord) -> TrainJobStatus:
    progress_items = decode_progress(record)
    steps = _steps_from_dicts(progress_items)
    result_payload = decode_result(record)
    result = TrainResponse(**result_payload) if isinstance(result_payload, dict) else None
    return TrainJobStatus(
        job_id=record.job_id,
        status=record.status,
        training_type=record.training_type,
        data_source=record.data_source,
        progress=steps,
        result=result,
        error=record.error_text,
    )


def _run_training_job(
    *,
    job_id: str,
    training_type: str,
    data_source: str,
    save_models: bool,
    output_dir: str,
    window_size: int,
) -> None:
    with get_session() as db:
        record = get_train_job(db, job_id)
        if record is None:
            return

        progress_items = decode_progress(record)
        steps = _steps_from_dicts(progress_items)

        def _persist_progress(current_steps: list[TrainStep]) -> None:
            with _DB_LOCK:
                update_train_job_progress(db, record, _steps_to_dicts(current_steps))

        try:
            result = _execute_training(
                training_type=training_type,
                data_source=data_source,
                save_models=save_models,
                output_dir=output_dir,
                window_size=window_size,
                progress_sink=steps,
                on_progress=_persist_progress,
            )
            with _DB_LOCK:
                finalize_train_job(
                    db,
                    record,
                    status="completed",
                    result=result.dict(),
                    error_text=None,
                    progress_items=_steps_to_dicts(steps),
                )
        except Exception as exc:  # pragma: no cover - background reporting
            with _DB_LOCK:
                finalize_train_job(
                    db,
                    record,
                    status="failed",
                    result=None,
                    error_text=str(exc),
                    progress_items=_steps_to_dicts(steps),
                )

@app.post("/aeos/energy-deviation", response_model=EnergyDeviationResponse)
def energy_deviation(payload: EnergyDeviationRequest) -> EnergyDeviationResponse:
    if payload.baseline == 0:
        raise HTTPException(status_code=400, detail="baseline must be non-zero")
    deviation = calculate_deviation(payload.actual, payload.baseline)
    anomaly = None
    if payload.threshold_pct is not None:
        anomaly = abs(deviation) >= payload.threshold_pct
    return EnergyDeviationResponse(deviation_pct=deviation, anomaly=anomaly)


@app.post("/aeos/energy-deviation/batch", response_model=EnergyDeviationBatchResponse)
def energy_deviation_batch(
    payload: EnergyDeviationBatchRequest,
) -> EnergyDeviationBatchResponse:
    results: list[EnergyDeviationResponse] = []
    for item in payload.items:
        if item.baseline == 0:
            raise HTTPException(status_code=400, detail="baseline must be non-zero")
        deviation = calculate_deviation(item.actual, item.baseline)
        anomaly = None
        if item.threshold_pct is not None:
            anomaly = abs(deviation) >= item.threshold_pct
        results.append(EnergyDeviationResponse(deviation_pct=deviation, anomaly=anomaly))
    return EnergyDeviationBatchResponse(results=results)


@app.post("/train", response_model=TrainResponse)
def train(
    training_type: str = Form("all"),
    data_path: Optional[str] = Form(None),
    save_models: bool = Form(True),
    output_dir: str = Form("models/"),
    window_size: int = Form(24),
    file: Optional[UploadFile] = File(None),
    admin_user: User = Depends(require_role("admin")),
) -> TrainResponse:
    allowed_types = {"failure_risk", "rul", "anomaly", "all"}
    if training_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid training_type")

    data_source = data_path or "data/processed/training_data.csv"
    if file is not None:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only .csv uploads are supported")
        data_source = str(_save_upload(file, uploaded_by=admin_user.username))
    elif data_path:
        _record_dataset_path(data_source, uploaded_by=admin_user.username, source="data_path")

    try:
        return _execute_training(
            training_type=training_type,
            data_source=data_source,
            save_models=save_models,
            output_dir=output_dir,
            window_size=window_size,
        )
    except Exception as exc:  # pragma: no cover - used for API error reporting
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/train/async", response_model=TrainJobStatus)
def train_async(
    training_type: str = Form("all"),
    data_path: Optional[str] = Form(None),
    save_models: bool = Form(True),
    output_dir: str = Form("models/"),
    window_size: int = Form(24),
    file: Optional[UploadFile] = File(None),
    admin_user: User = Depends(require_role("admin")),
) -> TrainJobStatus:
    allowed_types = {"failure_risk", "rul", "anomaly", "all"}
    if training_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid training_type")

    data_source = data_path or "data/processed/training_data.csv"
    if file is not None:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only .csv uploads are supported")
        data_source = str(_save_upload(file, uploaded_by=admin_user.username))
    elif data_path:
        _record_dataset_path(data_source, uploaded_by=admin_user.username, source="data_path")

    job_id = uuid.uuid4().hex
    with get_session() as db:
        with _DB_LOCK:
            record = create_train_job(
                db,
                job_id=job_id,
                status="running",
                training_type=training_type,
                data_source=data_source,
                save_models=save_models,
                output_dir=output_dir,
                window_size=window_size,
                created_by=admin_user.username,
            )

    thread = threading.Thread(
        target=_run_training_job,
        kwargs={
            "job_id": job_id,
            "training_type": training_type,
            "data_source": data_source,
            "save_models": save_models,
            "output_dir": output_dir,
            "window_size": window_size,
        },
        daemon=True,
    )
    thread.start()

    return _train_job_to_status(record)


@app.get("/train/status/{job_id}", response_model=TrainJobStatus)
def train_status(
    job_id: str,
    _admin_user: User = Depends(require_role("admin")),
) -> TrainJobStatus:
    with get_session() as db:
        record = get_train_job(db, job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return _train_job_to_status(record)


@app.post("/failure-risk/predict", response_model=FailureRiskResponse)
def predict_failure_risk(
    window_size: int = Form(24),
    step: int = Form(6),
    threshold: float = Form(0.7),
    data_path: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> FailureRiskResponse:
    if window_size < 1:
        raise HTTPException(status_code=400, detail="window_size must be >= 1")
    if step < 1:
        raise HTTPException(status_code=400, detail="step must be >= 1")

    data_source = data_path or "data/processed/training_data.csv"
    if file is not None:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only .csv uploads are supported")
        data_source = str(_save_upload(file, uploaded_by=None, source="failure_risk_predict"))
    elif data_path:
        _record_dataset_path(data_source, uploaded_by=None, source="failure_risk_predict")

    pipeline = TrainingPipeline(config_path="configs/base.yaml")
    pipeline.load_data(data_source)
    pipeline.aeos_preprocessing()
    pipeline.feature_engineering(window_size=window_size, step=step)

    features = pipeline.features.copy()
    if "target_failure" in features.columns:
        features = features.drop(columns=["target_failure"])

    if features.empty:
        raise HTTPException(status_code=400, detail="No features available for prediction")

    model = _load_failure_risk_model()

    scores = _predict_failure_risk_scores(model, features)

    timestamps = []
    rig_ids = []
    if "timestamp" in pipeline.data.columns:
        indices = list(range(0, len(pipeline.data) - window_size + 1, step))
        for start in indices:
            end_idx = start + window_size - 1
            if end_idx >= len(pipeline.data):
                continue
            row = pipeline.data.iloc[end_idx]
            timestamps.append(row["timestamp"])
            if "rig_id" in pipeline.data.columns:
                rig_ids.append(row["rig_id"])

    predictions: list[FailureRiskPrediction] = []
    for idx, score in enumerate(scores):
        pred = FailureRiskPrediction(
            risk_score=float(score),
            high_risk=float(score) >= float(threshold),
        )
        if idx < len(timestamps):
            pred.timestamp = (
                timestamps[idx].isoformat()
                if hasattr(timestamps[idx], "isoformat")
                else str(timestamps[idx])
            )
        if idx < len(rig_ids):
            pred.rig_id = rig_ids[idx]
        predictions.append(pred)

    return FailureRiskResponse(
        status="success",
        threshold=float(threshold),
        predictions=predictions,
    )


@app.post("/decision/risk")
def decision_risk(payload: DecisionRiskRequest) -> dict:
    predictions = {
        "failure_risk": payload.failure_risk,
        "anomaly_score": payload.anomaly_score,
        "rul": payload.rul,
        "weights": payload.weights,
        "max_rul": payload.max_rul,
    }
    if payload.metadata:
        predictions["metadata"] = payload.metadata
    return calculate_overall_risk(payload.equipment_id, predictions)


@app.post("/fintech/report")
def fintech_report(
    payload: FintechReportRequest,
    _admin_user: User = Depends(require_role("admin")),
) -> dict:
    return generate_financial_report(payload.periods)


@app.post("/fintech/report/pdf")
def fintech_report_pdf(
    payload: FintechReportRequest,
    _admin_user: User = Depends(require_role("admin")),
) -> StreamingResponse:
    report = generate_financial_report(payload.periods)
    pdf_bytes = generate_financial_report_pdf(report)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=rigvisionx-fintech-report.pdf"},
    )


@app.post("/fintech/forecast")
def fintech_forecast(payload: FintechForecastRequest) -> dict:
    frame = pd.DataFrame(payload.history)
    forecast = forecast_costs(frame, periods=payload.periods)
    return {"forecast": forecast.to_dict(orient="records")}


@app.post("/fintech/roi")
def fintech_roi(payload: FintechRoiRequest) -> dict:
    roi_pct = calculate_roi(
        payload.implementation_cost,
        payload.annual_savings,
        years=payload.years,
    )
    response = {"roi_pct": roi_pct}
    if payload.monthly_savings and payload.monthly_savings > 0:
        response["break_even_months"] = calculate_break_even_period(
            payload.implementation_cost,
            payload.monthly_savings,
        )
    return response


@app.post("/fintech/compare")
def fintech_compare(payload: FintechCompareRequest) -> dict:
    return compare_scenarios(
        payload.preventive_cost,
        payload.failure_cost,
        payload.failure_probability,
    )


@app.post("/fintech/cost-savings")
def fintech_cost_savings(payload: FintechSavingsRequest) -> dict:
    savings = calculate_cost_savings(
        payload.baseline_failures,
        payload.preventive_costs,
        payload.actual_failures,
    )
    return {"savings": savings}


@app.post("/reports/system/pdf")
def system_report_pdf(
    payload: SystemReportRequest,
    _admin_user: User = Depends(require_role("admin")),
) -> StreamingResponse:
    data = pd.read_json(io.StringIO(payload.dataset), orient="split")
    selections = {
        "rig_id": payload.rig_id,
        "metrics": payload.metrics,
        "table_metric": payload.table_metric,
        "anomaly_metrics": payload.anomaly_metrics,
        "anomaly_threshold": payload.anomaly_threshold,
        "risk_metrics": payload.risk_metrics,
        "risk_window_size": payload.risk_window_size,
        "risk_threshold": payload.risk_threshold,
        "title": payload.title,
    }
    pdf_bytes = generate_system_report_pdf(data, selections)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=rigvisionx-system-report.pdf"},
    )


@app.post("/explainability/feature-importance")
def get_feature_importance(
    model_type: str = Form("failure_risk"),
    top_n: int = Form(15),
) -> dict:
    """
    Get global feature importance for a model using SHAP.

    Args:
        model_type: Model to explain ("failure_risk" or "rul")
        top_n: Number of top features to return

    Returns:
        Feature importance rankings
    """
    if model_type not in ["failure_risk", "rul"]:
        raise HTTPException(
            status_code=400,
            detail="model_type must be 'failure_risk' or 'rul'"
        )

    # Load explainer
    explainer_path = MODEL_DIR / f"{model_type}_explainer.pkl"
    if not explainer_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Explainer not found for model '{model_type}'. Train the model first."
        )

    try:
        explainer = SHAPExplainer.load(str(explainer_path))

        # Load some data for feature importance calculation
        pipeline = TrainingPipeline(config_path="configs/base.yaml")
        pipeline.load_data("data/processed/training_data.csv")
        pipeline.aeos_preprocessing()
        pipeline.feature_engineering(window_size=24, step=6)

        features = pipeline.features.copy()
        # Remove target columns
        for col in ["target_failure", "target_horizon"]:
            if col in features.columns:
                features = features.drop(columns=[col])

        if features.empty:
            raise HTTPException(
                status_code=400,
                detail="No features available"
            )

        # Get feature importance
        importance_data = explainer.get_feature_importance(features, top_n=top_n)

        return {
            "status": "success",
            "model_type": model_type,
            "top_features": importance_data["top_features"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute feature importance: {str(e)}"
        )


@app.post("/explainability/explain-prediction")
def explain_prediction(
    window_size: int = Form(24),
    step: int = Form(6),
    model_type: str = Form("failure_risk"),
    prediction_index: int = Form(0),
    data_path: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> dict:
    """
    Explain a specific prediction using SHAP.

    Args:
        window_size: Window size for feature engineering
        step: Step size for windowing
        model_type: Model to explain ("failure_risk" or "rul")
        prediction_index: Index of prediction to explain (default: latest)
        data_path: Optional path to data file
        file: Optional uploaded CSV file

    Returns:
        Detailed explanation of the prediction
    """
    if model_type not in ["failure_risk", "rul"]:
        raise HTTPException(
            status_code=400,
            detail="model_type must be 'failure_risk' or 'rul'"
        )

    # Load data
    data_source = data_path or "data/processed/training_data.csv"
    if file is not None:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only .csv uploads are supported")
        data_source = str(_save_upload(file, uploaded_by=None, source="explain_prediction"))

    # Process features
    pipeline = TrainingPipeline(config_path="configs/base.yaml")
    pipeline.load_data(data_source)
    pipeline.aeos_preprocessing()
    pipeline.feature_engineering(window_size=window_size, step=step)

    features = pipeline.features.copy()
    # Remove target columns
    for col in ["target_failure", "target_horizon"]:
        if col in features.columns:
            features = features.drop(columns=[col])

    if features.empty:
        raise HTTPException(status_code=400, detail="No features available")

    # Get single prediction to explain
    if prediction_index >= len(features):
        prediction_index = len(features) - 1

    X_single = features.iloc[[prediction_index]]

    # Load explainer
    explainer_path = MODEL_DIR / f"{model_type}_explainer.pkl"
    if not explainer_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Explainer not found for model '{model_type}'. Train the model first."
        )

    try:
        explainer = SHAPExplainer.load(str(explainer_path))

        # Explain single prediction
        explanation = explainer.explain_single_prediction(X_single)

        # Add timestamp if available
        timestamp = None
        if "timestamp" in pipeline.data.columns:
            indices = list(range(0, len(pipeline.data) - window_size + 1, step))
            if prediction_index < len(indices):
                start = indices[prediction_index]
                end_idx = start + window_size - 1
                if end_idx < len(pipeline.data):
                    ts = pipeline.data.iloc[end_idx]["timestamp"]
                    timestamp = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

        return {
            "status": "success",
            "model_type": model_type,
            "prediction_index": prediction_index,
            "timestamp": timestamp,
            "explanation": explanation
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to explain prediction: {str(e)}"
        )

