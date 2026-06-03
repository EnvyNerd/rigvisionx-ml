from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

from .services import build_summary, load_dataset, predict_failure_risk, run_training
from .services.paths import resolve_existing_path
from .services.jobs import get_job, list_jobs, start_training_job
from .storage import log_history, read_history, save_upload

router = APIRouter()


class EdaResponse(BaseModel):
    status: str = "success"
    dataset: dict[str, Any]
    summary: dict[str, Any]


class TrainResponse(BaseModel):
    status: str
    training_type: str
    data_source: str
    data_shape: Optional[list[int]] = None
    features_shape: Optional[list[int]] = None
    models_trained: list[str]
    evaluation_results: Optional[dict[str, Any]] = None
    output_dir: Optional[str] = None


class FailureRiskPrediction(BaseModel):
    timestamp: Optional[str] = None
    rig_id: Optional[str] = None
    risk_score: float
    high_risk: bool


class FailureRiskResponse(BaseModel):
    status: str
    threshold: float
    predictions: list[FailureRiskPrediction]

class TrainJobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    payload: dict[str, Any]
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class HistoryResponse(BaseModel):
    records: list[dict[str, Any]]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

def _require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if os.environ.get("RIGVISIONX_DISABLE_AUTH") == "1":
        return
    expected = os.environ.get("RIGVISIONX_API_KEY")
    if not expected:
        return
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _resolve_data_source(
    data_path: Optional[str],
    file: Optional[UploadFile],
) -> tuple[str, dict[str, Any]]:
    if file is not None:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only .csv uploads are supported")
        saved = save_upload(file)
        return str(saved), {"source": "upload", "filename": file.filename}
    if data_path:
        resolved = str(resolve_existing_path(data_path))
        return resolved, {"source": "path", "path": resolved}
    raise HTTPException(status_code=400, detail="Provide data_path or file")


@router.post("/eda/summary", response_model=EdaResponse, dependencies=[Depends(_require_api_key)])
def eda_summary(
    data_path: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> EdaResponse:
    data_source, meta = _resolve_data_source(data_path, file)
    try:
        data = load_dataset(data_source)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    summary = build_summary(data)
    log_history(
        "eda_summary",
        data_source=data_source,
        metadata={"rows": int(data.shape[0]), "columns": int(data.shape[1]), **meta},
    )
    return EdaResponse(
        dataset={"path": data_source, "rows": data.shape[0], "columns": data.shape[1]},
        summary=summary,
    )


@router.post("/train", response_model=TrainResponse, dependencies=[Depends(_require_api_key)])
def train_model(
    training_type: str = Form("all"),
    data_path: Optional[str] = Form(None),
    save_models: bool = Form(True),
    output_dir: str = Form("models/"),
    window_size: int = Form(24),
    file: Optional[UploadFile] = File(None),
) -> TrainResponse:
    data_source, meta = _resolve_data_source(data_path, file)
    try:
        result = run_training(
            training_type=training_type,
            data_source=data_source,
            save_models=save_models,
            output_dir=output_dir,
            window_size=window_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    log_history(
        "train_sync",
        data_source=data_source,
        metadata={
            "training_type": training_type,
            "save_models": bool(save_models),
            "output_dir": output_dir,
            "window_size": int(window_size),
            **meta,
        },
    )
    return TrainResponse(**result)


@router.post(
    "/predict/failure-risk",
    response_model=FailureRiskResponse,
    dependencies=[Depends(_require_api_key)],
)
def failure_risk_predict(
    window_size: int = Form(24),
    step: int = Form(6),
    threshold: float = Form(0.7),
    data_path: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> FailureRiskResponse:
    data_source, meta = _resolve_data_source(data_path, file)
    try:
        result = predict_failure_risk(
            data_source=data_source,
            window_size=window_size,
            step=step,
            threshold=threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    log_history(
        "predict_failure_risk",
        data_source=data_source,
        metadata={
            "window_size": int(window_size),
            "step": int(step),
            "threshold": float(threshold),
            **meta,
        },
    )
    return FailureRiskResponse(**result)


@router.post("/train/async", response_model=TrainJobStatus, dependencies=[Depends(_require_api_key)])
def train_async(
    training_type: str = Form("all"),
    data_path: Optional[str] = Form(None),
    save_models: bool = Form(True),
    output_dir: str = Form("models/"),
    window_size: int = Form(24),
    file: Optional[UploadFile] = File(None),
) -> TrainJobStatus:
    data_source, meta = _resolve_data_source(data_path, file)
    payload = {
        "training_type": training_type,
        "data_source": data_source,
        "save_models": save_models,
        "output_dir": output_dir,
        "window_size": window_size,
    }
    record = start_training_job(payload)
    log_history(
        "train_async",
        data_source=data_source,
        metadata={
            "training_type": training_type,
            "save_models": bool(save_models),
            "output_dir": output_dir,
            "window_size": int(window_size),
            "job_id": record["job_id"],
            **meta,
        },
    )
    return TrainJobStatus(**record)


@router.get("/train/status/{job_id}", response_model=TrainJobStatus, dependencies=[Depends(_require_api_key)])
def train_status(job_id: str) -> TrainJobStatus:
    record = get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return TrainJobStatus(**record)


@router.get("/train/jobs", response_model=list[TrainJobStatus], dependencies=[Depends(_require_api_key)])
def train_jobs(limit: int = 20) -> list[TrainJobStatus]:
    return [TrainJobStatus(**item) for item in list_jobs(limit=limit)]


@router.get("/history", response_model=HistoryResponse, dependencies=[Depends(_require_api_key)])
def history(limit: int = 50) -> HistoryResponse:
    records = read_history(limit=limit)
    return HistoryResponse(records=records)
