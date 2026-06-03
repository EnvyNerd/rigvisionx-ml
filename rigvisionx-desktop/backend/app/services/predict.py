from __future__ import annotations

from typing import Any
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from rigvisionx.train_pipeline import TrainingPipeline

from .paths import resolve_repo_path


def _load_model(model_dir: Path) -> object:
    model_path = model_dir / "failure_risk_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Failure risk model not found")
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def _predict_scores(model: object, features: pd.DataFrame) -> list[float]:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(features))
        if proba.ndim == 1:
            return proba.astype(float).tolist()
        if proba.shape[1] == 1:
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


def predict_failure_risk(
    *,
    data_source: str,
    window_size: int,
    step: int,
    threshold: float,
) -> dict[str, Any]:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if step < 1:
        raise ValueError("step must be >= 1")

    config_env = os.environ.get("RIGVISIONX_CONFIG")
    model_env = os.environ.get("RIGVISIONX_MODEL_DIR")
    config_path = resolve_repo_path(config_env or "configs/base.yaml")
    model_dir = Path(resolve_repo_path(model_env or "models"))

    pipeline = TrainingPipeline(config_path=str(config_path))
    pipeline.load_data(data_source)
    pipeline.aeos_preprocessing()
    pipeline.feature_engineering(window_size=window_size, step=step)

    features = pipeline.features.copy()
    for col in ["target_failure", "target_horizon"]:
        if col in features.columns:
            features = features.drop(columns=[col])

    if features.empty:
        raise ValueError("No features available for prediction")

    model = _load_model(model_dir)
    scores = _predict_scores(model, features)

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

    predictions: list[dict[str, Any]] = []
    for idx, score in enumerate(scores):
        entry: dict[str, Any] = {
            "risk_score": float(score),
            "high_risk": float(score) >= float(threshold),
        }
        if idx < len(timestamps):
            ts = timestamps[idx]
            entry["timestamp"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        if idx < len(rig_ids):
            entry["rig_id"] = rig_ids[idx]
        predictions.append(entry)

    return {
        "status": "success",
        "threshold": float(threshold),
        "predictions": predictions,
    }
