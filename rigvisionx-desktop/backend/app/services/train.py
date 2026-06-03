from __future__ import annotations

import os
from typing import Any

from rigvisionx.train_pipeline import TrainingPipeline

from .paths import resolve_repo_path


def run_training(
    *,
    training_type: str,
    data_source: str,
    save_models: bool,
    output_dir: str,
    window_size: int,
) -> dict[str, Any]:
    allowed_types = {"failure_risk", "rul", "anomaly", "all"}
    if training_type not in allowed_types:
        raise ValueError("Invalid training_type")

    config_env = os.environ.get("RIGVISIONX_CONFIG")
    config_path = resolve_repo_path(config_env or "configs/base.yaml")
    pipeline = TrainingPipeline(config_path=str(config_path))

    pipeline.load_data(data_source)
    pipeline.aeos_preprocessing()
    pipeline.feature_engineering(window_size=window_size)
    X_train, X_test, y_train, y_test = pipeline.prepare_data_splits()

    if training_type in {"failure_risk", "all"}:
        pipeline.train_failure_risk_model(X_train, y_train)

    if training_type in {"rul", "all"}:
        pipeline.train_rul_model(X_train, y_train)

    if training_type in {"anomaly", "all"}:
        pipeline.train_anomaly_model(X_train)

    pipeline.evaluate_models(X_test, y_test)

    if save_models:
        pipeline.save_models(output_dir)
        pipeline.save_results(output_dir)

    return {
        "status": "success",
        "training_type": training_type,
        "data_source": data_source,
        "data_shape": list(pipeline.data.shape) if pipeline.data is not None else None,
        "features_shape": list(pipeline.features.shape) if pipeline.features is not None else None,
        "models_trained": list(pipeline.models.keys()),
        "evaluation_results": pipeline.results,
        "output_dir": output_dir if save_models else None,
    }
