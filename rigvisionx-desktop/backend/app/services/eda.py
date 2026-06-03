from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    if data.empty:
        raise ValueError("Dataset is empty")
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    return data


def _to_builtin(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _sanitize_mapping(payload: dict) -> dict:
    cleaned: dict = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            cleaned[key] = _sanitize_mapping(value)
        else:
            cleaned[key] = _to_builtin(value)
    return cleaned


def build_summary(data: pd.DataFrame) -> dict[str, Any]:
    shape = [int(dim) for dim in data.shape]
    columns = list(data.columns)
    dtypes = {col: str(dtype) for col, dtype in data.dtypes.items()}
    missing = {col: int(count) for col, count in data.isna().sum().items()}

    describe = data.describe(include="all")
    describe = describe.replace({np.nan: None})
    describe_dict = _sanitize_mapping(describe.to_dict())

    numeric = data.select_dtypes(include=[np.number])
    correlations: dict[str, dict[str, float]] = {}
    if not numeric.empty and numeric.shape[1] > 1:
        corr = numeric.corr().replace({np.nan: None}).to_dict()
        correlations = _sanitize_mapping(corr)

    return {
        "shape": shape,
        "columns": columns,
        "dtypes": dtypes,
        "missing": missing,
        "describe": describe_dict,
        "correlations": correlations,
    }
