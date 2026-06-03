"""Fuse multiple risk predictions into overall risk score"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from rigvisionx.decision_engine.actions import generate_alert, get_recommended_actions
from rigvisionx.decision_engine.thresholds import RiskThresholds


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _normalize_anomaly_score(score: float) -> float:
    if score < 0:
        score = abs(score)
    if 0.0 <= score <= 1.0:
        return _clamp(score)
    # Treat as z-score style values; 3+ should be high risk.
    return _clamp(score / 3.0)


def _normalize_rul(rul: float, max_rul: float = 100.0) -> float:
    if max_rul <= 0:
        max_rul = 1.0
    normalized = 1.0 - (max(rul, 0.0) / max_rul)
    return _clamp(normalized)


def _normalize_components(
    failure_risk: Optional[float],
    anomaly_score: Optional[float],
    rul: Optional[float],
    max_rul: float,
) -> dict:
    components: dict[str, float] = {}
    if failure_risk is not None:
        components["failure"] = _clamp(failure_risk)
    if anomaly_score is not None:
        components["anomaly"] = _normalize_anomaly_score(anomaly_score)
    if rul is not None:
        components["rul"] = _normalize_rul(rul, max_rul=max_rul)
    return components


def _normalize_weights(weights: Optional[dict], components: dict) -> dict:
    if not components:
        return {}
    default_weights = {"failure": 0.5, "anomaly": 0.3, "rul": 0.2}
    weights = weights or default_weights
    weight_sum = sum(weights.get(key, 0.0) for key in components)
    if weight_sum <= 0:
        equal_weight = 1.0 / len(components)
        return {key: equal_weight for key in components}
    return {key: weights.get(key, 0.0) / weight_sum for key in components}


def fuse_risks(
    failure_risk: Optional[float],
    anomaly_score: Optional[float],
    rul: Optional[float],
    weights: Optional[dict] = None,
    max_rul: float = 100.0,
) -> float:
    """
    Combine multiple risk indicators into single risk score.

    Args:
        failure_risk: Probability of failure (0-1)
        anomaly_score: Anomaly detection score
        rul: Remaining useful life
        weights: Custom weights for each risk factor
        max_rul: Upper bound used to normalize RUL

    Returns:
        Fused risk score (0-1)
    """
    components = _normalize_components(
        failure_risk=failure_risk,
        anomaly_score=anomaly_score,
        rul=rul,
        max_rul=max_rul,
    )
    weights_used = _normalize_weights(weights, components)
    if not components:
        return 0.0
    return sum(components[key] * weights_used.get(key, 0.0) for key in components)


def calculate_overall_risk(equipment_id: str, predictions: dict) -> dict:
    """Calculate overall equipment risk."""
    predictions = predictions or {}
    failure_risk = _coerce_float(
        predictions.get("failure_risk") or predictions.get("failure_probability")
    )
    anomaly_score = _coerce_float(
        predictions.get("anomaly_score") or predictions.get("anomaly")
    )
    rul = _coerce_float(
        predictions.get("rul")
        or predictions.get("estimated_rul")
        or predictions.get("remaining_useful_life")
    )
    max_rul = _coerce_float(predictions.get("max_rul") or predictions.get("rul_max")) or 100.0

    components = _normalize_components(
        failure_risk=failure_risk,
        anomaly_score=anomaly_score,
        rul=rul,
        max_rul=max_rul,
    )
    weights_used = _normalize_weights(predictions.get("weights"), components)
    risk_score = sum(
        components[key] * weights_used.get(key, 0.0) for key in components
    ) if components else 0.0
    risk_level = RiskThresholds.get_risk_level(risk_score)
    actions = get_recommended_actions(risk_level)
    timestamp = datetime.now(timezone.utc).isoformat()

    alert = generate_alert(
        equipment_id,
        risk_level,
        {**predictions, "risk_score": risk_score, "risk_level": risk_level},
    )

    return {
        "equipment_id": equipment_id,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "components": components,
        "weights": weights_used,
        "actions": actions,
        "timestamp": timestamp,
        "alert": alert,
    }
