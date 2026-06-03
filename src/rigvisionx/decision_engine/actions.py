"""Recommended actions based on risk level"""

from datetime import datetime, timezone


def get_recommended_actions(risk_level: str) -> list:
    """Get recommended actions for risk level."""
    actions_map = {
        "OK": ["Monitor regularly"],
        "LOW": ["Schedule preventive inspection"],
        "MEDIUM": ["Increase monitoring frequency", "Plan maintenance"],
        "HIGH": ["Schedule urgent maintenance", "Prepare replacement"],
        "CRITICAL": ["Stop operation", "Emergency maintenance required"],
    }
    return actions_map.get(risk_level, [])


def generate_alert(equipment_id: str, risk_level: str, predictions: dict) -> dict:
    """Generate alert with recommended actions."""
    actions = get_recommended_actions(risk_level)
    created_at = datetime.now(timezone.utc).isoformat()
    summary = (
        f"{equipment_id} risk level {risk_level}"
        if equipment_id
        else f"Risk level {risk_level}"
    )
    return {
        "equipment_id": equipment_id,
        "risk_level": risk_level,
        "actions": actions,
        "predictions": predictions or {},
        "message": summary,
        "timestamp": created_at,
    }
