"""Tests for decision engine module"""

import pytest
from rigvisionx.decision_engine.thresholds import RiskThresholds
from rigvisionx.decision_engine.actions import get_recommended_actions, generate_alert
from rigvisionx.decision_engine.risk_fusion import fuse_risks, calculate_overall_risk


def test_risk_level_classification():
    """Test risk level classification."""
    assert RiskThresholds.get_risk_level(0.1) == "OK"
    assert RiskThresholds.get_risk_level(0.4) == "LOW"
    assert RiskThresholds.get_risk_level(0.7) == "MEDIUM"
    assert RiskThresholds.get_risk_level(0.85) == "HIGH"
    assert RiskThresholds.get_risk_level(0.98) == "CRITICAL"


def test_risk_threshold_constants():
    """Test threshold constants."""
    assert RiskThresholds.LOW == 0.3
    assert RiskThresholds.MEDIUM == 0.6
    assert RiskThresholds.HIGH == 0.8
    assert RiskThresholds.CRITICAL == 0.95


def test_recommended_actions_ok():
    """Test recommended actions for OK status."""
    actions = get_recommended_actions("OK")
    assert "Monitor regularly" in actions


def test_recommended_actions_critical():
    """Test recommended actions for CRITICAL status."""
    actions = get_recommended_actions("CRITICAL")
    assert len(actions) > 0
    assert any("Emergency" in action or "Stop" in action for action in actions)


def test_recommended_actions_low():
    """Test recommended actions for LOW risk."""
    actions = get_recommended_actions("LOW")
    assert len(actions) > 0
    assert "Schedule preventive inspection" in actions


def test_fuse_risks_defaults():
    score = fuse_risks(failure_risk=0.8, anomaly_score=0.5, rul=20, max_rul=100)
    assert 0.0 <= score <= 1.0
    assert pytest.approx(score, rel=1e-3) == 0.71


def test_calculate_overall_risk_outputs():
    result = calculate_overall_risk(
        "RIG_001",
        {"failure_risk": 0.8, "anomaly_score": 0.5, "rul": 20},
    )
    assert result["equipment_id"] == "RIG_001"
    assert result["risk_level"] == "MEDIUM"
    assert 0.0 <= result["risk_score"] <= 1.0
    assert "actions" in result


def test_generate_alert():
    alert = generate_alert("RIG_002", "HIGH", {"risk_score": 0.82})
    assert alert["equipment_id"] == "RIG_002"
    assert alert["risk_level"] == "HIGH"
    assert "actions" in alert
