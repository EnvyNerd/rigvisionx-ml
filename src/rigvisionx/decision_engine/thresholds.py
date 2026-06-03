"""Risk thresholds and decision rules"""


class RiskThresholds:
    """Manage risk decision thresholds."""

    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    CRITICAL = 0.95

    @staticmethod
    def get_risk_level(score: float) -> str:
        """Classify risk score into level."""
        if score < RiskThresholds.LOW:
            return "OK"
        elif score < RiskThresholds.MEDIUM:
            return "LOW"
        elif score < RiskThresholds.HIGH:
            return "MEDIUM"
        elif score < RiskThresholds.CRITICAL:
            return "HIGH"
        else:
            return "CRITICAL"
