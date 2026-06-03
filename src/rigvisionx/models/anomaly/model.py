"""Anomaly detection model architecture"""


class AnomalyModel:
    """Detects anomalies in equipment behavior."""

    def __init__(self, config: dict):
        self.config = config
        self.model = None

    def build(self):
        """Build the model."""
        pass

    def predict(self, features):
        """Detect anomalies."""
        pass
