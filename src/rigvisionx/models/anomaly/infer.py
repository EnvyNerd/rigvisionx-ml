"""Inference script for anomaly detection model"""

from .model import AnomalyModel


def infer_anomaly(model_path: str, features):
    """Run inference on anomaly detection model."""
    model = AnomalyModel({})
    # Load model from path
    predictions = model.predict(features)
    return predictions
