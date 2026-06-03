"""Inference script for failure risk model"""

from .model import FailureRiskModel


def infer_failure_risk(model_path: str, features):
    """Run inference on failure risk model."""
    model = FailureRiskModel({})
    # Load model from path
    predictions = model.predict(features)
    return predictions
