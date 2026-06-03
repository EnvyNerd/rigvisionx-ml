"""Inference script for RUL model"""

from .model import RULModel


def infer_rul(model_path: str, features):
    """Run inference on RUL model."""
    model = RULModel({})
    # Load model from path
    predictions = model.predict(features)
    return predictions
