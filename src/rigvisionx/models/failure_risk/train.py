"""Training script for failure risk model"""

import yaml
from .model import FailureRiskModel


def train_failure_risk_model(config_path: str):
    """Train the failure risk model."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = FailureRiskModel(config)
    model.build()
    # Training logic here
    pass


if __name__ == "__main__":
    train_failure_risk_model("configs/model_failure.yaml")
