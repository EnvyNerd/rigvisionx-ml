"""Training script for anomaly detection model"""

import yaml
from .model import AnomalyModel


def train_anomaly_model(config_path: str):
    """Train the anomaly detection model."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = AnomalyModel(config)
    model.build()
    # Training logic here
    pass


if __name__ == "__main__":
    train_anomaly_model("configs/model_anomaly.yaml")
