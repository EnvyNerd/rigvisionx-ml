"""Training script for RUL model"""

import yaml
from .model import RULModel


def train_rul_model(config_path: str):
    """Train the RUL model."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = RULModel(config)
    model.build()
    # Training logic here
    pass


if __name__ == "__main__":
    train_rul_model("configs/model_rul.yaml")
