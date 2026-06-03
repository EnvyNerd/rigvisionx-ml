"""
Multi-output model for TerraEnergy AI.

Simultaneously predicts:
1. Failure horizon (classification: Normal/Caution/Warning/Critical)
2. Remaining Useful Life (regression: hours to failure)
"""

from .model import MultiOutputModel
from .train import train_multi_output_model

__all__ = ["MultiOutputModel", "train_multi_output_model"]

