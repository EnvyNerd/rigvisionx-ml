"""
Explainability module for TerraEnergy AI.

Provides SHAP-based explanations for model predictions.
"""

from .shap_explainer import SHAPExplainer
from .failure_horizon import generate_failure_horizon_labels

__all__ = ["SHAPExplainer", "generate_failure_horizon_labels"]

