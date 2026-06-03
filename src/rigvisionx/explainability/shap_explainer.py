"""
SHAP-based explainability for TerraEnergy AI models.

Provides interpretability for failure risk, RUL, and anomaly detection models.
"""

import logging
from typing import Dict, Optional, Union, List
import numpy as np
import pandas as pd
import shap
import pickle
from pathlib import Path


class SHAPExplainer:
    """
    SHAP explainer for TerraEnergy AI predictive maintenance models.

    Provides:
    - Feature importance analysis
    - Individual prediction explanations
    - Global model interpretability
    - Waterfall plots and force plots
    """

    def __init__(self, model, model_type: str = "tree", logger: Optional[logging.Logger] = None):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model (RandomForest, XGBoost, etc.)
            model_type: Type of explainer ("tree", "kernel", "linear")
            logger: Optional logger instance
        """
        self.model = model
        self.model_type = model_type
        self.logger = logger or logging.getLogger(__name__)
        self.explainer: Optional[shap.Explainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self.expected_value: Optional[float] = None

    def initialize_explainer(self, X_background: pd.DataFrame, max_samples: int = 100):
        """
        Initialize SHAP explainer with background data.

        Args:
            X_background: Background dataset for SHAP (training data sample)
            max_samples: Maximum samples to use for background (faster computation)
        """
        self.logger.info(f"Initializing SHAP explainer (type: {self.model_type})")

        # Sample background data for efficiency
        if len(X_background) > max_samples:
            X_background = X_background.sample(n=max_samples, random_state=42)

        try:
            if self.model_type == "tree":
                # For tree-based models (RandomForest, XGBoost, LightGBM)
                self.explainer = shap.TreeExplainer(self.model, X_background)
                self.logger.info("TreeExplainer initialized")
            elif self.model_type == "kernel":
                # For any model (model-agnostic but slower)
                self.explainer = shap.KernelExplainer(
                    self.model.predict,
                    X_background
                )
                self.logger.info("KernelExplainer initialized")
            elif self.model_type == "linear":
                # For linear models
                self.explainer = shap.LinearExplainer(self.model, X_background)
                self.logger.info("LinearExplainer initialized")
            else:
                raise ValueError(f"Unsupported explainer type: {self.model_type}")

        except Exception as e:
            self.logger.error(f"Failed to initialize SHAP explainer: {e}")
            # Fallback to KernelExplainer
            self.logger.info("Falling back to KernelExplainer")
            self.explainer = shap.KernelExplainer(
                self.model.predict,
                shap.sample(X_background, min(50, len(X_background)))
            )

    def explain(self, X: pd.DataFrame) -> Dict:
        """
        Generate SHAP explanations for predictions.

        Args:
            X: Input features to explain

        Returns:
            Dictionary with SHAP values and expected value
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")

        self.logger.info(f"Computing SHAP values for {len(X)} samples")

        try:
            # Compute SHAP values
            shap_values = self.explainer.shap_values(X)

            # Handle multi-output models (e.g., binary classification with 2 classes)
            if isinstance(shap_values, list):
                # Use positive class SHAP values for binary classification
                shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

            self.shap_values = shap_values

            # Get expected value (base value)
            if hasattr(self.explainer, 'expected_value'):
                expected_value = self.explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[1] if len(expected_value) == 2 else expected_value[0]
                self.expected_value = float(expected_value)
            else:
                self.expected_value = 0.0

            self.logger.info("SHAP values computed successfully")

            return {
                "shap_values": self.shap_values,
                "expected_value": self.expected_value,
                "feature_names": X.columns.tolist()
            }

        except Exception as e:
            self.logger.error(f"Failed to compute SHAP values: {e}")
            raise

    def get_feature_importance(self, X: pd.DataFrame, top_n: int = 10) -> Dict:
        """
        Get global feature importance using mean absolute SHAP values.

        Args:
            X: Input features
            top_n: Number of top features to return

        Returns:
            Dictionary with feature importance rankings
        """
        if self.shap_values is None:
            self.explain(X)

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            "feature": X.columns,
            "importance": mean_abs_shap
        }).sort_values("importance", ascending=False)

        top_features = feature_importance.head(top_n)

        self.logger.info(f"Top {top_n} features computed")

        return {
            "top_features": top_features.to_dict(orient="records"),
            "all_features": feature_importance.to_dict(orient="records")
        }

    def explain_single_prediction(self, X_single: pd.DataFrame, feature_names: Optional[List[str]] = None) -> Dict:
        """
        Explain a single prediction with detailed SHAP breakdown.

        Args:
            X_single: Single sample (1 row DataFrame)
            feature_names: Optional list of feature names to focus on

        Returns:
            Dictionary with prediction explanation
        """
        if len(X_single) != 1:
            raise ValueError("X_single must contain exactly one sample")

        if self.explainer is None:
            raise ValueError("Explainer not initialized")

        # Compute SHAP values for single prediction
        shap_values = self.explainer.shap_values(X_single)

        # Handle multi-output
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

        shap_values = shap_values[0]  # Extract single prediction

        # Get expected value
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1] if len(expected_value) == 2 else expected_value[0]

        # Create explanation dictionary
        explanation = {
            "base_value": float(expected_value),
            "predicted_value": float(expected_value + shap_values.sum()),
            "feature_contributions": []
        }

        # Sort features by absolute SHAP value
        feature_shap = pd.DataFrame({
            "feature": X_single.columns,
            "value": X_single.iloc[0].values,
            "shap_value": shap_values
        }).sort_values(by="shap_value", key=abs, ascending=False)

        # Filter to specific features if requested
        if feature_names:
            feature_shap = feature_shap[feature_shap["feature"].isin(feature_names)]

        for _, row in feature_shap.iterrows():
            explanation["feature_contributions"].append({
                "feature": row["feature"],
                "value": float(row["value"]),
                "shap_contribution": float(row["shap_value"]),
                "impact": "increases" if row["shap_value"] > 0 else "decreases"
            })

        return explanation

    def get_summary_data(self, X: pd.DataFrame, top_n: int = 15) -> Dict:
        """
        Get summary data for visualization.

        Args:
            X: Input features
            top_n: Number of top features to include

        Returns:
            Dictionary with summary data for plotting
        """
        if self.shap_values is None:
            self.explain(X)

        # Get top features by mean absolute SHAP
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

        top_features = X.columns[top_indices].tolist()
        top_shap_values = self.shap_values[:, top_indices]
        top_feature_values = X.iloc[:, top_indices].values

        return {
            "shap_values": top_shap_values.tolist(),
            "feature_values": top_feature_values.tolist(),
            "feature_names": top_features,
            "expected_value": self.expected_value
        }

    def save(self, output_path: str):
        """Save explainer to disk."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        explainer_data = {
            "model": self.model,
            "model_type": self.model_type,
            "expected_value": self.expected_value
        }

        with open(output_path, 'wb') as f:
            pickle.dump(explainer_data, f)

        self.logger.info(f"Explainer saved to {output_path}")

    @classmethod
    def load(cls, input_path: str, logger: Optional[logging.Logger] = None):
        """Load explainer from disk."""
        with open(input_path, 'rb') as f:
            explainer_data = pickle.load(f)

        instance = cls(
            model=explainer_data["model"],
            model_type=explainer_data["model_type"],
            logger=logger
        )
        instance.expected_value = explainer_data["expected_value"]

        if logger:
            logger.info(f"Explainer loaded from {input_path}")

        return instance

