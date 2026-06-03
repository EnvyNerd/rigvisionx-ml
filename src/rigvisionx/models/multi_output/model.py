"""
Multi-output predictive maintenance model.

Combines failure horizon classification and RUL regression in a single model.
"""

import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class MultiOutputModel:
    """
    Multi-task model for predictive maintenance.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 15,
        min_samples_split: int = 5,
        random_state: int = 42,
        logger: Optional[logging.Logger] = None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.logger = logger or logging.getLogger(__name__)

        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1
        )

        self.regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )

        self.is_fitted = False
        self.feature_names = None

        # ✅ NEW: store expected class mapping
        self.horizon_map = {
            0: "Normal",
            1: "Caution",
            2: "Warning",
            3: "Critical"
        }

    def fit(self, X: pd.DataFrame, y_horizon: pd.Series, y_rul: pd.Series):
        self.logger.info(f"Training multi-output model with {len(X)} samples")

        self.feature_names = X.columns.tolist()

        self.logger.info("Training failure horizon classifier...")
        self.classifier.fit(X, y_horizon)

        self.logger.info("Training RUL regressor...")
        self.regressor.fit(X, y_rul)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        horizon_pred = self.classifier.predict(X)
        horizon_proba = self.classifier.predict_proba(X)
        rul_pred = self.regressor.predict(X)

        return {
            "horizon": horizon_pred,
            "horizon_proba": horizon_proba,
            "rul": rul_pred
        }

    def predict_with_labels(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        SAFE version — dynamically handles missing classes
        """
        predictions = self.predict(X)

        horizon_pred = predictions["horizon"]
        proba = predictions["horizon_proba"]
        rul = predictions["rul"]

        # ✅ CRITICAL FIX: use actual model classes
        model_classes = list(self.classifier.classes_)
        class_to_index = {cls: i for i, cls in enumerate(model_classes)}

        # Prepare probability columns safely
        def safe_proba(class_id):
            if class_id in class_to_index:
                return proba[:, class_to_index[class_id]]
            else:
                return np.zeros(len(proba))

        results = pd.DataFrame({
            "horizon_class": horizon_pred,
            "horizon_label": [self.horizon_map.get(h, "Unknown") for h in horizon_pred],

            # ✅ SAFE dynamic mapping
            "horizon_proba_normal": safe_proba(0),
            "horizon_proba_caution": safe_proba(1),
            "horizon_proba_warning": safe_proba(2),
            "horizon_proba_critical": safe_proba(3),

            "rul_hours": rul,
            "rul_days": rul / 24
        })

        return results

    def feature_importance(self, top_n: int = 15) -> Dict:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        clf_importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.classifier.feature_importances_
        }).sort_values("importance", ascending=False).head(top_n)

        reg_importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.regressor.feature_importances_
        }).sort_values("importance", ascending=False).head(top_n)

        return {
            "horizon_classifier": clf_importance.to_dict(orient="records"),
            "rul_regressor": reg_importance.to_dict(orient="records")
        }

    def evaluate(self, X, y_horizon, y_rul) -> Dict:
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix,
            mean_absolute_error,
            mean_squared_error,
            r2_score
        )

        predictions = self.predict(X)

        horizon_pred = predictions["horizon"]
        accuracy = accuracy_score(y_horizon, horizon_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_horizon, horizon_pred, average="weighted", zero_division=0
        )
        conf_matrix = confusion_matrix(y_horizon, horizon_pred)

        rul_pred = predictions["rul"]
        mae = mean_absolute_error(y_rul, rul_pred)
        rmse = np.sqrt(mean_squared_error(y_rul, rul_pred))
        r2 = r2_score(y_rul, rul_pred)

        mape = np.mean(np.abs((y_rul - rul_pred) / (y_rul + 1e-8))) * 100

        return {
            "classification": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "confusion_matrix": conf_matrix.tolist()
            },
            "regression": {
                "mae_hours": float(mae),
                "rmse_hours": float(rmse),
                "r2_score": float(r2),
                "mape_percent": float(mape)
            }
        }

    def get_config(self) -> Dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
            "n_features": len(self.feature_names) if self.feature_names else None
        }