"""
Complete ML training pipeline integrating AEOS, feature engineering, and model training.

Pipeline flow:
1. Load configuration and data
2. AEOS preprocessing (baseline establishment, data cleaning, deviation detection)
3. Feature engineering (windowing, aggregation, feature creation)
4. Model training (failure_risk, RUL, anomaly detection)
5. Evaluation and reporting
"""

import os
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import AEOS modules
from rigvisionx.aeos.baseline import calculate_baseline, update_baseline
from rigvisionx.aeos.preprocessing import clean_data, normalize_data, handle_outliers
from rigvisionx.aeos.energy_deviation import calculate_deviation, detect_anomalous_consumption

# Import feature engineering
from rigvisionx.features.windowing import create_windows, aggregate_by_window
from rigvisionx.features.feature_set import FeatureSet

# Import models
from rigvisionx.models.failure_risk.model import FailureRiskModel
from rigvisionx.models.rul.model import RULModel
from rigvisionx.models.anomaly.model import AnomalyModel

# Import utilities
from rigvisionx.utils.metrics import classification_metrics, regression_metrics
from rigvisionx.utils.splits import temporal_train_test_split, stratified_split
from rigvisionx.utils.logging import setup_logging

# Import explainability
from rigvisionx.explainability import SHAPExplainer, generate_failure_horizon_labels


class TrainingPipeline:
    """Orchestrates complete ML training pipeline."""

    def __init__(self, config_path: str = "configs/base.yaml"):
        """Initialize pipeline with configuration."""
        self.logger = setup_logging(__name__, level="INFO")
        self.config = self._load_config(config_path)
        self.data: Optional[pd.DataFrame] = None
        self.baseline: Optional[Dict] = None
        self.features: Optional[pd.DataFrame] = None
        self.feature_set = FeatureSet()
        self.models = {}
        self.results = {}
        self.explainers = {}  # Store SHAP explainers for each model

    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration file."""
        resolved = self._resolve_config_path(config_path)
        self.logger.info(f"Loading configuration from {resolved}")
        with open(resolved, 'r') as f:
            return yaml.safe_load(f)

    def _resolve_config_path(self, config_path: str) -> Path:
        path = Path(config_path)
        if path.is_absolute():
            return path
        if path.exists():
            return path

        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / config_path
            if candidate.exists():
                return candidate
            if (parent / "configs").exists():
                candidate = parent / config_path
                if candidate.exists():
                    return candidate

        return path

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load training data from CSV."""
        self.logger.info(f"Loading data from {data_path}")
        if not os.path.exists(data_path):
            self.logger.warning(f"Data file not found: {data_path}. Generating synthetic data.")
            return self._generate_synthetic_data()

        self.data = pd.read_csv(data_path)
        self.logger.info(f"Loaded data shape: {self.data.shape}")
        return self.data

    def _generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic sensor data for demonstration."""
        self.logger.info(f"Generating {n_samples} synthetic data samples")
        np.random.seed(42)

        timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="1H")
        data = pd.DataFrame({
            "timestamp": timestamps,
            "sensor_id": "SENSOR_001",
            "power_consumption": np.random.normal(100, 15, n_samples) + np.linspace(0, 50, n_samples),
            "temperature": np.random.normal(60, 5, n_samples),
            "vibration": np.random.normal(0.5, 0.1, n_samples),
            "pressure": np.random.normal(50, 5, n_samples),
        })

        # Add some anomalies and failure indicators
        data.loc[800:850, "power_consumption"] = np.random.normal(150, 20, 51)  # High consumption anomaly
        data.loc[900:920, "vibration"] = np.random.normal(1.5, 0.3, 21)  # High vibration
        data["failure_flag"] = 0
        data.loc[950:, "failure_flag"] = 1  # Failure period

        # Generate failure horizon labels
        self.logger.info("Generating failure horizon labels")
        data = generate_failure_horizon_labels(
            data,
            failure_column="failure_flag",
            timestamp_column="timestamp",
            numeric=True
        )

        self.data = data
        return self.data

    def aeos_preprocessing(self) -> pd.DataFrame:
        """
        Step 1: AEOS preprocessing pipeline.
        - Establish baseline
        - Clean data
        - Normalize values
        - Handle outliers
        - Detect energy deviations
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: AEOS PREPROCESSING")
        self.logger.info("=" * 60)

        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Set timestamp as index for time series operations
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
        self.data = self.data.sort_values("timestamp").reset_index(drop=True)

        # Extract numeric columns (exclude IDs and flags)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if "failure_flag" in numeric_cols:
            numeric_cols.remove("failure_flag")

        sensor_data = self.data[numeric_cols].copy()

        # Step 1a: Establish baseline (from first 30% of data)
        baseline_size = int(len(sensor_data) * 0.3)
        baseline_data = sensor_data.iloc[:baseline_size]
        self.logger.info(f"Establishing baseline from {baseline_size} samples")
        
        # Simple baseline: mean and std of first 30%
        self.baseline = {
            col: {
                "mean": baseline_data[col].mean(),
                "std": baseline_data[col].std(),
                "min": baseline_data[col].min(),
                "max": baseline_data[col].max(),
            }
            for col in baseline_data.columns
        }
        self.logger.info("Baseline established")

        # Step 1b: Data cleaning and normalization
        self.logger.info("Cleaning and normalizing data")
        for col in numeric_cols:
            # Remove obvious outliers (>3 sigma)
            baseline = self.baseline[col]
            mean, std = baseline["mean"], baseline["std"]
            self.data = self.data[
                (self.data[col] >= mean - 3 * std) & (self.data[col] <= mean + 3 * std)
            ]

        self.logger.info(f"Data shape after cleaning: {self.data.shape}")

        # Step 1c: Normalize energy-related columns
        energy_cols = [col for col in numeric_cols if "power" in col.lower() or "energy" in col.lower()]
        for col in energy_cols:
            baseline = self.baseline[col]
            self.data[f"{col}_normalized"] = (self.data[col] - baseline["mean"]) / baseline["std"]

        self.logger.info("AEOS preprocessing completed")
        return self.data

    def feature_engineering(self, window_size: int = 24, step: int = 6) -> pd.DataFrame:
        """
        Step 2: Feature engineering pipeline.
        - Create time windows
        - Aggregate statistics
        - Engineer domain-specific features
        - Calculate deviations
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: FEATURE ENGINEERING")
        self.logger.info("=" * 60)

        if self.data is None:
            raise ValueError("Data not preprocessed. Call aeos_preprocessing() first.")
        if self.data.empty:
            raise ValueError("No data available after preprocessing.")

        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        if len(self.data) < window_size:
            self.logger.warning(
                "Window size (%s) exceeds data length (%s). Using window size %s.",
                window_size,
                len(self.data),
                len(self.data),
            )
            window_size = len(self.data)

        features_list = []

        # Get numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if "failure_flag" in numeric_cols:
            numeric_cols.remove("failure_flag")

        # Create rolling window features
        self.logger.info(f"Creating windowed features (window={window_size}, step={step})")

        for i in range(0, len(self.data) - window_size + 1, step):
            window_data = self.data.iloc[i:i + window_size]
            feature_dict = {}

            # Basic statistics per column
            for col in numeric_cols:
                feature_dict[f"{col}_mean"] = window_data[col].mean()
                feature_dict[f"{col}_std"] = window_data[col].std()
                feature_dict[f"{col}_min"] = window_data[col].min()
                feature_dict[f"{col}_max"] = window_data[col].max()
                feature_dict[f"{col}_range"] = window_data[col].max() - window_data[col].min()
                feature_dict[f"{col}_trend"] = window_data[col].iloc[-1] - window_data[col].iloc[0]

            # Energy deviation features
            if "power_consumption" in numeric_cols and self.baseline:
                power_col = "power_consumption"
                baseline = self.baseline[power_col]
                avg_power = window_data[power_col].mean()
                deviation = calculate_deviation(avg_power, baseline["mean"])
                feature_dict["power_deviation"] = deviation
                feature_dict["power_anomaly"] = 1 if abs(deviation) > 20 else 0

            # Timestamp information
            window_end = window_data.iloc[-1]["timestamp"]
            feature_dict["hour_of_day"] = window_end.hour
            feature_dict["day_of_week"] = window_end.dayofweek

            # Label (use last value of failure_flag if available)
            if "failure_flag" in self.data.columns:
                feature_dict["target_failure"] = self.data.iloc[i + window_size - 1]["failure_flag"]

            # Add failure horizon labels if available
            if "horizon_numeric" in self.data.columns:
                feature_dict["target_horizon"] = self.data.iloc[i + window_size - 1]["horizon_numeric"]

            features_list.append(feature_dict)

        self.features = pd.DataFrame(features_list)
        if self.features.empty:
            raise ValueError("Feature engineering produced no windows. Check window_size and data length.")
        self.logger.info(f"Features created: shape {self.features.shape}")
        self.logger.info(f"Features: {self.features.columns.tolist()}")

        return self.features

    def prepare_data_splits(self, test_size: float = 0.2, temporal: bool = True) -> Tuple:
        """
        Prepare train/test splits maintaining temporal order.

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: DATA SPLITTING")
        self.logger.info("=" * 60)

        if self.features is None or self.features.empty:
            raise ValueError("Features not engineered. Call feature_engineering() first.")

        # Separate features and target
        target_cols = []
        if "target_failure" in self.features.columns:
            target_cols.append("target_failure")
        if "target_horizon" in self.features.columns:
            target_cols.append("target_horizon")

        if target_cols:
            X = self.features.drop(columns=target_cols)
            # Use horizon labels if available, otherwise binary failure
            y = self.features["target_horizon"] if "target_horizon" in self.features.columns else self.features["target_failure"]
        else:
            X = self.features
            y = None

        # Split data
        if len(X) < 2:
            self.logger.warning(
                "Insufficient samples (%s) for train/test split. Using all data for training.",
                len(X),
            )
            X_train, X_test = X, X.iloc[0:0]
            if y is not None:
                y_train, y_test = y, y.iloc[0:0]
            else:
                y_train, y_test = None, None
        elif temporal:
            self.logger.info("Using temporal train/test split")
            split_idx = int(len(X) * (1 - test_size))
            if split_idx <= 0:
                split_idx = 1
            if split_idx >= len(X):
                split_idx = len(X) - 1
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            if y is not None:
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            else:
                y_train, y_test = None, None
        else:
            self.logger.info("Using stratified train/test split")
            if y is not None:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, stratify=y, random_state=42
                )
            else:
                X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
                y_train, y_test = None, None

        self.logger.info(f"Training set: {X_train.shape}")
        self.logger.info(f"Test set: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def train_failure_risk_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train failure risk classification model."""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING: FAILURE RISK MODEL")
        self.logger.info("=" * 60)

        if y_train is None or X_train.empty:
            self.logger.warning("No target labels available for failure risk model")
            return {}
        if y_train.empty:
            self.logger.warning("Empty training labels; skipping failure risk model")
            return {}

        # Initialize model
        config = self._load_config("configs/model_failure.yaml")
        model = FailureRiskModel(config)

        # For demo: use simple sklearn classifier
        from sklearn.ensemble import RandomForestClassifier

        self.logger.info("Training RandomForest classifier for failure risk")
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        self.models["failure_risk"] = clf
        self.logger.info("Failure risk model trained")

        return {"model": clf, "type": "classifier"}

    def train_rul_model(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> Dict:
        """Train RUL (Remaining Useful Life) regression model."""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING: RUL MODEL")
        self.logger.info("=" * 60)

        if X_train.empty:
            self.logger.warning("Empty training data; skipping RUL model")
            return {}

        config = self._load_config("configs/model_rul.yaml")
        model = RULModel(config)

        # For demo: generate synthetic RUL target (decreasing towards end)
        if y_train is None:
            y_train = pd.Series(
                np.linspace(100, 10, len(X_train)),
                index=X_train.index
            )

        from sklearn.ensemble import RandomForestRegressor

        self.logger.info("Training RandomForest regressor for RUL")
        reg = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        reg.fit(X_train, y_train)

        self.models["rul"] = reg
        self.logger.info("RUL model trained")

        return {"model": reg, "type": "regressor"}

    def train_anomaly_model(self, X_train: pd.DataFrame) -> Dict:
        """Train unsupervised anomaly detection model."""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING: ANOMALY DETECTION MODEL")
        self.logger.info("=" * 60)

        if X_train.empty:
            self.logger.warning("Empty training data; skipping anomaly model")
            return {}

        config = self._load_config("configs/model_anomaly.yaml")
        model = AnomalyModel(config)

        from sklearn.ensemble import IsolationForest

        self.logger.info("Training IsolationForest for anomaly detection")
        anomaly_detector = IsolationForest(
            contamination=config.get("training", {}).get("contamination", 0.05),
            random_state=42,
            n_jobs=-1
        )
        anomaly_detector.fit(X_train)

        self.models["anomaly"] = anomaly_detector
        self.logger.info("Anomaly detection model trained")

        return {"model": anomaly_detector, "type": "unsupervised"}

    def evaluate_models(self, X_test: pd.DataFrame, y_test: Optional[pd.Series] = None) -> Dict:
        """Evaluate all trained models."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 4: MODEL EVALUATION")
        self.logger.info("=" * 60)

        results = {}
        if X_test.empty:
            self.logger.warning("Empty test data; skipping evaluation")
            self.results = results
            return results

        # Evaluate failure risk model
        if "failure_risk" in self.models and y_test is not None and not y_test.empty:
            self.logger.info("Evaluating failure risk model")
            clf = self.models["failure_risk"]
            y_pred = clf.predict(X_test)
            metrics = classification_metrics(y_test, y_pred)
            results["failure_risk"] = metrics
            self.logger.info(f"Failure Risk Metrics: {metrics}")

        # Evaluate RUL model
        if "rul" in self.models:
            self.logger.info("Evaluating RUL model")
            reg = self.models["rul"]
            # Synthetic RUL target
            y_test_rul = pd.Series(
                np.linspace(100, 10, len(X_test)),
                index=X_test.index
            )
            y_pred = reg.predict(X_test)
            metrics = regression_metrics(y_test_rul, y_pred)
            results["rul"] = metrics
            self.logger.info(f"RUL Metrics: {metrics}")

        # Evaluate anomaly model
        if "anomaly" in self.models:
            self.logger.info("Evaluating anomaly detection model")
            anomaly_detector = self.models["anomaly"]
            predictions = anomaly_detector.predict(X_test)
            anomaly_score = anomaly_detector.score_samples(X_test)
            results["anomaly"] = {
                "anomalies_detected": int((predictions == -1).sum()),
                "normal_samples": int((predictions == 1).sum()),
                "anomaly_percentage": float(((predictions == -1).sum() / len(X_test)) * 100),
                "mean_anomaly_score": float(anomaly_score.mean()),
            }
            self.logger.info(f"Anomaly Detection Results: {results['anomaly']}")

        self.results = results
        return results

    def generate_explanations(self, X_train: pd.DataFrame) -> None:
        """
        Generate SHAP explanations for all models.

        Args:
            X_train: Training features for background data
        """
        self.logger.info("=" * 60)
        self.logger.info("GENERATING SHAP EXPLANATIONS")
        self.logger.info("=" * 60)

        # Generate explainer for failure risk model
        if "failure_risk" in self.models:
            self.logger.info("Creating SHAP explainer for failure risk model")
            explainer = SHAPExplainer(
                model=self.models["failure_risk"],
                model_type="tree",
                logger=self.logger
            )
            explainer.initialize_explainer(X_train, max_samples=100)
            self.explainers["failure_risk"] = explainer
            self.logger.info("Failure risk explainer created")

        # Generate explainer for RUL model
        if "rul" in self.models:
            self.logger.info("Creating SHAP explainer for RUL model")
            explainer = SHAPExplainer(
                model=self.models["rul"],
                model_type="tree",
                logger=self.logger
            )
            explainer.initialize_explainer(X_train, max_samples=100)
            self.explainers["rul"] = explainer
            self.logger.info("RUL explainer created")

        self.logger.info("SHAP explainers generated successfully")

    def save_models(self, output_dir: str = "models/") -> None:
        """Save trained models to disk."""
        self.logger.info("=" * 60)
        self.logger.info("SAVING MODELS")
        self.logger.info("=" * 60)

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for model_name, model in self.models.items():
            model_path = os.path.join(output_dir, f"{model_name}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            self.logger.info(f"Saved {model_name} model to {model_path}")

        # Save explainers
        for explainer_name, explainer in self.explainers.items():
            explainer_path = os.path.join(output_dir, f"{explainer_name}_explainer.pkl")
            explainer.save(explainer_path)
            self.logger.info(f"Saved {explainer_name} explainer to {explainer_path}")

        # Save baseline
        baseline_path = os.path.join(output_dir, "baseline.pkl")
        with open(baseline_path, 'wb') as f:
            pickle.dump(self.baseline, f)
        self.logger.info(f"Saved baseline to {baseline_path}")

    def save_results(self, output_dir: str = "models/") -> None:
        """Save evaluation results and pipeline summary."""
        self.logger.info("Saving results summary")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "data_shape": self.data.shape if self.data is not None else None,
            "features_shape": self.features.shape if self.features is not None else None,
            "baseline": self.baseline,
            "evaluation_results": self.results,
        }

        results_path = os.path.join(output_dir, "training_summary.yaml")
        with open(results_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)

        self.logger.info(f"Saved training summary to {results_path}")

    def run(self, data_path: str = None, window_size: int = 24, output_dir: str = "models/") -> Dict:
        """
        Execute complete training pipeline.

        Args:
            data_path: Path to training data CSV (optional, uses synthetic if not provided)
            window_size: Time window size for feature engineering
            output_dir: Directory to save models and results

        Returns:
            Dictionary with pipeline results
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING ML TRAINING PIPELINE")
        self.logger.info("=" * 60)
        self.logger.info(f"Timestamp: {datetime.now()}")

        try:
            # 1. Load data
            self.load_data(data_path or "data/processed/training_data.csv")

            # 2. AEOS preprocessing
            self.aeos_preprocessing()

            # 3. Feature engineering
            self.feature_engineering(window_size=window_size)

            # 4. Prepare data splits
            X_train, X_test, y_train, y_test = self.prepare_data_splits()

            # 5. Train models
            self.train_failure_risk_model(X_train, y_train)
            self.train_rul_model(X_train, y_train)
            self.train_anomaly_model(X_train)

            # 6. Evaluate models
            self.evaluate_models(X_test, y_test)

            # 7. Generate SHAP explanations
            self.generate_explanations(X_train)

            # 8. Save artifacts
            self.save_models(output_dir)
            self.save_results(output_dir)

            self.logger.info("=" * 60)
            self.logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)

            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "data_shape": self.data.shape,
                "features_shape": self.features.shape,
                "models_trained": list(self.models.keys()),
                "results": self.results,
            }

        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
            }


def main():
    """Main entry point for training pipeline."""
    # Initialize pipeline
    pipeline = TrainingPipeline(config_path="configs/base.yaml")

    # Run pipeline
    results = pipeline.run(
        data_path=None,  # Use synthetic data if file not found
        window_size=24,
        output_dir="models/"
    )

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Status: {results['status']}")
    if results['status'] == 'success':
        print(f"Data shape: {results['data_shape']}")
        print(f"Features shape: {results['features_shape']}")
        print(f"Models trained: {', '.join(results['models_trained'])}")
        print("\nEvaluation Results:")
        for model_name, metrics in results['results'].items():
            print(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")


if __name__ == "__main__":
    main()
