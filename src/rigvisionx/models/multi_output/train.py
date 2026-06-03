"""
Training pipeline for multi-output model with failure horizon labeling.
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from rigvisionx.explainability import generate_failure_horizon_labels
from rigvisionx.models.multi_output.model import MultiOutputModel


def prepare_failure_horizon_data(
    df: pd.DataFrame,
    failure_column: str = "failure_flag",
    rul_column: str = "estimated_rul_hours",
    timestamp_column: str = "timestamp",
    horizons: Optional[Dict[str, Tuple[float, float]]] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Prepare data with failure horizon labels.

    Args:
        df: Input dataframe
        failure_column: Column with binary failure flags
        rul_column: Column with RUL estimates (if available)
        timestamp_column: Timestamp column
        horizons: Custom horizon definitions
        logger: Optional logger

    Returns:
        DataFrame with horizon labels added
    """
    logger = logger or logging.getLogger(__name__)

    # Default horizons: Critical (0-24h), Warning (24-72h), Caution (72-168h), Normal (>168h)
    if horizons is None:
        horizons = {
            "Critical": (0, 24),      # 3
            "Warning": (24, 72),      # 2
            "Caution": (72, 168),     # 1
            "Normal": (168, np.inf)   # 0
        }

    logger.info("Preparing failure horizon labels...")

    # Check if we have RUL column already
    if rul_column in df.columns:
        logger.info(f"Using existing RUL column: {rul_column}")
        df["time_to_failure"] = df[rul_column]
    else:
        # Generate TTF from failure flags
        logger.info("Generating time-to-failure from failure flags...")
        df = generate_failure_horizon_labels(
            df,
            failure_column=failure_column,
            timestamp_column=timestamp_column,
            numeric=False
        )

    # Create numeric horizon labels based on RUL
    df["horizon_numeric"] = 0  # Default: Normal

    # Map based on time_to_failure
    rul_values = df["time_to_failure"]

    # Critical: 0-24h → 3
    df.loc[rul_values <= 24, "horizon_numeric"] = 3

    # Warning: 24-72h → 2
    df.loc[(rul_values > 24) & (rul_values <= 72), "horizon_numeric"] = 2

    # Caution: 72-168h → 1
    df.loc[(rul_values > 72) & (rul_values <= 168), "horizon_numeric"] = 1

    # Normal: >168h → 0 (already default)

    # Create text labels
    horizon_map = {0: "Normal", 1: "Caution", 2: "Warning", 3: "Critical"}
    df["horizon_label"] = df["horizon_numeric"].map(horizon_map)

    # Log distribution
    logger.info("Horizon distribution:")
    for label, count in df["horizon_numeric"].value_counts().sort_index().items():
        pct = (count / len(df)) * 100
        logger.info(f"  {horizon_map[label]}: {count} samples ({pct:.1f}%)")

    return df


def train_multi_output_model(
    df: pd.DataFrame,
    feature_columns: list,
    target_horizon_col: str = "horizon_numeric",
    target_rul_col: str = "time_to_failure",
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: Optional[int] = 15,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None
) -> Tuple[MultiOutputModel, Dict]:
    """
    Train multi-output model for failure prediction.

    Args:
        df: Dataframe with features and targets
        feature_columns: List of feature column names
        target_horizon_col: Column with horizon labels (0-3)
        target_rul_col: Column with RUL values (hours)
        test_size: Fraction for test set
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed
        logger: Optional logger

    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    logger = logger or logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("TRAINING MULTI-OUTPUT MODEL")
    logger.info("=" * 60)

    # Extract features and targets
    X = df[feature_columns].copy()
    y_horizon = df[target_horizon_col].copy()
    y_rul = df[target_rul_col].copy()

    logger.info(f"Dataset: {len(X)} samples, {len(feature_columns)} features")
    logger.info(f"Target horizon distribution: {y_horizon.value_counts().sort_index().to_dict()}")
    logger.info(f"Target RUL range: {y_rul.min():.1f}h - {y_rul.max():.1f}h")

    # Check for class imbalance
    class_counts = y_horizon.value_counts()
    if len(class_counts) < 4:
        logger.warning(f"⚠ Only {len(class_counts)} horizon classes present in data")
        logger.warning("  Dataset may be too small or imbalanced for robust training")

    # Split data
    if len(X) < 10:
        logger.warning("⚠ Very small dataset. Using all data for training (no test set)")
        X_train, X_test = X, X.iloc[0:0]
        y_horizon_train, y_horizon_test = y_horizon, y_horizon.iloc[0:0]
        y_rul_train, y_rul_test = y_rul, y_rul.iloc[0:0]
    else:
        logger.info(f"Splitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")

        # Use stratified split if possible
        try:
            X_train, X_test, y_horizon_train, y_horizon_test, y_rul_train, y_rul_test = train_test_split(
                X, y_horizon, y_rul,
                test_size=test_size,
                stratify=y_horizon,
                random_state=random_state
            )
            logger.info("Using stratified split")
        except ValueError:
            # Fallback to non-stratified if classes too imbalanced
            logger.warning("Cannot use stratified split (class imbalance). Using random split.")
            X_train, X_test, y_horizon_train, y_horizon_test, y_rul_train, y_rul_test = train_test_split(
                X, y_horizon, y_rul,
                test_size=test_size,
                random_state=random_state
            )

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Initialize and train model
    model = MultiOutputModel(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        logger=logger
    )

    logger.info("Training model...")
    model.fit(X_train, y_horizon_train, y_rul_train)

    # Evaluate on test set
    results = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": len(feature_columns),
        "feature_names": feature_columns
    }

    if len(X_test) > 0:
        logger.info("Evaluating on test set...")
        metrics = model.evaluate(X_test, y_horizon_test, y_rul_test)
        results["metrics"] = metrics

        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Classification Accuracy: {metrics['classification']['accuracy']:.3f}")
        logger.info(f"Classification F1-Score: {metrics['classification']['f1_score']:.3f}")
        logger.info(f"RUL MAE: {metrics['regression']['mae_hours']:.1f} hours")
        logger.info(f"RUL RMSE: {metrics['regression']['rmse_hours']:.1f} hours")
        logger.info(f"RUL R²: {metrics['regression']['r2_score']:.3f}")
    else:
        logger.info("No test set available for evaluation")
        results["metrics"] = None

    # Feature importance
    logger.info("Computing feature importance...")
    importance = model.feature_importance(top_n=10)
    results["feature_importance"] = importance

    logger.info("\nTop 5 Features (Horizon Classification):")
    for i, feat in enumerate(importance["horizon_classifier"][:5], 1):
        logger.info(f"  {i}. {feat['feature']}: {feat['importance']:.4f}")

    logger.info("\nTop 5 Features (RUL Regression):")
    for i, feat in enumerate(importance["rul_regressor"][:5], 1):
        logger.info(f"  {i}. {feat['feature']}: {feat['importance']:.4f}")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    return model, results
