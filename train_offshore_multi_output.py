"""
Complete training script for multi-output model on offshore rig dataset.

Usage:
    python train_offshore_multi_output.py

Trains a model that predicts:
1. Failure Horizon (4-class): Normal/Caution/Warning/Critical
2. RUL (regression): Remaining useful life in hours

Works with: critical_offshore_rig_dataset_100.csv
"""

import sys
import pickle
from pathlib import Path
import pandas as pd
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rigvisionx.models.multi_output.train import (
    prepare_failure_horizon_data,
    train_multi_output_model
)
from feature_pipeline import engineer_offshore_features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("TerraEnergy AI Multi-Output Model Training")
    print("Offshore Rig Predictive Maintenance")
    print("=" * 80)

    # ========================================================================
    # 1. Load Data
    # ========================================================================
    logger.info("Step 1: Loading offshore rig dataset...")

    # Try multiple possible locations
    data_paths = [
        "critical_offshore_sample.csv",
        "data/uploads/20260122_114109_critical_offshore_rig_dataset_100.csv",
        "data/raw/critical_offshore_rig_dataset_100.csv"
    ]

    df = None
    for path in data_paths:
        if Path(path).exists():
            logger.info(f"Loading data from: {path}")
            df = pd.read_csv(path)
            break

    if df is None:
        logger.error(" Dataset not found. Tried:")
        for path in data_paths:
            logger.error(f"  - {path}")
        logger.error("\nPlease ensure critical_offshore_rig_dataset_100.csv is available")
        return

    logger.info(f" Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Check for failures
    failure_count = df["failure_flag"].sum()
    logger.info(f"  Failures in dataset: {failure_count}")

    if failure_count < 2:
        logger.warning(" WARNING: Only 1 failure event in dataset!")
        logger.warning("  Model will train but classification may not be robust.")
        logger.warning("  For production, collect more failure data.")

    # ========================================================================
    # 2. Feature Engineering
    # ========================================================================
    logger.info("\nStep 2: Feature engineering...")

    df_with_features, feature_names = engineer_offshore_features(
        df,
        window_sizes=[3, 6, 12],
        logger=logger
    )

    logger.info(f" Features engineered: {len(feature_names)} features")
    logger.info(f"  Total columns: {df_with_features.shape[1]}")

    # ========================================================================
    # 3. Prepare Failure Horizon Labels
    # ========================================================================
    logger.info("\nStep 3: Preparing failure horizon labels...")

    df_labeled = prepare_failure_horizon_data(
        df_with_features,
        failure_column="failure_flag",
        rul_column="estimated_rul_hours",  # Use existing RUL from dataset
        timestamp_column="timestamp",
        logger=logger
    )

    logger.info(f" Horizon labels prepared")

    # ========================================================================
    # 4. Train Multi-Output Model
    # ========================================================================
    logger.info("\nStep 4: Training multi-output model...")

    model, results = train_multi_output_model(
        df=df_labeled,
        feature_columns=feature_names,
        target_horizon_col="horizon_numeric",
        target_rul_col="time_to_failure",
        test_size=0.2,
        n_estimators=100,
        max_depth=15,
        random_state=42,
        logger=logger
    )

    # ========================================================================
    # 5. Save Model and Results
    # ========================================================================
    logger.info("\nStep 5: Saving model and results...")

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    # Save model
    model_path = output_dir / "multi_output_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f" Model saved: {model_path}")

    # Save feature names
    feature_path = output_dir / "feature_names.pkl"
    with open(feature_path, "wb") as f:
        pickle.dump(feature_names, f)
    logger.info(f" Feature names saved: {feature_path}")

    # Save results
    results_path = output_dir / "multi_output_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f" Results saved: {results_path}")

    # ========================================================================
    # 6. Display Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 80)

    print("\nDataset:")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Training samples: {results['train_samples']}")
    print(f"  - Test samples: {results['test_samples']}")
    print(f"  - Features: {results['n_features']}")

    if results.get("metrics"):
        metrics = results["metrics"]
        print("\nClassification Performance (Failure Horizon):")
        print(f"  - Accuracy: {metrics['classification']['accuracy']:.1%}")
        print(f"  - Precision: {metrics['classification']['precision']:.1%}")
        print(f"  - Recall: {metrics['classification']['recall']:.1%}")
        print(f"  - F1-Score: {metrics['classification']['f1_score']:.3f}")

        print("\nRegression Performance (RUL):")
        print(f"  - MAE: {metrics['regression']['mae_hours']:.1f} hours ({metrics['regression']['mae_hours']/24:.1f} days)")
        print(f"  - RMSE: {metrics['regression']['rmse_hours']:.1f} hours")
        print(f"  - R2 Score: {metrics['regression']['r2_score']:.3f}")
        print(f"  - MAPE: {metrics['regression']['mape_percent']:.1f}%")

    print("\nTop 5 Important Features (Classification):")
    for i, feat in enumerate(results["feature_importance"]["horizon_classifier"][:5], 1):
        print(f"  {i}. {feat['feature']}: {feat['importance']:.4f}")

    print("\nTop 5 Important Features (Regression):")
    for i, feat in enumerate(results["feature_importance"]["rul_regressor"][:5], 1):
        print(f"  {i}. {feat['feature']}: {feat['importance']:.4f}")

    print("\nSaved Files:")
    print(f"  - {model_path}")
    print(f"  - {feature_path}")
    print(f"  - {results_path}")

    print("\nMulti-output model training successful!")
    print(f"\nNext steps:")
    print(f"  1. Run predictions: python predict_offshore_multi_output.py")
    print(f"  2. Integrate with API: see API integration guide")
    print(f"  3. Deploy to production")

    print("=" * 80)


if __name__ == "__main__":
    main()

