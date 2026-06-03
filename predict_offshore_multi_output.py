"""
Prediction script for multi-output model.

Demonstrates inference with the trained model.

Usage:
    python predict_offshore_multi_output.py
"""

import sys
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from feature_pipeline import engineer_offshore_features


def ensure_probability_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all expected probability columns exist.
    Missing ones are filled with 0.0 (safe fallback).
    """
    expected_proba_cols = [
        "horizon_proba_normal",
        "horizon_proba_caution",
        "horizon_proba_warning",
        "horizon_proba_critical",
    ]

    predictions = predictions.copy()

    for col in expected_proba_cols:
        if col not in predictions.columns:
            predictions[col] = 0.0

    return predictions


def main():
    """Run predictions on test data."""
    print("=" * 80)
    print("TerraEnergy AI Multi-Output Model - Prediction Demo")
    print("=" * 80)

    # Load model
    model_path = Path("models/multi_output_model.pkl")
    feature_path = Path("models/feature_names.pkl")

    if not model_path.exists():
        print("❌ Model not found. Please run training first:")
        print("   python train_offshore_multi_output.py")
        return

    print("\n📦 Loading model...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(feature_path, "rb") as f:
        feature_names = pickle.load(f)

    print(f"✓ Model loaded: {model.get_config()['n_features']} features")

    # Load test data
    print("\n📊 Loading test data...")
    data_paths = [
        "critical_offshore_sample.csv",
        "data/uploads/20260122_114109_critical_offshore_rig_dataset_100.csv"
    ]

    df = None
    for path in data_paths:
        if Path(path).exists():
            df = pd.read_csv(path)
            print(f"✓ Loaded: {path}")
            break

    if df is None:
        print("❌ Test data not found")
        return

    # Engineer features
    print("\n⚙️  Engineering features...")
    df_features, _ = engineer_offshore_features(df, window_sizes=[3, 6, 12])

    # Select only model features
    X = df_features[feature_names]

    # Make predictions
    print("\n🔮 Making predictions...")
    predictions = model.predict_with_labels(X)

    # ✅ SAFETY FIX: ensure all probability columns exist
    predictions = ensure_probability_columns(predictions)

    # Combine with original data
    results = pd.concat([
        df[["timestamp", "rig_id", "equipment", "estimated_rul_hours", "failure_flag"]],
        predictions
    ], axis=1)

    # Display results
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)

    # Summary statistics
    print(f"\n📈 Prediction Summary:")
    print(f"  Total predictions: {len(results)}")

    print(f"\n  Horizon Distribution:")
    unique_labels = results["horizon_label"].dropna().unique()

    for label in ["Normal", "Caution", "Warning", "Critical"]:
        count = (results["horizon_label"] == label).sum()
        pct = (count / len(results)) * 100 if len(results) > 0 else 0
        print(f"    - {label}: {count} ({pct:.1f}%)")

    print(f"\n  RUL Statistics:")
    print(f"    - Mean: {results['rul_hours'].mean():.1f} hours ({results['rul_days'].mean():.1f} days)")
    print(f"    - Min: {results['rul_hours'].min():.1f} hours")
    print(f"    - Max: {results['rul_hours'].max():.1f} hours")

    # Show critical predictions
    critical = results[results["horizon_label"] == "Critical"]
    if len(critical) > 0:
        print(f"\n⚠️  CRITICAL ALERTS ({len(critical)} samples):")
        print(critical[["timestamp", "rig_id", "horizon_label", "rul_hours", "rul_days"]]
              .head(10)
              .to_string(index=False))

    # Show warning predictions
    warning = results[results["horizon_label"] == "Warning"]
    if len(warning) > 0:
        print(f"\n🟠 WARNING ALERTS ({len(warning)} samples):")
        print(warning[["timestamp", "rig_id", "horizon_label", "rul_hours", "rul_days"]]
              .head(5)
              .to_string(index=False))

    # Compare with actual RUL (if available)
    if "estimated_rul_hours" in results.columns:
        print(f"\n📊 Prediction Accuracy (vs. estimated RUL):")
        mae = np.abs(results["rul_hours"] - results["estimated_rul_hours"]).mean()
        rmse = np.sqrt(((results["rul_hours"] - results["estimated_rul_hours"]) ** 2).mean())
        print(f"  - MAE: {mae:.1f} hours ({mae/24:.1f} days)")
        print(f"  - RMSE: {rmse:.1f} hours ({rmse/24:.1f} days)")

    # Save results
    output_path = Path("predictions_multi_output.csv")
    results.to_csv(output_path, index=False)
    print(f"\n💾 Predictions saved to: {output_path}")

    # Show sample predictions with probabilities
    print(f"\n🎯 Sample Predictions (with probabilities):")
    sample_indices = [0, len(results)//2, -1]
    sample = results.iloc[sample_indices]

    for _, row in sample.iterrows():
        print(f"\n  Timestamp: {row['timestamp']}")
        print(f"  Prediction: {row['horizon_label']} (RUL: {row['rul_hours']:.1f}h / {row['rul_days']:.1f}d)")
        print(f"  Probabilities:")
        print(f"    - Normal:   {row['horizon_proba_normal']:.1%}")
        print(f"    - Caution:  {row['horizon_proba_caution']:.1%}")
        print(f"    - Warning:  {row['horizon_proba_warning']:.1%}")
        print(f"    - Critical: {row['horizon_proba_critical']:.1%}")

    print("\n" + "=" * 80)
    print("✅ Prediction complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()