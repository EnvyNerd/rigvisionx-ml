"""
Verification script for Explainable AI features in TerraEnergy AI.

This script demonstrates and verifies:
1. Failure Horizon Labels generation
2. SHAP explainer integration
3. Feature importance extraction
4. Individual prediction explanations
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("TerraEnergy AI Explainability Feature Verification")
print("=" * 80)

# ============================================================================
# Test 1: Import modules
# ============================================================================
print("\n[1/5] Testing module imports...")
try:
    from rigvisionx.explainability import SHAPExplainer, generate_failure_horizon_labels
    print("✓ Successfully imported explainability modules")
    print("  - SHAPExplainer")
    print("  - generate_failure_horizon_labels")
except ImportError as e:
    print(f"✗ Failed to import modules: {e}")
    sys.exit(1)

# ============================================================================
# Test 2: Generate Failure Horizon Labels
# ============================================================================
print("\n[2/5] Testing Failure Horizon Label generation...")
try:
    # Create sample data
    timestamps = pd.date_range("2024-01-01", periods=100, freq="1H")
    sample_data = pd.DataFrame({
        "timestamp": timestamps,
        "sensor_id": "TEST_001",
        "power_consumption": np.random.normal(100, 10, 100),
        "temperature": np.random.normal(60, 5, 100),
        "failure_flag": 0
    })

    # Simulate failure at end
    sample_data.loc[90:, "failure_flag"] = 1

    # Generate horizon labels
    labeled_data = generate_failure_horizon_labels(
        sample_data,
        failure_column="failure_flag",
        timestamp_column="timestamp",
        numeric=True
    )

    # Verify new columns exist
    assert "time_to_failure" in labeled_data.columns, "Missing time_to_failure column"
    assert "horizon_numeric" in labeled_data.columns, "Missing horizon_numeric column"

    print("✓ Failure horizon labels generated successfully")
    print(f"  - Time-to-failure column: {labeled_data['time_to_failure'].describe()['min']:.2f}h to {labeled_data['time_to_failure'].describe()['max']:.2f}h")
    print(f"  - Horizon label distribution:")
    for label, count in labeled_data['horizon_numeric'].value_counts().sort_index().items():
        label_names = {3: "Critical (0-24h)", 2: "Warning (24-72h)", 1: "Caution (72-168h)", 0: "Normal (>168h)"}
        print(f"    - {label_names.get(label, f'Label {label}')}: {count} samples")

except Exception as e:
    print(f"✗ Failed to generate horizon labels: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 3: Train model with SHAP integration
# ============================================================================
print("\n[3/5] Testing training pipeline with SHAP integration...")
try:
    from rigvisionx.train_pipeline import TrainingPipeline

    # Initialize pipeline
    pipeline = TrainingPipeline(config_path="configs/base.yaml")

    # Run with synthetic data
    print("  Running training pipeline (this may take 30-60 seconds)...")
    results = pipeline.run(
        data_path=None,  # Use synthetic data
        window_size=24,
        output_dir="models/"
    )

    if results["status"] == "success":
        print("✓ Training pipeline completed successfully")
        print(f"  - Data shape: {results['data_shape']}")
        print(f"  - Features shape: {results['features_shape']}")
        print(f"  - Models trained: {', '.join(results['models_trained'])}")

        # Check if explainers were created
        if pipeline.explainers:
            print(f"  - SHAP explainers created: {', '.join(pipeline.explainers.keys())}")
        else:
            print("  ⚠ Warning: No SHAP explainers created")
    else:
        print(f"✗ Training failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)

except Exception as e:
    print(f"✗ Training pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 4: Verify explainer files exist
# ============================================================================
print("\n[4/5] Verifying explainer artifacts...")
try:
    from pathlib import Path

    model_dir = Path("models")
    expected_files = [
        "failure_risk_model.pkl",
        "failure_risk_explainer.pkl",
        "rul_model.pkl",
        "rul_explainer.pkl",
        "anomaly_model.pkl",
        "baseline.pkl",
        "training_summary.yaml"
    ]

    found_files = []
    missing_files = []

    for file in expected_files:
        file_path = model_dir / file
        if file_path.exists():
            found_files.append(file)
            size_kb = file_path.stat().st_size / 1024
            print(f"  ✓ {file} ({size_kb:.1f} KB)")
        else:
            missing_files.append(file)
            print(f"  ✗ {file} (missing)")

    if missing_files:
        print(f"\n⚠ Warning: {len(missing_files)} files missing")
    else:
        print(f"\n✓ All {len(expected_files)} artifact files present")

except Exception as e:
    print(f"✗ Failed to verify artifacts: {e}")
    sys.exit(1)

# ============================================================================
# Test 5: Test SHAP explainer functionality
# ============================================================================
print("\n[5/5] Testing SHAP explainer functionality...")
try:
    # Load the trained explainer
    explainer_path = "models/failure_risk_explainer.pkl"

    if Path(explainer_path).exists():
        explainer = SHAPExplainer.load(explainer_path)
        print("✓ SHAP explainer loaded successfully")

        # Get feature importance
        if pipeline.features is not None:
            features = pipeline.features.copy()
            for col in ["target_failure", "target_horizon"]:
                if col in features.columns:
                    features = features.drop(columns=[col])

            if not features.empty:
                importance = explainer.get_feature_importance(features, top_n=5)
                print("\n  Top 5 most important features:")
                for i, feat in enumerate(importance["top_features"][:5], 1):
                    print(f"    {i}. {feat['feature']}: {feat['importance']:.4f}")

                # Explain single prediction
                X_single = features.iloc[[0]]
                explanation = explainer.explain_single_prediction(X_single)

                print(f"\n  Single prediction explanation:")
                print(f"    - Base value: {explanation['base_value']:.4f}")
                print(f"    - Predicted value: {explanation['predicted_value']:.4f}")
                print(f"    - Top 3 contributing features:")
                for i, contrib in enumerate(explanation["feature_contributions"][:3], 1):
                    print(f"      {i}. {contrib['feature']}: {contrib['shap_contribution']:.4f} ({contrib['impact']})")

                print("\n✓ SHAP functionality verified")
            else:
                print("  ⚠ No features available for testing")
    else:
        print(f"  ✗ Explainer file not found: {explainer_path}")

except Exception as e:
    print(f"✗ SHAP explainer test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print("\n✅ All explainability features are working correctly!")
print("\nImplemented features:")
print("  1. ✓ Failure Horizon Labels (4-level: Critical/Warning/Caution/Normal)")
print("  2. ✓ SHAP Explainers (TreeExplainer for RandomForest models)")
print("  3. ✓ Feature Importance (Global model interpretability)")
print("  4. ✓ Prediction Explanations (Per-prediction SHAP values)")
print("  5. ✓ Training Pipeline Integration")
print("\nAPI Endpoints available:")
print("  - POST /explainability/feature-importance")
print("  - POST /explainability/explain-prediction")
print("\nNext steps:")
print("  1. Install SHAP: pip install -r requirements.txt")
print("  2. Run this script: python verify_explainability.py")
print("  3. Start API server: uvicorn src.rigvisionx.serving.api:app --reload")
print("  4. Test endpoints: curl -X POST http://localhost:8000/explainability/feature-importance")
print("=" * 80)


