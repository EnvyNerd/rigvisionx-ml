# TerraEnergy AI Explainability Features Guide

## Overview

TerraEnergy AI now includes **Explainable AI (SHAP)** and **Failure Horizon Labels** for interpretable predictive maintenance.

## Features Implemented

### 1. Failure Horizon Labels
Multi-class failure prediction with time-based horizons:

| Horizon | Time Range | Risk Level | Action |
|---------|-----------|------------|---------|
| **Critical** | 0-24 hours | 🔴 High | Immediate maintenance required |
| **Warning** | 24-72 hours | 🟠 Medium | Schedule maintenance soon |
| **Caution** | 72-168 hours | 🟡 Low | Monitor closely |
| **Normal** | >168 hours | 🟢 Safe | Routine monitoring |

### 2. SHAP Explanations
- **Global Feature Importance**: Which sensors/features matter most?
- **Individual Predictions**: Why did the model predict this specific failure?
- **Feature Contributions**: How much does each sensor reading contribute?

## Quick Verification

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Verification Script
```bash
python verify_explainability.py
```

**Expected Output:**
```
[1/5] Testing module imports...
✓ Successfully imported explainability modules

[2/5] Testing Failure Horizon Label generation...
✓ Failure horizon labels generated successfully
  - Horizon label distribution:
    - Critical (0-24h): 10 samples
    - Warning (24-72h): 20 samples
    ...

[3/5] Testing training pipeline with SHAP integration...
✓ Training pipeline completed successfully
  - SHAP explainers created: failure_risk, rul

[4/5] Verifying explainer artifacts...
  ✓ failure_risk_explainer.pkl (234.5 KB)
  ✓ rul_explainer.pkl (189.2 KB)

[5/5] Testing SHAP explainer functionality...
✓ SHAP functionality verified
  Top 5 most important features:
    1. power_consumption_mean: 0.1234
    2. vibration_std: 0.0987
    ...
```

### Step 3: Test Training Pipeline
```bash
# Using Python module
python -m rigvisionx.train_pipeline

# Or direct script
python src/rigvisionx/train_pipeline.py
```

**Check for these logs:**
```
===========================================
GENERATING SHAP EXPLANATIONS
===========================================
Creating SHAP explainer for failure risk model
TreeExplainer initialized
Failure risk explainer created
```

### Step 4: Test API Endpoints

#### Start the API Server
```bash
uvicorn src.rigvisionx.serving.api:app --reload --host 0.0.0.0 --port 8000
```

#### Test Feature Importance
```bash
# Using curl
curl -X POST "http://localhost:8000/explainability/feature-importance" \
  -F "model_type=failure_risk" \
  -F "top_n=10"

# Using Python test script
python test_api_explainability.py
```

**Expected Response:**
```json
{
  "status": "success",
  "model_type": "failure_risk",
  "top_features": [
    {
      "feature": "power_consumption_mean",
      "importance": 0.1234
    },
    {
      "feature": "vibration_std",
      "importance": 0.0987
    }
  ]
}
```

#### Test Prediction Explanation
```bash
curl -X POST "http://localhost:8000/explainability/explain-prediction" \
  -F "model_type=failure_risk" \
  -F "window_size=24" \
  -F "step=6" \
  -F "prediction_index=0"
```

**Expected Response:**
```json
{
  "status": "success",
  "model_type": "failure_risk",
  "explanation": {
    "base_value": 0.3456,
    "predicted_value": 0.7890,
    "feature_contributions": [
      {
        "feature": "power_consumption_mean",
        "value": 125.5,
        "shap_contribution": 0.1234,
        "impact": "increases"
      }
    ]
  }
}
```

## File Structure Verification

Check that these files exist after training:

```
rigvision-ml/
├── models/
│   ├── failure_risk_model.pkl          ✓ Trained model
│   ├── failure_risk_explainer.pkl      ✓ SHAP explainer (NEW)
│   ├── rul_model.pkl                   ✓ Trained model
│   ├── rul_explainer.pkl               ✓ SHAP explainer (NEW)
│   ├── anomaly_model.pkl               ✓ Trained model
│   ├── baseline.pkl                    ✓ AEOS baseline
│   └── training_summary.yaml           ✓ Training metrics
├── src/rigvisionx/explainability/      ✓ NEW MODULE
│   ├── __init__.py
│   ├── shap_explainer.py               ✓ SHAP integration
│   └── failure_horizon.py              ✓ Horizon labels
└── requirements.txt                    ✓ Updated with shap>=0.45
```

## Code Verification

### Check 1: Import Test
```python
# This should work without errors
from rigvisionx.explainability import SHAPExplainer, generate_failure_horizon_labels
print("✓ Explainability modules available")
```

### Check 2: Failure Horizon Labels
```python
import pandas as pd
import numpy as np
from rigvisionx.explainability import generate_failure_horizon_labels

# Create test data
df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=100, freq="1H"),
    "sensor_id": "TEST_001",
    "power": np.random.normal(100, 10, 100),
    "failure_flag": 0
})
df.loc[90:, "failure_flag"] = 1

# Generate labels
result = generate_failure_horizon_labels(df, failure_column="failure_flag")

# Verify columns
assert "time_to_failure" in result.columns
assert "failure_horizon" in result.columns
print("✓ Failure horizon labels working")
```

### Check 3: SHAP Explainer
```python
from rigvisionx.explainability import SHAPExplainer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Train simple model
X = pd.DataFrame(np.random.randn(100, 5), columns=[f"feat_{i}" for i in range(5)])
y = np.random.randint(0, 2, 100)
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Create explainer
explainer = SHAPExplainer(model, model_type="tree")
explainer.initialize_explainer(X, max_samples=50)

# Get feature importance
importance = explainer.get_feature_importance(X, top_n=5)
print("✓ SHAP explainer working")
print(f"  Top feature: {importance['top_features'][0]['feature']}")
```

## API Documentation

Once the server is running, access:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Look for these new endpoints:
- `POST /explainability/feature-importance`
- `POST /explainability/explain-prediction`

## Common Issues & Solutions

### Issue 1: "Module 'shap' not found"
```bash
# Solution: Install SHAP
pip install shap>=0.45
```

### Issue 2: "Explainer not found for model"
```bash
# Solution: Train models first
python src/rigvisionx/train_pipeline.py
```

### Issue 3: Slow SHAP computation
- **Expected**: First SHAP computation takes 10-30 seconds
- **Solution**: Background samples are limited to 100 for efficiency
- **Note**: Subsequent predictions are faster (cached)

### Issue 4: API returns 404
```bash
# Check if models exist
ls models/failure_risk_explainer.pkl

# Retrain if missing
python -m rigvisionx.train_pipeline
```

## Integration with Existing System

The explainability features integrate seamlessly:

1. **Training**: Automatically creates SHAP explainers during `pipeline.run()`
2. **Inference**: Load explainer with model for predictions
3. **API**: New endpoints available alongside existing ones
4. **No Breaking Changes**: Existing code continues to work

## Use Cases

### Use Case 1: Feature Engineering Validation
```python
# Which sensors actually matter?
importance = explainer.get_feature_importance(features, top_n=20)
# Focus instrumentation on top 10 features
```

### Use Case 2: Root Cause Analysis
```python
# Why did this specific prediction occur?
explanation = explainer.explain_single_prediction(X_sample)
# "High vibration_std (value: 1.5) increased failure risk by +0.23"
```

### Use Case 3: Maintenance Scheduling
```python
# Horizon labels enable graduated response
Critical (0-24h) → Emergency shutdown
Warning (24-72h) → Schedule next maintenance window
Caution (72-168h) → Monitor closely
Normal (>168h) → Routine checks
```

## Performance Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| SHAP initialization | 5-10s | One-time per model load |
| Feature importance (1000 samples) | 3-5s | Depends on model complexity |
| Single prediction explanation | 0.1-0.5s | Very fast |
| Training with SHAP | +20-30s | Automatic during pipeline |

## Next Steps

1. ✅ Verify features work: `python verify_explainability.py`
2. ✅ Train models: `python src/rigvisionx/train_pipeline.py`
3. ✅ Start API: `uvicorn src.rigvisionx.serving.api:app --reload`
4. ✅ Test endpoints: `python test_api_explainability.py`
5. 📊 Visualize results in dashboard (integrate SHAP plots)
6. 📄 Generate interpretable reports for stakeholders

## Support

If features aren't working:
1. Check this guide's verification steps
2. Review log files for errors
3. Ensure all dependencies installed: `pip install -r requirements.txt`
4. Verify Python 3.9+ installed

---

**Status**: ✅ Fully Implemented
**Version**: TerraEnergy AI v2.0 (Interpretable Predictive Maintenance)
**Last Updated**: 2026-01-29


