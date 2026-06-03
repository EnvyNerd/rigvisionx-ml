"""
API endpoint testing script for explainability features.

Run this after starting the API server with:
    uvicorn src.rigvisionx.serving.api:app --reload
"""

import requests
import json
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

print("=" * 80)
print("TerraEnergy AI Explainability API Testing")
print("=" * 80)

# ============================================================================
# Test 1: Health check
# ============================================================================
print("\n[1/3] Testing API health...")
try:
    response = requests.get(f"{API_BASE_URL}/health", timeout=5)
    if response.status_code == 200:
        print("✓ API is running")
        print(f"  Response: {response.json()}")
    else:
        print(f"✗ API returned status {response.status_code}")
except requests.exceptions.ConnectionError:
    print("✗ Cannot connect to API. Make sure server is running:")
    print("  uvicorn src.rigvisionx.serving.api:app --reload")
    exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# ============================================================================
# Test 2: Feature Importance Endpoint
# ============================================================================
print("\n[2/3] Testing /explainability/feature-importance endpoint...")
try:
    response = requests.post(
        f"{API_BASE_URL}/explainability/feature-importance",
        data={
            "model_type": "failure_risk",
            "top_n": 10
        },
        timeout=30
    )

    if response.status_code == 200:
        result = response.json()
        print("✓ Feature importance retrieved successfully")
        print(f"  Model: {result.get('model_type', 'N/A')}")
        print(f"  Top features:")

        for i, feature in enumerate(result.get("top_features", [])[:5], 1):
            print(f"    {i}. {feature['feature']}: {feature['importance']:.4f}")

    elif response.status_code == 404:
        print("⚠ Explainer not found. Run training first:")
        print("  python src/rigvisionx/train_pipeline.py")
        print(f"  Response: {response.json()}")
    else:
        print(f"✗ Request failed with status {response.status_code}")
        print(f"  Response: {response.text}")

except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# Test 3: Explain Prediction Endpoint
# ============================================================================
print("\n[3/3] Testing /explainability/explain-prediction endpoint...")
try:
    # Test with default synthetic data
    response = requests.post(
        f"{API_BASE_URL}/explainability/explain-prediction",
        data={
            "window_size": 24,
            "step": 6,
            "model_type": "failure_risk",
            "prediction_index": 0
        },
        timeout=30
    )

    if response.status_code == 200:
        result = response.json()
        print("✓ Prediction explanation retrieved successfully")
        print(f"  Model: {result.get('model_type', 'N/A')}")
        print(f"  Prediction index: {result.get('prediction_index', 'N/A')}")

        explanation = result.get("explanation", {})
        print(f"\n  Explanation:")
        print(f"    Base value: {explanation.get('base_value', 0):.4f}")
        print(f"    Predicted value: {explanation.get('predicted_value', 0):.4f}")

        contributions = explanation.get("feature_contributions", [])
        if contributions:
            print(f"\n  Top 5 feature contributions:")
            for i, contrib in enumerate(contributions[:5], 1):
                impact_symbol = "↑" if contrib["impact"] == "increases" else "↓"
                print(f"    {i}. {contrib['feature']}: {contrib['shap_contribution']:+.4f} {impact_symbol}")
                print(f"       (feature value: {contrib['value']:.4f})")

    elif response.status_code == 404:
        print("⚠ Explainer not found. Run training first:")
        print("  python src/rigvisionx/train_pipeline.py")
        print(f"  Response: {response.json()}")
    else:
        print(f"✗ Request failed with status {response.status_code}")
        print(f"  Response: {response.text}")

except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("API TESTING COMPLETE")
print("=" * 80)
print("\n📚 API Documentation:")
print(f"  OpenAPI docs: {API_BASE_URL}/docs")
print(f"  ReDoc: {API_BASE_URL}/redoc")
print("\n🔍 Explainability Endpoints:")
print(f"  POST {API_BASE_URL}/explainability/feature-importance")
print(f"  POST {API_BASE_URL}/explainability/explain-prediction")
print("=" * 80)


