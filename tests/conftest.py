"""Pytest configuration and fixtures"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_sensor_data():
    """Generate sample sensor data for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="H")
    return pd.DataFrame({
        "timestamp": dates,
        "sensor_id": "SENSOR_001",
        "value": np.random.normal(100, 10, 100),
        "unit": "kW",
    })


@pytest.fixture
def sample_features():
    """Generate sample feature matrix."""
    return np.random.randn(50, 20)


@pytest.fixture
def sample_labels():
    """Generate sample labels."""
    return np.random.randint(0, 2, 50)
