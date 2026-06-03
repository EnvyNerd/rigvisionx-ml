"""Tests for utility functions"""

import pytest
import numpy as np
import pandas as pd
from rigvisionx.utils.metrics import classification_metrics, regression_metrics


def test_classification_metrics(sample_labels):
    """Test classification metrics calculation."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    
    metrics = classification_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert metrics["accuracy"] == 0.8


def test_regression_metrics():
    """Test regression metrics calculation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2])
    
    metrics = regression_metrics(y_true, y_pred)
    assert "mse" in metrics
    assert "rmse" in metrics
    assert metrics["mse"] > 0
    assert metrics["rmse"] > 0


def test_metrics_perfect_prediction():
    """Test metrics with perfect predictions."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
