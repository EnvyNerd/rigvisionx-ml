"""Energy deviation detection and analysis"""

import pandas as pd


def calculate_deviation(actual: float, baseline: float) -> float:
    """Calculate deviation from baseline."""
    return ((actual - baseline) / baseline) * 100


def detect_anomalous_consumption(data: pd.DataFrame, baseline: dict, threshold: float = 15.0):
    """Detect anomalous energy consumption."""
    pass


def analyze_deviation_patterns(data: pd.DataFrame):
    """Analyze patterns in energy deviations."""
    pass
