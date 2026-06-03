"""Data splitting utilities"""

import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit


def temporal_train_test_split(data: pd.DataFrame, test_size: float = 0.2):
    """
    Split time series data into train/test maintaining temporal order.

    Args:
        data: DataFrame with datetime index
        test_size: Proportion of data for testing

    Returns:
        train_data, test_data
    """
    split_idx = int(len(data) * (1 - test_size))
    return data.iloc[:split_idx], data.iloc[split_idx:]


def stratified_split(data: pd.DataFrame, labels: pd.Series, test_size: float = 0.2):
    """Stratified split maintaining class distribution."""
    return train_test_split(data, labels, test_size=test_size, stratify=labels, random_state=42)


def cross_validation_splits(data: pd.DataFrame, n_splits: int = 5):
    """Generate cross-validation splits for time series."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(data))
