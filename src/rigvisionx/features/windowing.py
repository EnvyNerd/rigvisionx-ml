"""Time-window feature creation"""

import pandas as pd


def create_windows(data: pd.DataFrame, window_size: int, step: int = 1) -> list:
    """Create sliding windows from time series data."""
    pass


def aggregate_by_window(data: pd.DataFrame, window_size: int, agg_func: str = "mean"):
    """Aggregate data within time windows."""
    pass
