"""CSV replay adapter for testing/simulation"""

import pandas as pd


def load_csv(filepath: str) -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv(filepath)


def replay_data(dataframe: pd.DataFrame, speed: float = 1.0):
    """Replay CSV data at specified speed."""
    pass
