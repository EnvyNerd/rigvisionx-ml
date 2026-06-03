"""
Failure Horizon Label Generation for Predictive Maintenance.

Generates time-to-failure labels with configurable horizons,
enabling more nuanced failure prediction beyond binary classification.
"""

import logging
from typing import Optional, Union, Tuple, Dict
import numpy as np
import pandas as pd


class FailureHorizonLabeler:
    """
    Generate failure horizon labels for predictive maintenance.

    Instead of binary failure labels, creates multi-class labels based on
    time-to-failure horizons (e.g., "Critical", "Warning", "Normal").

    This enables:
    - Early warning systems (predict failures hours/days ahead)
    - Graduated risk assessment
    - Better maintenance planning
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize failure horizon labeler."""
        self.logger = logger or logging.getLogger(__name__)

    def generate_ttf_labels(
        self,
        df: pd.DataFrame,
        failure_column: str = "failure_flag",
        timestamp_column: str = "timestamp",
        unit_column: Optional[str] = None,
        ttf_column: str = "time_to_failure"
    ) -> pd.DataFrame:
        """
        Generate time-to-failure (TTF) labels.

        Args:
            df: Input dataframe with failure flags
            failure_column: Column indicating failure (1 = failure, 0 = normal)
            timestamp_column: Timestamp column
            unit_column: Optional column for equipment unit/asset ID
            ttf_column: Name for output TTF column

        Returns:
            DataFrame with TTF column added (in hours or time units)
        """
        self.logger.info("Generating time-to-failure labels")

        df = df.copy()
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df = df.sort_values(timestamp_column).reset_index(drop=True)

        if unit_column:
            # Handle multiple units separately
            df[ttf_column] = np.nan
            for unit in df[unit_column].unique():
                mask = df[unit_column] == unit
                df.loc[mask, ttf_column] = self._compute_ttf(
                    df[mask], failure_column, timestamp_column
                )
        else:
            # Single unit
            df[ttf_column] = self._compute_ttf(df, failure_column, timestamp_column)

        self.logger.info(f"TTF labels generated: min={df[ttf_column].min():.2f}, max={df[ttf_column].max():.2f}")

        return df

    def _compute_ttf(
        self,
        df: pd.DataFrame,
        failure_column: str,
        timestamp_column: str
    ) -> pd.Series:
        """
        Compute time-to-failure for a single unit.

        Args:
            df: Sorted dataframe for single unit
            failure_column: Failure flag column
            timestamp_column: Timestamp column

        Returns:
            Series with TTF values (hours)
        """
        ttf = pd.Series(index=df.index, dtype=float)

        # Find failure points
        failure_indices = df[df[failure_column] == 1].index.tolist()

        if not failure_indices:
            # No failures: set TTF to large value
            ttf[:] = 1e6
            return ttf

        # For each point, find time to next failure
        for idx in df.index:
            # Find next failure
            next_failures = [f for f in failure_indices if f >= idx]

            if next_failures:
                next_failure_idx = next_failures[0]
                time_diff = df.loc[next_failure_idx, timestamp_column] - df.loc[idx, timestamp_column]
                ttf.loc[idx] = time_diff.total_seconds() / 3600  # Convert to hours
            else:
                # No future failures
                ttf.loc[idx] = 1e6

        return ttf

    def generate_horizon_labels(
        self,
        df: pd.DataFrame,
        ttf_column: str = "time_to_failure",
        horizons: Optional[Dict[str, Tuple[float, float]]] = None,
        label_column: str = "failure_horizon"
    ) -> pd.DataFrame:
        """
        Generate failure horizon labels from TTF.

        Args:
            df: DataFrame with time_to_failure column
            ttf_column: Time-to-failure column name
            horizons: Dictionary mapping label -> (min_hours, max_hours)
                      Default: {"Critical": (0, 24), "Warning": (24, 72), "Normal": (72, inf)}
            label_column: Output column name for horizon labels

        Returns:
            DataFrame with horizon labels added
        """
        self.logger.info("Generating failure horizon labels")

        df = df.copy()

        # Default horizons
        if horizons is None:
            horizons = {
                "Critical": (0, 24),      # Failure within 24 hours
                "Warning": (24, 72),      # Failure within 24-72 hours
                "Caution": (72, 168),     # Failure within 3-7 days
                "Normal": (168, np.inf)   # No failure expected
            }

        self.logger.info(f"Horizons: {horizons}")

        # Assign labels based on TTF
        df[label_column] = "Unknown"

        for label, (min_hours, max_hours) in horizons.items():
            mask = (df[ttf_column] >= min_hours) & (df[ttf_column] < max_hours)
            df.loc[mask, label_column] = label

        # Log distribution
        label_counts = df[label_column].value_counts()
        self.logger.info(f"Horizon label distribution:\n{label_counts}")

        return df

    def generate_numeric_horizon(
        self,
        df: pd.DataFrame,
        ttf_column: str = "time_to_failure",
        horizons: Optional[Dict[str, Tuple[float, float]]] = None,
        label_column: str = "horizon_numeric"
    ) -> pd.DataFrame:
        """
        Generate numeric failure horizon labels (0, 1, 2, 3...).

        Args:
            df: DataFrame with time_to_failure column
            ttf_column: Time-to-failure column name
            horizons: Dictionary mapping label -> (min_hours, max_hours)
            label_column: Output column name

        Returns:
            DataFrame with numeric horizon labels (higher = more critical)
        """
        self.logger.info("Generating numeric horizon labels")

        df = df.copy()

        # Default horizons (ordered from most to least critical)
        if horizons is None:
            horizons = {
                "Critical": (0, 24),      # Label 3
                "Warning": (24, 72),      # Label 2
                "Caution": (72, 168),     # Label 1
                "Normal": (168, np.inf)   # Label 0
            }

        # Assign numeric labels (reverse order for criticality)
        df[label_column] = 0

        for idx, (label, (min_hours, max_hours)) in enumerate(reversed(list(horizons.items()))):
            mask = (df[ttf_column] >= min_hours) & (df[ttf_column] < max_hours)
            df.loc[mask, label_column] = len(horizons) - 1 - idx

        self.logger.info(f"Numeric horizon distribution:\n{df[label_column].value_counts().sort_index()}")

        return df


def generate_failure_horizon_labels(
    df: pd.DataFrame,
    failure_column: str = "failure_flag",
    timestamp_column: str = "timestamp",
    unit_column: Optional[str] = None,
    horizons: Optional[Dict[str, Tuple[float, float]]] = None,
    numeric: bool = False
) -> pd.DataFrame:
    """
    Convenience function to generate failure horizon labels.

    Args:
        df: Input dataframe with failure flags
        failure_column: Column indicating failure
        timestamp_column: Timestamp column
        unit_column: Optional equipment unit/asset ID column
        horizons: Custom horizon definitions
        numeric: Return numeric labels instead of text

    Returns:
        DataFrame with TTF and horizon labels added
    """
    labeler = FailureHorizonLabeler()

    # Generate TTF
    df = labeler.generate_ttf_labels(
        df,
        failure_column=failure_column,
        timestamp_column=timestamp_column,
        unit_column=unit_column
    )

    # Generate horizon labels
    if numeric:
        df = labeler.generate_numeric_horizon(
            df,
            horizons=horizons
        )
    else:
        df = labeler.generate_horizon_labels(
            df,
            horizons=horizons
        )

    return df
