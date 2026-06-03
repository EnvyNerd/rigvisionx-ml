"""
Feature engineering pipeline for offshore rig dataset.

Works with critical_offshore_rig_dataset_100.csv format.
"""

import logging
from typing import Optional
import numpy as np
import pandas as pd


class FeaturePipeline:
    """
    Feature engineering pipeline for predictive maintenance.

    Handles:
    - Rolling window statistics
    - Trend calculations
    - Rate of change features
    - Interaction features
    - Time-based features
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize feature pipeline."""
        self.logger = logger or logging.getLogger(__name__)
        self.feature_columns = None

    def engineer_features(
        self,
        df: pd.DataFrame,
        window_sizes: list = [3, 6, 12],
        timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Engineer features from raw sensor data.

        Args:
            df: Input dataframe with sensor readings
            window_sizes: Rolling window sizes (in hours)
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Engineering features...")

        df = df.copy()

        # Ensure timestamp is datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        # Identify sensor columns
        sensor_columns = [
            "power_kw",
            "vibration_rms",
            "temperature_c",
            "bearing_temp_c",
            "pressure_psi",
            "flow_rate_m3h",
            "rotational_speed_rpm"
        ]

        # Keep only columns that exist
        sensor_columns = [col for col in sensor_columns if col in df.columns]
        self.logger.info(f"Processing {len(sensor_columns)} sensor columns")

        # Add existing computed columns if present
        if "energy_deviation" in df.columns:
            sensor_columns.append("energy_deviation")
        if "anomaly_score" in df.columns:
            sensor_columns.append("anomaly_score")
        if "failure_indicator" in df.columns:
            sensor_columns.append("failure_indicator")

        # 1. Rolling window statistics
        for window in window_sizes:
            self.logger.info(f"Computing rolling features (window={window}h)...")
            for col in sensor_columns:
                # Mean
                df[f"{col}_mean_{window}h"] = df[col].rolling(window=window, min_periods=1).mean()

                # Std deviation
                df[f"{col}_std_{window}h"] = df[col].rolling(window=window, min_periods=1).std().fillna(0)

                # Min/Max
                df[f"{col}_min_{window}h"] = df[col].rolling(window=window, min_periods=1).min()
                df[f"{col}_max_{window}h"] = df[col].rolling(window=window, min_periods=1).max()

                # Range
                df[f"{col}_range_{window}h"] = df[f"{col}_max_{window}h"] - df[f"{col}_min_{window}h"]

        # 2. Trend features (rate of change)
        self.logger.info("Computing trend features...")
        for col in sensor_columns:
            # 1-hour change
            df[f"{col}_diff_1h"] = df[col].diff(1).fillna(0)

            # 3-hour change
            df[f"{col}_diff_3h"] = df[col].diff(3).fillna(0)

            # Percentage change
            df[f"{col}_pct_change"] = df[col].pct_change().fillna(0).replace([np.inf, -np.inf], 0)

        # 3. Interaction features
        self.logger.info("Computing interaction features...")

        # Power efficiency (power per flow)
        if "power_kw" in df.columns and "flow_rate_m3h" in df.columns:
            df["power_efficiency"] = df["power_kw"] / (df["flow_rate_m3h"] + 1e-6)
            df["power_efficiency"] = df["power_efficiency"].replace([np.inf, -np.inf], 0)

        # Vibration per speed
        if "vibration_rms" in df.columns and "rotational_speed_rpm" in df.columns:
            df["vibration_per_speed"] = df["vibration_rms"] / (df["rotational_speed_rpm"] + 1e-6)
            df["vibration_per_speed"] = df["vibration_per_speed"].replace([np.inf, -np.inf], 0)

        # Temperature differential
        if "bearing_temp_c" in df.columns and "temperature_c" in df.columns:
            df["temp_differential"] = df["bearing_temp_c"] - df["temperature_c"]

        # Pressure-flow relationship
        if "pressure_psi" in df.columns and "flow_rate_m3h" in df.columns:
            df["pressure_flow_ratio"] = df["pressure_psi"] / (df["flow_rate_m3h"] + 1e-6)
            df["pressure_flow_ratio"] = df["pressure_flow_ratio"].replace([np.inf, -np.inf], 0)

        # 4. Time-based features
        self.logger.info("Computing time-based features...")
        df["hour_of_day"] = df[timestamp_col].dt.hour
        df["day_of_week"] = df[timestamp_col].dt.dayofweek
        df["day_of_month"] = df[timestamp_col].dt.day

        # 5. Operational mode encoding (if present)
        if "operational_mode" in df.columns:
            self.logger.info("Encoding operational mode...")
            mode_dummies = pd.get_dummies(df["operational_mode"], prefix="mode")
            df = pd.concat([df, mode_dummies], axis=1)

        # 6. Cumulative features
        self.logger.info("Computing cumulative features...")
        if "power_kw" in df.columns:
            df["cumulative_power"] = df["power_kw"].cumsum()

        if "vibration_rms" in df.columns:
            df["cumulative_vibration"] = df["vibration_rms"].cumsum()

        # 7. Equipment age proxy (row index as proxy for operating hours)
        df["operating_hours"] = np.arange(len(df))

        self.logger.info(f"Feature engineering complete: {df.shape[1]} columns")

        return df

    def select_features(
        self,
        df: pd.DataFrame,
        exclude_columns: list = None
    ) -> pd.DataFrame:
        """
        Select only feature columns (exclude metadata and targets).

        Args:
            df: DataFrame with all columns
            exclude_columns: Additional columns to exclude

        Returns:
            DataFrame with only feature columns
        """
        if exclude_columns is None:
            exclude_columns = []

        # Default exclusions
        default_exclusions = [
            "timestamp",
            "rig_id",
            "equipment",
            "operational_mode",
            "failure_flag",
            "failure_indicator",
            "estimated_rul_hours",
            "time_to_failure",
            "horizon_numeric",
            "horizon_label"
        ]

        all_exclusions = set(default_exclusions + exclude_columns)

        # Select feature columns
        feature_cols = [col for col in df.columns if col not in all_exclusions]

        self.feature_columns = feature_cols
        self.logger.info(f"Selected {len(feature_cols)} feature columns")

        return df[feature_cols]

    def get_feature_names(self) -> list:
        """Get list of feature column names."""
        if self.feature_columns is None:
            raise ValueError("Features not yet selected. Call select_features() first.")
        return self.feature_columns


def engineer_offshore_features(
    df: pd.DataFrame,
    window_sizes: list = [3, 6, 12],
    logger: Optional[logging.Logger] = None
) -> tuple:
    """
    Convenience function to engineer features from offshore dataset.

    Args:
        df: Raw offshore data
        window_sizes: Rolling window sizes
        logger: Optional logger

    Returns:
        Tuple of (full_dataframe, feature_columns)
    """
    pipeline = FeaturePipeline(logger=logger)

    # Engineer features
    df_features = pipeline.engineer_features(df, window_sizes=window_sizes)

    # Get feature column names
    feature_df = pipeline.select_features(df_features)
    feature_names = pipeline.get_feature_names()

    return df_features, feature_names
