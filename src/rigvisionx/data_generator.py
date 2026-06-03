"""
Synthetic Oil-Rig Dataset Generator

Generates realistic industrial equipment sensor data for demonstration and testing:
- Multiple oil rigs with various operational states
- Sensor readings (power, temperature, vibration, pressure, flow rate)
- Failure patterns and anomalies
- Seasonal and daily patterns
- Degradation trends
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class OilRigDatasetGenerator:
    """Generate synthetic oil rig sensor data."""

    def __init__(self, random_seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        np.random.seed(random_seed)

    def _samples_per_day(self, frequency: str) -> int:
        """Return the number of samples in one day for the given frequency."""
        delta = pd.Timedelta(frequency)
        day = pd.Timedelta(days=1)
        if delta <= pd.Timedelta(0):
            raise ValueError("frequency must be a positive time delta")
        if day % delta != pd.Timedelta(0):
            raise ValueError("frequency must evenly divide one day")
        return int(day / delta)

    def generate_rig_data(
        self,
        rig_id: str = "RIG_001",
        n_days: int = 90,
        frequency: str = "1H",
        failure_day: Optional[int] = None,
        anomaly_windows: Optional[List[Tuple[int, int]]] = None,
    ) -> pd.DataFrame:
        """
        Generate sensor data for a single oil rig.

        Args:
            rig_id: Identifier for the rig
            n_days: Number of days of data to generate
            frequency: Pandas frequency string (e.g., '1H' for hourly)
            failure_day: Day (1-indexed) when failure occurs (optional)
            anomaly_windows: List of (start_day, end_day) tuples for anomalies

        Returns:
            DataFrame with sensor readings
        """
        if failure_day is not None and (failure_day < 1 or failure_day > n_days):
            failure_day = None

        # Generate timestamps
        normalized_frequency = frequency.replace("H", "h")
        samples_per_day = self._samples_per_day(normalized_frequency)
        n_samples = n_days * samples_per_day
        timestamps = pd.date_range(
            start="2024-01-01",
            periods=n_samples,
            freq=normalized_frequency,  # Handle pandas frequency change
        )

        data = pd.DataFrame({"timestamp": timestamps})
        data["rig_id"] = rig_id
        data["hour"] = data["timestamp"].dt.hour
        data["day"] = data["timestamp"].dt.day
        data["day_of_week"] = data["timestamp"].dt.dayofweek

        # Base operational conditions (vary by hour and day)
        base_load_profile = self._create_load_profile(timestamps)
        data["base_load"] = base_load_profile

        # Sensor 1: Power Consumption (kW)
        data["power_consumption"] = self._generate_power_consumption(
            base_load_profile,
            timestamps,
            samples_per_day,
            failure_day,
            anomaly_windows,
        )

        # Sensor 2: Temperature (degC)
        data["temperature"] = self._generate_temperature(
            data["power_consumption"],
            timestamps,
            samples_per_day,
            failure_day,
            anomaly_windows,
        )

        # Sensor 3: Vibration (mm/s)
        data["vibration"] = self._generate_vibration(
            data["power_consumption"],
            timestamps,
            samples_per_day,
            failure_day,
            anomaly_windows,
        )

        # Sensor 4: Pressure (bar)
        data["pressure"] = self._generate_pressure(
            data["power_consumption"],
            timestamps,
            samples_per_day,
            failure_day,
            anomaly_windows,
        )

        # Sensor 5: Flow Rate (m3/h)
        data["flow_rate"] = self._generate_flow_rate(
            data["power_consumption"],
            timestamps,
            samples_per_day,
            failure_day,
            anomaly_windows,
        )

        # Sensor 6: Bearing Temperature (degC)
        data["bearing_temperature"] = self._generate_bearing_temperature(
            data["temperature"],
            samples_per_day,
            failure_day,
            anomaly_windows,
        )

        # Operational Status
        data["operational_mode"] = self._generate_operational_mode(data["hour"])

        # Failure indicator
        data["failure_indicator"] = 0
        if failure_day:
            failure_idx = (failure_day - 1) * samples_per_day
            data.loc[failure_idx:, "failure_indicator"] = 1
        data["failure_flag"] = data["failure_indicator"]

        # RUL (Remaining Useful Life) - decreases over time
        data["estimated_rul"] = self._generate_rul(
            n_samples,
            samples_per_day,
            failure_day,
            n_days,
        )

        return data

    def _create_load_profile(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Create realistic daily load profile (higher during day, lower at night)."""
        hours = timestamps.hour.to_numpy()
        # Peak load during business hours (8-18), low at night
        load_profile = 60 + 35 * np.sin((hours - 6) * np.pi / 12)
        load_profile = np.clip(load_profile, 40, 95)
        # Add weekly pattern (slightly lower on weekends)
        weekly = 1 - 0.1 * np.sin(timestamps.dayofweek.to_numpy() * np.pi / 7)
        return load_profile * weekly

    def _generate_power_consumption(
        self,
        base_load: np.ndarray,
        timestamps: pd.DatetimeIndex,
        samples_per_day: int,
        failure_day: Optional[int] = None,
        anomaly_windows: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """Generate power consumption with trend, noise, and anomalies."""
        n_samples = len(timestamps)

        # Base consumption with trend (slight degradation over time)
        degradation = np.linspace(0, 15, n_samples)
        power = base_load + degradation + np.random.normal(0, 5, n_samples)

        # Add seasonal pattern
        day_of_year = timestamps.dayofyear.to_numpy()
        seasonal = 10 * np.sin(day_of_year * 2 * np.pi / 365)
        power += seasonal

        # Add anomalies
        if anomaly_windows:
            for start_day, end_day in anomaly_windows:
                start_idx = (start_day - 1) * samples_per_day
                if start_idx >= n_samples:
                    continue
                end_idx = min(end_day * samples_per_day, n_samples)
                if end_idx <= start_idx:
                    continue
                power[start_idx:end_idx] += np.random.normal(20, 8, end_idx - start_idx)

        # Add failure ramp
        if failure_day:
            failure_idx = (failure_day - 1) * samples_per_day
            if failure_idx >= n_samples:
                return np.clip(power, 20, 150)
            # Power spikes before failure
            ramp = np.linspace(0, 40, n_samples - failure_idx)
            power[failure_idx:] += ramp

        return np.clip(power, 20, 150)

    def _generate_temperature(
        self,
        power_consumption: np.ndarray,
        timestamps: pd.DatetimeIndex,
        samples_per_day: int,
        failure_day: Optional[int] = None,
        anomaly_windows: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """Generate temperature (correlated with power consumption)."""
        n_samples = len(power_consumption)
        # Temperature increases with power
        temp = 30 + power_consumption * 0.6 + np.random.normal(0, 2, len(power_consumption))

        # Add daily cooling cycle
        hours = timestamps.hour.to_numpy()
        daily_cycle = 8 * np.sin((hours - 6) * np.pi / 12)
        temp += daily_cycle

        # Add anomalies
        if anomaly_windows:
            for start_day, end_day in anomaly_windows:
                start_idx = (start_day - 1) * samples_per_day
                if start_idx >= n_samples:
                    continue
                end_idx = min(end_day * samples_per_day, n_samples)
                if end_idx <= start_idx:
                    continue
                temp[start_idx:end_idx] += np.random.normal(15, 5, end_idx - start_idx)

        # Temperature spikes before failure
        if failure_day:
            failure_idx = (failure_day - 1) * samples_per_day
            if failure_idx >= n_samples:
                return np.clip(temp, 15, 95)
            ramp = np.linspace(0, 25, len(temp) - failure_idx)
            temp[failure_idx:] += ramp

        return np.clip(temp, 15, 95)

    def _generate_vibration(
        self,
        power_consumption: np.ndarray,
        timestamps: pd.DatetimeIndex,
        samples_per_day: int,
        failure_day: Optional[int] = None,
        anomaly_windows: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """Generate vibration (mm/s) - increases with power and degradation."""
        n_samples = len(power_consumption)
        # Base vibration from equipment operation
        vibration = 0.3 + power_consumption * 0.003 + np.random.normal(0, 0.05, len(power_consumption))

        # Add degradation trend
        degradation = np.linspace(0, 0.4, len(power_consumption))
        vibration += degradation

        # Add anomalies (increased vibration)
        if anomaly_windows:
            for start_day, end_day in anomaly_windows:
                start_idx = (start_day - 1) * samples_per_day
                if start_idx >= n_samples:
                    continue
                end_idx = min(end_day * samples_per_day, n_samples)
                if end_idx <= start_idx:
                    continue
                vibration[start_idx:end_idx] += np.random.normal(0.5, 0.15, end_idx - start_idx)

        # Vibration increases significantly before failure
        if failure_day:
            failure_idx = (failure_day - 1) * samples_per_day
            if failure_idx >= n_samples:
                return np.clip(vibration, 0.1, 5.0)
            ramp = np.linspace(0, 2.5, len(vibration) - failure_idx)
            vibration[failure_idx:] += ramp

        return np.clip(vibration, 0.1, 5.0)

    def _generate_pressure(
        self,
        power_consumption: np.ndarray,
        timestamps: pd.DatetimeIndex,
        samples_per_day: int,
        failure_day: Optional[int] = None,
        anomaly_windows: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """Generate system pressure (bar)."""
        n_samples = len(power_consumption)
        # Pressure correlates with power consumption
        pressure = 30 + power_consumption * 0.25 + np.random.normal(0, 2, len(power_consumption))

        # Add anomalies (pressure drops or spikes)
        if anomaly_windows:
            for start_day, end_day in anomaly_windows:
                start_idx = (start_day - 1) * samples_per_day
                if start_idx >= n_samples:
                    continue
                end_idx = min(end_day * samples_per_day, n_samples)
                if end_idx <= start_idx:
                    continue
                pressure[start_idx:end_idx] += np.random.normal(-5, 3, end_idx - start_idx)

        # Pressure instability before failure
        if failure_day:
            failure_idx = (failure_day - 1) * samples_per_day
            if failure_idx >= n_samples:
                return np.clip(pressure, 10, 120)
            instability = np.random.normal(0, 3, len(pressure) - failure_idx)
            pressure[failure_idx:] += instability

        return np.clip(pressure, 10, 120)

    def _generate_flow_rate(
        self,
        power_consumption: np.ndarray,
        timestamps: pd.DatetimeIndex,
        samples_per_day: int,
        failure_day: Optional[int] = None,
        anomaly_windows: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """Generate flow rate (m3/h)."""
        n_samples = len(power_consumption)
        # Flow rate increases with power consumption
        flow = 50 + power_consumption * 0.3 + np.random.normal(0, 3, len(power_consumption))

        # Add anomalies (flow reduction)
        if anomaly_windows:
            for start_day, end_day in anomaly_windows:
                start_idx = (start_day - 1) * samples_per_day
                if start_idx >= n_samples:
                    continue
                end_idx = min(end_day * samples_per_day, n_samples)
                if end_idx <= start_idx:
                    continue
                flow[start_idx:end_idx] -= np.random.normal(10, 5, end_idx - start_idx)

        # Flow reduction before failure
        if failure_day:
            failure_idx = (failure_day - 1) * samples_per_day
            if failure_idx >= n_samples:
                return np.clip(flow, 10, 150)
            ramp = np.linspace(0, 30, len(flow) - failure_idx)
            flow[failure_idx:] -= ramp

        return np.clip(flow, 10, 150)

    def _generate_bearing_temperature(
        self,
        temperature: np.ndarray,
        samples_per_day: int,
        failure_day: Optional[int] = None,
        anomaly_windows: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """Generate bearing temperature (slightly higher variance)."""
        n_samples = len(temperature)
        bearing_temp = temperature + np.random.normal(5, 3, len(temperature))

        # Bearing temp spikes in anomalies
        if anomaly_windows:
            for start_day, end_day in anomaly_windows:
                start_idx = (start_day - 1) * samples_per_day
                if start_idx >= n_samples:
                    continue
                end_idx = min(end_day * samples_per_day, n_samples)
                if end_idx <= start_idx:
                    continue
                bearing_temp[start_idx:end_idx] += np.random.normal(20, 8, end_idx - start_idx)

        # Critical bearing temp before failure
        if failure_day:
            failure_idx = (failure_day - 1) * samples_per_day
            if failure_idx >= n_samples:
                return np.clip(bearing_temp, 15, 110)
            ramp = np.linspace(0, 30, len(bearing_temp) - failure_idx)
            bearing_temp[failure_idx:] += ramp

        return np.clip(bearing_temp, 15, 110)

    def _generate_operational_mode(self, hours: np.ndarray) -> np.ndarray:
        """Generate operational mode based on time of day."""
        modes = np.where(
            (hours >= 6) & (hours < 22),
            "HIGH_PRODUCTION",
            np.where((hours >= 22) | (hours < 6), "STANDBY", "MAINTENANCE"),
        )
        return modes

    def _generate_rul(
        self,
        n_samples: int,
        samples_per_day: int,
        failure_day: Optional[int],
        n_days: int,
    ) -> np.ndarray:
        """Generate Remaining Useful Life (days)."""
        if failure_day:
            failure_idx = (failure_day - 1) * samples_per_day
            rul = np.linspace(n_days - failure_day, 0, n_samples)
            # Smooth decrease
            rul = np.maximum(rul, 0)
        else:
            # No failure, RUL stays high
            rul = np.linspace(n_days, n_days + 100, n_samples)

        return rul

    def generate_multi_rig_dataset(
        self,
        n_rigs: int = 5,
        n_days: int = 90,
        failure_probability: float = 0.2,
        anomaly_probability: float = 0.4,
    ) -> pd.DataFrame:
        """
        Generate dataset for multiple rigs with varied conditions.

        Args:
            n_rigs: Number of rigs
            n_days: Days of data per rig
            failure_probability: Probability a rig has failure
            anomaly_probability: Probability a rig has anomalies

        Returns:
            Combined DataFrame for all rigs
        """
        all_data = []

        for i in range(n_rigs):
            rig_id = f"RIG_{i + 1:03d}"

            # Random failure
            failure_day = None
            if np.random.random() < failure_probability:
                failure_day = np.random.randint(30, n_days)

            # Random anomalies
            anomaly_windows = None
            if np.random.random() < anomaly_probability:
                n_anomalies = np.random.randint(1, 3)
                anomaly_windows = []
                for _ in range(n_anomalies):
                    start = np.random.randint(1, n_days - 10)
                    end = start + np.random.randint(2, 8)
                    anomaly_windows.append((start, end))

            rig_data = self.generate_rig_data(
                rig_id=rig_id,
                n_days=n_days,
                failure_day=failure_day,
                anomaly_windows=anomaly_windows,
            )

            all_data.append(rig_data)

        return pd.concat(all_data, ignore_index=True)

    def save_dataset(self, data: pd.DataFrame, output_path: str) -> None:
        """Save dataset to CSV file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        print(f"  Shape: {data.shape}")
        if "rig_id" in data.columns:
            print(f"  Rigs: {data['rig_id'].nunique()}")
        print(f"  Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")

    def generate_sample_scenarios(self) -> Dict[str, pd.DataFrame]:
        """Generate various sample scenarios for testing."""
        scenarios = {}

        # Scenario 1: Normal operation (no failures)
        print("Generating Scenario 1: Normal Operation...")
        scenarios["normal_operation"] = self.generate_rig_data(
            rig_id="RIG_NORMAL",
            n_days=60,
            failure_day=None,
            anomaly_windows=None,
        )

        # Scenario 2: Equipment with gradual degradation
        print("Generating Scenario 2: Degradation...")
        scenarios["degradation"] = self.generate_rig_data(
            rig_id="RIG_DEGRADATION",
            n_days=90,
            failure_day=85,
        )

        # Scenario 3: Equipment with anomalies
        print("Generating Scenario 3: Anomalies...")
        scenarios["anomalies"] = self.generate_rig_data(
            rig_id="RIG_ANOMALIES",
            n_days=90,
            anomaly_windows=[(10, 15), (45, 48), (75, 78)],
        )

        # Scenario 4: Multiple rigs in fleet
        print("Generating Scenario 4: Fleet Data...")
        scenarios["fleet"] = self.generate_multi_rig_dataset(
            n_rigs=10,
            n_days=90,
            failure_probability=0.3,
            anomaly_probability=0.5,
        )

        return scenarios


def main():
    """CLI for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic oil rig datasets")
    parser.add_argument(
        "--scenario",
        choices=["single", "fleet", "all"],
        default="all",
        help="Type of dataset to generate",
    )
    parser.add_argument("--n-rigs", type=int, default=5, help="Number of rigs for fleet scenario")
    parser.add_argument("--n-days", type=int, default=90, help="Number of days of data")
    parser.add_argument("--output-dir", default="data/generated/", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    generator = OilRigDatasetGenerator(random_seed=args.seed)

    print("=" * 70)
    print("OIL RIG SYNTHETIC DATASET GENERATOR")
    print("=" * 70)

    # Generate datasets
    if args.scenario in ["single", "all"]:
        print("\n[1/2] Generating single rig dataset...")
        single_rig = generator.generate_rig_data(
            rig_id="RIG_001",
            n_days=args.n_days,
            failure_day=75,
            anomaly_windows=[(20, 25), (50, 55)],
        )
        generator.save_dataset(single_rig, f"{args.output_dir}single_rig.csv")
        print("Sample of single rig data:")
        print(single_rig.head())

    if args.scenario in ["fleet", "all"]:
        print("\n[2/2] Generating fleet dataset...")
        fleet = generator.generate_multi_rig_dataset(
            n_rigs=args.n_rigs,
            n_days=args.n_days,
        )
        generator.save_dataset(fleet, f"{args.output_dir}fleet_data.csv")
        print("\nFleet Statistics:")
        print(f"  Total records: {len(fleet)}")
        print(f"  Rigs: {fleet['rig_id'].unique()}")
        print(f"  Failures: {(fleet['failure_indicator'] == 1).sum()} records")

    # Generate sample scenarios
    print("\n" + "=" * 70)
    print("GENERATING SAMPLE SCENARIOS")
    print("=" * 70)
    scenarios = generator.generate_sample_scenarios()

    for scenario_name, data in scenarios.items():
        output_path = f"{args.output_dir}scenario_{scenario_name}.csv"
        generator.save_dataset(data, output_path)

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"All datasets saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
