"""Data schema definitions"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class SensorReading:
    """Sensor data reading."""
    timestamp: datetime
    sensor_id: str
    value: float
    unit: str
    quality: int = 100  # Data quality score (0-100)


@dataclass
class EventData:
    """Event data structure."""
    timestamp: datetime
    event_type: str
    source: str
    details: Optional[dict] = None
