"""Tests for data ingestion module"""

import pytest
from datetime import datetime
from rigvisionx.ingest.schema import SensorReading, EventData


def test_sensor_reading_creation():
    """Test creating a sensor reading."""
    reading = SensorReading(
        timestamp=datetime.now(),
        sensor_id="SENSOR_001",
        value=100.5,
        unit="kW"
    )
    assert reading.sensor_id == "SENSOR_001"
    assert reading.value == 100.5
    assert reading.unit == "kW"
    assert reading.quality == 100


def test_event_data_creation():
    """Test creating event data."""
    event = EventData(
        timestamp=datetime.now(),
        event_type="alarm",
        source="PLC_01",
        details={"severity": "high"}
    )
    assert event.event_type == "alarm"
    assert event.source == "PLC_01"
    assert event.details["severity"] == "high"


def test_sensor_reading_custom_quality():
    """Test sensor reading with custom quality."""
    reading = SensorReading(
        timestamp=datetime.now(),
        sensor_id="SENSOR_002",
        value=50.0,
        unit="kW",
        quality=85
    )
    assert reading.quality == 85
