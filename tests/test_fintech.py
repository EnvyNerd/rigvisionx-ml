"""Tests for fintech reporting module."""

import pandas as pd

from rigvisionx.fintech.reporting import generate_financial_report, forecast_costs


def test_generate_financial_report_totals():
    data = [
        {
            "period": "2024-01",
            "maintenance_cost": 1000,
            "failure_cost": 5000,
            "downtime_cost": 2000,
            "preventive_costs": 500,
            "savings": 800,
        },
        {
            "period": "2024-02",
            "maintenance_cost": 2000,
            "failure_cost": 0,
            "downtime_cost": 1000,
            "preventive_costs": 300,
        },
    ]
    report = generate_financial_report(data)

    assert report["periods"] == 2
    assert report["totals"]["total_cost"] == 11800
    assert report["totals"]["net_cost"] == 11000
    assert len(report["breakdown"]) == 2


def test_forecast_costs():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="D"),
            "total_cost": [100, 120, 130, 140],
        }
    )
    forecast = forecast_costs(df, periods=3)
    assert len(forecast) == 3
    assert "forecast_cost" in forecast.columns
