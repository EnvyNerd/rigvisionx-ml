"""Financial reporting and analysis"""

from __future__ import annotations

import io
import os
from typing import Any

import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

TRADEMARK_TEXT = os.environ.get(
    "REPORT_TRADEMARK_TEXT",
    os.environ.get("DASH_TRADEMARK_TEXT", "TerraEnergy AI 2026."),
)
REPORT_LINK_URL = os.environ.get(
    "REPORT_LINK_URL",
    "https://terraenergy-ai.vercel.app/",
)
REPORT_LINK_LABEL = os.environ.get("REPORT_LINK_LABEL", "Go to TerraEnergy AI")


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def generate_financial_report(period_data: list) -> dict:
    """Generate financial performance report."""
    period_data = period_data or []
    totals = {
        "maintenance_cost": 0.0,
        "failure_cost": 0.0,
        "downtime_cost": 0.0,
        "preventive_costs": 0.0,
        "savings": 0.0,
        "total_cost": 0.0,
        "net_cost": 0.0,
    }
    breakdown: list[dict] = []

    for idx, entry in enumerate(period_data):
        entry = entry or {}
        maintenance = _coerce_float(entry.get("maintenance_cost"))
        failure = _coerce_float(entry.get("failure_cost"))
        downtime = _coerce_float(entry.get("downtime_cost"))
        preventive = _coerce_float(entry.get("preventive_costs"))
        savings = _coerce_float(entry.get("savings") or entry.get("cost_savings"))
        reported_total = entry.get("total_cost")

        if reported_total is not None:
            total_cost = _coerce_float(reported_total)
        else:
            total_cost = maintenance + failure + downtime + preventive
            if total_cost == 0.0:
                total_cost = _coerce_float(entry.get("cost"))

        net_cost = total_cost - savings
        label = entry.get("period") or entry.get("timestamp") or f"period_{idx + 1}"

        breakdown.append(
            {
                "period": label,
                "maintenance_cost": maintenance,
                "failure_cost": failure,
                "downtime_cost": downtime,
                "preventive_costs": preventive,
                "savings": savings,
                "total_cost": total_cost,
                "net_cost": net_cost,
            }
        )

        totals["maintenance_cost"] += maintenance
        totals["failure_cost"] += failure
        totals["downtime_cost"] += downtime
        totals["preventive_costs"] += preventive
        totals["savings"] += savings
        totals["total_cost"] += total_cost
        totals["net_cost"] += net_cost

    periods = len(breakdown)
    averages = {
        key: (value / periods if periods else 0.0) for key, value in totals.items()
    }

    worst_period = None
    if breakdown:
        worst_period = max(breakdown, key=lambda item: item.get("net_cost", 0.0))

    return {
        "periods": periods,
        "totals": totals,
        "averages": averages,
        "worst_period": worst_period,
        "breakdown": breakdown,
    }


def _draw_header_footer(
    pdf: canvas.Canvas,
    width: float,
    height: float,
    text: str,
    link_url: str,
    link_label: str,
) -> None:
    if not text:
        return
    pdf.saveState()
    pdf.setFont("Helvetica-Oblique", 8)
    margin = 0.75 * inch
    top_y = height - 0.45 * inch
    pdf.drawString(margin, top_y, text)
    if link_url and link_label:
        text_width = pdf.stringWidth(link_label, "Helvetica-Oblique", 8)
        right_x = width - margin
        left_x = right_x - text_width
        pdf.drawRightString(right_x, top_y, link_label)
        pdf.linkURL(link_url, (left_x, top_y - 2, right_x, top_y + 10), relative=0)
    pdf.restoreState()


def generate_financial_report_pdf(report: dict, title: str = "TerraEnergy AI FinTech Report") -> bytes:
    """Render a financial report to PDF bytes."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    x = 0.75 * inch
    y = height - 0.75 * inch

    _draw_header_footer(
        pdf,
        width,
        height,
        TRADEMARK_TEXT,
        REPORT_LINK_URL,
        REPORT_LINK_LABEL,
    )
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(x, y, title)
    y -= 0.35 * inch

    pdf.setFont("Helvetica", 10)
    pdf.drawString(x, y, f"Periods: {report.get('periods', 0)}")
    y -= 0.25 * inch

    totals = report.get("totals", {})
    averages = report.get("averages", {})
    worst = report.get("worst_period")

    def write_line(label: str, value: float) -> None:
        nonlocal y
        if y <= 0.75 * inch:
            pdf.showPage()
            _draw_header_footer(
                pdf,
                width,
                height,
                TRADEMARK_TEXT,
                REPORT_LINK_URL,
                REPORT_LINK_LABEL,
            )
            pdf.setFont("Helvetica", 10)
            y = height - 0.75 * inch
        pdf.drawString(x, y, f"{label}: {value:,.2f}")
        y -= 0.22 * inch

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(x, y, "Totals")
    y -= 0.25 * inch
    pdf.setFont("Helvetica", 10)
    for key in [
        "maintenance_cost",
        "failure_cost",
        "downtime_cost",
        "preventive_costs",
        "savings",
        "total_cost",
        "net_cost",
    ]:
        if key in totals:
            write_line(key.replace("_", " ").title(), float(totals[key]))

    y -= 0.1 * inch
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(x, y, "Averages")
    y -= 0.25 * inch
    pdf.setFont("Helvetica", 10)
    for key in [
        "maintenance_cost",
        "failure_cost",
        "downtime_cost",
        "preventive_costs",
        "savings",
        "total_cost",
        "net_cost",
    ]:
        if key in averages:
            write_line(key.replace("_", " ").title(), float(averages[key]))

    if worst:
        y -= 0.1 * inch
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(x, y, "Worst Period")
        y -= 0.25 * inch
        pdf.setFont("Helvetica", 10)
        pdf.drawString(x, y, f"Period: {worst.get('period', 'n/a')}")
        y -= 0.22 * inch
        write_line("Net Cost", float(worst.get("net_cost", 0.0)))

    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def calculate_cost_savings(baseline_failures: int, preventive_costs: float, actual_failures: int) -> float:
    """Calculate realized cost savings."""
    avoided_failures = baseline_failures - actual_failures
    savings = (avoided_failures * 10000) - preventive_costs  # Assuming $10k per failure
    return savings


def forecast_costs(historical_data: pd.DataFrame, periods: int = 12) -> pd.DataFrame:
    """Forecast maintenance costs."""
    if historical_data is None or historical_data.empty:
        return pd.DataFrame(columns=["period", "forecast_cost", "method"])

    data = historical_data.copy()
    cost_column = None
    for candidate in ("total_cost", "cost", "maintenance_cost"):
        if candidate in data.columns:
            cost_column = candidate
            break

    if cost_column is None:
        cost_components = [
            col
            for col in ["maintenance_cost", "failure_cost", "downtime_cost", "preventive_costs"]
            if col in data.columns
        ]
        if cost_components:
            data["total_cost"] = data[cost_components].sum(axis=1)
            cost_column = "total_cost"
        else:
            raise ValueError("historical_data must include cost columns")

    data = data.dropna(subset=[cost_column])
    if data.empty:
        return pd.DataFrame(columns=["period", "forecast_cost", "method"])

    costs = data[cost_column].astype(float).to_numpy()
    x = np.arange(len(costs))
    if len(costs) > 1:
        slope, intercept = np.polyfit(x, costs, 1)
    else:
        slope, intercept = 0.0, costs[-1]

    future_x = np.arange(len(costs), len(costs) + max(int(periods), 0))
    forecast = slope * future_x + intercept
    forecast = np.maximum(forecast, 0.0)

    time_col = None
    for candidate in ("timestamp", "date", "period"):
        if candidate in data.columns:
            time_col = candidate
            break

    if time_col and time_col != "period":
        timestamps = pd.to_datetime(data[time_col], errors="coerce")
        timestamps = timestamps.dropna()
        if not timestamps.empty:
            timestamps = timestamps.sort_values()
            deltas = timestamps.diff().dropna()
            delta = deltas.median() if not deltas.empty else pd.Timedelta(days=1)
            future_times = [timestamps.iloc[-1] + delta * (idx + 1) for idx in range(len(forecast))]
            return pd.DataFrame(
                {"timestamp": future_times, "forecast_cost": forecast, "method": "linear"}
            )

    return pd.DataFrame(
        {
            "period": list(range(len(costs) + 1, len(costs) + len(forecast) + 1)),
            "forecast_cost": forecast,
            "method": "linear",
        }
    )

