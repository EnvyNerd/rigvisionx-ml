"""System-wide report generation for TerraEnergy AI."""

from __future__ import annotations

import io
import os
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


matplotlib.use("Agg")
TRADEMARK_TEXT = os.environ.get(
    "REPORT_TRADEMARK_TEXT",
    os.environ.get("DASH_TRADEMARK_TEXT", "TerraEnergy AI 2026."),
)
REPORT_LINK_URL = os.environ.get(
    "REPORT_LINK_URL",
    "https://terraenergy-ai.vercel.app/",
)
REPORT_LINK_LABEL = os.environ.get("REPORT_LINK_LABEL", "Go to TerraEnergy AI")


def _figure_image(fig: plt.Figure, width: float = 6.8 * inch) -> Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    img = Image(buffer)
    img.drawWidth = width
    img.drawHeight = width * (img.imageHeight / img.imageWidth)
    return img


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _build_metric_table(data: pd.DataFrame, metric: str, limit: int = 20) -> list[list[Any]]:
    if metric not in data.columns:
        return []
    columns = []
    if "timestamp" in data.columns:
        columns.append("timestamp")
    if "rig_id" in data.columns:
        columns.append("rig_id")
    columns.append(metric)

    rows = data[columns].copy()
    if "timestamp" in rows.columns:
        rows["timestamp"] = rows["timestamp"].astype(str)
    rows = rows.tail(limit)

    table = [columns]
    for row in rows.itertuples(index=False):
        table.append([str(value) if value is not None else "" for value in row])
    return table


def _build_anomaly_table(
    data: pd.DataFrame,
    metrics: list[str],
    threshold: float,
    limit: int = 30,
) -> list[list[Any]]:
    anomalies: list[dict[str, Any]] = []
    threshold_value = float(threshold or 3.0)

    for metric in metrics:
        if metric not in data.columns:
            continue
        values = _safe_numeric(data[metric])
        mean = values.mean()
        std = values.std()
        if pd.isna(std) or std == 0:
            continue
        z_scores = (values - mean) / std
        mask = z_scores.abs() >= threshold_value
        if not mask.any():
            continue
        for idx in data[mask].index:
            row = data.loc[idx]
            anomalies.append(
                {
                    "timestamp": row["timestamp"] if "timestamp" in data.columns else str(idx),
                    "rig_id": row["rig_id"] if "rig_id" in data.columns else "n/a",
                    "metric": metric,
                    "value": float(values.loc[idx]) if pd.notna(values.loc[idx]) else None,
                    "z_score": float(z_scores.loc[idx]) if pd.notna(z_scores.loc[idx]) else None,
                }
            )

    anomalies.sort(key=lambda item: abs(item["z_score"] or 0.0), reverse=True)
    anomalies = anomalies[:limit]
    table = [["timestamp", "rig_id", "metric", "value", "z_score"]]
    for item in anomalies:
        table.append(
            [
                str(item["timestamp"]),
                str(item["rig_id"]),
                str(item["metric"]),
                f"{item['value']:.3f}" if item["value"] is not None else "",
                f"{item['z_score']:.2f}" if item["z_score"] is not None else "",
            ]
        )
    return table


def _plot_time_series(data: pd.DataFrame, metrics: list[str]) -> plt.Figure | None:
    metrics = [metric for metric in metrics if metric in data.columns]
    if not metrics:
        return None
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x_values = data["timestamp"] if "timestamp" in data.columns else data.index
    for metric in metrics:
        ax.plot(x_values, _safe_numeric(data[metric]), label=metric)
    ax.set_title("Time-Series Metrics")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    return fig


def _plot_risk(
    data: pd.DataFrame,
    metrics: list[str],
    window_size: int,
    threshold: float,
) -> plt.Figure | None:
    metrics = [metric for metric in metrics if metric in data.columns]
    if not metrics:
        return None

    z_frame = pd.DataFrame(index=data.index)
    for metric in metrics:
        values = _safe_numeric(data[metric])
        if "rig_id" in data.columns:
            means = values.groupby(data["rig_id"]).transform("mean")
            stds = values.groupby(data["rig_id"]).transform("std")
            z_scores = (values - means) / stds.replace(0, pd.NA)
        else:
            std = values.std()
            z_scores = (values - values.mean()) / (std if std else pd.NA)
        z_frame[metric] = z_scores

    if z_frame.empty:
        return None

    risk_raw = z_frame.abs().mean(axis=1)
    risk_score = risk_raw / (risk_raw + 1)
    risk_score = risk_score.fillna(0.0)
    window = max(int(window_size), 1)
    risk_smoothed = risk_score.rolling(window=window, min_periods=1).mean()
    threshold_value = float(threshold or 0.7)

    x_values = data["timestamp"] if "timestamp" in data.columns else data.index
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(x_values, risk_smoothed, label="Risk score")
    ax.axhline(threshold_value, linestyle="--", color="red", label="Threshold")
    ax.set_title("Failure Risk (Derived)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Risk Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    return fig


def _draw_header_footer(canvas_obj, doc) -> None:
    if not TRADEMARK_TEXT:
        return
    canvas_obj.saveState()
    canvas_obj.setFont("Helvetica-Oblique", 8)
    width, height = doc.pagesize
    margin = 0.7 * inch
    top_y = height - 0.45 * inch
    canvas_obj.drawString(margin, top_y, TRADEMARK_TEXT)
    if REPORT_LINK_URL and REPORT_LINK_LABEL:
        link_label = REPORT_LINK_LABEL
        text_width = canvas_obj.stringWidth(link_label, "Helvetica-Oblique", 8)
        right_x = width - margin
        left_x = right_x - text_width
        canvas_obj.drawRightString(right_x, top_y, link_label)
        canvas_obj.linkURL(
            REPORT_LINK_URL,
            (left_x, top_y - 2, right_x, top_y + 10),
            relative=0,
        )
    canvas_obj.restoreState()


def generate_system_report_pdf(
    data: pd.DataFrame,
    selections: dict[str, Any],
) -> bytes:
    """Generate a PDF system report for the selected dataset and settings."""
    report_title = selections.get("title") or "TerraEnergy AI Report"
    rig_id = selections.get("rig_id")
    metrics = selections.get("metrics") or []
    table_metric = selections.get("table_metric")
    anomaly_metrics = selections.get("anomaly_metrics") or []
    anomaly_threshold = selections.get("anomaly_threshold", 3.0)
    risk_metrics = selections.get("risk_metrics") or []
    risk_window_size = selections.get("risk_window_size", 12)
    risk_threshold = selections.get("risk_threshold", 0.7)

    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

    if rig_id and rig_id != "ALL" and "rig_id" in data.columns:
        data = data[data["rig_id"] == rig_id]

    styles = getSampleStyleSheet()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    story: list[Any] = []

    story.append(Paragraph(report_title, styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))

    rows = len(data)
    rigs = data["rig_id"].nunique() if "rig_id" in data.columns else "n/a"
    date_range = "n/a"
    if "timestamp" in data.columns and not data.empty:
        min_ts = data["timestamp"].min()
        max_ts = data["timestamp"].max()
        date_range = f"{min_ts} to {max_ts}"

    summary_text = f"Rows: {rows} | Rigs: {rigs} | Date range: {date_range}"
    story.append(Paragraph(summary_text, styles["BodyText"]))
    story.append(Spacer(1, 0.2 * inch))

    time_fig = _plot_time_series(data, metrics)
    if time_fig:
        story.append(Paragraph("Time-Series Overview", styles["Heading2"]))
        story.append(_figure_image(time_fig))
        story.append(Spacer(1, 0.2 * inch))

    risk_fig = _plot_risk(data, risk_metrics, risk_window_size, risk_threshold)
    if risk_fig:
        story.append(Paragraph("Failure Risk", styles["Heading2"]))
        story.append(_figure_image(risk_fig))
        story.append(Spacer(1, 0.2 * inch))

    metric_table = _build_metric_table(data, table_metric or "")
    if metric_table:
        story.append(Paragraph("Metric Readings", styles["Heading2"]))
        table = Table(metric_table, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 0.2 * inch))

    anomaly_table = _build_anomaly_table(data, anomaly_metrics, anomaly_threshold)
    if len(anomaly_table) > 1:
        story.append(Paragraph("Anomaly Alerts", styles["Heading2"]))
        table = Table(anomaly_table, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.append(table)

    doc.build(story, onFirstPage=_draw_header_footer, onLaterPages=_draw_header_footer)
    return buffer.getvalue()

