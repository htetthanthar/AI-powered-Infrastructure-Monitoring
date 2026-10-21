"""
charts.py
---------
Interactive Plotly charts for the Streamlit dashboard: hover for exact
values, drag to zoom, double-click to reset, click legend items to hide a
series. `uirevision` keeps your zoom and pan while the chart updates live.

(The matplotlib versions in anomaly_detector.py / forecasting_engine.py are
still used for the saved report screenshots.)
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LABELS = {"cpu_usage": "CPU (%)", "memory_usage": "Memory (%)",
          "network_mbps": "Network (Mbps)", "response_time_ms": "Response time (ms)",
          "error_rate_percent": "Error rate (%)", "disk_usage": "Disk (%)"}
COLOURS = {"cpu_usage": "#2a6fdb", "memory_usage": "#2e9e6a",
           "network_mbps": "#8a5cd1", "response_time_ms": "#d9822b",
           "error_rate_percent": "#c2417b", "disk_usage": "#5b6b7a"}
RED = "#d62728"


def anomaly_figure(result: pd.DataFrame, metrics: list[str], features: list[str],
                   leak_window: tuple | None = None, time_fmt: str = "%d %b %H:%M",
                   minor_times: set | None = None,
                   min_std: dict | None = None) -> go.Figure:
    """One panel per metric; detected anomalies marked with red crosses.

    A detected reading is marked on a panel when that metric was unusual
    (> 2.5 standard deviations from normal) or was the reading's main metric.
    Minor flags (no metric moved meaningfully) are shown as small hollow
    circles on their main metric only.
    """
    fig = make_subplots(rows=len(metrics), cols=1, shared_xaxes=True,
                        vertical_spacing=0.04,
                        subplot_titles=[LABELS[m] for m in metrics])
    flagged = result[result["is_anomaly"]]
    is_minor = flagged["timestamp"].isin(minor_times or set())
    anomalies, minor = flagged[~is_minor], flagged[is_minor]
    med = result[features].median()
    std = result[features].std().fillna(0)
    if min_std:
        std = std.clip(lower=pd.Series(min_std)[features])
    std = std.replace(0, 1e-9)

    for row, m in enumerate(metrics, start=1):
        fig.add_trace(go.Scatter(
            x=result["timestamp"], y=result[m], mode="lines", name=LABELS[m],
            line=dict(color=COLOURS[m], width=1.4),
            hovertemplate=f"{LABELS[m]}: %{{y:.1f}}<extra></extra>"), row=row, col=1)
        if not anomalies.empty and m in features:
            unusual = anomalies[((anomalies[m] - med[m]).abs() > 2.5 * std[m])
                                | (anomalies["main_metric"] == m)]
            fig.add_trace(go.Scatter(
                x=unusual["timestamp"], y=unusual[m], mode="markers",
                name="Detected anomaly", legendgroup="anomaly",
                showlegend=(row == 1),
                marker=dict(symbol="x", size=10, color=RED, line=dict(width=1)),
                customdata=unusual[["anomaly_score"]],
                hovertemplate=(f"<b>Anomaly</b><br>{LABELS[m]}: %{{y:.1f}}"
                               "<br>score: %{customdata[0]:.3f}<extra></extra>")),
                row=row, col=1)
        if not minor.empty and m in features:
            mm = minor[minor["main_metric"] == m]
            fig.add_trace(go.Scatter(
                x=mm["timestamp"], y=mm[m], mode="markers",
                name="Minor flag", legendgroup="minor", showlegend=(row == 1),
                marker=dict(symbol="circle-open", size=8, color=RED),
                hovertemplate=(f"Minor flag<br>{LABELS[m]}: %{{y:.1f}}"
                               "<extra></extra>")), row=row, col=1)
        if leak_window is not None:
            fig.add_vrect(x0=leak_window[0], x1=leak_window[1], row=row, col=1,
                          fillcolor="#e8a33d", opacity=0.15, line_width=0)

    fig.update_layout(
        height=170 * len(metrics) + 60, margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified", uirevision="anomaly",
        legend=dict(orientation="h", y=1.04, x=1, xanchor="right", yanchor="bottom"),
        showlegend=True)
    fig.update_xaxes(hoverformat=time_fmt)
    return fig


def forecast_figure(fc, warning: float, critical: float) -> go.Figure:
    """Disk history, regression fit, 30-day forecast band and thresholds."""
    h, f = fc.history, fc.forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pd.concat([f["date"], f["date"][::-1]]),
                             y=pd.concat([f["upper"], f["lower"][::-1]]),
                             fill="toself", fillcolor="rgba(232,113,10,0.15)",
                             line=dict(width=0), hoverinfo="skip",
                             name="95% prediction band"))
    fig.add_trace(go.Scatter(x=h["date"], y=h["disk_usage"], mode="lines",
                             name="Historical disk usage",
                             line=dict(color="#2a6fdb", width=1.6),
                             hovertemplate="%{x|%d %b %Y}: %{y:.1f}%<extra>history</extra>"))
    fig.add_trace(go.Scatter(x=h["date"], y=h["fitted"], mode="lines",
                             name="Linear Regression fit",
                             line=dict(color="#777", width=1, dash="dot"),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f["date"], y=f["forecast_disk_usage"], mode="lines",
                             name="Forecast (30 days)",
                             line=dict(color="#e8710a", width=2.4, dash="dash"),
                             hovertemplate="%{x|%d %b %Y}: %{y:.1f}%<extra>forecast</extra>"))
    fig.add_hline(y=warning, line=dict(color="#d98c00", dash="dash", width=1.4),
                  annotation_text=f"Warning {warning:.0f}%", annotation_position="top left")
    fig.add_hline(y=critical, line=dict(color=RED, dash="dot", width=1.4),
                  annotation_text=f"Critical {critical:.0f}%", annotation_position="top left")

    if fc.warning_date is not None and fc.warning_date <= f["date"].iloc[-1] \
            and fc.latest < warning:
        fig.add_trace(go.Scatter(
            x=[fc.warning_date], y=[warning], mode="markers+text",
            marker=dict(size=11, color="#d98c00"), name=f"Reaches {warning:.0f}%",
            text=[f"{fc.warning_date:%d %b} (in {fc.days_to_warning} days)"],
            textposition="top left",
            hovertemplate=f"Projected to reach {warning:.0f}%<br>%{{x|%d %b %Y}}<extra></extra>"))

    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", uirevision="forecast",
                      yaxis=dict(title="Disk usage (%)",
                                 range=[min(50, h["disk_usage"].min() - 3), 100]),
                      legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"))
    return fig
