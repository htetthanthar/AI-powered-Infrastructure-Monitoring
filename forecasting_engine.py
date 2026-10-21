"""
forecasting_engine.py
---------------------
Task 2 - Capacity forecasting for disk storage with Linear Regression.

Builds on the Module 26 demo app's "predictive health check" (Linear
Regression on a time index, warning / critical thresholds, first crossing
time) and extends it to the brief:

    1. Load 90 days of daily disk utilisation from CSV
    2. Analyse the history (growth rate, variation)
    3. Train Linear Regression (day number -> disk %)
    4. Forecast the next 30 days (the demo forecast 24 x 5 minutes)
    5. Find when the 85% warning (and 95% critical) threshold is reached

Run directly to print the analysis and save the chart:
    python forecasting_engine.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from data_generator import STORAGE_CSV, generate_storage_data

WARNING_THRESHOLD = 85.0   # brief: 85% warning (demo app default was 90)
CRITICAL_THRESHOLD = 95.0
FORECAST_DAYS = 30


@dataclass
class Forecast:
    history: pd.DataFrame        # date, disk_usage, time_index, fitted
    forecast: pd.DataFrame       # date, time_index, forecast_disk_usage, lower, upper
    growth_per_day: float
    r2: float
    mae: float
    latest: float
    warning_date: pd.Timestamp | None     # trend reaches warning threshold
    critical_date: pd.Timestamp | None    # trend reaches critical threshold
    earliest_warning: pd.Timestamp | None # upper band reaches warning
    days_to_warning: int | None
    status: str                           # NORMAL / WARNING / HIGH RISK


# --------------------------------------------------------------------------
def load_storage(path: Path = STORAGE_CSV) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        generate_storage_data().to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["date"])


def analyse_history(df: pd.DataFrame) -> dict:
    s = df["disk_usage"]
    return {
        "days": len(df),
        "first_date": df["date"].iloc[0], "last_date": df["date"].iloc[-1],
        "start_pct": float(s.iloc[0]), "latest_pct": float(s.iloc[-1]),
        "min_pct": float(s.min()), "max_pct": float(s.max()),
        "avg_daily_change": float(s.diff().mean()),
        "daily_change_std": float(s.diff().std()),
        "last_30d_growth": float(s.iloc[-1] - s.iloc[-31]),
    }


def forecast_disk(df: pd.DataFrame, days: int = FORECAST_DAYS,
                  warning: float = WARNING_THRESHOLD,
                  critical: float = CRITICAL_THRESHOLD) -> Forecast:
    """Fit Linear Regression on the time index and forecast `days` ahead."""
    history = df.copy()
    history["time_index"] = np.arange(len(history))

    model = LinearRegression()
    model.fit(history[["time_index"]], history["disk_usage"])
    history["fitted"] = model.predict(history[["time_index"]])
    resid_std = float(np.std(history["disk_usage"] - history["fitted"], ddof=2))

    future_idx = np.arange(len(history), len(history) + days)
    future = pd.DataFrame({
        "date": pd.date_range(history["date"].iloc[-1] + pd.Timedelta(days=1),
                              periods=days, freq="D"),
        "time_index": future_idx,
    })
    future["forecast_disk_usage"] = model.predict(future[["time_index"]])
    future["lower"] = future["forecast_disk_usage"] - 1.96 * resid_std
    future["upper"] = future["forecast_disk_usage"] + 1.96 * resid_std

    slope, intercept = float(model.coef_[0]), float(model.intercept_)

    def crossing(level):
        """Date the trend line reaches `level` (can be beyond the horizon)."""
        if slope <= 0:
            return None
        day = int(np.ceil((level - intercept) / slope))
        return history["date"].iloc[0] + pd.Timedelta(days=day)

    warning_date, critical_date = crossing(warning), crossing(critical)
    hit = future.loc[future["upper"] >= warning, "date"]
    earliest = hit.iloc[0] if not hit.empty else None
    last = history["date"].iloc[-1]
    days_to_warning = max((warning_date - last).days, 0) if warning_date is not None else None

    # Same status rule as the demo app, applied to the 30-day forecast
    if (future["forecast_disk_usage"] >= critical).any():
        status = "HIGH RISK"
    elif (future["forecast_disk_usage"] >= warning).any():
        status = "WARNING"
    else:
        status = "NORMAL"

    y = history["disk_usage"]
    return Forecast(
        history=history, forecast=future, growth_per_day=slope,
        r2=float(r2_score(y, history["fitted"])),
        mae=float(mean_absolute_error(y, history["fitted"])),
        latest=float(y.iloc[-1]), warning_date=warning_date,
        critical_date=critical_date, earliest_warning=earliest,
        days_to_warning=days_to_warning, status=status,
    )


def plot_forecast(fc: Forecast, warning: float = WARNING_THRESHOLD,
                  critical: float = CRITICAL_THRESHOLD,
                  save_path: Path | None = None):
    """History, regression fit, 30-day forecast and threshold lines."""
    h, f = fc.history, fc.forecast
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(h["date"], h["disk_usage"], color="#2a6fdb", lw=1.4,
            label="Historical disk usage")
    ax.plot(h["date"], h["fitted"], color="#555", lw=1, ls=":",
            label="Linear Regression fit")
    ax.plot(f["date"], f["forecast_disk_usage"], color="#e8710a", lw=2,
            ls="--", label=f"Forecast disk usage ({len(f)} days)")
    ax.fill_between(f["date"], f["lower"], f["upper"], color="#e8710a",
                    alpha=0.15, label="95% prediction band")
    ax.axhline(warning, color="#d98c00", ls="--", lw=1.4,
               label=f"Warning threshold: {warning:.0f}%")
    ax.axhline(critical, color="#d62728", ls=":", lw=1.4,
               label=f"Critical threshold: {critical:.0f}%")
    ax.axvline(h["date"].iloc[-1], color="#999", lw=0.8)

    if fc.warning_date is not None and fc.warning_date <= f["date"].iloc[-1]:
        ax.scatter([fc.warning_date], [warning], color="#d98c00", s=60, zorder=4)
        ax.annotate(f"Projected to reach {warning:.0f}%\n"
                    f"{fc.warning_date:%d %b %Y} (in {fc.days_to_warning} days)",
                    xy=(fc.warning_date, warning), xytext=(-170, 28),
                    textcoords="offset points", fontsize=9, color="#8a5a00",
                    arrowprops=dict(arrowstyle="->", color="#8a5a00"))

    ax.set_title("Predictive health check: disk usage forecast (next 30 days)",
                 loc="left")
    ax.set_xlabel("Date")
    ax.set_ylabel("Disk usage (%)")
    ax.set_ylim(min(50, h["disk_usage"].min() - 3), 100)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def run_pipeline(warning: float = WARNING_THRESHOLD,
                 critical: float = CRITICAL_THRESHOLD):
    df = load_storage()
    return analyse_history(df), forecast_disk(df, warning=warning, critical=critical)


if __name__ == "__main__":
    stats, fc = run_pipeline()
    print("Historical storage analysis")
    print(f"  period            : {stats['first_date']:%d %b %Y} - {stats['last_date']:%d %b %Y} ({stats['days']} days)")
    print(f"  usage             : {stats['start_pct']:.1f}% -> {stats['latest_pct']:.1f}%")
    print(f"  avg daily change  : {stats['avg_daily_change']:+.3f} points (std {stats['daily_change_std']:.2f})")
    print(f"  last 30 days      : {stats['last_30d_growth']:+.1f} points")
    print("\nLinear Regression")
    print(f"  growth per day    : {fc.growth_per_day:.3f} points")
    print(f"  R^2 / MAE         : {fc.r2:.3f} / {fc.mae:.2f} points")
    print(f"  forecast day 30   : {fc.forecast['forecast_disk_usage'].iloc[-1]:.1f}% on {fc.forecast['date'].iloc[-1]:%d %b %Y}")
    print(f"  85% reached       : {fc.warning_date:%d %b %Y} ({fc.days_to_warning} days)")
    if fc.earliest_warning is not None:
        print(f"  85% earliest      : {fc.earliest_warning:%d %b %Y} (upper band)")
    print(f"  95% reached       : {fc.critical_date:%d %b %Y}")
    print(f"  health status     : {fc.status}")
    plot_forecast(fc, save_path=Path(__file__).parent / "screenshots" / "capacity_forecast_chart.png")
    print("Saved screenshots/capacity_forecast_chart.png")
