"""
data_generator.py
-----------------
Creates the two datasets used by the PredictiveAI project.

1. data/infra_metrics.csv
   200 readings at 5-minute intervals of CPU %, Memory %, Network Mbps,
   Response time (ms) and Error rate (%). CPU, Memory, Response time and
   Error rate are generated exactly as in the Module 26 demo app
   (np.random.seed(42), same distributions); Network Mbps is added because
   the assignment brief asks for CPU, Memory and Network metrics.

2. data/storage_usage.csv
   90 days of daily disk utilisation with a steady upward trend, used for
   the 30-day capacity forecast. (The demo app simulated disk usage per
   5 minutes; a 30-day forecast needs daily history.)

The data is clean here - synthetic anomalies are injected later by
anomaly_detector.py, as Task 1 requires.

Run directly to (re)create the CSV files:
    python data_generator.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
METRICS_CSV = DATA_DIR / "infra_metrics.csv"
STORAGE_CSV = DATA_DIR / "storage_usage.csv"

N_POINTS = 200
START = "2026-01-01 09:00"


def generate_infrastructure_data() -> pd.DataFrame:
    """Infrastructure telemetry, following the Module 26 demo app."""
    np.random.seed(42)

    timestamps = pd.date_range(start=START, periods=N_POINTS, freq="5min")

    # Same draws, in the same order, as the demo app
    cpu_usage = np.random.normal(loc=45, scale=8, size=N_POINTS)
    memory_usage = np.random.normal(loc=60, scale=6, size=N_POINTS)
    response_time = np.random.normal(loc=200, scale=30, size=N_POINTS)
    error_rate = np.random.normal(loc=1.5, scale=0.5, size=N_POINTS)

    # Network throughput (added for the brief), separate generator so the
    # demo app's values above stay identical
    net_rng = np.random.default_rng(7)
    network = net_rng.normal(loc=150, scale=20, size=N_POINTS)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "network_mbps": network,
        "response_time_ms": response_time,
        "error_rate_percent": error_rate,
    })

    df["cpu_usage"] = df["cpu_usage"].clip(0, 100)
    df["memory_usage"] = df["memory_usage"].clip(0, 100)
    df["network_mbps"] = df["network_mbps"].clip(0, None)
    df["response_time_ms"] = df["response_time_ms"].clip(50, None)
    df["error_rate_percent"] = df["error_rate_percent"].clip(0, None)
    num = df.columns.drop("timestamp")
    df[num] = df[num].round(2)
    return df


def generate_storage_data(days: int = 90, end: str = "2026-01-01") -> pd.DataFrame:
    """Daily disk utilisation: steady growth plus noise, as in the demo app
    (np.linspace trend + normal noise), but one reading per day."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=end, periods=days, freq="D")
    disk = np.linspace(55, 78, days) + rng.normal(0, 1.2, days)
    return pd.DataFrame({"date": dates,
                         "disk_usage": disk.clip(0, 100).round(2)})


def next_reading(ts: pd.Timestamp, rng: np.random.Generator,
                 spike_chance: float = 0.08) -> dict:
    """One new live reading for the dashboard's simulated live feed.

    Same distributions as generate_infrastructure_data(); occasionally a
    spike is added so live anomaly detection can be seen working.
    """
    row = {
        "timestamp": ts,
        "cpu_usage": float(np.clip(rng.normal(45, 8), 0, 100)),
        "memory_usage": float(np.clip(rng.normal(60, 6), 0, 100)),
        "network_mbps": float(max(rng.normal(150, 20), 0)),
        "response_time_ms": float(max(rng.normal(200, 30), 50)),
        "error_rate_percent": float(max(rng.normal(1.5, 0.5), 0)),
    }
    if rng.random() < spike_chance:
        row.update(cpu_usage=float(rng.uniform(88, 98)),
                   response_time_ms=float(rng.uniform(450, 600)),
                   error_rate_percent=float(rng.uniform(5, 9)))
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    generate_infrastructure_data().to_csv(METRICS_CSV, index=False)
    generate_storage_data().to_csv(STORAGE_CSV, index=False)
    print(f"Wrote {METRICS_CSV} and {STORAGE_CSV}")


if __name__ == "__main__":
    main()
