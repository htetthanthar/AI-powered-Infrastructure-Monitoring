"""
anomaly_detector.py
-------------------
Task 1 - Infrastructure anomaly detection with an Isolation Forest.

Builds on the Module 26 demo app's anomaly detection (same data, same
injected incident spikes, StandardScaler + IsolationForest) and extends it:

    1. Load the telemetry dataset from CSV (CPU, Memory, Network, ...)
    2. Inject synthetic anomalies:
         - the demo app's incident spikes (CPU, memory, latency, errors)
           with a network surge added
         - a gradual memory leak (new)
    3. Add a 30-minute memory average so a slow leak can be detected
    4. Train the Isolation Forest and flag abnormal readings
    5. Check the result against the injected anomalies and plot it

Run directly to train, print the evaluation and save the chart:
    python anomaly_detector.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend; Streamlit draws the same figure
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from data_generator import METRICS_CSV, generate_infrastructure_data

# Metrics used as model inputs (the demo app's four + network)
FEATURES = ["cpu_usage", "memory_usage", "network_mbps",
            "response_time_ms", "error_rate_percent"]
SMOOTH_WINDOW = 6  # 6 x 5 min = 30 minutes

# Demo app's injected incident spikes (row positions and values)
SPIKE_INDICES = [40, 41, 42, 120, 121, 170]
SPIKE_VALUES = {
    "cpu_usage": [92, 95, 97, 88, 91, 96],
    "memory_usage": [85, 88, 90, 83, 87, 92],
    "network_mbps": [420, 460, 510, 400, 440, 530],   # added
    "response_time_ms": [500, 540, 580, 480, 520, 600],
    "error_rate_percent": [6, 7, 8, 5.5, 6.5, 9],
}
# Gradual memory leak (new): rows 140-164 (~2 hours), +0 -> +30 points,
# then the service restarts and memory returns to normal
LEAK_START, LEAK_LEN, LEAK_SIZE = 140, 25, 30.0


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
def load_metrics(path: Path = METRICS_CSV) -> pd.DataFrame:
    """Load telemetry CSV; generate it first if it is missing."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        generate_infrastructure_data().to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["timestamp"])


# --------------------------------------------------------------------------
# 2. Inject synthetic anomalies
# --------------------------------------------------------------------------
def inject_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Add the demo app's spikes plus a memory leak; label them."""
    df = df.copy()
    df["injected"] = "normal"

    for col, values in SPIKE_VALUES.items():
        df.loc[SPIKE_INDICES, col] = values
    df.loc[SPIKE_INDICES, "injected"] = "Incident spike"

    leak = list(range(LEAK_START, LEAK_START + LEAK_LEN))
    ramp = np.linspace(0, LEAK_SIZE, LEAK_LEN)
    df.loc[leak, "memory_usage"] = (df.loc[leak, "memory_usage"] + ramp).clip(0, 100).round(2)
    df.loc[leak, "injected"] = "Memory leak"
    return df


# --------------------------------------------------------------------------
# 3. Features
# --------------------------------------------------------------------------
def build_features(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """The metrics plus memory averaged over the last 6 readings
    (30 minutes for the 5-minute dataset).

    A spike is extreme at a single moment, so the raw values isolate it.
    A memory leak is a slow rise hidden in noisy readings (memory varies by
    about +/-6 points reading to reading); averaging the last 6 readings
    cuts that noise so the rise stands out sooner.
    """
    X = df[features or FEATURES].copy()
    X["memory_avg_30min"] = df["memory_usage"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
    return X


# --------------------------------------------------------------------------
# 4. Train + detect
# --------------------------------------------------------------------------
class AnomalyDetector:
    """StandardScaler + IsolationForest, as in the demo app.

    `features` defaults to the five dataset metrics; the dashboard's live
    mode for a real computer uses CPU, memory and network only.
    """

    def __init__(self, contamination: float = 0.08, n_estimators: int = 200,
                 random_state: int = 42, features: list[str] | None = None):
        self.features = list(features or FEATURES)
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=n_estimators,
                                     contamination=contamination,
                                     random_state=random_state)
        self.normal = None

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        self.model.fit(self.scaler.fit_transform(build_features(df, self.features)))
        self.normal = df[self.features].agg(["median", "std"])
        return self

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.scaler.transform(build_features(df, self.features))
        out = df.copy()
        out["anomaly_score"] = (-self.model.decision_function(X)).round(4)
        out["anomaly_prediction"] = self.model.predict(X)
        out["is_anomaly"] = out["anomaly_prediction"] == -1
        # Root-cause hint: the metric furthest from its normal level
        std = self.normal.loc["std"].replace(0, 1e-9)
        z = (df[self.features] - self.normal.loc["median"]) / std
        out["main_metric"] = z.abs().idxmax(axis=1)
        return out


def evaluate(result: pd.DataFrame) -> dict:
    """Precision / recall against the injected anomalies."""
    truth = result["injected"].isin(["Incident spike", "Memory leak"])
    pred = result["is_anomaly"]
    tp, fp, fn = int((truth & pred).sum()), int((~truth & pred).sum()), int((truth & ~pred).sum())
    by_type = {k: f"{int(g['is_anomaly'].sum())}/{len(g)}"
               for k, g in result[truth].groupby("injected")}
    leak = result[result["injected"] == "Memory leak"]
    first = leak.loc[leak["is_anomaly"], "timestamp"]
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "flagged": int(pred.sum()), "false_alarms": fp,
        "injected": int(truth.sum()), "detected_by_type": by_type,
        "leak_first_flag": first.min() if not first.empty else None,
        "leak_start": leak["timestamp"].min(),
    }


# --------------------------------------------------------------------------
# 5. Plot
# --------------------------------------------------------------------------
LABELS = {"cpu_usage": "CPU (%)", "memory_usage": "Memory (%)",
          "network_mbps": "Network (Mbps)"}
COLOURS = {"cpu_usage": "#2a6fdb", "memory_usage": "#2e9e6a",
           "network_mbps": "#8a5cd1"}


def plot_anomalies(result: pd.DataFrame, save_path: Path | None = None):
    """CPU, Memory and Network over time with detected anomalies marked.

    A detected reading is marked on a panel when that metric was unusual
    (more than 2.5 standard deviations from normal) or was the reading's
    main metric, so a memory leak is not shown as a CPU problem.
    """
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    anomalies = result[result["is_anomaly"]]
    med, std = result[FEATURES].median(), result[FEATURES].std()

    for ax, m in zip(axes, LABELS):
        ax.plot(result["timestamp"], result[m], color=COLOURS[m], lw=1.1)
        unusual = anomalies[((anomalies[m] - med[m]).abs() > 2.5 * std[m])
                            | (anomalies["main_metric"] == m)]
        ax.scatter(unusual["timestamp"], unusual[m], marker="x", s=55,
                   color="#d62728", lw=1.6, zorder=3)
        ax.set_ylabel(LABELS[m])
        ax.grid(alpha=0.3)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.08)

    # Shade the injected memory leak window for reference
    leak = result[result["injected"] == "Memory leak"]["timestamp"]
    if not leak.empty:
        for ax in axes:
            ax.axvspan(leak.min(), leak.max(), color="#e8a33d", alpha=0.14, lw=0)
        lo, _ = axes[1].get_ylim()
        axes[1].text(leak.min(), lo + 1.5, " injected memory leak",
                     fontsize=8, color="#9a5b00", va="bottom")

    handles = [plt.Line2D([], [], color="#d62728", marker="x", ls="", ms=7,
                          mew=1.6, label="Detected anomaly"),
               plt.Rectangle((0, 0), 1, 1, color="#e8a33d", alpha=0.25,
                             label="Injected memory leak window")]
    fig.legend(handles=handles, loc="upper right", ncol=2, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle("AI-based anomaly detection - Isolation Forest", x=0.01,
                 ha="left", y=0.995, fontsize=12)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    axes[-1].set_xlabel("Time")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def run_pipeline(contamination: float = 0.08):
    """Load -> inject -> train -> detect. Also used by app.py."""
    df = inject_anomalies(load_metrics())
    detector = AnomalyDetector(contamination=contamination).fit(df)
    return detector, detector.detect(df)


if __name__ == "__main__":
    _, result = run_pipeline()
    ev = evaluate(result)
    print("Isolation Forest results")
    print(f"  readings analysed  : {len(result)}")
    print(f"  injected anomalies : {ev['injected']}")
    print(f"  readings flagged   : {ev['flagged']} ({ev['false_alarms']} false alarms)")
    print(f"  precision          : {ev['precision']:.0%}")
    print(f"  recall             : {ev['recall']:.0%}")
    print(f"  detected by type   : {ev['detected_by_type']}")
    if ev["leak_first_flag"] is not None:
        lag = (ev["leak_first_flag"] - ev["leak_start"]).total_seconds() / 60
        print(f"  memory leak flagged: {ev['leak_first_flag']:%H:%M} ({lag:.0f} min after it began)")
    plot_anomalies(result, Path(__file__).parent / "screenshots" / "anomaly_detection_chart.png")
    print("Saved screenshots/anomaly_detection_chart.png")
