"""
app.py
------
Task 3 - AI-Powered DevOps Monitoring Dashboard (Streamlit), interactive
and live.

Based on the Module 26 demo app (same four sections), now driven by the
anomaly_detector and forecasting_engine modules:

    1. Current System Health      - CPU, Memory, Disk (+ Network ...)
    2. AI-Based Anomaly Detection - interactive Isolation Forest chart
    3. Predictive Health Check    - interactive 30-day disk forecast (85% line)
    4. AI-Assisted Operational Recommendation - generated from both models

Two data sources (sidebar):
    * Simulated (course data) - the demo app's data plus injected anomalies,
      streaming a new 5-minute reading on every refresh
    * This computer (live)    - real CPU / memory / disk / network readings
      from the machine running the dashboard; the Isolation Forest learns
      this computer's normal behaviour from the first 30 readings

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from anomaly_detector import (FEATURES, AnomalyDetector, evaluate,
                              inject_anomalies, load_metrics)
from charts import anomaly_figure, forecast_figure
from data_generator import next_reading
from forecasting_engine import (CRITICAL_THRESHOLD, WARNING_THRESHOLD,
                                analyse_history, forecast_disk, load_storage)
from live_metrics import LIVE_FEATURES, SystemSampler, top_processes

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(page_title="AI DevOps Monitoring Dashboard", layout="wide")

st.title("AI-Powered Infrastructure Monitoring Dashboard")
st.caption("Anomaly detection (Isolation Forest), predictive capacity "
           "forecasting (Linear Regression) and operational recommendations")

NAMES = {"cpu_usage": "CPU", "memory_usage": "Memory", "network_mbps": "Network",
         "response_time_ms": "Response time", "error_rate_percent": "Error rate",
         "disk_usage": "Disk"}
UNITS = {"cpu_usage": "%", "memory_usage": "%", "network_mbps": " Mbps",
         "response_time_ms": " ms", "error_rate_percent": "%", "disk_usage": "%"}
RANK = {"NORMAL": 0, "WARNING": 1, "HIGH RISK": 2}
MIN_TRAIN = 30        # live mode: readings before the model first trains
SKIP_FIRST = 3        # live mode: start-up readings left out of training
RETRAIN_EVERY = 20    # live mode: retrain after this many new readings
MAX_ROWS = 2000       # keep memory use bounded during long live sessions
# Smallest change that counts as meaningful, per metric. Stops tiny wobbles
# on a very steady metric (e.g. memory fixed at 16.4%) being called incidents.
MIN_STD = {"cpu_usage": 2.0, "memory_usage": 1.0, "network_mbps": 0.5,
           "response_time_ms": 5.0, "error_rate_percent": 0.1, "disk_usage": 0.5}


# ---------------------------------------------------------
# Cached data and models (simulated source)
# ---------------------------------------------------------
@st.cache_data
def get_metrics() -> pd.DataFrame:
    return inject_anomalies(load_metrics())


@st.cache_resource
def train_detector(contamination: float) -> AnomalyDetector:
    return AnomalyDetector(contamination=contamination).fit(get_metrics())


@st.cache_data
def get_forecast(warning: float, critical: float):
    df = load_storage()
    return analyse_history(df), forecast_disk(df, warning=warning, critical=critical)


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.header("Dashboard Settings")
source = st.sidebar.radio(
    "Data source", ["Simulated (course data)", "This computer (live)"],
    help="Simulated: the demo app's data with injected anomalies (matches the "
         "report). This computer: real readings from the machine running the "
         "dashboard.")
REAL = source.startswith("This computer")

st.sidebar.subheader("Live feed")
live = st.sidebar.toggle("Live updates", value=True,
                         help="Add a new reading and refresh every section "
                              "automatically. Switch off to pause and zoom "
                              "into the charts.")
refresh = st.sidebar.slider("Refresh every (seconds)", 1, 10, 2 if REAL else 3)
window = st.sidebar.slider("Chart window (latest readings)", 30, 600,
                           150 if REAL else 200, step=10)
metric_options = LIVE_FEATURES + ["disk_usage"] if REAL else FEATURES
shown = st.sidebar.multiselect(
    "Metrics to chart", metric_options,
    default=["cpu_usage", "memory_usage", "network_mbps"],
    format_func=lambda m: NAMES[m]) or ["cpu_usage"]
if st.sidebar.button("Inject test spike", help="Adds one abnormal reading so "
                                               "you can watch the model flag it."):
    st.session_state.spike_pending = True
if st.sidebar.button("Reset data", help="Start the live data again from the beginning."):
    for k in ("real_data", "real_detector", "real_total", "real_trained_at",
              "sampler") if REAL else ("sim_data", "rng"):
        st.session_state.pop(k, None)

st.sidebar.subheader("Model and thresholds")
contamination = st.sidebar.slider(
    "Expected anomaly rate (contamination)", 0.01, 0.15, 0.03 if REAL else 0.08,
    0.01, help="Share of readings the Isolation Forest treats as anomalous. "
               "The demo app used 0.03.")
warning_threshold = st.sidebar.slider("Disk Warning Threshold (%)", 70, 95,
                                      int(WARNING_THRESHOLD))
critical_threshold = st.sidebar.slider("Disk Critical Threshold (%)", 80, 100,
                                       int(CRITICAL_THRESHOLD))
show_raw_data = st.sidebar.checkbox("Show raw data", value=False)

st.sidebar.caption("Charts: hover for values, drag to zoom, double-click to "
                   "reset, click a legend item to hide it. While live updates "
                   "are on, charts follow the newest readings; switch them off "
                   "to zoom and explore.")

storage_stats, fc = get_forecast(float(warning_threshold), float(critical_threshold))


# ---------------------------------------------------------
# Data feeds
# ---------------------------------------------------------
def sim_tick(spike: bool) -> None:
    """Append one simulated 5-minute reading (the demo app's distributions)."""
    if "sim_data" not in st.session_state:
        st.session_state.sim_data = get_metrics()
        st.session_state.rng = np.random.default_rng(2026)
        if not spike:
            return
    df = st.session_state.sim_data
    row = next_reading(df["timestamp"].iloc[-1] + pd.Timedelta(minutes=5),
                       st.session_state.rng, spike_chance=1.0 if spike else 0.08)
    row["injected"] = "test spike" if spike else "live"
    st.session_state.sim_data = pd.concat([df, pd.DataFrame([row])],
                                          ignore_index=True).tail(MAX_ROWS)


def real_tick(spike: bool) -> None:
    """Append one real reading from this computer."""
    if "sampler" not in st.session_state:
        st.session_state.sampler = SystemSampler()
        st.session_state.real_data = pd.DataFrame()
        st.session_state.real_total = 0
    row = st.session_state.sampler.read()
    # The first few readings include the dashboard's own start-up load
    row["injected"] = "start-up" if st.session_state.real_total < SKIP_FIRST else "live"
    if spike:
        # Abnormal on all three live metrics so it stands out on any machine
        past = st.session_state.real_data
        normal_past = past[past["injected"] != "test spike"] if len(past) else past
        peak_net = float(normal_past["network_mbps"].max()) if len(normal_past) else 1.0
        row.update(cpu_usage=99.0,
                   memory_usage=min(row["memory_usage"] + 25.0, 99.0),
                   network_mbps=round(max(peak_net * 5, 50.0), 3),
                   injected="test spike")
    st.session_state.real_data = pd.concat(
        [st.session_state.real_data, pd.DataFrame([row])],
        ignore_index=True).tail(MAX_ROWS)
    st.session_state.real_total += 1


def real_result() -> tuple[pd.DataFrame, AnomalyDetector | None]:
    """Score real readings; (re)train the model on this computer's data."""
    data = st.session_state.real_data
    total = st.session_state.real_total
    det = st.session_state.get("real_detector")
    stale = (det is None
             or total - st.session_state.get("real_trained_at", 0) >= RETRAIN_EVERY
             or st.session_state.get("real_contamination") != contamination)
    if (data["injected"] == "live").sum() >= MIN_TRAIN and stale:
        # Learn "normal" without the start-up readings and test spikes
        normal = data[data["injected"] == "live"]
        det = AnomalyDetector(contamination=contamination,
                              features=LIVE_FEATURES).fit(normal)
        # Range check alongside the model: Isolation Forest gives a reading
        # beyond the largest value it has seen the same score as the readings
        # at that edge, and it cannot split on a metric that never changed
        # while it was learning. So readings far outside the learned range
        # (3 x the normal variation beyond the 1st-99th percentile) are
        # flagged as well.
        spread = normal[LIVE_FEATURES].std().fillna(0).clip(
            lower=pd.Series(MIN_STD)[LIVE_FEATURES])
        det.range_low = normal[LIVE_FEATURES].quantile(0.01) - 3 * spread
        det.range_high = normal[LIVE_FEATURES].quantile(0.99) + 3 * spread
        st.session_state.update(real_detector=det, real_trained_at=total,
                                real_contamination=contamination)
    if det is None:
        out = data.copy()
        out["anomaly_score"], out["is_anomaly"] = 0.0, False
        out["main_metric"] = "cpu_usage"
        return out, None
    out = det.detect(data)
    breach = ((data[LIVE_FEATURES] > det.range_high)
              | (data[LIVE_FEATURES] < det.range_low)).any(axis=1)
    out["out_of_range"] = breach.to_numpy()
    out["is_anomaly"] = (out["is_anomaly"] | out["out_of_range"]) & (out["injected"] != "start-up")
    return out, det


# ---------------------------------------------------------
# Incidents and the generated recommendation
# ---------------------------------------------------------
def find_incidents(result: pd.DataFrame, features: list[str],
                   gap: pd.Timedelta) -> list[dict]:
    """Group detected readings that are close in time and describe them."""
    a = result[result["is_anomaly"]]
    if a.empty:
        return []
    med = result[features].median()
    std = result[features].std().fillna(0).clip(lower=pd.Series(MIN_STD)[features])
    overload_cols = ["cpu_usage", "response_time_ms", "error_rate_percent"]
    can_overload = set(overload_cols) <= set(features)
    z_all = (a[features] - med) / std
    spike = ((z_all[overload_cols] > 2.5).all(axis=1) if can_overload
             else z_all["cpu_usage"] > 2.5)
    # A new group starts after a time gap or when the pattern changes
    groups = ((a["timestamp"].diff() > gap) | (spike != spike.shift())).cumsum()
    incidents = []
    for _, g in a.groupby(groups):
        z = ((g[features] - med) / std).max()
        high = [m for m in features if z[m] > 2.5]
        if can_overload and set(overload_cols) <= set(high):
            kind = "Service overload spike"
        elif high == ["memory_usage"] or (len(g) >= 3 and z["memory_usage"] == z.max()):
            kind = "Memory growth (possible leak)"
        elif "cpu_usage" in high:
            kind = "CPU spike"
        elif "network_mbps" in high:
            kind = "Network burst"
        else:
            kind = "Unusual reading"
        incidents.append({
            "start": g["timestamp"].min(), "end": g["timestamp"].max(),
            "readings": len(g), "kind": kind, "times": set(g["timestamp"]),
            "detail": ", ".join(f"{NAMES[m]} {g[m].max():.0f}{UNITS[m]}"
                                for m in (high or [g["main_metric"].iloc[0]])),
        })
    return incidents


ACTIONS = {
    "Service overload spike": "Check the top CPU processes and recent deployments, "
                              "review load-balancer logs for the traffic surge, and "
                              "scale out if the load is genuine.",
    "Memory growth (possible leak)": "Capture a heap dump, restart the affected "
                                     "service in a quiet window, and raise a bug "
                                     "for the release that introduced it.",
    "CPU spike": "Find the process using the CPU (Task Manager on Windows, `top` "
                 "on Linux/macOS), check recent deployments or scheduled jobs, "
                 "and scale out if the load is genuine.",
    "Network burst": "Find what is sending the traffic (Resource Monitor > Network, "
                     "or load-balancer logs) and rate-limit it if it is not expected.",
}


def recommend(result: pd.DataFrame, incidents: list[dict], time_fmt: str,
              disk_now: float) -> tuple[str, list[str], list[str]]:
    """Health status, findings and actions generated from both models."""
    findings, actions = [], []

    # Anomaly detection
    significant = [i for i in incidents if i["kind"] != "Unusual reading"]
    anomaly_status = "WARNING" if significant else "NORMAL"
    latest_time = result["timestamp"].iloc[-1] if len(result) else None
    if latest_time is not None and any(latest_time in i["times"] for i in significant):
        anomaly_status = "HIGH RISK"
        findings.append(f"The latest reading ({result['timestamp'].iloc[-1]:{time_fmt}}) "
                        "is anomalous.")
        actions.append("Investigate the current reading now.")
    for inc in significant:
        span = f"{inc['start']:{time_fmt}}" + (
            f"-{inc['end']:{time_fmt}}" if inc["end"] != inc["start"] else "")
        findings.append(f"{inc['kind']} at {span} ({inc['detail']}).")
    for kind in dict.fromkeys(i["kind"] for i in significant):
        actions.append(ACTIONS[kind])
    if REAL and any(i["kind"].startswith("Memory") for i in significant):
        top = ", ".join(f"{n} {v:.0f}%" for n, v in top_processes())
        findings.append(f"Top memory users right now: {top}.")

    # Disk
    if REAL:
        # Live disk level against the thresholds (a 30-day forecast needs
        # days of history, so section 3 uses the disk dataset)
        disk_status = ("HIGH RISK" if disk_now >= critical_threshold else
                       "WARNING" if disk_now >= warning_threshold else "NORMAL")
        if disk_status == "NORMAL":
            findings.append(f"Disk is {disk_now:.1f}%, below the "
                            f"{warning_threshold}% warning threshold.")
        else:
            findings.append(f"Disk is {disk_now:.1f}%, above the "
                            f"{warning_threshold if disk_status == 'WARNING' else critical_threshold}% threshold.")
            actions.append("Free disk space now (empty temp files and Recycle Bin, "
                           "remove old logs and downloads) or add storage.")
    else:
        disk_status = fc.status
        if fc.status == "NORMAL":
            findings.append(f"Disk is {fc.latest:.1f}% and not expected to reach "
                            f"{warning_threshold}% within 30 days.")
        elif fc.latest >= warning_threshold:
            findings.append(f"Disk is {fc.latest:.1f}%, already above the "
                            f"{warning_threshold}% warning threshold.")
            actions.append("Free disk space now (clean old logs, archive unused "
                           "files) and increase disk capacity.")
        else:
            findings.append(
                f"Disk is {fc.latest:.1f}% and growing {fc.growth_per_day:.2f}% per day; "
                f"it reaches {warning_threshold}% on {fc.warning_date:%d %b %Y} "
                f"(in {fc.days_to_warning} days)"
                + (f", possibly as early as {fc.earliest_warning:%d %b}"
                   if fc.earliest_warning is not None else "") + ".")
            lead = max((fc.days_to_warning or 0) - 7, 1)
            actions.append(f"Plan disk capacity expansion within {lead} days; "
                           "meanwhile schedule log cleanup and archive unused files.")
            if fc.status == "HIGH RISK":
                actions.append("Critical threshold is inside the forecast window: "
                               "increase disk capacity now.")

    if not actions:
        actions = ["Continue monitoring.",
                   "Review dashboard trends during the next operations check."]
    status = max(anomaly_status, disk_status, key=RANK.get)
    return status, findings, actions



# ---------------------------------------------------------
# Section 4 visuals: risk score gauge and risk factors
# ---------------------------------------------------------
STATUS_BAND = {"NORMAL": (0, 39), "WARNING": (40, 69), "HIGH RISK": (70, 100)}
STATUS_COLOUR = {"NORMAL": "#2e9e6a", "WARNING": "#e8a33d", "HIGH RISK": "#e05252"}
FACTOR_POINTS = [            # (finding starts with / contains, points)
    ("The latest reading", 30),
    ("Service overload spike", 10),
    ("CPU spike", 10),
    ("Memory growth", 15),
    ("Network burst", 8),
]


def risk_assessment(status: str, findings: list[str]) -> tuple[int, list[tuple[str, int]]]:
    """Turn the recommendation's findings into scored risk factors.

    Each finding adds points; the total is kept inside the band of the
    health status (NORMAL 0-39, WARNING 40-69, HIGH RISK 70-100), so the
    gauge always agrees with the status.
    """
    factors = []
    for f in findings:
        points = next((p for key, p in FACTOR_POINTS if f.startswith(key)), 0)
        if f.startswith("Disk"):
            if "already above" in f or "above the" in f:
                points = 30
            elif "reaches" in f:
                points = 30 if fc.status == "HIGH RISK" else 20
        factors.append((f, points))
    low, high = STATUS_BAND[status]
    score = int(min(max(sum(p for _, p in factors), low), high))
    return score, factors


def risk_gauge(score: int, status: str) -> go.Figure:
    colour = STATUS_COLOUR[status]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"font": {"size": 46, "color": colour}},
        title={"text": f"<b>{status.title()}</b>", "font": {"size": 18, "color": colour}},
        gauge={"axis": {"range": [0, 100], "tickvals": [0, 40, 70, 100]},
               "bar": {"color": colour, "thickness": 0.3},
               "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0, 40], "color": "rgba(46,158,106,0.15)"},
                         {"range": [40, 70], "color": "rgba(232,163,61,0.18)"},
                         {"range": [70, 100], "color": "rgba(224,82,82,0.18)"}]}))
    fig.update_layout(height=250, margin=dict(l=25, r=25, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def risk_factors_html(factors: list[tuple[str, int]]) -> str:
    rows = []
    for text, points in factors:
        colour = ("#e05252" if points >= 20 else "#e8a33d" if points >= 10
                  else "#e8a33d" if points > 0 else "#2e9e6a")
        score = f"Score <b style='color:{colour}'>+{points}</b>" if points else \
                "<span style='opacity:0.6'>Score +0</span>"
        rows.append(
            "<div style='display:flex;justify-content:space-between;gap:16px;"
            "padding:7px 0;border-bottom:1px solid rgba(128,128,128,0.2)'>"
            f"<span><span style='color:{colour}'>&#9679;</span>&nbsp; {text}</span>"
            f"<span style='white-space:nowrap'>{score}</span></div>")
    return "".join(rows)


# ---------------------------------------------------------
# Section 3 (called inside the live fragment)
# ---------------------------------------------------------
def forecast_section() -> None:
    st.subheader("3. Predictive Health Check: Disk Usage Forecast")
    if REAL:
        st.info("A 30-day forecast needs days of disk history, so this section "
                "uses the 90-day disk dataset (data/storage_usage.csv). Your "
                "computer's live disk usage is shown in section 1.")
    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(forecast_figure(fc, warning_threshold, critical_threshold),
                        key="forecast_chart",
                        config={"displaylogo": False, "scrollZoom": True})
    with right:
        st.write("**Model Used:** Linear Regression")
        st.write(f"**Latest Disk Usage:** {fc.latest:.1f}%")
        st.write(f"**Estimated Growth Rate:** {fc.growth_per_day:.3f}% per day")
        st.write(f"**Model Fit:** R² {fc.r2:.3f}, mean error {fc.mae:.2f} points")
        st.write(f"**Forecast in 30 days:** {fc.forecast['forecast_disk_usage'].iloc[-1]:.1f}%")
        if fc.latest >= critical_threshold:
            st.error(f"Disk is already above the critical threshold ({critical_threshold}%).")
        elif fc.status == "HIGH RISK":
            st.error(f"Critical threshold ({critical_threshold}%) may be crossed on "
                     f"{fc.critical_date:%d %b %Y}.")
        elif fc.latest >= warning_threshold:
            st.warning(f"Disk is already above the warning threshold ({warning_threshold}%).")
        elif fc.status == "WARNING":
            st.warning(f"Warning threshold ({warning_threshold}%) may be crossed on "
                       f"{fc.warning_date:%d %b %Y} - in {fc.days_to_warning} days.")
        else:
            st.success("Disk usage is not expected to cross the configured "
                       "thresholds in the next 30 days.")


# ---------------------------------------------------------
# Sections 1-4: one fragment, so in live mode they refresh together and
# always describe the same latest reading
# ---------------------------------------------------------
@st.fragment(run_every=refresh if live else None)
def live_sections() -> None:
    spike = st.session_state.pop("spike_pending", False)

    if REAL:
        if live or spike or "sampler" not in st.session_state:
            real_tick(spike)
        result, det = real_result()
        features, time_fmt = LIVE_FEATURES, "%H:%M:%S"
        lookback, lookback_label = pd.Timedelta(minutes=1), "vs 1 min ago"
        gap = pd.Timedelta(seconds=max(10, 3 * refresh))
    else:
        if live or spike or "sim_data" not in st.session_state:
            sim_tick(spike)
        detector = train_detector(contamination)
        result, det = detector.detect(st.session_state.sim_data), detector
        features, time_fmt = FEATURES, "%H:%M"
        lookback, lookback_label = pd.Timedelta(hours=1), "vs 1h ago"
        gap = pd.Timedelta(minutes=15)

    view = result.tail(window)
    incidents = find_incidents(view, features, gap) if det is not None else []
    latest = view.iloc[-1]
    earlier = view[view["timestamp"] <= latest["timestamp"] - lookback]
    before = earlier.iloc[-1] if not earlier.empty else view.iloc[0]
    disk_now = float(latest["disk_usage"]) if REAL else fc.latest

    # 1. Current System Health
    st.subheader("1. Current System Health")
    st.caption(("Live readings from this computer" if REAL else "Simulated course data")
               + f" · latest reading {latest['timestamp']:%d %b %Y {time_fmt}}"
               + (f" · 🔴 live, every {refresh}s" if live else " · paused"))
    c = st.columns(6)
    c[0].metric("CPU Usage", f"{latest['cpu_usage']:.1f}%",
                f"{latest['cpu_usage'] - before['cpu_usage']:+.1f} {lookback_label}",
                delta_color="inverse")
    c[1].metric("Memory Usage", f"{latest['memory_usage']:.1f}%",
                f"{latest['memory_usage'] - before['memory_usage']:+.1f} {lookback_label}",
                delta_color="inverse")
    if REAL:
        c[2].metric("Disk Usage", f"{disk_now:.1f}%",
                    f"drive {st.session_state.sampler.disk_path}", delta_color="off")
        c[3].metric("Network (Mbps)", f"{latest['network_mbps']:.2f}")
        c[4].metric("Readings", st.session_state.real_total)
        c[5].metric("Model", "Learning" if det is None else "Active",
                    help=f"Learns from the first {MIN_TRAIN} readings, then "
                         f"retrains every {RETRAIN_EVERY} readings.")
    else:
        c[2].metric("Disk Usage", f"{fc.latest:.1f}%",
                    f"{fc.growth_per_day:+.2f} per day", delta_color="inverse")
        c[3].metric("Network", f"{latest['network_mbps']:.0f} Mbps")
        c[4].metric("Response Time", f"{latest['response_time_ms']:.0f} ms")
        c[5].metric("Error Rate", f"{latest['error_rate_percent']:.2f}%")

    # 2. AI-Based Anomaly Detection
    st.subheader("2. AI-Based Anomaly Detection")
    left, right = st.columns([2, 1])
    with left:
        leak = None
        if not REAL:
            lk = view.loc[view["injected"] == "Memory leak", "timestamp"]
            leak = (lk.min(), lk.max()) if not lk.empty else None
        minor = set().union(*[i["times"] for i in incidents
                              if i["kind"] == "Unusual reading"])
        st.plotly_chart(anomaly_figure(view, shown, features, leak, time_fmt,
                                       minor_times=minor, min_std=MIN_STD),
                        key="anomaly_chart",
                        config={"displaylogo": False, "scrollZoom": True})
    with right:
        st.write("**Model Used:** Isolation Forest")
        if REAL and det is None:
            n = int((st.session_state.real_data["injected"] == "live").sum())
            st.progress(min(n / MIN_TRAIN, 1.0),
                        text=f"Learning this computer's normal behaviour: "
                             f"{n}/{MIN_TRAIN} readings")
        significant = [i for i in incidents if i["kind"] != "Unusual reading"]
        n_minor = len(incidents) - len(significant)
        st.write(f"**Readings in view:** {len(view)}")
        st.write(f"**Anomalies Detected:** {int(view['is_anomaly'].sum())} readings: "
                 f"{len(significant)} incidents"
                 + (f" + {n_minor} minor flag{'s' if n_minor != 1 else ''}" if n_minor else ""))
        if significant:
            st.warning("Unusual infrastructure behaviour detected.")
        elif det is not None:
            st.success("No unusual behaviour detected.")
        if significant:
            st.dataframe(pd.DataFrame([{
                "Time": f"{i['start']:{time_fmt}}"
                        + (f"-{i['end']:{time_fmt}}" if i["end"] != i["start"] else ""),
                "Type": i["kind"]} for i in significant][::-1]),
                hide_index=True, height=200)
        if not REAL:
            ev = evaluate(result[~result["injected"].isin(["live", "test spike"])])
            st.caption(f"Check against injected anomalies: precision {ev['precision']:.0%}, "
                       f"recall {ev['recall']:.0%} (spikes "
                       f"{ev['detected_by_type'].get('Incident spike', '-')}, "
                       f"leak readings {ev['detected_by_type'].get('Memory leak', '-')}).")

    # 3. Predictive Health Check
    forecast_section()

    # 4. AI-Assisted Operational Recommendation
    st.subheader("4. AI-Assisted Operational Recommendation")
    status, findings, actions = recommend(view, incidents, time_fmt, disk_now)
    score, factors = risk_assessment(status, findings)
    g_col, f_col = st.columns([1, 1.4])
    with g_col:
        st.markdown("**Risk Assessment**")
        st.plotly_chart(risk_gauge(score, status), key="risk_gauge",
                        config={"displayModeBar": False})
    with f_col:
        st.markdown("**Risk Factors**")
        st.markdown(risk_factors_html(factors), unsafe_allow_html=True)
    st.markdown("**Recommended actions:**\n" + "\n".join(f"- {a}" for a in actions))
    st.session_state.latest_result = result


live_sections()


# ---------------------------------------------------------
# Optional raw data
# ---------------------------------------------------------
if show_raw_data:
    st.subheader("Raw Infrastructure Data")
    st.dataframe(st.session_state.latest_result)
    st.subheader("Disk Forecast Table")
    st.dataframe(fc.forecast.round({c: 2 for c in ["forecast_disk_usage", "lower", "upper"]}))
