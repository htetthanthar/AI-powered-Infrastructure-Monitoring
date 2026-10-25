# AI-Powered Infrastructure Monitoring

Research Project – DevOps and Cloud with AI.

This project extends the **Module 26 AI-Powered Infrastructure Monitoring demo app** into the solution the assignment asks for:

1. **Anomaly detection** – an **Isolation Forest** (scikit-learn) finds abnormal CPU, Memory and Network behaviour, including injected incident spikes and a gradual memory leak.
2. **Capacity forecasting** – **Linear Regression** forecasts disk utilisation for the **next 30 days** and shows when the **85% warning threshold** will be reached.
3. **Interactive live Streamlit dashboard** – current CPU / Memory / Disk utilisation, interactive (Plotly) anomaly and forecast charts, and an AI-generated operational recommendation, all updating live. Data can come from the simulated course dataset or from **your own computer** in real time.

---

## Project structure

```
ai-powered-infrastructure-monitoring/
├── app.py                    # Streamlit dashboard (Task 3)
├── anomaly_detector.py       # Isolation Forest anomaly detection (Task 1)
├── forecasting_engine.py     # Linear Regression 30-day disk forecast (Task 2)
├── data_generator.py         # Creates the two datasets (based on the demo app's data)
├── live_metrics.py           # Real CPU / memory / disk / network readings from this computer (psutil)
├── charts.py                 # Interactive Plotly charts for the dashboard
├── requirements.txt
├── README.md
├── data/
│   ├── infra_metrics.csv     # 200 readings, every 5 minutes
│   └── storage_usage.csv     # 90 days of daily disk usage
├── diagrams/                 # architecture, ML, data, CI/CD and frontend/backend diagrams (PNG + SVG)
└── screenshots/
    ├── system_architecture.png
    ├── ml_pipeline.png
    ├── data_pipeline.png
    ├── frontend_backend_data.png
    ├── fullstack_workflow.png
    ├── cicd_pipeline.png
    ├── anomaly_detection_chart.png
    ├── capacity_forecast_chart.png
    ├── streamlit_dashboard_1.png
    ├── streamlit_dashboard_2.png
    ├── streamlit_dashboard_3.png
    └── solution_architecture.png
```

---

## Live demo

The dashboard can be opened online (deployed on Streamlit Community Cloud): **link to be added after deployment**

In the online version, **"This computer (live)" shows the cloud server's own CPU, memory, disk and network**, not your computer. To monitor your own computer, run the project locally (below).

---

## Requirements

- Python **3.9 or newer**
- streamlit, pandas, numpy, scikit-learn, matplotlib, plotly, psutil (see `requirements.txt`)

Tested on Python 3.9 and on the Module 26 environment (Python 3.14, Streamlit 1.58, pandas 3.0, scikit-learn 1.9).

---

## Installation

```bash
# 1. Get the code and open a terminal in the project folder
git clone https://github.com/htetthanthar/ai-powered-infrastructure-monitoring.git
cd ai-powered-infrastructure-monitoring

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate             # Windows
# source .venv/bin/activate        # macOS / Linux

# 3. Install the dependencies
pip install -r requirements.txt
```

---

## Running the dashboard

```bash
streamlit run app.py
```

The dashboard opens in your browser at **http://localhost:8501**.

Sidebar controls:

| Control | What it does |
|---|---|
| Data source | **Simulated (course data)** – the demo app's data with injected anomalies (matches the report). **This computer (live)** – real readings from the machine running the dashboard |
| Live updates | On: a new reading arrives and every section refreshes automatically. Off: pause, so you can zoom into the charts |
| Refresh every (seconds) | How often a new reading is added (1–10 s) |
| Chart window | How many of the latest readings the anomaly chart shows |
| Metrics to chart | Choose which metrics get a panel in the anomaly chart |
| Inject test spike | Adds one abnormal reading so you can watch the model flag it |
| Reset data | Starts the data for the current source again |
| Expected anomaly rate | Isolation Forest `contamination` (default 0.08 simulated, 0.03 live) |
| Disk Warning / Critical Threshold (%) | Default **85** / 95 |
| Show raw data | Shows the scored readings and the 30-day forecast table |

**Using the charts:** hover for exact values, drag to zoom, double-click to reset, click a legend item to hide it. While live updates are on, the charts follow the newest readings; switch **Live updates** off to zoom and explore.

### "This computer (live)" mode

- Reads CPU %, memory %, disk % (system drive, e.g. `C:\`) and network Mbps from your computer with `psutil`, every refresh.
- For the first **30 readings** (about a minute at 2 s) the dashboard shows "Learning this computer's normal behaviour". The Isolation Forest then trains on your machine's own readings (CPU, memory, network) and retrains every 20 readings.
- Try it: open a few heavy apps or browser tabs, or click **Inject test spike**, and watch the reading get flagged and the recommendation change.
- A **range check** runs beside the model: a reading far outside the range the model learned (3 × the normal variation beyond the 1st–99th percentile) is flagged too. Isolation Forest alone scores readings beyond its training range like the edge of that range, and cannot see changes in a metric that never moved while it was learning.
- Small wobbles on a very steady metric are shown as hollow **minor flags** and do not change the health status.
- Response time and error rate come from an application, not the operating system, so they appear only in simulated mode. The disk forecast (section 3) always uses the 90-day disk dataset, because a 30-day forecast needs days of history.

---

## Running each part on its own (optional)

```bash
python data_generator.py        # re-create the CSV files in data/
python anomaly_detector.py      # train the Isolation Forest, print results, save the anomaly chart
python forecasting_engine.py    # train Linear Regression, print the forecast, save the forecast chart
```

If the `data/` folder is missing, the scripts create the datasets automatically.

---

## How it works

### Task 1 – Anomaly detection (`anomaly_detector.py`)

- **Data:** 200 readings at 5-minute intervals (1 Jan 2026 09:00 – 2 Jan 01:35). CPU, Memory, Response time and Error rate are generated exactly as in the demo app (`np.random.seed(42)`); Network Mbps is added because the brief asks for it.
- **Synthetic anomalies:**
  - The demo app's **incident spikes** at 12:20–12:30, 19:00–19:05 and 23:10 (CPU 88–97%, memory 83–92%, response time 480–600 ms, errors 5.5–9%), now with a network surge (400–530 Mbps).
  - A new **memory leak** from 20:40 to 22:40 (memory climbs by up to 30 points, then the service restarts).
- **Model:** `StandardScaler` + `IsolationForest(n_estimators=200, contamination=0.08)` on the five metrics plus a 30-minute memory average (to see the slow leak through noisy readings).
- **Result:** all 6 spike readings detected; the leak flagged at 22:00 (80 min in, while still rising); precision 81%, recall 42%, 3 false alarms.

### Task 2 – Capacity forecasting (`forecasting_engine.py`)

- **Data:** 90 days of daily disk usage (4 Oct 2025 – 1 Jan 2026), 55.4% → 78.5%.
- **Model:** `LinearRegression` on the day number. R² = 0.982, mean error 0.74 points, growth 0.257 points/day.
- **Forecast:** 85.7% in 30 days; the trend reaches **85% on 29 Jan 2026 (in 28 days)**, possibly as early as 22 Jan.

### Task 3 – Dashboard (`app.py`, `charts.py`, `live_metrics.py`)

Same four sections as the demo app: Current System Health, AI-Based Anomaly Detection, Predictive Health Check, and AI-Assisted Operational Recommendation. All four refresh together on every live update, so they always describe the same latest reading. The recommendation is generated from both models: each detected incident is labelled (service overload spike, CPU spike, network burst or memory growth) and matched to actions, and the disk forecast (or, in live mode, your current disk level) sets the capacity action. Health status is NORMAL, WARNING or HIGH RISK.

### Changes from the demo app

| Demo app | This project |
|---|---|
| Single `app.py` | Separate `anomaly_detector.py`, `forecasting_engine.py`, `data_generator.py`, `app.py` |
| Data generated in memory | Datasets saved as CSV and loaded |
| No network metric | Network Mbps added |
| Spikes only | Spikes plus a gradual memory leak |
| Forecast 24 × 5 minutes (2 hours) | Forecast 30 days of daily disk usage |
| Warning 90%, critical 95% | Warning **85%**, critical 95% |
| Fixed recommendation text by disk status | Recommendation generated from detected incidents and the forecast |
| No check of detection quality | Precision / recall against the injected anomalies |
| Static matplotlib charts | Interactive Plotly charts (hover, zoom, pan) |
| No live data | Live stream of simulated readings, or real readings from your own computer |
| No `requirements.txt` / README | Both included |
