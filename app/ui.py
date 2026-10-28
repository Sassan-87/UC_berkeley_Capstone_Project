"""SECOM early-warning screen: a Streamlit front end that calls the model API.

Start the API first (python -m fastapi dev app/api.py), then, in a second terminal:
    python -m streamlit run app/ui.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"
APP_DIR = Path(__file__).parent

st.set_page_config(page_title="SECOM early-warning screen", layout="wide")
st.title("Early-warning screen for failing production runs")
st.caption("Prototype built on the SECOM capstone model. It flags runs for an engineer to review, "
           "it doesn't decide on its own whether a run is good or bad.")


# ---------- 1. is the API running? ----------
try:
    model_info = requests.get(API_URL, timeout=5).json()
except requests.exceptions.RequestException:
    st.error("The model API isn't running. Start it in a second terminal with "
             "`python -m fastapi dev app/api.py`, then refresh this page.")
    st.stop()
FEATURES = model_info["features"]


@st.cache_data
def load_reference():
    """Readings of the five key sensors in passing training runs (saved by the notebook)."""
    return pd.read_csv(APP_DIR / "reference_pass_runs.csv")


@st.cache_data
def score_runs(runs, threshold):
    """Send the runs to the API and get back a fail probability and an alarm for each one."""
    payload = {"runs": json.loads(runs[FEATURES].to_json(orient="records")),   # NaN becomes null
               "threshold": threshold}
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
    response.raise_for_status()
    return pd.DataFrame(response.json()["results"], index=runs.index)


# ---------- 2. choose the runs and the threshold ----------
st.sidebar.header("Settings")
source = st.sidebar.radio("Runs to screen", ["Sample: 314 test runs", "Upload a CSV file"])
if source == "Upload a CSV file":
    uploaded = st.sidebar.file_uploader("CSV with one row per run and the 274 model features", type="csv")
    if uploaded is None:
        st.info("Upload a CSV file in the sidebar, or switch back to the sample runs.")
        st.stop()
    runs = pd.read_csv(uploaded)
else:
    runs = pd.read_csv(APP_DIR / "sample_runs.csv")

missing = [f for f in FEATURES if f not in runs.columns]
if missing:
    st.error(f"The file is missing {len(missing)} of the model's features, for example {missing[:5]}.")
    st.stop()
if "run_id" not in runs.columns or not runs["run_id"].is_unique:
    runs["run_id"] = range(len(runs))   # give every run its own id
runs = runs.set_index("run_id")
has_result = "fail" in runs.columns     # the sample runs include the real result, an uploaded file may not

threshold = st.sidebar.slider(
    "Alarm threshold", min_value=0.05, max_value=0.40, value=round(model_info["threshold"], 2), step=0.01,
    help="A run raises an alarm when its fail probability is at or above this value. A lower threshold "
         "catches more fails but raises more false alarms. 0.17 was chosen in the analysis to catch "
         "about half of the fails.")

results = runs[["fail"]].join(score_runs(runs, threshold)) if has_result else score_runs(runs, threshold)
tab_batch, tab_run = st.tabs(["Screen a batch of runs", "Look at one run"])


# ---------- 3. tab 1: the whole batch ----------
with tab_batch:
    n_alarms = int(results["alarm"].sum())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Runs screened", len(results))
    col2.metric("Alarms raised", n_alarms)
    if has_result:
        caught = int((results["alarm"] & (results["fail"] == 1)).sum())
        col3.metric("Fails caught", f"{caught} of {int(results['fail'].sum())}")
        col4.metric("False alarms per caught fail", f"{(n_alarms - caught) / caught:.1f}" if caught else "-")

    # every run as a dot: fails to the right of the line are caught, passes to the right are false alarms
    plot_data = results.reset_index()
    plot_data["real result"] = plot_data["fail"].map({0: "Pass", 1: "Fail"}) if has_result else "Unknown"
    fig = px.strip(plot_data, x="fail_probability", y="real result", color="real result",
                   hover_data=["run_id"], color_discrete_map={"Pass": "#1f77b4", "Fail": "#d62728"},
                   labels={"fail_probability": "Fail probability from the model", "real result": "Real result"},
                   title="Each dot is a run. Runs right of the dashed line raise an alarm.")
    fig.add_vline(x=threshold, line_dash="dash", line_color="gray")
    fig.update_layout(showlegend=False, height=320)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Runs, highest risk first")
    only_alarms = st.checkbox("Show only runs with an alarm")
    table = results.sort_values("fail_probability", ascending=False).reset_index()
    if only_alarms:
        table = table[table["alarm"]]
    table["alarm"] = table["alarm"].map({True: "ALARM", False: ""})
    if has_result:
        table["fail"] = table["fail"].map({0: "pass", 1: "fail"})
    st.dataframe(table.rename(columns={"run_id": "Run", "fail_probability": "Fail probability",
                                       "alarm": "Alarm", "fail": "Real result"}),
                 column_config={"Fail probability": st.column_config.ProgressColumn(
                     format="%.2f", min_value=0.0, max_value=1.0)},
                 hide_index=True, width="stretch")


# ---------- 4. tab 2: one run and its five key sensors ----------
with tab_run:
    reference = load_reference()
    ranked = results.sort_values("fail_probability", ascending=False)
    run_id = st.selectbox(
        "Run (highest risk first)", ranked.index,
        format_func=lambda r: f"Run {r}: fail probability {ranked.loc[r, 'fail_probability']:.2f}"
                              + ("  (ALARM)" if ranked.loc[r, "alarm"] else ""))
    run = results.loc[run_id]

    col1, col2, col3 = st.columns(3)
    col1.metric("Fail probability", f"{run['fail_probability']:.2f}")
    col2.metric("Alarm", "Yes" if run["alarm"] else "No")
    if has_result:
        col3.metric("Real result", "Fail" if run["fail"] == 1 else "Pass")

    # where each key sensor sits compared with passing runs (share of passing runs with a lower reading)
    rows = []
    for sensor in reference.columns:
        reading = runs.loc[run_id, sensor]
        normal = reference[sensor].dropna()
        rows.append({"Sensor": sensor,
                     "Reading": reading,
                     "Typical range in passing runs": f"{normal.quantile(0.25):.4g} to {normal.quantile(0.75):.4g}",
                     "Higher than % of passing runs": round((normal < reading).mean() * 100) if pd.notna(reading) else None})
    sensor_table = pd.DataFrame(rows)

    fig = px.bar(sensor_table, x="Higher than % of passing runs", y="Sensor", orientation="h", range_x=[0, 100],
                 title=f"Run {run_id}: the five key sensors compared with passing runs")
    fig.add_vline(x=50, line_dash="dot", line_color="gray")    # a typical passing run
    fig.add_vline(x=90, line_dash="dash", line_color="red")    # unusually high
    fig.update_layout(height=320,
                      xaxis_title="Higher than % of passing runs (gray line = typical, red line = unusually high)",
                      yaxis={"categoryorder": "array", "categoryarray": sensor_table["Sensor"][::-1]})
    st.plotly_chart(fig, width="stretch")
    st.dataframe(sensor_table, hide_index=True, width="stretch")
    st.caption("These are the five sensors the analysis linked most strongly to fails, and in all five, failing "
               "runs read higher. A reading above 90% of passing runs is worth a look first. It's a pointer, "
               "not a diagnosis. A missing reading shows as empty and is filled with the training median by the model.")
