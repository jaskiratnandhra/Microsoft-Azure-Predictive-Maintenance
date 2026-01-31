# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from src.inference import predict_failure_risk, build_features_for_machine

# -------------------------
# App settings
# -------------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")

telemetry_path = os.path.join(DATA_DIR, "PdM_telemetry.csv")
machines_path = os.path.join(DATA_DIR, "PdM_machines.csv")

@st.cache_data
def load_data():
    telemetry = pd.read_csv(telemetry_path)
    machines = pd.read_csv(machines_path)
    telemetry["datetime"] = pd.to_datetime(telemetry["datetime"])
    return telemetry, machines

telemetry, machines = load_data()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Controls")
machine_ids = sorted(telemetry["machineID"].unique().tolist())
selected_machine = st.sidebar.selectbox("Select Machine ID", machine_ids)

# Optional: show last N hours
hours = st.sidebar.slider("Show last N hours of telemetry", min_value=24, max_value=720, value=168, step=24)

st.title("🔧 Predictive Maintenance (Next 24h Failure Risk)")

# -------------------------
# Risk prediction
# -------------------------
col1, col2, col3 = st.columns([1,1,2])

with col1:
    st.subheader("Machine Info")
    mrow = machines[machines["machineID"] == selected_machine]
    if not mrow.empty:
        st.write(f"**Model:** {mrow.iloc[0].get('model', 'N/A')}")
        st.write(f"**Age:** {mrow.iloc[0].get('age', 'N/A')}")
    else:
        st.write("No metadata found.")

with col2:
    st.subheader("Predicted Risk")
    risk = predict_failure_risk(int(selected_machine))

    # Color label
    if risk < 0.20:
        label = "LOW"
        st.success(f"Risk: {risk:.3f} ({label})")
    elif risk < 0.60:
        label = "MEDIUM"
        st.warning(f"Risk: {risk:.3f} ({label})")
    else:
        label = "HIGH"
        st.error(f"Risk: {risk:.3f} ({label})")

    st.caption("Risk = probability of failure in next 24 hours")

with col3:
    st.subheader("Recommended Action")
    if risk < 0.20:
        st.write("✅ Continue normal operations. Monitor standard telemetry.")
    elif risk < 0.60:
        st.write("🟡 Investigate: check recent error history and sensor trends.")
        st.write("Consider scheduling a maintenance inspection soon.")
    else:
        st.write("🔴 High risk: prioritize inspection and prepare for proactive maintenance.")
        st.write("Check components, error codes, and abnormal vibration/pressure.")

st.divider()

# -------------------------
# Telemetry charts
# -------------------------
st.subheader("📈 Telemetry Trends")

df_m = telemetry[telemetry["machineID"] == selected_machine].sort_values("datetime")

# last N hours
end_time = df_m["datetime"].max()
start_time = end_time - pd.Timedelta(hours=hours)
df_plot = df_m[df_m["datetime"] >= start_time]

sensor_cols = ["volt", "rotate", "pressure", "vibration"]

cA, cB = st.columns(2)

for i, col in enumerate(sensor_cols):
    ax_col = cA if i % 2 == 0 else cB
    with ax_col:
        fig = plt.figure()
        plt.plot(df_plot["datetime"], df_plot[col])
        plt.xticks(rotation=30)
        plt.title(col)
        plt.xlabel("datetime")
        plt.ylabel(col)
        st.pyplot(fig, clear_figure=True)

st.divider()

# -------------------------
# Feature snapshot (debug / explainability)
# -------------------------
with st.expander("Show latest engineered features used for prediction"):
    X = build_features_for_machine(int(selected_machine))
    st.dataframe(X.T.rename(columns={X.index[0]: "value"}))
