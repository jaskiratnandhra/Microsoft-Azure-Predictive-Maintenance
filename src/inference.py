# src/inference.py
import os
import pandas as pd
import numpy as np
from joblib import load

# ============================
# Paths
# ============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "Data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "pdm_model.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.joblib")

# ============================
# Load artifacts
# ============================
model = load(MODEL_PATH)
feature_columns = load(FEATURES_PATH)

# ============================
# Load raw data
# ============================
telemetry = pd.read_csv(os.path.join(DATA_DIR, "PdM_telemetry.csv"))
errors = pd.read_csv(os.path.join(DATA_DIR, "PdM_errors.csv"))
machines = pd.read_csv(os.path.join(DATA_DIR, "PdM_machines.csv"))

telemetry["datetime"] = pd.to_datetime(telemetry["datetime"])
errors["datetime"] = pd.to_datetime(errors["datetime"])

# ============================
# Feature generation (LATEST ROW ONLY)
# ============================
def build_features_for_machine(machine_id: int) -> pd.DataFrame:
    df = telemetry[telemetry["machineID"] == machine_id].copy()
    if df.empty:
        raise ValueError("Invalid machineID")

    df = df.sort_values("datetime")
    df = df.merge(machines, on="machineID", how="left")

    # ----------------------------
    # Error features
    # ----------------------------
    errs = errors[errors["machineID"] == machine_id]
    err_dummies = pd.get_dummies(errs["errorID"], prefix="err")
    errs = pd.concat([errs[["datetime"]], err_dummies], axis=1)

    errs = errs.groupby("datetime", as_index=False).sum()
    df = df.merge(errs, on="datetime", how="left")

    err_cols = [c for c in df.columns if c.startswith("err_")]
    df[err_cols] = df[err_cols].fillna(0)

    for lb in [24, 72]:
        df[f"errors_sum_{lb}h"] = (
            df[err_cols]
            .rolling(lb, min_periods=1)
            .sum()
            .sum(axis=1)
        )

    # ----------------------------
    # Rolling telemetry features
    # ----------------------------
    sensors = ["volt", "rotate", "pressure", "vibration"]
    for w in [3, 6, 12, 24]:
        for s in sensors:
            df[f"{s}_mean_{w}h"] = df[s].rolling(w, 1).mean()
            df[f"{s}_std_{w}h"] = df[s].rolling(w, 1).std().fillna(0)
            df[f"{s}_min_{w}h"] = df[s].rolling(w, 1).min()
            df[f"{s}_max_{w}h"] = df[s].rolling(w, 1).max()

    # Take latest timestamp only
    latest = df.iloc[-1:]

    # One-hot model column if needed
    if "model" in latest.columns:
        latest = pd.get_dummies(latest, columns=["model"], drop_first=True)

    # Align columns exactly with training
    X = latest.reindex(columns=feature_columns, fill_value=0)

    return X

# ============================
# Predict
# ============================
def predict_failure_risk(machine_id: int) -> float:
    X = build_features_for_machine(machine_id)
    risk = model.predict_proba(X)[0, 1]
    return float(risk)

# ============================
# CLI test
# ============================
if __name__ == "__main__":
    test_machine = 1
    risk = predict_failure_risk(test_machine)
    print(f"Machine {test_machine} failure risk (next 24h): {risk:.3f}")
