# src/train.py
import os
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# ============================
# Paths
# ============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "Data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================
# Config
# ============================
HORIZON_HOURS = 24
ROLL_WINDOWS = [3, 6, 12, 24]
ERROR_LOOKBACK = [24, 72]

# ============================
# Helpers
# ============================
def load_csv(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def assert_columns(df, required, fname):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{fname} missing columns {missing}")

# ============================
# Build Dataset
# ============================
def build_master_table():
    telemetry = load_csv("PdM_telemetry.csv")
    errors = load_csv("PdM_errors.csv")
    failures = load_csv("PdM_failures.csv")
    machines = pd.read_csv(os.path.join(DATA_DIR, "PdM_machines.csv"))

    assert_columns(
        telemetry,
        ["machineID", "datetime", "volt", "rotate", "pressure", "vibration"],
        "PdM_telemetry.csv",
    )

    df = telemetry.sort_values(["machineID", "datetime"]).copy()
    df = df.merge(machines, on="machineID", how="left")

    # ============================
    # Error features
    # ============================
    errors_small = errors[["machineID", "datetime", "errorID"]].copy()
    err_dummies = pd.get_dummies(errors_small["errorID"], prefix="err")
    errors_small = pd.concat(
        [errors_small[["machineID", "datetime"]], err_dummies], axis=1
    )

    errors_hourly = (
        errors_small.groupby(["machineID", "datetime"], as_index=False).sum()
    )

    df = df.merge(errors_hourly, on=["machineID", "datetime"], how="left")
    err_cols = [c for c in df.columns if c.startswith("err_")]
    df[err_cols] = df[err_cols].fillna(0)

    for lb in ERROR_LOOKBACK:
        df[f"errors_sum_{lb}h"] = (
            df.groupby("machineID")[err_cols]
            .rolling(lb, min_periods=1)
            .sum()
            .sum(axis=1)
            .reset_index(level=0, drop=True)
        )

    # ============================
    # Telemetry rolling features
    # ============================
    sensors = ["volt", "rotate", "pressure", "vibration"]

    for w in ROLL_WINDOWS:
        for s in sensors:
            g = df.groupby("machineID")[s]
            df[f"{s}_mean_{w}h"] = g.rolling(w, 1).mean().reset_index(0, drop=True)
            df[f"{s}_std_{w}h"] = (
                g.rolling(w, 1).std().reset_index(0, drop=True).fillna(0)
            )
            df[f"{s}_min_{w}h"] = g.rolling(w, 1).min().reset_index(0, drop=True)
            df[f"{s}_max_{w}h"] = g.rolling(w, 1).max().reset_index(0, drop=True)

    # ============================
    # Label: failure in next 24h (FIXED)
    # ============================
    df["label_next_24h"] = 0
    fail_times = failures.groupby("machineID")["datetime"].apply(list).to_dict()

    dt = df["datetime"].values.astype("datetime64[ns]")
    dt_h = (df["datetime"] + pd.Timedelta(hours=HORIZON_HOURS)).values.astype(
        "datetime64[ns]"
    )

    labels = np.zeros(len(df), dtype=np.int8)

    for mid, idx in df.groupby("machineID").groups.items():
        ft = fail_times.get(mid, [])
        if not ft:
            continue

        ft = np.array(pd.to_datetime(ft).values.astype("datetime64[ns]"))
        ft.sort()

        t = dt[idx]
        th = dt_h[idx]

        pos = np.searchsorted(ft, t, side="right")

        has_future = np.zeros(len(pos), dtype=bool)
        valid = pos < len(ft)

        has_future[valid] = ft[pos[valid]] <= th[valid]
        labels[idx] = has_future.astype(np.int8)

    df["label_next_24h"] = labels
    return df

# ============================
# Train
# ============================
def train_model():
    df = build_master_table()

    if "model" in df.columns:
        df = pd.get_dummies(df, columns=["model"], drop_first=True)

    target = "label_next_24h"
    drop_cols = ["machineID", "datetime", target]

    cutoff = pd.Timestamp("2015-10-01")
    train = df[df["datetime"] < cutoff]
    test = df[df["datetime"] >= cutoff]

    X_train = train.drop(columns=drop_cols)
    y_train = train[target].astype(int)
    X_test = test.drop(columns=drop_cols)
    y_test = test[target].astype(int)

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=10,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    print(f"\n✅ ROC-AUC: {auc:.4f}\n")
    print(classification_report(y_test, (proba >= 0.5).astype(int), digits=4))

    dump(model, os.path.join(MODEL_DIR, "pdm_model.joblib"))
    dump(list(X_train.columns), os.path.join(MODEL_DIR, "feature_columns.joblib"))

    print("\n✅ Model saved to /models")

if __name__ == "__main__":
    train_model()
