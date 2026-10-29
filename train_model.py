from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "transformer_load.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
OVERLOAD_THRESHOLD = 0.90
TEST_FRACTION = 0.20

def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path)
    required = {"timestamp", "load_kva", "capacity_kva"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "load_kva", "capacity_kva"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if (df["capacity_kva"] <= 0).any():
        raise ValueError("capacity_kva must be positive.")
    for col in ["temperature_c", "voltage_v", "current_a"]:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].interpolate(limit_direction="both")
        if df[col].isna().all():
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(df[col].median())
    df["load_pct"] = 100 * df["load_kva"] / df["capacity_kva"]
    return df

def make_features(df):
    out = df.copy()
    ts = out["timestamp"]
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    out["hour_sin"] = np.sin(2*np.pi*out["hour"]/24)
    out["hour_cos"] = np.cos(2*np.pi*out["hour"]/24)
    out["dow_sin"] = np.sin(2*np.pi*out["day_of_week"]/7)
    out["dow_cos"] = np.cos(2*np.pi*out["day_of_week"]/7)
    # Lag features use only observations at or before the prediction origin.
    for lag in [1, 2, 3, 24, 48, 168]:
        out[f"load_lag_{lag}"] = out["load_kva"].shift(lag)
    out["load_roll_mean_24"] = out["load_kva"].shift(1).rolling(24, min_periods=6).mean()
    out["load_roll_std_24"] = out["load_kva"].shift(1).rolling(24, min_periods=6).std()
    out["load_roll_mean_168"] = out["load_kva"].shift(1).rolling(168, min_periods=24).mean()
    return out

FEATURES = [
    "capacity_kva", "temperature_c", "voltage_v", "current_a",
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "load_lag_1", "load_lag_2", "load_lag_3",
    "load_lag_24", "load_lag_48", "load_lag_168",
    "load_roll_mean_24", "load_roll_std_24", "load_roll_mean_168"
]

def main():
    df = load_and_clean()
    feat = make_features(df).dropna(subset=FEATURES + ["load_kva"]).reset_index(drop=True)
    if len(feat) < 100:
        raise ValueError("Not enough rows after lag feature creation; provide more historical data.")
    split = int(len(feat) * (1 - TEST_FRACTION))
    train, test = feat.iloc[:split], feat.iloc[split:]
    X_train, X_test = train[FEATURES], test[FEATURES]
    y_train, y_test = train["load_kva"], test["load_kva"]

    reg = RandomForestRegressor(
        n_estimators=250, min_samples_leaf=2, max_features=0.9,
        random_state=42, n_jobs=-1
    )
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    metrics = {
        "model": "RandomForestRegressor",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "mae_kva": float(mean_absolute_error(y_test, pred)),
        "rmse_kva": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
        "overload_threshold_pct": OVERLOAD_THRESHOLD * 100
    }

    # Classification target is defined from observed load, independent of regression prediction.
    y_train_risk = (train["load_pct"] >= OVERLOAD_THRESHOLD * 100).astype(int)
    y_test_risk = (test["load_pct"] >= OVERLOAD_THRESHOLD * 100).astype(int)
    clf = RandomForestClassifier(
        n_estimators=250, min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train_risk)
    risk_pred = clf.predict(X_test)
    risk_metrics = {
        "precision": float(precision_score(y_test_risk, risk_pred, zero_division=0)),
        "recall": float(recall_score(y_test_risk, risk_pred, zero_division=0)),
        "f1": float(f1_score(y_test_risk, risk_pred, zero_division=0)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y_test_risk, risk_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test_risk, risk_pred, zero_division=0, output_dict=True)
    }
    metrics["risk_classifier"] = risk_metrics

    joblib.dump(reg, ARTIFACTS / "load_regressor.joblib")
    joblib.dump(clf, ARTIFACTS / "risk_classifier.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    test_results = pd.DataFrame({
        "timestamp": test["timestamp"].to_numpy(),
        "actual_load_kva": y_test.to_numpy(),
        "predicted_load_kva": pred,
        "actual_load_pct": test["load_pct"].to_numpy(),
        "predicted_risk": risk_pred,
        "actual_risk": y_test_risk.to_numpy()
    })
    test_results.to_csv(ARTIFACTS / "test_predictions.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(test_results["timestamp"], test_results["actual_load_kva"], label="Actual", linewidth=1.2)
    plt.plot(test_results["timestamp"], test_results["predicted_load_kva"], label="Predicted", linewidth=1.2)
    plt.title("Actual vs Predicted Transformer Load (Chronological Test Set)")
    plt.xlabel("Timestamp"); plt.ylabel("Load (kVA)"); plt.legend(); plt.tight_layout()
    plt.savefig(ARTIFACTS / "actual_vs_predicted.png", dpi=160); plt.close()

    importance = pd.Series(reg.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(9, 7))
    importance.tail(15).plot(kind="barh")
    plt.title("Top Feature Importances — Random Forest")
    plt.xlabel("Importance"); plt.tight_layout()
    plt.savefig(ARTIFACTS / "feature_importance.png", dpi=160); plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved models and reports to: {ARTIFACTS}")

if __name__ == "__main__":
    main()
