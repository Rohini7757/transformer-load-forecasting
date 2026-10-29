from pathlib import Path
import json
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from train_model import load_and_clean, make_features, FEATURES, OVERLOAD_THRESHOLD, ARTIFACTS, DATA_PATH

st.set_page_config(page_title="Transformer Load & Risk Monitor", page_icon="⚡", layout="wide")
st.title("⚡ AI-Based Transformer Load Forecasting")
st.caption("Educational prototype • validate with real utility data before operational use")

uploaded = st.sidebar.file_uploader("Upload historical CSV (optional)", type=["csv"])
threshold_pct = st.sidebar.slider("Overload warning threshold (% capacity)", 70, 110, 90, 1)
horizon = st.sidebar.slider("Forecast horizon (hours)", 1, 24, 6, 1)

@st.cache_data
def read_uploaded(file_bytes):
    import io
    return pd.read_csv(io.BytesIO(file_bytes))

if uploaded:
    raw = read_uploaded(uploaded.getvalue())
    tmp = DATA_PATH.parent / "uploaded_for_dashboard.csv"
    raw.to_csv(tmp, index=False)
    data_path = tmp
else:
    data_path = DATA_PATH

if not (ARTIFACTS / "load_regressor.joblib").exists() or not (ARTIFACTS / "risk_classifier.joblib").exists():
    with st.spinner("Training demonstration models for first run..."):
        subprocess.run([sys.executable, str(Path(__file__).with_name("train_model.py"))], check=True)

reg = joblib.load(ARTIFACTS / "load_regressor.joblib")
clf = joblib.load(ARTIFACTS / "risk_classifier.joblib")
df = load_and_clean(data_path)
feat = make_features(df).dropna(subset=FEATURES).reset_index(drop=True)

if feat.empty:
    st.error("Not enough data to create lag features. Provide at least 7 days of hourly data.")
    st.stop()

latest = feat.iloc[-1]
# For this compact demo, next-hour forecast is based on the latest available feature row.
# The horizon is displayed as a requested horizon, but recursively feeding predictions is not implemented.
predicted = float(reg.predict(feat[FEATURES].tail(1))[0])
predicted_pct = predicted / float(latest["capacity_kva"]) * 100
risk_prob = float(clf.predict_proba(feat[FEATURES].tail(1))[0][1]) if hasattr(clf, "predict_proba") else float(clf.predict(feat[FEATURES].tail(1))[0])
latest_load = float(latest["load_kva"])
capacity = float(latest["capacity_kva"])
actual_pct = latest_load / capacity * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest observed load", f"{latest_load:,.1f} kVA", f"{actual_pct:.1f}% capacity")
c2.metric("Model estimate", f"{predicted:,.1f} kVA", f"{predicted_pct:.1f}% capacity")
c3.metric("Rated capacity", f"{capacity:,.0f} kVA")
c4.metric("Estimated overload probability", f"{risk_prob*100:.1f}%")

if predicted_pct >= threshold_pct or risk_prob >= 0.5:
    st.error(f"Risk alert: estimated loading or classifier risk is elevated. Review operating conditions and confirm using approved procedures.")
else:
    st.success("No high-risk alert from this demonstration model at the selected threshold.")

st.subheader("Recent load history")
recent = df.tail(24*7)
st.line_chart(recent.set_index("timestamp")[["load_kva"]])
st.subheader("Prediction evaluation")
pred_path = ARTIFACTS / "test_predictions.csv"
if pred_path.exists():
    test = pd.read_csv(pred_path, parse_dates=["timestamp"])
    st.line_chart(test.set_index("timestamp")[["actual_load_kva", "predicted_load_kva"]])
    mpath = ARTIFACTS / "metrics.json"
    if mpath.exists():
        metrics = json.loads(mpath.read_text())
        st.write("Chronological holdout metrics")
        st.json({
            "MAE (kVA)": round(metrics["mae_kva"], 3),
            "RMSE (kVA)": round(metrics["rmse_kva"], 3),
            "R²": round(metrics["r2"], 3),
            "Risk precision": round(metrics["risk_classifier"]["precision"], 3),
            "Risk recall": round(metrics["risk_classifier"]["recall"], 3),
            "Risk F1": round(metrics["risk_classifier"]["f1"], 3)
        })
        with st.expander("Feature importance plot"):
            image = ARTIFACTS / "feature_importance.png"
            if image.exists(): st.image(str(image), use_container_width=True)
else:
    st.info("Run `python train_model.py` to generate evaluation plots and metrics.")

st.caption("Prototype only: the sample dataset is synthetic. Forecast horizon is currently a one-step model estimate; recursive multi-step forecasting is future work.")
