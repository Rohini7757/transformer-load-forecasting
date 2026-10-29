# AI-Based Power Transformer Load Forecasting and Overload Risk Prediction

A runnable internship capstone project with a time-series forecasting pipeline, overload-risk classifier, model evaluation, and Streamlit dashboard.

## Important data note
`data/transformer_load.csv` is a **simulated demonstration dataset**, not utility/field measurements. Replace it with approved historical transformer data before making operational decisions. The simulated data covers 180 days at hourly intervals and includes occasional high-load episodes.

## Features
- Data validation and time-aware feature engineering
- Lag, rolling, calendar, and temperature features
- Chronological train/test split to avoid time leakage
- Random Forest regression baseline and overload classifier
- MAE, RMSE, R², precision, recall, F1 and confusion matrix
- Actual-versus-predicted plot and feature importance
- Interactive Streamlit dashboard with forecast and risk alert

## Quick start
Use Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The training script saves models and evaluation artifacts under `artifacts/`. The dashboard will train models automatically if artifacts are missing.

## Use real data
Provide a CSV with at least:
- `timestamp`: date/time parseable by pandas
- `load_kva`: measured apparent load
- `capacity_kva`: rated transformer capacity
- Optional: `temperature_c`, `voltage_v`, `current_a`

Keep timestamps consistent, sort chronologically, and use a consistent sampling interval. Do not use future measurements as features. For a fair test, evaluate on a later time period not used for training.

## Project structure
- `data/transformer_load.csv` — simulated sample data
- `train_model.py` — preprocessing, feature engineering, training and evaluation
- `app.py` — Streamlit dashboard
- `report.md` — editable project report template with methodology and limitations
- `artifacts/` — generated at training time

## Safety
The default overload threshold is 90% of rated capacity (configurable). This is a demonstration threshold, not a universal engineering limit. Confirm alarm thresholds, thermal ratings, and operating procedures with the utility/qualified electrical engineer. This project is not a protection relay and must not be used to control equipment.
