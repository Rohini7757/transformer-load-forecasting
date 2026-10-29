# Internship Capstone Report
## AI-Based Power Transformer Load Forecasting and Overload Risk Prediction

**Student:** [Your name]  
**Department / Institution:** [Fill in]  
**Internship organization:** [Fill in]  
**Guide:** [Fill in]  
**Date:** [Fill in]

## Abstract
This project develops a prototype machine-learning pipeline for estimating transformer load and identifying potential overload risk from historical time-stamped operating data. It combines data cleaning, calendar and lag feature engineering, a Random Forest regression model, a Random Forest classification model, chronological holdout evaluation, and a Streamlit dashboard. The included dataset is simulated for demonstration; field validation is required before any operational use.

## 1. Problem statement
Transformer loading varies with time, weather, and consumer demand. A useful forecasting tool can help operators review expected loading and investigate possible overload periods. The goal is to estimate future load in kVA and flag observations at or above a configurable percentage of rated capacity.

## 2. Objectives
1. Prepare and validate time-series transformer data.
2. Explore load patterns by hour, weekday, and temperature.
3. Engineer lag, rolling-statistic, and calendar features.
4. Train a load regression model and an overload-risk classifier.
5. Evaluate with MAE, RMSE, R², precision, recall, F1, and a confusion matrix.
6. Display current data, predictions, and alerts in a dashboard.

## 3. System architecture
CSV data → validation and cleaning → time-based feature engineering → chronological train/test split → regression and classification models → evaluation artifacts → Streamlit dashboard.

## 4. Dataset and data dictionary
The supplied `data/transformer_load.csv` contains 180 days of simulated hourly records.

| Field | Meaning | Unit |
|---|---|---|
| timestamp | Observation time | ISO date/time |
| load_kva | Apparent transformer load | kVA |
| capacity_kva | Rated capacity | kVA |
| temperature_c | Ambient temperature | °C |
| voltage_v | Line voltage (simulated) | V |
| current_a | Approximate line current (simulated) | A |

The simulated data includes noise and a few high-load episodes. It is not real utility data. Replace it with a documented, permissioned dataset for a real study.

## 5. Methodology
### Preprocessing
- Parse timestamps, remove invalid timestamps and duplicate times, sort chronologically.
- Validate positive rated capacity.
- Interpolate/fill missing optional sensor values.
- Calculate loading percentage as `100 × load_kva / capacity_kva`.

### Feature engineering
Calendar features include hour, day of week, month, weekend flag, and cyclical encodings. Historical features include 1-, 2-, 3-, 24-, 48-, and 168-hour lags plus 24-hour and 168-hour rolling means. Rolling statistics are shifted by one observation so the current target does not leak into its predictors.

### Models
- **Regression:** Random Forest Regressor estimates load in kVA.
- **Classification:** Random Forest Classifier predicts whether measured loading is at least 90% of rated capacity. The threshold is configurable and must be approved by a qualified engineer for real assets.

### Validation
The dataset is sorted by time and split chronologically: the earliest 80% is training data and the latest 20% is test data. Random shuffling is avoided to reduce time leakage.

## 6. Evaluation
Run `python train_model.py`. The script writes MAE, RMSE, R², precision, recall, F1, confusion matrix, and classification report to `artifacts/metrics.json`. It also saves `artifacts/test_predictions.csv`, `artifacts/actual_vs_predicted.png`, and `artifacts/feature_importance.png`.

**Insert measured results here after running the code. Do not report invented metrics.** Discuss whether the model underpredicts peak loads, and review recall carefully because missed overloads can be more consequential than false alarms.

## 7. Dashboard
Run `streamlit run app.py`. The dashboard displays recent load history, a model estimate, estimated overload probability, a threshold-based alert, and holdout evaluation plots/metrics. It supports uploading a CSV with the documented schema.

## 8. Limitations
- The provided data is synthetic and cannot establish real-world accuracy.
- The current dashboard estimate is a one-step model estimate; selecting a longer horizon does not yet recursively forecast multiple future timestamps.
- Sensor drift, irregular sampling, changing transformer ratings, maintenance events, and seasonal distribution shifts require additional handling.
- A classifier trained on a small number of overload events may have unstable recall.
- The 90% warning threshold is illustrative, not a universal engineering standard.

## 9. Future work
- Validate on real SCADA/AMI data with permission and data governance.
- Compare Random Forest against gradient boosting, XGBoost, and seasonal naïve baselines.
- Add walk-forward validation and prediction intervals.
- Implement true recursive or direct multi-horizon forecasts.
- Add model drift monitoring, alert logging, and role-based dashboard access.
- Validate alarm thresholds and procedures with a qualified power engineer.

## 10. Conclusion
The project provides an end-to-end educational prototype for transformer load estimation and overload-risk screening. Its workflow demonstrates preprocessing, time-aware features, supervised learning, evaluation, and dashboard reporting. Operational use requires real data, rigorous validation, and engineering review.

## References
1. Scikit-learn documentation: https://scikit-learn.org/stable/
2. Pandas documentation: https://pandas.pydata.org/docs/
3. Streamlit documentation: https://docs.streamlit.io/
