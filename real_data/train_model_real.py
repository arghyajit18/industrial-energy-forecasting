"""
Real-data pipeline — Step 3: Forecasting model.

IMPORTANT modeling decision: this dataset's Lagging/Leading reactive power,
power factor, and CO2 columns are all measured AT THE SAME TIMESTAMP as
Usage_kWh (CO2 is in fact a near-deterministic function of Usage_kWh — see
correlation of 0.99 in EDA, so it's excluded entirely as target leakage).
A genuine *forecast* (predicting consumption ahead of time) can only use
information available in advance: time-of-day/week/month, and past usage
(lags/rolling mean). So the primary model below uses ONLY those features —
this is what the SAIL problem statement actually asks for ("forecast energy
consumption... enabling real-time property prediction").

A secondary same-timestep "estimation" model (using the simultaneous
electrical readings) is included separately purely to show how tightly
usage correlates with reactive power/power factor — it is explicitly NOT a
forecast and is labeled as such.
"""
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

df = pd.read_csv("real_data/clean_steel_industry_data.csv", parse_dates=["timestamp"])

FORECAST_FEATURES = [
    "hour", "dayofweek", "is_weekend", "month",
    "usage_lag_1step", "usage_lag_1day", "usage_roll_mean_1day",
]
ESTIMATION_ONLY_FEATURES = [
    "Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh",
    "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor",
]
target = "Usage_kWh"

split_idx = int(len(df) * 0.8)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]

def evaluate(features, label):
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    lr = LinearRegression().fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_mape = mean_absolute_percentage_error(np.maximum(y_test, 0.5), np.maximum(lr_pred, 0.5)) * 100

    xgb = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_mape = mean_absolute_percentage_error(np.maximum(y_test, 0.5), np.maximum(xgb_pred, 0.5)) * 100

    importance = pd.Series(xgb.feature_importances_, index=features).sort_values(ascending=False)
    print(f"\n=== {label} ===")
    print(f"Linear Regression: MAE={lr_mae:.2f} kWh, MAPE={lr_mape:.1f}%")
    print(f"XGBoost:           MAE={xgb_mae:.2f} kWh, MAPE={xgb_mape:.1f}%")
    print(importance)
    return xgb, xgb_pred, {"linear_regression": {"MAE_kWh": round(lr_mae, 2), "MAPE_pct": round(lr_mape, 1)},
                            "xgboost": {"MAE_kWh": round(xgb_mae, 2), "MAPE_pct": round(xgb_mape, 1)}}, importance

# --- Primary: genuine forecast (time + lag features only) ---
xgb_forecast, forecast_pred, forecast_metrics, forecast_importance = evaluate(
    FORECAST_FEATURES, "PRIMARY MODEL — genuine forecast (time + lag features only)")

# --- Secondary: same-timestep estimation (includes simultaneous electrical readings) ---
xgb_estimate, estimate_pred, estimate_metrics, estimate_importance = evaluate(
    FORECAST_FEATURES + ESTIMATION_ONLY_FEATURES,
    "SECONDARY MODEL — same-timestep estimation (NOT a forecast — uses simultaneous readings)")

# --- Save forecast results (this is the one that matters for the resume claim) ---
results = test[["timestamp", "Usage_kWh"]].copy()
results["predicted_kwh"] = forecast_pred
results["residual"] = results["Usage_kWh"] - results["predicted_kwh"]
results.to_csv("real_data/outputs/predictions.csv", index=False)

forecast_importance.to_csv("real_data/outputs/feature_importance.csv", header=["importance"])
joblib.dump(xgb_forecast, "real_data/outputs/xgb_forecast_model.joblib")

metrics = {
    "primary_forecast_model": forecast_metrics,
    "secondary_estimation_model_not_a_forecast": estimate_metrics,
    "train_set_size_rows": len(train),
    "test_set_size_rows": len(test),
    "note": "Primary model uses only time+lag features (genuine forecast). Secondary model "
            "uses simultaneous electrical readings for comparison only and should not be "
            "presented as a forecast.",
}
with open("real_data/outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved predictions.csv, feature_importance.csv, metrics.json, xgb_forecast_model.joblib")
