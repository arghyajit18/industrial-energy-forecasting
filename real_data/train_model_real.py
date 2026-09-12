"""
Real-data pipeline — Step 3: Forecasting model.

Primary model uses only time + lag features (genuine forecast).
Secondary same-timestep estimation uses simultaneous electrical readings
for comparison only and must not be presented as a forecast.
Matches the synthetic pipeline: GridSearch-tuned XGBoost, LightGBM
comparison, R2, SHAP, residual diagnostics, Fourier + rolling features.
"""
import pandas as pd
import numpy as np
import json
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

df = pd.read_csv("real_data/clean_steel_industry_data.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# --- Richer features to match synthetic pipeline (computed here so no re-clean needed) ---
if "minute" not in df.columns:
    if "NSM" in df.columns:
        df["minute"] = (df["NSM"] % 3600) // 60
    else:
        df["minute"] = df["timestamp"].dt.minute
if "day" not in df.columns:
    df["day"] = df["timestamp"].dt.day
if "year" not in df.columns:
    df["year"] = df["timestamp"].dt.year
if "is_weekday" not in df.columns:
    df["is_weekday"] = (df["dayofweek"] < 5).astype(int)

decimal_hour = df["hour"] + df["minute"] / 60
df["hour_sin"] = np.sin(2 * np.pi * decimal_hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * decimal_hour / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

if "usage_roll_std_1day" not in df.columns:
    df["usage_roll_std_1day"] = df["Usage_kWh"].rolling(96, min_periods=1).std().fillna(0)

FORECAST_FEATURES = [
    "hour", "minute", "dayofweek", "is_weekend", "is_weekday", "month", "day", "year",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "usage_lag_1step", "usage_lag_1day", "usage_roll_mean_1day", "usage_roll_std_1day",
]
FORECAST_FEATURES = [c for c in FORECAST_FEATURES if c in df.columns]
ESTIMATION_ONLY_FEATURES = [
    "Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh",
    "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor",
]
target = "Usage_kWh"

split_idx = int(len(df) * 0.8)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]

param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}


def evaluate(features, label, tune=True):
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    lr = LinearRegression().fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_mape = mean_absolute_percentage_error(np.maximum(y_test, 0.5), np.maximum(lr_pred, 0.5)) * 100
    lr_r2 = r2_score(y_test, lr_pred)

    if tune:
        grid = GridSearchCV(
            XGBRegressor(random_state=42),
            param_grid, cv=3, scoring="neg_mean_absolute_error",
            verbose=0, n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        xgb = grid.best_estimator_
        best_params = grid.best_params_
        print(f"GridSearch best params ({label}): {best_params}")
        print(f"GridSearch best MAE: {-grid.best_score_:.2f} kWh")
    else:
        best_params = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
                       "subsample": 0.8, "colsample_bytree": 0.8}
        xgb = XGBRegressor(**best_params, random_state=42)
        xgb.fit(X_train, y_train)

    xgb_pred = xgb.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_mape = mean_absolute_percentage_error(np.maximum(y_test, 0.5), np.maximum(xgb_pred, 0.5)) * 100
    xgb_r2 = r2_score(y_test, xgb_pred)

    lgb = LGBMRegressor(
        n_estimators=300, max_depth=-1, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1,
    )
    lgb.fit(X_train, y_train)
    lgb_pred = lgb.predict(X_test)
    lgb_mae = mean_absolute_error(y_test, lgb_pred)
    lgb_mape = mean_absolute_percentage_error(np.maximum(y_test, 0.5), np.maximum(lgb_pred, 0.5)) * 100
    lgb_r2 = r2_score(y_test, lgb_pred)

    importance = pd.Series(xgb.feature_importances_, index=features).sort_values(ascending=False)
    print(f"\n=== {label} ===")
    print(f"Linear Regression: MAE={lr_mae:.2f} kWh, MAPE={lr_mape:.1f}%, R2={lr_r2:.4f}")
    print(f"XGBoost:           MAE={xgb_mae:.2f} kWh, MAPE={xgb_mape:.1f}%, R2={xgb_r2:.4f}")
    print(f"LightGBM:          MAE={lgb_mae:.2f} kWh, MAPE={lgb_mape:.1f}%, R2={lgb_r2:.4f}")
    print(importance)
    return (xgb, lgb, xgb_pred, lgb_pred, lr_pred,
            {"linear_regression": {"MAE_kWh": round(lr_mae, 2), "MAPE_pct": round(lr_mape, 1), "R2": round(lr_r2, 4)},
             "xgboost": {"MAE_kWh": round(xgb_mae, 2), "MAPE_pct": round(xgb_mape, 1), "R2": round(xgb_r2, 4)},
             "lightgbm": {"MAE_kWh": round(lgb_mae, 2), "MAPE_pct": round(lgb_mape, 1), "R2": round(lgb_r2, 4)},
             "grid_search_best_params": best_params},
            importance)


# --- Primary: genuine forecast (time + lag features only) ---
(xgb_forecast, lgb_forecast, forecast_pred, lgb_forecast_pred, lr_forecast_pred,
 forecast_metrics, forecast_importance) = evaluate(
    FORECAST_FEATURES, "PRIMARY MODEL - genuine forecast (time + lag features only)", tune=True)

# --- Secondary: same-timestep estimation (includes simultaneous electrical readings) ---
(xgb_estimate, _, estimate_pred, _, _,
 estimate_metrics, estimate_importance) = evaluate(
    FORECAST_FEATURES + ESTIMATION_ONLY_FEATURES,
    "SECONDARY MODEL - same-timestep estimation (NOT a forecast)", tune=False)

# --- SHAP values (primary XGBoost) ---
explainer = shap.TreeExplainer(xgb_forecast, feature_perturbation="tree_path_dependent")
shap_values = explainer.shap_values(test[FORECAST_FEATURES])

# --- Save forecast results ---
results = test[["timestamp", "Usage_kWh"]].copy()
results["predicted_kwh"] = forecast_pred
results["residual"] = results["Usage_kWh"] - results["predicted_kwh"]
results.to_csv("real_data/outputs/predictions.csv", index=False)

forecast_importance.to_csv("real_data/outputs/feature_importance.csv", header=["importance"])
joblib.dump(xgb_forecast, "real_data/outputs/xgb_forecast_model.joblib")
joblib.dump(lgb_forecast, "real_data/outputs/lgb_forecast_model.joblib")

# SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, test[FORECAST_FEATURES], feature_names=FORECAST_FEATURES, show=False)
plt.savefig("real_data/outputs/shap_summary.png", bbox_inches="tight", dpi=100)
plt.close()

# Residual analysis
X_test_primary, y_test_primary = test[FORECAST_FEATURES], test[target]
lr_resid = lr_forecast_pred - y_test_primary.values
xgb_resid = forecast_pred - y_test_primary.values
lgb_resid = lgb_forecast_pred - y_test_primary.values

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
sns.histplot(lr_resid, kde=True)
plt.title("Linear Regression Residuals")
plt.subplot(1, 3, 2)
sns.histplot(xgb_resid, kde=True)
plt.title("XGBoost Residuals")
plt.subplot(1, 3, 3)
sns.histplot(lgb_resid, kde=True)
plt.title("LightGBM Residuals")
plt.tight_layout()
plt.savefig("real_data/outputs/residual_plots.png", bbox_inches="tight", dpi=100)
plt.close()

residual_stats = {
    "linear_regression": {"mean_residual": float(np.mean(lr_resid)), "std_residual": float(np.std(lr_resid)),
                          "min_residual": float(np.min(lr_resid)), "max_residual": float(np.max(lr_resid))},
    "xgboost": {"mean_residual": float(np.mean(xgb_resid)), "std_residual": float(np.std(xgb_resid)),
                "min_residual": float(np.min(xgb_resid)), "max_residual": float(np.max(xgb_resid))},
    "lightgbm": {"mean_residual": float(np.mean(lgb_resid)), "std_residual": float(np.std(lgb_resid)),
                 "min_residual": float(np.min(lgb_resid)), "max_residual": float(np.max(lgb_resid))},
}
with open("real_data/outputs/residual_stats.json", "w") as f:
    json.dump(residual_stats, f, indent=2)

lr_mae_f = forecast_metrics["linear_regression"]["MAE_kWh"]
xgb_mae_f = forecast_metrics["xgboost"]["MAE_kWh"]
metrics = {
    "primary_forecast_model": forecast_metrics,
    "secondary_estimation_model_not_a_forecast": estimate_metrics,
    "improvement_over_baseline_pct": round((lr_mae_f - xgb_mae_f) / lr_mae_f * 100, 1),
    "forecast_features": FORECAST_FEATURES,
    "train_set_size_rows": len(train),
    "test_set_size_rows": len(test),
    "note": "Primary model uses only time+lag features (genuine forecast). Secondary model "
            "uses simultaneous electrical readings for comparison only and should not be "
            "presented as a forecast.",
}
with open("real_data/outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved predictions.csv, feature_importance.csv, metrics.json, "
      "xgb_forecast_model.joblib, lgb_forecast_model.joblib, shap_summary.png, "
      "residual_plots.png, residual_stats.json")
