
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

df = pd.read_csv("data/clean_energy_data.csv", parse_dates=["timestamp"])

features = [
    "production_load_t", "equipment_utilization_pct", "ambient_temp_c",
    "equipment_run_frac", "hour", "dayofweek", "is_weekend", "month",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "energy_lag_1h", "energy_lag_24h", "energy_roll_mean_24h",
]
target = "energy_kwh"

# Chronological split: last 20% of the year as test set
split_idx = int(len(df) * 0.8)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

# Baseline: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_mape = mean_absolute_percentage_error(y_test, lr_pred) * 100
lr_r2 = r2_score(y_test, lr_pred)

# Hyperparameter Tuning with GridSearch 
param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}

grid_search = GridSearchCV(
    XGBRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    verbose=0,
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)
best_xgb = grid_search.best_estimator_
xgb_pred = best_xgb.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_mape = mean_absolute_percentage_error(y_test, xgb_pred) * 100
xgb_r2 = r2_score(y_test, xgb_pred)
print(f"GridSearch best params: {grid_search.best_params_}")
print(f"GridSearch best MAE: {-grid_search.best_score_:.1f} kWh")

#  Comparison: LightGBM 
lgb = LGBMRegressor(
    n_estimators=300, max_depth=-1, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    verbosity=-1,
)
lgb.fit(X_train, y_train)
lgb_pred = lgb.predict(X_test)
lgb_mae = mean_absolute_error(y_test, lgb_pred)
lgb_mape = mean_absolute_percentage_error(y_test, lgb_pred) * 100
lgb_r2 = r2_score(y_test, lgb_pred)

#  Feature importance (XGBoost - grid-tuned)
xgb_importance = pd.Series(best_xgb.feature_importances_, index=features).sort_values(ascending=False)

#  SHAP values (XGBoost - grid-tuned)
explainer = shap.TreeExplainer(best_xgb, feature_perturbation="tree_path_dependent")
shap_values = explainer.shap_values(X_test)

# Save predictions for dashboard
results = test[["timestamp", "energy_kwh"]].copy()
results["predicted_kwh"] = xgb_pred
results["residual"] = results["energy_kwh"] - results["predicted_kwh"]
results.to_csv("outputs/predictions.csv", index=False)

xgb_importance.to_csv("outputs/feature_importance.csv", header=["importance"])

# Save SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
plt.savefig("outputs/shap_summary.png", bbox_inches="tight", dpi=100)
plt.close()

# Residual analysis
lr_residuals = lr_pred - y_test
xgb_residuals = xgb_pred - y_test
lgb_residuals = lgb_pred - y_test

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
sns.histplot(lr_residuals, kde=True)
plt.title("Linear Regression Residuals")
plt.subplot(1, 3, 2)
sns.histplot(xgb_residuals, kde=True)
plt.title("XGBoost Residuals")
plt.subplot(1, 3, 3)
sns.histplot(lgb_residuals, kde=True)
plt.title("LightGBM Residuals")
plt.tight_layout()
plt.savefig("outputs/residual_plots.png", bbox_inches="tight", dpi=100)
plt.close()

# Save residual statistics
residual_stats = {
    "linear_regression": {
        "mean_residual": float(lr_residuals.mean()),
        "std_residual": float(lr_residuals.std()),
        "min_residual": float(lr_residuals.min()),
        "max_residual": float(lr_residuals.max()),
    },
    "xgboost": {
        "mean_residual": float(xgb_residuals.mean()),
        "std_residual": float(xgb_residuals.std()),
        "min_residual": float(xgb_residuals.min()),
        "max_residual": float(xgb_residuals.max()),
    },
    "lightgbm": {
        "mean_residual": float(lgb_residuals.mean()),
        "std_residual": float(lgb_residuals.std()),
        "min_residual": float(lgb_residuals.min()),
        "max_residual": float(lgb_residuals.max()),
    },
}
with open("outputs/residual_stats.json", "w") as f:
    json.dump(residual_stats, f, indent=2)

metrics = {
    "baseline_linear_regression": {
        "MAE_kWh": round(lr_mae, 2), 
        "MAPE_pct": round(lr_mape, 2), 
        "R2": round(lr_r2, 4)
    },
    "xgboost (grid-tuned)": {
        "MAE_kWh": round(xgb_mae, 2), 
        "MAPE_pct": round(xgb_mape, 2), 
        "R2": round(xgb_r2, 4)
    },
    "lightgbm": {
        "MAE_kWh": round(lgb_mae, 2), 
        "MAPE_pct": round(lgb_mape, 2), 
        "R2": round(lgb_r2, 4)
    },
    "improvement_over_baseline_pct": round((lr_mae - xgb_mae) / lr_mae * 100, 1),
    "test_set_size_hours": len(test),
    "train_set_size_hours": len(train),
    "grid_search_best_params": grid_search.best_params_,
}
with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Save the best model from grid search
joblib.dump(best_xgb, "outputs/xgb_model.joblib")
joblib.dump(lgb, "outputs/lgb_model.joblib")

print("Model performance:")
print(f"Linear Regression baseline: MAE={lr_mae:.1f} kWh, MAPE={lr_mape:.2f}%, R2={lr_r2:.4f}")
print(f"XGBoost model:              MAE={xgb_mae:.1f} kWh, MAPE={xgb_mape:.2f}%, R2={xgb_r2:.4f}")
print(f"LightGBM model:             MAE={lgb_mae:.1f} kWh, MAPE={lgb_mape:.2f}%, R2={lgb_r2:.4f}")
print(f"\nImprovement over baseline: {metrics['improvement_over_baseline_pct']}%")
print("\nFeature importance:")
print(xgb_importance)
print("\nSHAP summary (feature importance):")
print("SHAP summary plot saved to outputs/shap_summary.png")
