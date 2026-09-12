"""
Real-data pipeline — Step 1: Clean and engineer features for the real
Steel Industry Energy Consumption dataset (DAEWOO Steel Co., South Korea,
15-minute interval, 2018, via UCI/Kaggle).

Column notes:
- Load_Type (Light/Medium/Maximum) is a *label derived from Usage_kWh itself*
  by the dataset's authors — it must NOT be used as a model input feature
  (that would leak the target). It's kept for segment-level analysis instead
  (the equivalent of "equipment ranking" in the synthetic pipeline).
- NSM = seconds since midnight, already a clean time-of-day feature.
- 96 rows/day (15-min interval) is used for the 24h lag/rolling window,
  vs 24 rows/day (hourly) in the synthetic pipeline.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("real_data/raw_steel_industry_data.csv")

# Parse timestamp (format: DD/MM/YYYY HH:MM)
df["timestamp"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
df = df.sort_values("timestamp").reset_index(drop=True)
df = df.drop(columns=["date"])

# Missing values (none found on inspection, but handle defensively)
n_missing_before = df["Usage_kWh"].isna().sum()
df["Usage_kWh"] = df["Usage_kWh"].ffill().bfill()

# Outlier capping at 99.5th percentile (sensor spikes)
cap = df["Usage_kWh"].quantile(0.995)
n_capped = (df["Usage_kWh"] > cap).sum()
df["Usage_kWh"] = df["Usage_kWh"].clip(upper=cap)

# Time features
df["hour"] = df["NSM"] // 3600
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = (df["WeekStatus"] == "Weekend").astype(int)
df["month"] = df["timestamp"].dt.month

# Lag / rolling features (96 steps/day at 15-min resolution)
df["usage_lag_1step"] = df["Usage_kWh"].shift(1)          # 15 min ago
df["usage_lag_1day"] = df["Usage_kWh"].shift(96)           # same time yesterday
df["usage_roll_mean_1day"] = df["Usage_kWh"].rolling(96, min_periods=1).mean()

# Power factor sanity: power factor is reported 0-100 in this dataset
# (Confirmed by inspecting value ranges — treated as %.)

df = df.dropna().reset_index(drop=True)

df.to_csv("real_data/clean_steel_industry_data.csv", index=False)

print(f"Missing values filled: {n_missing_before}")
print(f"Outliers capped at {cap:.1f} kWh: {n_capped}")
print(f"Final clean dataset: {len(df)} rows, {df.shape[1]} columns")
print("Saved -> real_data/clean_steel_industry_data.csv")
