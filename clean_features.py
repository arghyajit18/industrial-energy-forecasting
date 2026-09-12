"""
Step 2: Clean the raw data and engineer features for forecasting.

Dataset:
Steel Industry Energy Consumption

- Handles timestamps
- Handles missing values
- Caps extreme energy values
- Adds time-based features
- Adds Fourier seasonal features
- Adds lag features
- Adds rolling 24-hour features
"""

import pandas as pd
import numpy as np


# 1. LOAD DATA

df = pd.read_csv("data/raw_energy_data.csv")


# 2. CLEAN COLUMN NAMES

df.columns = (
    df.columns
    .str.strip()
    .str.replace("\ufeff", "", regex=False)
)

print("Columns found in CSV:")
print(df.columns.tolist())


# 3. HANDLE TIMESTAMP

# Your current CSV already contains "timestamp".
# If timestamp does not exist but "date" exists,
# create timestamp from date.

if "timestamp" in df.columns:

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

elif "date" in df.columns:

    df["timestamp"] = pd.to_datetime(
        df["date"],
        dayfirst=True,
        errors="coerce"
    )

else:

    raise ValueError(
        "ERROR: Neither 'timestamp' nor 'date' "
        "was found in the CSV."
    )


# 4. SORT BY TIME

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# 5. ENERGY COLUMN

# Your dataset uses Usage_kWh.

if "Usage_kWh" in df.columns:

    df["energy_kwh"] = pd.to_numeric(
        df["Usage_kWh"],
        errors="coerce"
    )

elif "energy_kwh" in df.columns:

    df["energy_kwh"] = pd.to_numeric(
        df["energy_kwh"],
        errors="coerce"
    )

else:

    raise ValueError(
        "ERROR: Neither 'Usage_kWh' nor "
        "'energy_kwh' was found."
    )


# 6. REMOVE OLD ENGINEERED FEATURES

# Your current CSV already contains old versions of
# lag and rolling features.
#
# We remove them and recreate them correctly.

old_features = [
    "hour",
    "minute",
    "dayofweek",
    "is_weekend",
    "month",
    "day",
    "year",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "energy_lag_15min",
    "energy_lag_1h",
    "energy_lag_24h",
    "energy_roll_mean_24h",
    "energy_roll_std_24h",
    "is_weekday",
    "usage_lag_1step",
    "usage_lag_1day",
    "usage_roll_mean_1day"
]

for col in old_features:

    if col in df.columns:

        df = df.drop(
            columns=col
        )


# 7. MISSING VALUES

n_missing = df["energy_kwh"].isna().sum()

# Fill energy missing values
df["energy_kwh"] = (
    df["energy_kwh"]
    .ffill()
    .bfill()
)


# Fill missing numerical sensor values
numeric_cols = df.select_dtypes(
    include=[np.number]
).columns

for col in numeric_cols:

    df[col] = (
        df[col]
        .ffill()
        .bfill()
    )


# 8. OUTLIER CAPPING

# Cap extreme energy readings at 99.5 percentile.

cap = df["energy_kwh"].quantile(
    0.995
)

n_capped = (
    df["energy_kwh"] > cap
).sum()

df["energy_kwh"] = (
    df["energy_kwh"]
    .clip(upper=cap)
)


# 9. TIME FEATURES

df["hour"] = (
    df["timestamp"].dt.hour
)

df["minute"] = (
    df["timestamp"].dt.minute
)

df["dayofweek"] = (
    df["timestamp"].dt.dayofweek
)

df["day"] = (
    df["timestamp"].dt.day
)

df["month"] = (
    df["timestamp"].dt.month
)

df["year"] = (
    df["timestamp"].dt.year
)


# Weekend
df["is_weekend"] = (
    df["dayofweek"] >= 5
).astype(int)


# Weekday
df["is_weekday"] = (
    df["dayofweek"] < 5
).astype(int)


# 10. FOURIER FEATURES

decimal_hour = (
    df["hour"] +
    df["minute"] / 60
)


# Daily seasonality
df["hour_sin"] = np.sin(
    2 * np.pi * decimal_hour / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * decimal_hour / 24
)


# Weekly seasonality
df["dow_sin"] = np.sin(
    2 * np.pi * df["dayofweek"] / 7
)

df["dow_cos"] = np.cos(
    2 * np.pi * df["dayofweek"] / 7
)


# Monthly seasonality
df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)


# 11. LAG FEATURES

# Your dataset has 15-minute measurements.
#
# 15 minutes = 1 row
# 1 hour     = 4 rows
# 24 hours   = 96 rows

df["energy_lag_15min"] = (
    df["energy_kwh"].shift(1)
)

df["energy_lag_1h"] = (
    df["energy_kwh"].shift(4)
)

df["energy_lag_24h"] = (
    df["energy_kwh"].shift(96)
)


# 12. ROLLING FEATURES

# 24 hours = 96 observations

df["energy_roll_mean_24h"] = (
    df["energy_kwh"]
    .rolling(
        window=96,
        min_periods=1
    )
    .mean()
)

df["energy_roll_std_24h"] = (
    df["energy_kwh"]
    .rolling(
        window=96,
        min_periods=1
    )
    .std()
)


# 13. REMOVE INVALID ROWS

# Lag_24h creates missing values for the first 96 rows.

df = df.dropna().reset_index(
    drop=True
)


# 14. SAVE CLEAN DATA

df.to_csv(
    "data/clean_energy_data.csv",
    index=False
)


# 15. SUMMARY

print()
print("=" * 55)
print("DATA CLEANING COMPLETED SUCCESSFULLY")
print("=" * 55)

print(
    f"Missing energy values filled: {n_missing}"
)

print(
    f"Outliers capped at: {cap:.2f} kWh"
)

print(
    f"Number of outliers capped: {n_capped}"
)

print(
    f"Final dataset rows: {len(df)}"
)

print(
    f"Final dataset columns: {df.shape[1]}"
)

print(
    "Saved -> data/clean_energy_data.csv"
)

print("=" * 55)