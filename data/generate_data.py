"""
Generates a synthetic hourly energy-consumption dataset for a steel plant utility area,
broken down by named equipment (needed for equipment-level ranking, peak-demand
attribution, and process-optimization recommendations).

Synthetic data stands in for plant meter data that was not available here.
This generator encodes realistic relationships (production load, equipment run-hours,
shift patterns, ambient temperature, and idle-loss noise) so the full pipeline
(cleaning -> EDA -> forecasting -> optimization) can be built and demonstrated end-to-end.
To use real plant data instead, swap this file's
output for the real CSV — the rest of the pipeline (clean_features.py, train_model.py,
dashboard/app.py) expects the same column schema and will work unchanged, as long as
per-equipment kWh columns are named `<equipment>_kwh`.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

start = "2024-01-01"
periods = 24 * 365  # one year, hourly
timestamps = pd.date_range(start=start, periods=periods, freq="h")

n = len(timestamps)
hour = timestamps.hour.values
dow = timestamps.dayofweek.values  # 0=Mon
day_of_year = timestamps.dayofyear.values

# Production load (tonnes/hr equivalent), with shift and weekday patterns
shift_factor = 0.85 + 0.3 * np.sin((hour - 6) / 24 * 2 * np.pi) ** 2
weekday_factor = np.where(dow < 5, 1.0, 0.75)  # lower on Sat/Sun
base_load = 100
production_load = (
    base_load * shift_factor * weekday_factor
    + np.random.normal(0, 4, n)
)
production_load = np.clip(production_load, 20, None)

# Equipment utilization (%) — correlated with production load
equipment_utilization = np.clip(
    40 + 0.5 * production_load + np.random.normal(0, 5, n), 10, 100
)

# Ambient temperature (deg C) — seasonal, affects cooling/ventilation load
ambient_temp = 25 + 10 * np.sin((day_of_year / 365) * 2 * np.pi - np.pi / 2) + np.random.normal(0, 1.5, n)

# Equipment run-hours in this hour (0-1, fraction of hour actively running)
equipment_run_frac = np.clip(equipment_utilization / 100 + np.random.normal(0, 0.05, n), 0, 1)

# Per-equipment consumption breakdown
# Furnace: dominant load, tracks production almost directly, always drawing base power
furnace_kwh = 120 + 5.0 * production_load + np.random.normal(0, 8, n)

# Compressor bank: scales with equipment utilization, has its own duty cycle
compressor_kwh = 60 + 2.1 * equipment_utilization + np.random.normal(0, 6, n)

# Ventilation / cooling: driven mainly by ambient temperature, small baseline
ventilation_kwh = 40 + 1.8 * np.maximum(ambient_temp - 22, 0) + np.random.normal(0, 4, n)

# Auxiliary/misc equipment (lighting, standby drives, small motors): this is where
# idle-loss wastage concentrates — low correlation with production, but spikes when
# equipment is left running (low run-fraction) instead of shut down.
idle_loss = np.where(
    (equipment_run_frac < 0.3) & (production_load < 50),
    np.random.uniform(50, 150, n),  # equipment idling but still drawing power
    0,
)
auxiliary_kwh = 60 + idle_loss + np.random.normal(0, 5, n)

for arr in (furnace_kwh, compressor_kwh, ventilation_kwh, auxiliary_kwh):
    arr[:] = np.clip(arr, 10, None)

energy_kwh_true = furnace_kwh + compressor_kwh + ventilation_kwh + auxiliary_kwh

df = pd.DataFrame({
    "timestamp": timestamps,
    "production_load_t": production_load.round(2),
    "equipment_utilization_pct": equipment_utilization.round(2),
    "ambient_temp_c": ambient_temp.round(2),
    "equipment_run_frac": equipment_run_frac.round(3),
    "furnace_kwh": furnace_kwh.round(2),
    "compressor_kwh": compressor_kwh.round(2),
    "ventilation_kwh": ventilation_kwh.round(2),
    "auxiliary_kwh": auxiliary_kwh.round(2),
    "energy_kwh": energy_kwh_true.round(2),
})

# Inject a few missing values and sensor-glitch outliers on the main meter,
# like real data (sub-meters on individual equipment are assumed cleaner/more reliable)
missing_idx = np.random.choice(n, size=int(0.01 * n), replace=False)
df.loc[missing_idx, "energy_kwh"] = np.nan
glitch_idx = np.random.choice(n, size=15, replace=False)
df.loc[glitch_idx, "energy_kwh"] *= 3.5

df.to_csv("data/raw_energy_data.csv", index=False)
print(f"Generated {len(df)} rows -> data/raw_energy_data.csv")
print(df.head())
