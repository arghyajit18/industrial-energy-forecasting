"""
Step 3: Exploratory data analysis — trend, hourly/weekly patterns, correlations.
Saves plots to outputs/ for the report and dashboard.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/clean_energy_data.csv", parse_dates=["timestamp"])
out = "outputs"

sns.set_style("whitegrid")

# 1. Trend over time (first 30 days for readability)
fig, ax = plt.subplots(figsize=(12, 4))
sample = df[df["timestamp"] < df["timestamp"].min() + pd.Timedelta(days=30)]
ax.plot(sample["timestamp"], sample["energy_kwh"], linewidth=0.8, color="#2563eb")
ax.set_title("Hourly Energy Consumption — First 30 Days")
ax.set_ylabel("Energy (kWh)")
plt.tight_layout()
plt.savefig(f"{out}/01_trend_first_30_days.png", dpi=120)
plt.close()

# 2. Average consumption by hour of day
fig, ax = plt.subplots(figsize=(8, 4))
df.groupby("hour")["energy_kwh"].mean().plot(kind="bar", ax=ax, color="#2563eb")
ax.set_title("Average Energy Consumption by Hour of Day")
ax.set_ylabel("Avg Energy (kWh)")
plt.tight_layout()
plt.savefig(f"{out}/02_avg_by_hour.png", dpi=120)
plt.close()

# 3. Average consumption by day of week
fig, ax = plt.subplots(figsize=(8, 4))
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
means = df.groupby("dayofweek")["energy_kwh"].mean()
ax.bar(days, means.values, color="#16a34a")
ax.set_title("Average Energy Consumption by Day of Week")
ax.set_ylabel("Avg Energy (kWh)")
plt.tight_layout()
plt.savefig(f"{out}/03_avg_by_dayofweek.png", dpi=120)
plt.close()

# 4. Correlation heatmap
fig, ax = plt.subplots(figsize=(7, 6))
cols = ["energy_kwh", "production_load_t", "equipment_utilization_pct",
        "ambient_temp_c", "equipment_run_frac", "hour", "is_weekend"]
sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation with Energy Consumption")
plt.tight_layout()
plt.savefig(f"{out}/04_correlation_heatmap.png", dpi=120)
plt.close()

# 5. Scatter: production load vs energy, colored by equipment run fraction
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(df["production_load_t"], df["energy_kwh"], c=df["equipment_run_frac"],
                 cmap="viridis", s=4, alpha=0.5)
plt.colorbar(sc, label="Equipment run fraction")
ax.set_xlabel("Production Load (t)")
ax.set_ylabel("Energy (kWh)")
ax.set_title("Production Load vs Energy Consumption")
plt.tight_layout()
plt.savefig(f"{out}/05_load_vs_energy.png", dpi=120)
plt.close()

# 6. Equipment-level breakdown: average consumption by hour, stacked
EQUIPMENT_COLS = ["furnace_kwh", "compressor_kwh", "ventilation_kwh", "auxiliary_kwh"]
hourly_equip = df.groupby("hour")[EQUIPMENT_COLS].mean()
fig, ax = plt.subplots(figsize=(10, 5))
hourly_equip.plot(kind="bar", stacked=True, ax=ax,
                   color=["#2563eb", "#f97316", "#16a34a", "#dc2626"])
ax.set_title("Average Consumption by Hour — Equipment Breakdown")
ax.set_ylabel("Avg Energy (kWh)")
ax.legend(title="Equipment", labels=["Furnace", "Compressor", "Ventilation", "Auxiliary"])
plt.tight_layout()
plt.savefig(f"{out}/06_equipment_breakdown_by_hour.png", dpi=120)
plt.close()

# 7. Equipment share of total annual consumption (pie-equivalent bar)
totals = df[EQUIPMENT_COLS].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(["Furnace", "Compressor", "Ventilation", "Auxiliary"],
       [totals["furnace_kwh"], totals["compressor_kwh"], totals["ventilation_kwh"], totals["auxiliary_kwh"]],
       color=["#2563eb", "#f97316", "#16a34a", "#dc2626"])
ax.set_title("Total Annual Consumption by Equipment")
ax.set_ylabel("Total Energy (kWh)")
plt.tight_layout()
plt.savefig(f"{out}/07_equipment_annual_totals.png", dpi=120)
plt.close()

print("Saved 7 EDA plots to outputs/")
print("\nEquipment share of total annual consumption:")
print((totals / totals.sum() * 100).round(1).astype(str) + "%")
print("\nKey correlations with energy_kwh:")
print(df[cols].corr()["energy_kwh"].sort_values(ascending=False))
