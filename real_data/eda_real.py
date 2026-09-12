"""
Real-data pipeline — Step 2: EDA on the real Steel Industry Energy Consumption dataset.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("real_data/clean_steel_industry_data.csv", parse_dates=["timestamp"])
out = "real_data/outputs"
sns.set_style("whitegrid")

# 1. Trend — first 30 days
fig, ax = plt.subplots(figsize=(12, 4))
sample = df[df["timestamp"] < df["timestamp"].min() + pd.Timedelta(days=30)]
ax.plot(sample["timestamp"], sample["Usage_kWh"], linewidth=0.7, color="#2563eb")
ax.set_title("Real Plant: Usage (kWh) — First 30 Days (15-min intervals)")
ax.set_ylabel("Usage (kWh)")
plt.tight_layout()
plt.savefig(f"{out}/01_trend_first_30_days.png", dpi=120)
plt.close()

# 2. Average usage by hour
fig, ax = plt.subplots(figsize=(8, 4))
df.groupby("hour")["Usage_kWh"].mean().plot(kind="bar", ax=ax, color="#2563eb")
ax.set_title("Real Plant: Average Usage by Hour of Day")
ax.set_ylabel("Avg Usage (kWh)")
plt.tight_layout()
plt.savefig(f"{out}/02_avg_by_hour.png", dpi=120)
plt.close()

# 3. Average usage by day of week
fig, ax = plt.subplots(figsize=(8, 4))
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
df.groupby("dayofweek")["Usage_kWh"].mean().reindex(range(7)).plot(kind="bar", ax=ax, color="#16a34a")
ax.set_xticklabels(days, rotation=0)
ax.set_title("Real Plant: Average Usage by Day of Week")
ax.set_ylabel("Avg Usage (kWh)")
plt.tight_layout()
plt.savefig(f"{out}/03_avg_by_dayofweek.png", dpi=120)
plt.close()

# 4. Correlation heatmap
fig, ax = plt.subplots(figsize=(8, 7))
cols = ["Usage_kWh", "Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh",
        "CO2(tCO2)", "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor",
        "hour", "is_weekend"]
sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Real Plant: Feature Correlation with Usage_kWh")
plt.tight_layout()
plt.savefig(f"{out}/04_correlation_heatmap.png", dpi=120)
plt.close()

# 5. Load_Type breakdown (proxy for equipment/segment ranking — real data has no sub-metering)
fig, ax = plt.subplots(figsize=(7, 4))
totals = df.groupby("Load_Type")["Usage_kWh"].sum().sort_values(ascending=False)
ax.bar(totals.index, totals.values, color=["#dc2626", "#f97316", "#16a34a"])
ax.set_title("Real Plant: Total Annual Usage by Load Type")
ax.set_ylabel("Total Usage (kWh)")
plt.tight_layout()
plt.savefig(f"{out}/05_usage_by_loadtype.png", dpi=120)
plt.close()

# 6. Power factor vs Load_Type — the real efficiency signal (no per-equipment data,
# but this is the single most actionable real-plant metric available)
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="Load_Type", y="Lagging_Current_Power_Factor", ax=ax,
            order=["Light_Load", "Medium_Load", "Maximum_Load"])
ax.axhline(90, color="red", linestyle="--", linewidth=1, label="90% PF threshold (typical penalty line)")
ax.set_title("Real Plant: Lagging Power Factor by Load Type")
ax.legend()
plt.tight_layout()
plt.savefig(f"{out}/06_power_factor_by_loadtype.png", dpi=120)
plt.close()

print("Saved 6 EDA plots to real_data/outputs/")
print("\nTotal usage by Load_Type:")
print(totals)
print("\nCorrelations with Usage_kWh:")
print(df[cols].corr()["Usage_kWh"].sort_values(ascending=False))
print("\nMedian lagging power factor by Load_Type:")
print(df.groupby("Load_Type")["Lagging_Current_Power_Factor"].median())
