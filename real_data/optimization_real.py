"""
Real-data pipeline — Step 4: Optimization recommendations grounded in real
electrical measurements (no synthetic equipment IDs available in real data,
so the analysis uses what a real plant actually exposes: Load_Type segments,
peak-demand timing, and power factor).
"""
import pandas as pd
import numpy as np

df = pd.read_csv("real_data/clean_steel_industry_data.csv", parse_dates=["timestamp"])

# Segment ranking (Load_Type — closest real proxy to "major energy-consuming
# equipment" since this dataset has no sub-metering)
seg_totals = df.groupby("Load_Type")["Usage_kWh"].sum().sort_values(ascending=False)
seg_share = (seg_totals / seg_totals.sum() * 100).round(1)
seg_ranking = pd.DataFrame({"total_kwh": seg_totals.round(0), "share_pct": seg_share})
seg_ranking.to_csv("real_data/outputs/loadtype_ranking.csv")

# Peak demand analysis (top 5% of 15-min readings)
peak_threshold = df["Usage_kWh"].quantile(0.95)
peaks = df[df["Usage_kWh"] >= peak_threshold].copy()
peak_by_hour = peaks.groupby("hour").size().sort_values(ascending=False)
peak_by_dow = peaks.groupby("dayofweek").size().sort_values(ascending=False)
peaks.to_csv("real_data/outputs/peak_demand_periods.csv", index=False)

# Power factor analysis — THE real, actionable finding in this dataset.
# Indian/most state electricity boards penalize consumers when average power
# factor drops below ~0.90 (90%), and reward it above ~0.95. Low PF means
# more reactive current is drawn for the same real power, wasting capacity.
PF_THRESHOLD = 90.0
low_pf = df[df["Lagging_Current_Power_Factor"] < PF_THRESHOLD]
low_pf_pct_of_time = len(low_pf) / len(df) * 100
low_pf_by_loadtype = (df["Lagging_Current_Power_Factor"] < PF_THRESHOLD).groupby(df["Load_Type"]).mean().mul(100).sort_values(ascending=False)
median_pf_by_loadtype = df.groupby("Load_Type")["Lagging_Current_Power_Factor"].median()

print("Load type ranking (annual consumption):")
print(seg_ranking)

print(f"\nPeak demand:")
print(f"Peak threshold (95th pct): {peak_threshold:.1f} kWh, {len(peaks)} intervals flagged")
print(f"Most common peak hour: {peak_by_hour.idxmax()}:00 ({peak_by_hour.iloc[0]} intervals)")
print(f"Most common peak day: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][peak_by_dow.idxmax()]} "
      f"({peak_by_dow.iloc[0]} intervals)")

print(f"\nPower factor:")
print(f"Time spent below {PF_THRESHOLD}% lagging PF: {low_pf_pct_of_time:.1f}% of all readings")
print("Median lagging PF by Load_Type:")
print(median_pf_by_loadtype)
print(f"\n% of time below {PF_THRESHOLD}% PF, by Load_Type:")
print(low_pf_by_loadtype)

worst_segment = low_pf_by_loadtype.index[0]
worst_median_pf = median_pf_by_loadtype[worst_segment]

recommendations = f"""
OPTIMIZATION RECOMMENDATIONS, real data summary
SEGMENT RANKING (Load_Type — closest real proxy to equipment-level breakdown;
this dataset has a single main meter with no sub-metering)
{seg_ranking.to_string()}
-> {seg_ranking.index[0]} accounts for {seg_ranking.iloc[0]['share_pct']}% of annual consumption.

1. Power factor correction (highest-value, most concrete recommendation from
   real data): lagging power factor drops to a median of {worst_median_pf:.1f}%
   during {worst_segment} periods — well below the {PF_THRESHOLD:.0f}% threshold
   most electricity boards use for penalty billing. The plant spends
   {low_pf_pct_of_time:.1f}% of all 15-min intervals below {PF_THRESHOLD:.0f}% PF.
   Recommended action: size and install automatic power-factor-correction
   capacitor banks, switched in specifically during {worst_segment} periods
   (a static/fixed bank sized for Maximum_Load would over-compensate and
   cause leading-PF penalties at Light_Load — an automatic switched bank
   avoids this).

2. Peak demand timing: peak-demand periods concentrate around
   {peak_by_hour.idxmax()}:00 and on {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][peak_by_dow.idxmax()]}s.
   If the plant is on a demand-based tariff, shifting non-critical load away
   from this window reduces the peak demand charge component of the bill,
   independent of total energy consumed.

3. Load-type-based scheduling: since {seg_ranking.index[0]} periods dominate
   annual consumption ({seg_ranking.iloc[0]['share_pct']}%), even small
   percentage efficiency gains during these periods have outsized impact
   compared to the same gain during {seg_ranking.index[-1]} periods.
"""
with open("real_data/outputs/recommendations.txt", "w", encoding="utf-8") as f:
    f.write(recommendations)
print(recommendations)
