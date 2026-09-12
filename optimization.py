"""
Step 5: Flag wastage points, rank equipment by consumption and by contribution to
wastage, identify peak-demand hours and their dominant equipment, and turn all of
this into concrete recommendations (load balancing / equipment scheduling / process
optimization).
"""
import pandas as pd
import numpy as np

EQUIPMENT_COLS = ["furnace_kwh", "compressor_kwh", "ventilation_kwh", "auxiliary_kwh"]
EQUIPMENT_NAMES = {"furnace_kwh": "Furnace", "compressor_kwh": "Compressor bank",
                    "ventilation_kwh": "Ventilation/cooling", "auxiliary_kwh": "Auxiliary/misc"}

df = pd.read_csv("data/clean_energy_data.csv", parse_dates=["timestamp"])

# Flag hours where equipment is barely running / production load is low
# but consumption is still high relative to that load — this is the wastage signature.
df["expected_kwh_per_load"] = df["energy_kwh"] / df["production_load_t"].clip(lower=1)
kwh_threshold = df["expected_kwh_per_load"].quantile(0.85)
run_frac_threshold = df["equipment_run_frac"].quantile(0.20)

# Wastage signature: equipment utilization in the bottom 20% (relatively idle),
# but energy-per-unit-of-load is still in the top 15% — consumption isn't
# scaling down with production the way it should.
wastage = df[
    (df["equipment_run_frac"] < run_frac_threshold) &
    (df["expected_kwh_per_load"] > kwh_threshold)
].copy()

wastage_by_hour = wastage.groupby("hour").size().sort_values(ascending=False)
wastage_by_dow = wastage.groupby("dayofweek").size().sort_values(ascending=False)

total_wastage_kwh_estimate = (wastage["energy_kwh"] - wastage["production_load_t"] * df["expected_kwh_per_load"].median()).clip(lower=0).sum()
total_kwh = df["energy_kwh"].sum()
pct_of_total = total_wastage_kwh_estimate / total_kwh * 100

wastage.to_csv("outputs/flagged_wastage_periods.csv", index=False)

print(f"Flagged {len(wastage)} hours ({len(wastage)/len(df)*100:.1f}% of the year) as likely idle-loss wastage.")
print(f"Estimated recoverable energy: {total_wastage_kwh_estimate:,.0f} kWh "
      f"(~{pct_of_total:.1f}% of total annual consumption)")
print("\nMost common hours-of-day for wastage:")
print(wastage_by_hour.head(5))
print("\nMost common days for wastage:")
print(wastage_by_dow)

# Equipment-level ranking (annual totals + share)
equip_totals = df[EQUIPMENT_COLS].sum().sort_values(ascending=False)
equip_share = (equip_totals / equip_totals.sum() * 100).round(1)
equip_ranking = pd.DataFrame({"total_kwh": equip_totals.round(0), "share_pct": equip_share})
equip_ranking.index = [EQUIPMENT_NAMES[c] for c in equip_ranking.index]
equip_ranking.to_csv("outputs/equipment_ranking.csv")

# Which equipment drives the flagged wastage hours? (z-score vs its own baseline)
z_scores = {}
for col in EQUIPMENT_COLS:
    mu, sigma = df[col].mean(), df[col].std()
    z_scores[col] = (wastage[col] - mu) / sigma
z_df = pd.DataFrame(z_scores)
wastage_equipment_cause = z_df.idxmax(axis=1).map(EQUIPMENT_NAMES).value_counts()
top_wastage_equipment = wastage_equipment_cause.index[0]

# Peak demand analysis (top 5% of hours by total consumption)
peak_threshold = df["energy_kwh"].quantile(0.95)
peaks = df[df["energy_kwh"] >= peak_threshold].copy()
peak_by_hour = peaks.groupby("hour").size().sort_values(ascending=False)
peaks["dominant_equipment"] = peaks[EQUIPMENT_COLS].idxmax(axis=1).map(EQUIPMENT_NAMES)
peak_dominant_counts = peaks["dominant_equipment"].value_counts()
peaks.to_csv("outputs/peak_demand_hours.csv", index=False)

# Co-occurrence: how often are Furnace AND Compressor both in their own top 25% at once?
furnace_high = df["furnace_kwh"] >= df["furnace_kwh"].quantile(0.75)
compressor_high = df["compressor_kwh"] >= df["compressor_kwh"].quantile(0.75)
both_high_pct = (furnace_high & compressor_high).mean() * 100
expected_if_independent = furnace_high.mean() * compressor_high.mean() * 100
overlap_ratio = both_high_pct / expected_if_independent if expected_if_independent > 0 else 1

print("\nEquipment ranking (annual consumption):")
print(equip_ranking)
print(f"\nTop cause of flagged wastage hours: {top_wastage_equipment} "
      f"({wastage_equipment_cause.iloc[0]}/{len(wastage)} hours)")
print(f"\nPeak demand threshold (95th pct): {peak_threshold:.0f} kWh, {len(peaks)} hours flagged")
print(f"Peak hours dominated by:\n{peak_dominant_counts}")
print(f"\nFurnace & Compressor both in their own top 25% simultaneously: "
      f"{both_high_pct:.1f}% of hours ({overlap_ratio:.2f}x what independence would predict)")

dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
recommendations = f"""
OPTIMIZATION RECOMMENDATIONS (from flagged periods)
EQUIPMENT RANKING (major energy-consuming equipment)
{equip_ranking.to_string()}
-> {equip_ranking.index[0]} is the single largest consumer ({equip_ranking.iloc[0]['share_pct']}% of total)
   and the natural first target for any efficiency investment.

1. Load balancing: {wastage_by_hour.idxmax()}:00 hour shows the highest concentration
   of idle-loss events — audit equipment left running during this hour when
   production load is low. Equipment-level analysis attributes {wastage_equipment_cause.iloc[0]}/{len(wastage)}
   of these flagged hours to {top_wastage_equipment}, making it the priority for
   auto-shutdown/standby triggers (see recommendation 4).

2. Equipment scheduling: Weekday {dow_names[wastage_by_dow.idxmax()]}
   shows the most wastage hours — cross-check shift handover / maintenance-window
   timing against equipment shutdown procedures.

3. Process optimization: Furnace and Compressor bank are simultaneously in their
   own top-25%-load state {both_high_pct:.1f}% of hours — {overlap_ratio:.1f}x more
   often than if their duty cycles were independent. This overlap is what drives peak
   demand ({peak_dominant_counts.index[0]} dominates {peak_dominant_counts.iloc[0]}/{len(peaks)}
   flagged peak hours). Staggering compressor bank startup/cycling away from furnace
   peak-load windows would flatten peak demand without changing total output.

4. Estimated recoverable energy: ~{total_wastage_kwh_estimate:,.0f} kWh/year
   ({pct_of_total:.1f}% of total consumption) if idle equipment is shut down
   during low-load periods instead of left running.

5. Recommended action: install auto-shutdown/standby triggers on {top_wastage_equipment}
   whose utilization drops below 30% for more than 2 consecutive hours.
"""
with open("outputs/recommendations.txt", "w", encoding="utf-8") as f:
    f.write(recommendations)
print(recommendations)
