# Energy Consumption Forecasting & Optimization

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://industrial-energy-forecasting-r68qnqof8xkhnqm8e2y2m4.streamlit.app/)

**Live demo:** https://industrial-energy-forecasting-r68qnqof8xkhnqm8e2y2m4.streamlit.app/

A prototype ML pipeline that forecasts plant-utility energy consumption and
flags avoidable inefficiency.

**Two versions in this repo:**
- `./` (this folder) — synthetic data, full equipment-level breakdown (4 named
  sub-systems), built before real data was available. Use this to explain the
  *approach* end-to-end.
- `real_data/` — **real 2018 data from an actual steel plant** (DAEWOO Steel
  Co., South Korea, 34,944 rows, 15-min interval), adapted pipeline, and a
  genuine finding (power factor deficiency) with a concrete recommendation.
  **Lead with this one** in interviews/resume — it's real data, not simulated.

## Why synthetic data (this folder only)
Real SCADA/energy-meter data isn't accessible outside SAIL's internal systems
before the internship starts. `data/generate_data.py` generates one year of
hourly data with realistic relationships (production load, equipment
utilization, ambient temperature, shift/weekend patterns, and injected
idle-loss + sensor-glitch noise) so the full pipeline can be built and
demonstrated end-to-end. **See `real_data/README.md` for the real-data
version, which supersedes this for actual results.**

## Pipeline
```
data/generate_data.py   → raw_energy_data.csv         (synthetic data, split by equipment: furnace, compressor, ventilation, auxiliary)
clean_features.py       → clean_energy_data.csv        (cleaning + feature engineering + equipment share %)
eda.py                  → outputs/*.png                 (7 exploratory plots incl. equipment breakdown)
train_model.py           → outputs/metrics.json, predictions.csv, xgb_model.joblib
optimization.py          → outputs/flagged_wastage_periods.csv, equipment_ranking.csv,
                            peak_demand_hours.csv, recommendations.txt
dashboard/app.py         → interactive Streamlit dashboard (equipment breakdown + peak demand views)
```

Run in order:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit plotly
python data/generate_data.py
python clean_features.py
python eda.py
python train_model.py
python optimization.py
streamlit run dashboard/app.py
```

## Results

| Model | MAE (kWh) | MAPE |
|---|---|---|
| Linear Regression (baseline) | 11.5 | 1.25% |
| XGBoost | 10.9 | 1.20% |

**Top drivers of energy consumption** (feature importance): production load
(51%), equipment utilization (21%), day-of-week (17%) — confirming that
consumption tracks operational load, with a secondary weekday/weekend effect.

**Equipment-level breakdown** (identification of major energy-consuming
equipment): total annual consumption is now attributed to 4 named
sub-systems —

| Equipment | Share of total |
|---|---|
| Furnace | 62.5% |
| Compressor bank | 25.8% |
| Auxiliary/misc | 6.4% |
| Ventilation/cooling | 5.2% |

**Wastage detection**: 796 hours (9.1% of the year) flagged as idle-loss
periods, concentrated around shift-change windows (6-7 AM, 5-7 PM) and
weekends. Equipment-level attribution traces 402 of those 796 hours
specifically to the **Auxiliary/misc** sub-system — making it the clear
priority for auto-shutdown triggers, not a plant-wide guess. Estimated
recoverable energy: ~84,000 kWh/year (~1.0% of total consumption).

**Peak demand analysis**: top 5% of hours by total load (437 hours) are
dominated entirely by the Furnace. Furnace and Compressor bank are
simultaneously in their own top-25%-load state 17% of the time — 2.7x more
often than independent duty cycles would predict — which is what drives peak
demand spikes.

**Process optimization recommendation**: stagger compressor-bank
startup/cycling away from furnace peak-load windows to flatten peak demand
without reducing total output — directly addresses the "process
optimization" deliverable, derived from the co-occurrence analysis above
rather than asserted generically.

## Resume bullet
> Built an ML-based energy consumption forecasting pipeline (XGBoost, chronological
> train/test split) achieving 1.2% MAPE on hourly plant-utility data; built an
> equipment-level breakdown that attributed peak demand and idle-loss wastage
> (~1% of annual consumption) to specific sub-systems, and an interactive Streamlit
> dashboard for real-time monitoring and process-optimization recommendations.

## Next steps for the real internship
1. Replace synthetic data with actual plant SCADA/energy-meter exports.
2. Re-run `clean_features.py` — it already handles missing values and outlier capping.
3. Re-train (`train_model.py`) — expect XGBoost to pull ahead once real non-linearities show up.
4. Validate wastage thresholds in `optimization.py` against maintenance team feedback.
