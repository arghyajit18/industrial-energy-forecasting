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

## About this project
Industrial plants consume large amounts of electricity with highly variable load patterns. This project builds an end-to-end ML pipeline to forecast short-term energy consumption, explain key drivers, and surface actionable efficiency opportunities.
It includes two complementary pipelines: a synthetic hourly dataset with full equipment-level breakdown for method demonstration, and a real 15-minute steel-plant dataset for validated results (see `real_data/README.md`). Both share the same stages — cleaning and feature engineering, exploratory analysis, XGBoost forecasting, wastage and peak-demand optimization, and an interactive Streamlit dashboard with what-if forecasting.

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

**Wastage I noticed**: around 796 hours had high power draw even when production was low. A lot of it showed up around shift changes (6-7 AM, 5-7 PM) and on weekends. When I broke it down, about 402 of those hours came from the Auxiliary/misc side, so that is where I would put auto-shutdown first. Back-of-the-envelope saving is close to 84,000 kWh a year, about 1% of the total.

**Peak demand**: I looked at the top 5% of hours by load (437 hours) and the Furnace was on top every single time. What stood out is the Furnace and Compressor bank both run hot together around 17% of the time, which is way more than you would expect by chance (about 2.7x). That overlap is what seems to push the peaks up.

**What I would try next**: shift compressor starts away from furnace peak windows. It should cut the peaks without hurting output, since the issue looks like timing rather than total load.

## Tech stack
Python, pandas, scikit-learn, XGBoost, LightGBM, Streamlit, Plotly.
