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

## Results (real plant data - DAEWOO Steel, 2018, 34,944 readings at 15-min)

I ran everything on the real dataset, not the synthetic one. It is a full year of 15-minute meter readings, so you get the actual ups and downs of the plant.

My main model only uses stuff you would know ahead of time - time of day, day of week, and past usage. No cheating with same-time sensor readings.

| Model | MAE | MAPE |
|---|---|---|
| Linear Regression | 6.54 kWh | 55.8% |
| XGBoost | 5.22 kWh | 34.9% |

MAPE looks high but that is mostly because usage drops near zero when the plant is idle, so even a small miss looks huge in percent. MAE tells the real story. On average I am off by about 5 kWh against a mean around 27 kWh.

What mattered most was pretty simple. The last reading (15 mins ago) did most of the work, around 63%. Then same time yesterday, weekend flag, day of week and hour. Makes sense, this kind of load just follows its own recent history.

Split by load type came out like this:

| Load type | Share |
|---|---|
| Maximum_Load | 44.9% |
| Medium_Load | 38.9% |
| Light_Load | 16.2% |

The part I found most useful was power factor. The plant sits below 90% about 54.8% of the time, which is where penalty billing usually kicks in. Light_Load is the worst - median around 66.2%, and it is under 90% almost 80% of the time. Maximum is around 91.7% and Medium around 96.8%, so they are mostly fine. To me that says put an automatic capacitor bank that kicks in during Light_Load, not a fixed one. A fixed bank would overdo it when load is light.

For peaks I took the top 5% of readings (1,749 intervals, over 99.1 kWh). They pile up around 9 AM and on Thursdays. If the tariff has a demand charge, moving non-critical work out of that window should help the bill even if total energy stays the same.

## Tech stack
Python, pandas, scikit-learn, XGBoost, LightGBM, Streamlit, Plotly.
