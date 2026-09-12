# Energy Consumption Forecasting & Optimization

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://industrial-energy-forecasting-r68qnqof8xkhnqm8e2y2m4.streamlit.app/)

**Live demo:** https://industrial-energy-forecasting-r68qnqof8xkhnqm8e2y2m4.streamlit.app/

Steel plants pay for both total energy and peak demand plus power-factor penalties. This project forecasts short-term consumption and flags where money is lost.

**Outcomes on real plant data (DAEWOO Steel, 2018, 34,944 readings at 15-min):**
- Forecast error 5.22 kWh MAE with XGBoost using only time and lag features, chronological train/test split
- Power factor below the 90% penalty line 54.8% of the time, worst in Light_Load with median 66.2%
- Peak demand concentrated at 9 AM on Thursdays, top 5% over 99.1 kWh

## Demo screenshots

![Power factor by load type](real_data/outputs/06_power_factor_by_loadtype.png)
![Usage trend first 30 days](real_data/outputs/01_trend_first_30_days.png)

## Real pipeline - primary

Real 15-minute data from DAEWOO Steel Co., Gwangyang, South Korea. Source: UCI ML Repository / Kaggle Steel Industry Energy Consumption dataset.

Stages: cleaning and feature engineering, exploratory analysis, XGBoost forecasting with leakage avoidance, peak and power-factor optimization, Streamlit dashboard with what-if forecasting.

Run:
```bash
pip install -r requirements.txt
python real_data/eda_real.py
python real_data/train_model_real.py
python real_data/optimization_real.py
streamlit run real_data/dashboard_real.py
```

Results:

| Model | MAE | MAPE |
|---|---|---|
| Linear Regression | 6.54 kWh | 55.8% |
| XGBoost | 5.22 kWh | 34.9% |

MAPE is elevated because usage drops near zero when idle. MAE is the decision metric here, about 19% of the mean near 27 kWh.

The previous 15-minute reading carries about 63% of importance, followed by same time yesterday, weekend flag, day of week and hour. This is expected for an autocorrelated load series.

Load split:

| Load type | Share |
|---|---|
| Maximum_Load | 44.9% |
| Medium_Load | 38.9% |
| Light_Load | 16.2% |

Power factor is the main efficiency lever. Light_Load median is 66.2% and under 90% almost 80% of the time, against 91.7% for Maximum and 96.8% for Medium. An automatic switched capacitor bank for Light_Load periods avoids penalty charges without overcorrection. Peaks favor load shifting out of the 9 AM Thursday window under a demand tariff.

Business read: fewer PF penalty intervals, lower peak-demand charges, and a 15-minute-ahead forecast for scheduling. Even a 5 to 10% cut in penalty and peak exposure is material on an industrial bill.

## Synthetic pipeline - appendix

Hourly synthetic data with furnace, compressor, ventilation and auxiliary breakdown to demonstrate the full method end to end.

```
data/generate_data.py   -> raw_energy_data.csv
clean_features.py       -> clean_energy_data.csv
eda.py                  -> outputs/*.png
train_model.py          -> outputs/metrics.json, predictions.csv, xgb_model.joblib
optimization.py         -> flagged_wastage_periods.csv, equipment_ranking.csv, peak_demand_hours.csv
dashboard/app.py        -> Streamlit dashboard
```

Run:
```bash
pip install -r requirements.txt
python data/generate_data.py
python clean_features.py
python eda.py
python train_model.py
python optimization.py
streamlit run dashboard/app.py
```

## Tech stack

Forecasting: XGBoost, LightGBM, scikit-learn, chronological split, time plus lag features only to avoid target leakage.
Analysis: pandas, numpy, matplotlib, seaborn, SHAP and residual diagnostics.
Serving: Streamlit, Plotly, what-if sliders backed by serialized joblib models.

## Reproducibility

Tested on Python 3.12 on Windows. Install with `pip install -r requirements.txt`. Real pipeline runs in under 2 minutes. Synthetic `train_model.py` includes GridSearch and can take 20 to 40 minutes. Data and models are versioned in the repo. Outputs regenerate under `outputs/` and `real_data/outputs/`.
