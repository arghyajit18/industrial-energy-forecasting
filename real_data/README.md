# Real-Data Pipeline — Steel Industry Energy Consumption

Real 15-minute interval energy data from DAEWOO Steel Co. (Gwangyang, South
Korea), full year 2018, 34,944 clean rows after processing. Source: UCI ML
Repository / Kaggle "Steel Industry Energy Consumption" dataset.

This sits alongside `../` (the synthetic pipeline) — same structure, adapted
to the real schema. Run in order:
```bash
python real_data/clean_features_real.py
python real_data/eda_real.py
python real_data/train_model_real.py
python real_data/optimization_real.py
streamlit run real_data/dashboard_real.py
```

## Key differences from the synthetic pipeline

**No per-equipment sub-metering.** Real plants rarely expose that outside
internal SCADA — this dataset has one main meter. `Load_Type`
(Light/Medium/Maximum Load) is used as the closest real proxy for
segment-level ranking.

**Forecast vs. same-timestep estimation — kept strictly separate.**
`Lagging/Leading Current Reactive Power`, `Power Factor`, and `CO2(tCO2)` are
all measured at the *same timestamp* as `Usage_kWh` (CO2 is in fact a
near-deterministic function of usage — 0.99 correlation — so it's excluded
entirely as target leakage). A genuine forecast can only use information
available in advance, so:
- **Primary model** (the one that counts as "forecasting"): time features
  (hour, day-of-week, month, weekend) + lagged usage only.
- **Secondary model**: adds the simultaneous electrical readings, shown only
  to illustrate how tightly usage correlates with them — explicitly labeled
  as not a forecast in the code and the dashboard.

This distinction matters if you present this project to anyone with a
data-science background — using same-timestep sensor readings to "forecast"
the same timestep's target is a common mistake, and this pipeline
deliberately avoids it.

## Results

| Model | MAE (kWh) | MAPE |
|---|---|---|
| **Primary forecast** — Linear Regression | 6.54 | 55.8% |
| **Primary forecast** — XGBoost | **5.10** | **31.4%** |
| Secondary same-timestep estimation — XGBoost | 1.23 | 10.2% |

MAPE looks high in absolute terms because usage ranges from ~0 kWh (plant
idle) to 157 kWh (Maximum_Load) — small errors at low-usage hours blow up the
percentage metric. MAE (5.1 kWh average error against a 27 kWh mean) is the
more meaningful number here and is a legitimate, defensible result to quote.

**Forecast feature importance**: `usage_lag_1step` (63%) dominates — the most
recent reading is the strongest predictor, as expected for a
autocorrelated, high-frequency (15-min) series. Day-of-week and hour follow.

## The standout real finding: power factor

- **Median lagging power factor collapses to 66.2% during Light_Load
  periods**, vs. 91.7% at Maximum_Load and 96.8% at Medium_Load.
- The plant spends **54.8% of all 15-minute intervals below the 90% power
  factor threshold** most electricity boards use for penalty billing.
- **Recommendation**: an automatic (switched, not fixed) power-factor
  correction capacitor bank, engaged specifically during Light_Load periods.
  A fixed bank sized for Maximum_Load would over-correct at Light_Load and
  cause leading-PF penalties instead.

This is the strongest deliverable in the whole project — it's a concrete,
quantified, real electrical-engineering finding (not a generic ML metric),
directly relevant to an EEE background, and immediately actionable by a
plant's electrical maintenance team.

## Peak demand
Peak-demand periods (top 5% of 15-min readings) concentrate around **9:00 AM**
and on **Thursdays** — useful if the plant is on a demand-based (not just
energy-based) electricity tariff, since shifting non-critical load out of
this window reduces the peak-demand charge independent of total energy used.

## Resume bullet (real-data version)
> Built and validated an ML-based energy forecasting model (XGBoost, time+lag
> features only to avoid target leakage) on a real 15-min interval industrial
> dataset (34,944 rows), achieving 5.1 kWh MAE; identified a power-factor
> deficiency (54.8% of readings below the 90% penalty threshold, worst during
> low-load periods) and recommended automatic capacitor-bank correction.
