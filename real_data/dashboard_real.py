"""
Streamlit dashboard — real Steel Industry Energy Consumption data.
Run with: streamlit run real_data/dashboard_real.py
"""
import streamlit as st
import pandas as pd
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Energy Forecasting — Real Data", layout="wide")

DATA_DIR = "real_data"
OUT_DIR = "real_data/outputs"

FORECAST_FEATURES = [
    "hour", "dayofweek", "is_weekend", "month",
    "usage_lag_1step", "usage_lag_1day", "usage_roll_mean_1day",
]

@st.cache_data
def load_data():
    clean = pd.read_csv(f"{DATA_DIR}/clean_steel_industry_data.csv", parse_dates=["timestamp"])
    preds = pd.read_csv(f"{OUT_DIR}/predictions.csv", parse_dates=["timestamp"])
    importance = pd.read_csv(f"{OUT_DIR}/feature_importance.csv", index_col=0)
    with open(f"{OUT_DIR}/metrics.json") as f:
        metrics = json.load(f)
    seg_ranking = pd.read_csv(f"{OUT_DIR}/loadtype_ranking.csv", index_col=0)
    peaks = pd.read_csv(f"{OUT_DIR}/peak_demand_periods.csv", parse_dates=["timestamp"])
    return clean, preds, importance, metrics, seg_ranking, peaks

clean, preds, importance, metrics, seg_ranking, peaks = load_data()

@st.cache_resource
def load_models():
    import joblib, os
    xgb = joblib.load(f"{OUT_DIR}/xgb_forecast_model.joblib")
    lgb_path = f"{OUT_DIR}/lgb_forecast_model.joblib"
    lgb = joblib.load(lgb_path) if os.path.exists(lgb_path) else None
    return xgb, lgb

xgb_model, lgb_model = load_models()

def _model_features(model, fallback):
    for attr in ("feature_names_in_", "feature_names"):
        try:
            names = list(getattr(model, attr))
            if names and all(isinstance(n, str) for n in names):
                return names
        except Exception:
            pass
    try:
        names = list(model.get_booster().feature_names)
        if names:
            return names
    except Exception:
        pass
    return list(fallback)

try:
    _from_metrics = list(metrics.get("forecast_features", []))
except Exception:
    _from_metrics = []
FORECAST_FEATURES = _model_features(xgb_model, _from_metrics or FORECAST_FEATURES)

st.title("Energy Consumption Forecasting — Real Plant Data")
st.caption("DAEWOO Steel Co., South Korea — 15-min interval, 2018 (UCI/Kaggle Steel Industry Energy Consumption dataset)")

# KPI row
low_pf_pct = (clean["Lagging_Current_Power_Factor"] < 90).mean() * 100
c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecast MAPE", f"{metrics['primary_forecast_model']['xgboost']['MAPE_pct']}%")
c2.metric("Forecast MAE", f"{metrics['primary_forecast_model']['xgboost']['MAE_kWh']} kWh")
c3.metric("Time below 90% PF", f"{low_pf_pct:.1f}%")
c4.metric("Total annual usage", f"{clean['Usage_kWh'].sum()/1000:,.0f} MWh")

st.info("Forecast uses only time-of-day/week and past-usage features (a genuine "
        "forward-looking forecast). Simultaneous electrical readings (reactive power, "
        "power factor) are shown separately below as a diagnostic signal, not as forecast inputs.")

st.divider()

# Actual vs Forecast
st.subheader("Actual vs Forecast (test period)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=preds["timestamp"], y=preds["Usage_kWh"], name="Actual", line=dict(color="#2563eb")))
fig.add_trace(go.Scatter(x=preds["timestamp"], y=preds["predicted_kwh"], name="Forecast", line=dict(color="#f97316", dash="dot")))
fig.update_layout(height=400, xaxis_title="Time", yaxis_title="Usage (kWh)")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# What-If Scenario
st.subheader("What-If Scenario: Adjust Parameters & See Forecast Impact")
st.caption("Adjust the sliders below to see how changes in operating conditions affect the energy usage forecast")

col1, col2, col3 = st.columns(3)
with col1:
    adj_hour = st.slider("Hour of day", min_value=0, max_value=23,
                         value=int(clean["hour"].median()))
    adj_dow = st.selectbox("Day of week",
                           options=list(range(7)),
                           format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
                           index=int(clean["dayofweek"].median()))
with col2:
    adj_month = st.slider("Month", min_value=1, max_value=12,
                          value=int(clean["month"].median()))
    adj_lag_1step = st.slider("Usage 15 min ago (kWh)",
                              min_value=float(clean["usage_lag_1step"].min()),
                              max_value=float(clean["usage_lag_1step"].quantile(0.99)),
                              value=float(clean["usage_lag_1step"].median()))
with col3:
    adj_lag_1day = st.slider("Usage same time yesterday (kWh)",
                             min_value=float(clean["usage_lag_1day"].min()),
                             max_value=float(clean["usage_lag_1day"].quantile(0.99)),
                             value=float(clean["usage_lag_1day"].median()))
    adj_roll = st.slider("Rolling 24h mean (kWh)",
                         min_value=float(clean["usage_roll_mean_1day"].min()),
                         max_value=float(clean["usage_roll_mean_1day"].quantile(0.99)),
                         value=float(clean["usage_roll_mean_1day"].median()))
    roll_std_default = float(clean["Usage_kWh"].rolling(96, min_periods=1).std().fillna(0).median())
    adj_roll_std = st.slider("Rolling 24h std (kWh)", min_value=0.0,
                             max_value=float(clean["Usage_kWh"].std() * 2),
                             value=roll_std_default)

adj_is_weekend = 1 if adj_dow >= 5 else 0
adj_is_weekday = 1 if adj_dow < 5 else 0
adj_minute, adj_day, adj_year = 0, 15, 2018
adj_hour_sin = float(np.sin(2 * np.pi * (adj_hour + adj_minute / 60) / 24))
adj_hour_cos = float(np.cos(2 * np.pi * (adj_hour + adj_minute / 60) / 24))
adj_dow_sin = float(np.sin(2 * np.pi * adj_dow / 7))
adj_dow_cos = float(np.cos(2 * np.pi * adj_dow / 7))
adj_month_sin = float(np.sin(2 * np.pi * adj_month / 12))
adj_month_cos = float(np.cos(2 * np.pi * adj_month / 12))

_feat_map = {
    "hour": adj_hour, "minute": adj_minute, "dayofweek": adj_dow,
    "is_weekend": adj_is_weekend, "is_weekday": adj_is_weekday, "month": adj_month,
    "day": adj_day, "year": adj_year, "hour_sin": adj_hour_sin, "hour_cos": adj_hour_cos,
    "dow_sin": adj_dow_sin, "dow_cos": adj_dow_cos, "month_sin": adj_month_sin,
    "month_cos": adj_month_cos, "usage_lag_1step": adj_lag_1step,
    "usage_lag_1day": adj_lag_1day, "usage_roll_mean_1day": adj_roll,
    "usage_roll_std_1day": adj_roll_std,
}
xgb_if_features = pd.DataFrame([[_feat_map.get(c, 0) for c in FORECAST_FEATURES]],
                                columns=list(FORECAST_FEATURES))
try:
    xgb_if_pred = float(xgb_model.predict(xgb_if_features)[0])
except Exception:
    # Fallback for XGBoost version strictness on feature names
    xgb_if_pred = float(xgb_model.predict(xgb_if_features.values)[0])
try:
    lgb_if_pred = float(lgb_model.predict(xgb_if_features)[0]) if lgb_model is not None else None
except Exception:
    lgb_if_pred = float(lgb_model.predict(xgb_if_features.values)[0]) if lgb_model is not None else None

col_a, col_b = st.columns(2)
with col_a:
    st.metric("XGBoost Forecast (kWh per 15-min)", f"{xgb_if_pred:.1f}")
with col_b:
    if lgb_if_pred is not None:
        st.metric("LightGBM Forecast (kWh per 15-min)", f"{lgb_if_pred:.1f}")
    else:
        st.metric("Recent actual (last 96 intervals avg)", f"{clean['Usage_kWh'].tail(96).mean():.1f} kWh")

st.caption(f"Baseline annual mean: {clean['Usage_kWh'].mean():.1f} kWh per 15-min interval")

st.divider()

# Feature importance + hourly pattern
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Feature Importance")
    fig2 = px.bar(importance.sort_values("importance"), orientation="h", labels={"value": "Importance", "index": ""})
    fig2.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.subheader("Average Usage by Hour of Day")
    hourly = clean.groupby("hour")["Usage_kWh"].mean().reset_index()
    fig3 = px.bar(hourly, x="hour", y="Usage_kWh", labels={"Usage_kWh": "Avg kWh", "hour": "Hour of day"})
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# Segment ranking & peak demand
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Load Type Ranking (annual consumption)")
    st.dataframe(seg_ranking, use_container_width=True)
    fig4 = px.pie(seg_ranking, names=seg_ranking.index, values="share_pct", hole=0.4)
    fig4.update_layout(height=350)
    st.plotly_chart(fig4, use_container_width=True)

with col_d:
    st.subheader("Peak Demand Periods (top 5% of 15-min readings)")
    st.metric("Flagged peak intervals", f"{len(peaks):,}")
    peak_by_hour = peaks.groupby("hour").size().reset_index(name="count")
    fig5 = px.bar(peak_by_hour, x="hour", y="count",
                  labels={"count": "Peak intervals", "hour": "Hour of day"})
    fig5.update_layout(height=350)
    st.plotly_chart(fig5, use_container_width=True)

st.divider()

# Power factor diagnostic
st.subheader("Power Factor by Load Type")
fig6 = px.box(clean, x="Load_Type", y="Lagging_Current_Power_Factor",
              category_orders={"Load_Type": ["Light_Load", "Medium_Load", "Maximum_Load"]})
fig6.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="90% penalty threshold")
fig6.update_layout(height=350)
st.plotly_chart(fig6, use_container_width=True)

st.divider()

# Peak table + recommendations
st.subheader("Peak Demand Periods (latest)")
st.dataframe(
    peaks[["timestamp", "Usage_kWh", "Load_Type", "Lagging_Current_Power_Factor"]]
    .sort_values("timestamp", ascending=False).head(20),
    use_container_width=True,
)

with open(f"{OUT_DIR}/recommendations.txt", encoding="utf-8") as f:
    st.text(f.read())
