"""
Streamlit dashboard: Energy Consumption Forecasting & Optimization
Run with: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Energy Consumption Forecasting", layout="wide")

DATA_DIR = "data"
OUT_DIR = "outputs"

EQUIPMENT_COLS = ["furnace_kwh", "compressor_kwh", "ventilation_kwh", "auxiliary_kwh"]

@st.cache_data
def load_data():
    clean = pd.read_csv(f"{DATA_DIR}/clean_energy_data.csv", parse_dates=["timestamp"])
    preds = pd.read_csv(f"{OUT_DIR}/predictions.csv", parse_dates=["timestamp"])
    importance = pd.read_csv(f"{OUT_DIR}/feature_importance.csv", index_col=0)
    with open(f"{OUT_DIR}/metrics.json") as f:
        metrics = json.load(f)
    wastage = pd.read_csv(f"{OUT_DIR}/flagged_wastage_periods.csv", parse_dates=["timestamp"])
    equip_ranking = pd.read_csv(f"{OUT_DIR}/equipment_ranking.csv", index_col=0)
    peaks = pd.read_csv(f"{OUT_DIR}/peak_demand_hours.csv", parse_dates=["timestamp"])
    return clean, preds, importance, metrics, wastage, equip_ranking, peaks

clean, preds, importance, metrics, wastage, equip_ranking, peaks = load_data()

# Load models for what-if scenarios
@st.cache_resource
def load_models():
    import joblib
    xgb = joblib.load(f"{OUT_DIR}/xgb_model.joblib")
    lgb = joblib.load(f"{OUT_DIR}/lgb_model.joblib")
    return xgb, lgb

xgb_model, lgb_model = load_models()

st.title("Energy Consumption Forecasting & Optimization")
st.caption("Prototype dashboard — synthetic plant-utility data, XGBoost forecasting model")

# KPI row ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model MAPE", f"{metrics['xgboost (grid-tuned)']['MAPE_pct']}%")
c2.metric("Model MAE", f"{metrics['xgboost (grid-tuned)']['MAE_kWh']} kWh")
c3.metric("Flagged wastage hours", f"{len(wastage):,}")
c4.metric("Total annual consumption", f"{clean['energy_kwh'].sum()/1000:,.0f} MWh")

st.divider()

# Actual vs Predicted ---
st.subheader("Actual vs Predicted Consumption (test period)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=preds["timestamp"], y=preds["energy_kwh"], name="Actual", line=dict(color="#2563eb")))
fig.add_trace(go.Scatter(x=preds["timestamp"], y=preds["predicted_kwh"], name="Predicted", line=dict(color="#f97316", dash="dot")))
fig.update_layout(height=400, xaxis_title="Time", yaxis_title="Energy (kWh)")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# What-If Scenario ---
st.subheader("What-If Scenario: Adjust Parameters & See Forecast Impact")
st.caption("Adjust the sliders below to see how changes in operating parameters affect the energy consumption forecast")

col1, col2, col3 = st.columns(3)
with col1:
    adj_production_load = st.slider("Production load (tonnes/hr)", 
                                    min_value=int(clean['production_load_t'].min()), 
                                    max_value=int(clean['production_load_t'].max()),
                                    value=int(clean['production_load_t'].median()))
with col2:
    adj_equipment_util = st.slider("Equipment utilization (%)", 
                                   min_value=int(clean['equipment_utilization_pct'].min()), 
                                   max_value=int(clean['equipment_utilization_pct'].max()),
                                   value=int(clean['equipment_utilization_pct'].median()))
with col3:
    adj_ambient_temp = st.slider("Ambient temperature (°C)", 
                                 min_value=int(clean['ambient_temp_c'].min()), 
                                 max_value=int(clean['ambient_temp_c'].max()),
                                 value=int(clean['ambient_temp_c'].median()))

# Prepare input features for what-if
import numpy as np
hour_val = pd.Timestamp.now().hour
dow_val = pd.Timestamp.now().dayofweek
month_val = pd.Timestamp.now().month

hour_sin = np.sin(2 * np.pi * hour_val / 24)
hour_cos = np.cos(2 * np.pi * hour_val / 24)
dow_sin = np.sin(2 * np.pi * dow_val / 7)
dow_cos = np.cos(2 * np.pi * dow_val / 7)
month_sin = np.sin(2 * np.pi * month_val / 12)
month_cos = np.cos(2 * np.pi * month_val / 12)

# XGBoost what-if prediction
MODEL_FEATURES = [
    "production_load_t", "equipment_utilization_pct", "ambient_temp_c",
    "equipment_run_frac", "hour", "dayofweek", "is_weekend", "month",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "energy_lag_1h", "energy_lag_24h", "energy_roll_mean_24h",
]
xgb_if_df = pd.DataFrame([[
    adj_production_load, adj_equipment_util, adj_ambient_temp,
    0, hour_val, dow_val, 1 if dow_val >= 5 else 0, month_val,
    hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos,
    0, 0, 0,
]], columns=MODEL_FEATURES)
try:
    xgb_if_pred = float(xgb_model.predict(xgb_if_df)[0])
except Exception:
    xgb_if_pred = float(xgb_model.predict(xgb_if_df.values)[0])

# LightGBM what-if prediction
try:
    lgb_if_pred = float(lgb_model.predict(xgb_if_df)[0])
except Exception:
    lgb_if_pred = float(lgb_model.predict(xgb_if_df.values)[0])

col_a, col_b = st.columns(2)
with col_a:
    st.metric("XGBoost Forecast (kWh)", f"{xgb_if_pred:.1f}")
with col_b:
    st.metric("LightGBM Forecast (kWh)", f"{lgb_if_pred:.1f}")

st.caption(f"Baseline actual consumption around this time: {clean['energy_kwh'].tail(24).mean():.1f} kWh average")

st.divider()

# Feature importance + hourly pattern ---
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Feature Importance")
    fig2 = px.bar(importance.sort_values("importance"), orientation="h", labels={"value": "Importance", "index": ""})
    fig2.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.subheader("Average Consumption by Hour of Day")
    hourly = clean.groupby("hour")["energy_kwh"].mean().reset_index()
    fig3 = px.bar(hourly, x="hour", y="energy_kwh", labels={"energy_kwh": "Avg kWh", "hour": "Hour of day"})
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# Equipment breakdown & peak demand ---
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Equipment Ranking (annual consumption)")
    st.dataframe(equip_ranking, use_container_width=True)
    fig4 = px.pie(equip_ranking, names=equip_ranking.index, values="share_pct", hole=0.4)
    fig4.update_layout(height=350)
    st.plotly_chart(fig4, use_container_width=True)

with col_d:
    st.subheader("Peak Demand Hours (top 5% by total load)")
    st.metric("Flagged peak hours", f"{len(peaks):,}")
    dominant_counts = peaks["dominant_equipment"].value_counts().reset_index()
    dominant_counts.columns = ["equipment", "count"]
    fig5 = px.bar(dominant_counts, x="equipment", y="count",
                  labels={"count": "Peak hours dominated by"})
    fig5.update_layout(height=350)
    st.plotly_chart(fig5, use_container_width=True)

st.divider()

# Wastage / anomaly alerts ---
st.subheader("Flagged Wastage Periods (idle equipment, disproportionate draw)")
st.dataframe(
    wastage[["timestamp", "production_load_t", "equipment_utilization_pct", "energy_kwh"]]
    .sort_values("timestamp", ascending=False)
    .head(20),
    use_container_width=True,
)

with open(f"{OUT_DIR}/recommendations.txt") as f:
    st.text(f.read())
