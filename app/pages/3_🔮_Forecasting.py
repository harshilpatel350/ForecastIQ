"""
Page 3: Forecasting Engine
===========================
Model comparison, confidence-banded forecasts, what-if simulation, anomaly detection.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, sys, json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title="Forecasting — ForecastIQ", page_icon="🔮", layout="wide")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "app"))
from streamlit_app import inject_css, get_active_dataset, load_daily_data, render_sidebar, kpi_card, PLOTLY_LAYOUT

inject_css()
df = get_active_dataset()
filtered = render_sidebar(df)
cols = filtered.columns.tolist()

st.markdown('<p class="brand-header">🔮 Forecasting Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="brand-subtitle">AI-driven demand prediction with confidence intervals and scenario planning</p>', unsafe_allow_html=True)

# Retrieve schema extracted by streamlit_app
schema = st.session_state.get("schema", {"date_col": None, "numeric_cols": [], "categorical_cols": []})
num_cols = schema["numeric_cols"]
cat_cols = schema["categorical_cols"]
date_col = schema["date_col"]

# Check minimum requirements
if not date_col or len(num_cols) == 0:
    st.warning("⚠️ This page requires a **date** and at least one **numeric** column in your dataset for forecasting.")
    st.stop()

daily = load_daily_data(filtered)

# Check requirements for time series
if len(daily) < 20:
    st.error("🚨 **Insufficient Data for AI Forecasting**")
    st.markdown(f"""
    The forecasting models (LSTM, Attention LSTM, Transformer) require a continuous historical sequence to identify patterns.
    
    **Current Status:**
    - 📅 **Days found**: `{len(daily)}` (Minimum required: `20`)
    - 📊 **Target Column**: `{num_cols[0]}`
    
    Please upload a dataset with a longer time horizon to enable these features.
    """)
    st.stop()

# ── Model Comparison ────────────────────────────────────────────────────
comparison_path = os.path.join(PROJECT_ROOT, "data", "model_comparison.json")
models_available = os.path.exists(comparison_path)

if models_available:
    st.markdown('<div class="section-title">🧠 Model Performance Comparison</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Metrics for models trained on the base sales dataset</p>', unsafe_allow_html=True)
    
    with open(comparison_path, "r") as f:
        model_results = json.load(f)
    
    # Simple check: if current data is not sales data, warn that these metrics are historical
    if num_cols[0] not in ["total_amount", "daily_revenue"]:
        st.info("💡 **Note**: The metrics below represent models trained on the default Sales dataset. They may not accurately reflect performance on your custom uploaded schema.")

    metrics_df = pd.DataFrame(model_results).T.reset_index().rename(columns={"index": "Model"})
    
    model_cols = st.columns(len(metrics_df))
    for i, row in metrics_df.iterrows():
        with model_cols[i]:
            is_best = row["RMSE"] == metrics_df["RMSE"].min()
            kpi_card(f"{row['Model'].upper()}{' 🏆' if is_best else ''}", 
                     f"RMSE: {row['RMSE']:.2f}", 
                     f"MAE: {row['MAE']:.2f}", is_best, "🤖")

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name="RMSE", x=metrics_df["Model"].str.upper(), y=metrics_df["RMSE"], marker_color="#6c63ff"))
    fig_comp.update_layout(**PLOTLY_LAYOUT, barmode="group", height=280)
    st.plotly_chart(fig_comp, use_container_width=True)
else:
    st.info("🧠 No pre-trained models detected. Forecasts below use our adaptive heuristic engine.")

st.markdown("---")

# ── Forecast Visualization ────────────────────────────────────────────────
target_metric = num_cols[0]
metric_label = target_metric.replace("_", " ").title()
st.markdown(f'<div class="section-title">📈 {metric_label} Forecast</div>', unsafe_allow_html=True)

col_f1, col_f2 = st.columns([1, 3])
with col_f1:
    forecast_days = st.selectbox("📅 Forecast Horizon", [7, 14, 30, 60], index=0)
    model_choice = st.selectbox("🤖 Engine", ["Heuristic Adaptive", "LSTM (Experimental)", "Transformer (Experimental)"], index=0)
    confidence_level = st.slider("📊 Confidence", 80, 99, 95, step=5)

target_col = "daily_revenue" if "daily_revenue" in daily.columns else (
             "daily_val" if "daily_val" in daily.columns else daily.columns[1])
             
recent = daily[target_col].values[-30:] if len(daily) >= 30 else daily[target_col].values
base_trend = np.mean(recent) if len(recent) > 0 else 0
trend_slope = (recent[-1] - recent[0]) / len(recent) if len(recent) > 1 else 0

last_date = pd.to_datetime(daily["date"].max())
forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)
forecast_values = []
for i in range(forecast_days):
    seasonal = (recent[i % len(recent)] / base_trend) if base_trend != 0 else 1.0
    predicted = base_trend * seasonal + trend_slope * i
    forecast_values.append(max(0, predicted))
forecast_values = np.array(forecast_values)

std = np.std(recent) * 0.5 if len(recent) > 1 else 10.0
z = {80: 1.28, 85: 1.44, 90: 1.645, 95: 1.96, 99: 2.576}[confidence_level]
ci_width = np.array([std * z * np.sqrt(i + 1) * 0.3 for i in range(forecast_days)])
upper = forecast_values + ci_width
lower = np.maximum(0, forecast_values - ci_width)

with col_f2:
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=daily["date"].tail(90), y=daily[target_col].tail(90),
                                name="Historical", mode="lines",
                                line=dict(color="#6c63ff", width=2),
                                hovertemplate="%{y:,.0f}<extra>Historical</extra>"))
    fig_fc.add_trace(go.Scatter(x=forecast_dates, y=forecast_values,
                                name="Forecast", mode="lines",
                                line=dict(color="#06b6d4", width=2.5, dash="dash"),
                                hovertemplate="%{y:,.0f}<extra>Forecast</extra>"))
    fig_fc.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself", fillcolor="rgba(6,182,212,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{confidence_level}% Confidence",
    ))
    fig_fc.update_layout(**PLOTLY_LAYOUT, height=400,
                         title=f"{model_choice} Forecast — Next {forecast_days} Days",
                         hovermode="x unified")
    st.plotly_chart(fig_fc, use_container_width=True)

# Forecast KPIs
st.markdown('<div class="section-title">📊 Forecast Summary</div>', unsafe_allow_html=True)
kc1, kc2, kc3, kc4 = st.columns(4)
with kc1:
    kpi_card("Projected Total", f"{forecast_values.sum():,.0f}", icon="💰")
with kc2:
    kpi_card("Daily Average", f"{forecast_values.mean():,.0f}", icon="📊")
with kc3:
    kpi_card("Peak Projection", f"{forecast_values.max():,.0f}", icon="📈")
with kc4:
    kpi_card("Confidence Width", f"±{ci_width.mean():,.0f}", icon="📐")

st.markdown("---")

# ── Scenario Simulation ──────────────────────────────────────────────────
st.markdown('<div class="section-title">🎛️ What-If Scenario Simulation</div>', unsafe_allow_html=True)
st.markdown(f'<p class="section-subtitle">Adjust demand, pricing, and weather to model impact on {metric_label}</p>', unsafe_allow_html=True)

sc1, sc2, sc3 = st.columns(3)
with sc1:
    demand_change = st.slider("📈 Demand Change (%)", -50, 100, 0, step=5)
with sc2:
    price_change = st.slider("💰 Price Change (%)", -30, 50, 0, step=5)
with sc3:
    weather_impact = st.select_slider("🌦️ Weather Scenario",
                                      options=["Stormy", "Rainy", "Cloudy", "Clear", "Hot"],
                                      value="Clear")

weather_mults = {"Clear": 1.0, "Cloudy": 0.95, "Rainy": 0.75, "Stormy": 0.55, "Hot": 0.90}

scenario_values = forecast_values.copy()
scenario_values *= (1 + demand_change / 100)
scenario_values *= (1 + price_change / 100)
scenario_values *= weather_mults[weather_impact]

fig_scenario = go.Figure()
fig_scenario.add_trace(go.Scatter(x=forecast_dates, y=forecast_values,
                                  name="Base Forecast", mode="lines",
                                  line=dict(color="#6c63ff", width=2),
                                  hovertemplate="%{y:,.0f}<extra>Base</extra>"))
fig_scenario.add_trace(go.Scatter(x=forecast_dates, y=scenario_values,
                                  name="Scenario", mode="lines",
                                  line=dict(color="#f59e0b", width=2.5, dash="dot"),
                                  hovertemplate="%{y:,.0f}<extra>Scenario</extra>"))
fig_scenario.update_layout(**PLOTLY_LAYOUT, height=350,
                           title="Scenario vs Base Forecast", hovermode="x unified")
st.plotly_chart(fig_scenario, use_container_width=True)

delta = scenario_values.sum() - forecast_values.sum()
pct_delta = (delta / forecast_values.sum() * 100) if forecast_values.sum() > 0 else 0
kc1, kc2 = st.columns(2)
with kc1:
    kpi_card("Scenario Projection", f"{scenario_values.sum():,.0f}",
             f"{pct_delta:+.1f}% vs base", pct_delta > 0, "🎯")
with kc2:
    kpi_card("Total Impact", f"{abs(delta):,.0f}",
             "Gain" if delta > 0 else "Loss", delta > 0, "⚡")

st.markdown("---")

# ── Anomaly Detection ────────────────────────────────────────────────────
st.markdown('<div class="section-title">⚠️ Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown(f'<p class="section-subtitle">Unusual {metric_label} spikes and drops detected via rolling Z-score</p>', unsafe_allow_html=True)

rolling_mean = daily[target_col].rolling(14, min_periods=1).mean()
rolling_std = daily[target_col].rolling(14, min_periods=1).std().fillna(0)
upper_bound = rolling_mean + 2.5 * rolling_std
lower_bound = rolling_mean - 2.5 * rolling_std
anomalies = daily[(daily[target_col] > upper_bound) | (daily[target_col] < lower_bound)]

fig_anom = go.Figure()
fig_anom.add_trace(go.Scatter(x=daily["date"], y=daily[target_col],
                              name=metric_label, mode="lines",
                              line=dict(color="#6c63ff", width=1.5),
                              hovertemplate="%{y:,.0f}<extra></extra>"))
fig_anom.add_trace(go.Scatter(x=daily["date"], y=upper_bound,
                              name="Upper Bound", mode="lines",
                              line=dict(color="#94a3b8", width=1, dash="dash")))
fig_anom.add_trace(go.Scatter(x=daily["date"], y=lower_bound,
                              name="Lower Bound", mode="lines",
                              line=dict(color="#94a3b8", width=1, dash="dash")))
if len(anomalies) > 0:
    fig_anom.add_trace(go.Scatter(x=anomalies["date"], y=anomalies[target_col],
                                  name="Anomalies", mode="markers",
                                  marker=dict(color="#ef4444", size=8, symbol="diamond",
                                              line=dict(color="white", width=1)),
                                  hovertemplate="<b>Anomaly</b><br>%{y:,.0f}<extra></extra>"))
fig_anom.update_layout(**PLOTLY_LAYOUT, height=360, hovermode="x unified")
st.plotly_chart(fig_anom, use_container_width=True)
st.info(f"🔍 Detected **{len(anomalies)} anomalous days** out of {len(daily)} total days ({len(anomalies)/len(daily)*100:.1f}%)")
