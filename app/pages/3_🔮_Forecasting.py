"""
Page 3: Forecasting Engine
===========================
Industry-grade forecasting with Exponential Smoothing, Linear Regression,
and Auto-Regressive models. Live benchmarking on train/test split.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, sys, json, warnings
warnings.filterwarnings("ignore")

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
st.markdown('<p class="brand-subtitle">Production-grade time-series prediction with live model benchmarking and scenario planning</p>', unsafe_allow_html=True)

# Retrieve schema
schema = st.session_state.get("schema", {"date_col": None, "numeric_cols": [], "categorical_cols": []})
num_cols = schema["numeric_cols"]
cat_cols = schema["categorical_cols"]
date_col = schema["date_col"]

if not date_col or len(num_cols) == 0:
    st.warning("⚠️ This page requires a **date** column and at least one **numeric** column for forecasting.")
    st.stop()

daily = load_daily_data(filtered)

if len(daily) < 20:
    st.error("🚨 **Insufficient Historical Data**")
    st.markdown(f"""
    Time-series models require a minimum continuous sequence to capture trends and seasonality.

    | Requirement | Current | Minimum |
    |---|---|---|
    | Historical days | `{len(daily)}` | `20` |
    | Target metric | `{num_cols[0]}` | — |

    **Action**: Upload a dataset spanning at least 20 distinct dates.
    """)
    st.stop()

# ── Resolve target column ────────────────────────────────────────────────
target_col = "daily_revenue" if "daily_revenue" in daily.columns else (
             "daily_orders" if "daily_orders" in daily.columns else daily.columns[1])
target_label = target_col.replace("_", " ").title()

# ════════════════════════════════════════════════════════════════════════════
# FORECASTING MODELS (REAL STATISTICAL ENGINES)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner=False)
def run_ets_forecast(series, horizon):
    """Exponential Smoothing (ETS) — Holt-Winters with additive trend."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        # Try seasonal model if enough data
        sp = min(7, len(series) // 3)  # weekly seasonality
        if sp >= 2 and len(series) >= sp * 2:
            model = ExponentialSmoothing(
                series, trend="add", seasonal="add", seasonal_periods=sp,
                initialization_method="estimated"
            ).fit(optimized=True)
        else:
            model = ExponentialSmoothing(
                series, trend="add", initialization_method="estimated"
            ).fit(optimized=True)
        
        fcast = model.forecast(horizon)
        residuals = model.resid
        return np.maximum(0, fcast.values), residuals.values
    except Exception:
        # Fallback to simple exponential smoothing
        model = ExponentialSmoothing(
            series, initialization_method="estimated"
        ).fit(optimized=True)
        fcast = model.forecast(horizon)
        return np.maximum(0, fcast.values), model.resid.values


@st.cache_data(ttl=600, show_spinner=False)
def run_lr_forecast(series, horizon):
    """Linear Regression with time-based features."""
    from sklearn.linear_model import Ridge
    
    n = len(series)
    X_train = np.column_stack([
        np.arange(n),                            # trend
        np.sin(2 * np.pi * np.arange(n) / 7),   # weekly sin
        np.cos(2 * np.pi * np.arange(n) / 7),   # weekly cos
        np.sin(2 * np.pi * np.arange(n) / 30),  # monthly sin
        np.cos(2 * np.pi * np.arange(n) / 30),  # monthly cos
    ])
    y_train = series.values

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Forecast
    future_idx = np.arange(n, n + horizon)
    X_future = np.column_stack([
        future_idx,
        np.sin(2 * np.pi * future_idx / 7),
        np.cos(2 * np.pi * future_idx / 7),
        np.sin(2 * np.pi * future_idx / 30),
        np.cos(2 * np.pi * future_idx / 30),
    ])
    fcast = model.predict(X_future)
    residuals = y_train - model.predict(X_train)
    return np.maximum(0, fcast), residuals


@st.cache_data(ttl=600, show_spinner=False)
def run_ar_forecast(series, horizon):
    """Auto-Regressive model (AR) with automatic lag selection."""
    from statsmodels.tsa.ar_model import AutoReg
    try:
        max_lags = min(14, len(series) // 3)
        if max_lags < 1:
            max_lags = 1
        model = AutoReg(series, lags=max_lags).fit()
        fcast = model.predict(start=len(series), end=len(series) + horizon - 1)
        residuals = model.resid
        return np.maximum(0, fcast.values), residuals.values
    except Exception:
        # Fallback: simple AR(1)
        model = AutoReg(series, lags=1).fit()
        fcast = model.predict(start=len(series), end=len(series) + horizon - 1)
        return np.maximum(0, fcast.values), model.resid.values


def compute_metrics(actual, predicted):
    """Compute RMSE, MAE, MAPE."""
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.sum() > 0 else 0
    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE": round(mape, 1)}


# ── Train / Test Split ────────────────────────────────────────────────────
series = daily[target_col].values
split_idx = int(len(series) * 0.8)
train_series = pd.Series(series[:split_idx])
test_series = series[split_idx:]
test_horizon = len(test_series)

# ── Benchmark all models ─────────────────────────────────────────────────
model_runners = {
    "Exponential Smoothing": run_ets_forecast,
    "Linear Regression": run_lr_forecast,
    "Auto-Regressive": run_ar_forecast,
}

benchmark_results = {}
for model_name, runner in model_runners.items():
    try:
        pred, _ = runner(train_series, test_horizon)
        pred = pred[:len(test_series)]  # clip to test length
        metrics = compute_metrics(test_series, pred)
        benchmark_results[model_name] = metrics
    except Exception:
        benchmark_results[model_name] = {"RMSE": 999, "MAE": 999, "MAPE": 100}

# ═══════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🧠 Live Model Benchmarking</div>', unsafe_allow_html=True)
st.markdown(f'<p class="section-subtitle">80/20 train-test split on current dataset · {len(train_series)} train days · {test_horizon} test days</p>', unsafe_allow_html=True)

metrics_df = pd.DataFrame(benchmark_results).T.reset_index().rename(columns={"index": "Model"})
best_model_name = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]

model_cols_ui = st.columns(len(metrics_df))
for i, row in metrics_df.iterrows():
    with model_cols_ui[i]:
        is_best = row["Model"] == best_model_name
        badge = " 🏆" if is_best else ""
        kpi_card(
            f"{row['Model']}{badge}",
            f"RMSE: {row['RMSE']:.2f}",
            f"MAE: {row['MAE']:.2f} · MAPE: {row['MAPE']:.1f}%",
            is_best, "🤖"
        )

# Comparison chart
fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(name="RMSE", x=metrics_df["Model"], y=metrics_df["RMSE"],
                          marker_color="#6c63ff", hovertemplate="%{y:.2f}<extra>RMSE</extra>"))
fig_comp.add_trace(go.Bar(name="MAE", x=metrics_df["Model"], y=metrics_df["MAE"],
                          marker_color="#3b82f6", hovertemplate="%{y:.2f}<extra>MAE</extra>"))
fig_comp.update_layout(**PLOTLY_LAYOUT, barmode="group", height=300,
                       title="Model Performance — Lower is Better")
st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# FORECAST VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-title">📈 {target_label} Forecast</div>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">Select a model and forecast horizon to project future performance with confidence intervals</p>', unsafe_allow_html=True)

col_f1, col_f2 = st.columns([1, 3])
with col_f1:
    forecast_days = st.selectbox("📅 Forecast Horizon", [7, 14, 30, 60, 90], index=2)
    model_choice = st.selectbox("🤖 Model", list(model_runners.keys()), 
                                index=list(model_runners.keys()).index(best_model_name))
    confidence_level = st.slider("📊 Confidence Level", 80, 99, 95, step=5)

# Run the selected model on full data
full_series = pd.Series(series)
runner = model_runners[model_choice]
forecast_values, residuals = runner(full_series, forecast_days)

# Confidence intervals from residual standard deviation
residual_std = np.std(residuals) if len(residuals) > 1 else 1.0
z_scores = {80: 1.28, 85: 1.44, 90: 1.645, 95: 1.96, 99: 2.576}
z = z_scores[confidence_level]
ci_width = np.array([residual_std * z * np.sqrt(1 + i * 0.05) for i in range(forecast_days)])
upper = forecast_values + ci_width
lower = np.maximum(0, forecast_values - ci_width)

last_date = pd.to_datetime(daily["date"].max())
forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)

with col_f2:
    fig_fc = go.Figure()
    # Historical
    fig_fc.add_trace(go.Scatter(
        x=daily["date"].tail(90), y=daily[target_col].tail(90),
        name="Historical", mode="lines",
        line=dict(color="#6c63ff", width=2),
        hovertemplate="%{y:,.0f}<extra>Historical</extra>"))
    # Forecast
    fig_fc.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_values,
        name="Forecast", mode="lines",
        line=dict(color="#06b6d4", width=2.5, dash="dash"),
        hovertemplate="%{y:,.0f}<extra>Forecast</extra>"))
    # Confidence band
    fig_fc.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself", fillcolor="rgba(6,182,212,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{confidence_level}% CI",
    ))
    fig_fc.update_layout(**PLOTLY_LAYOUT, height=400, hovermode="x unified",
                         title=f"{model_choice} — Next {forecast_days} Days")
    st.plotly_chart(fig_fc, use_container_width=True)

# Forecast KPIs
st.markdown('<div class="section-title">📊 Forecast Summary</div>', unsafe_allow_html=True)
kc1, kc2, kc3, kc4 = st.columns(4)
with kc1:
    kpi_card("Projected Total", f"{forecast_values.sum():,.0f}", icon="💰")
with kc2:
    kpi_card("Daily Average", f"{forecast_values.mean():,.0f}", icon="📊")
with kc3:
    kpi_card("Peak Day", f"{forecast_values.max():,.0f}", icon="📈")
with kc4:
    kpi_card(f"CI Width (±)", f"{ci_width.mean():,.0f}", icon="📐")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO SIMULATION
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🎛️ What-If Scenario Simulation</div>', unsafe_allow_html=True)
st.markdown(f'<p class="section-subtitle">Adjust business variables to model their impact on projected {target_label}</p>', unsafe_allow_html=True)

sc1, sc2, sc3 = st.columns(3)
with sc1:
    demand_change = st.slider("📈 Demand Shift (%)", -50, 100, 0, step=5)
with sc2:
    price_change = st.slider("💰 Price Adjustment (%)", -30, 50, 0, step=5)
with sc3:
    external_impact = st.select_slider("🌍 External Factor",
                                       options=["Crisis", "Downturn", "Stable", "Growth", "Boom"],
                                       value="Stable")

impact_mults = {"Crisis": 0.55, "Downturn": 0.80, "Stable": 1.0, "Growth": 1.15, "Boom": 1.35}

scenario_values = forecast_values.copy()
scenario_values = scenario_values * (1 + demand_change / 100) * (1 + price_change / 100) * impact_mults[external_impact]

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
                           title="Base Forecast vs Scenario Projection", hovermode="x unified")
st.plotly_chart(fig_scenario, use_container_width=True)

delta = scenario_values.sum() - forecast_values.sum()
pct_delta = (delta / forecast_values.sum() * 100) if forecast_values.sum() > 0 else 0
kc1, kc2 = st.columns(2)
with kc1:
    kpi_card("Scenario Projection", f"{scenario_values.sum():,.0f}",
             f"{pct_delta:+.1f}% vs base", pct_delta > 0, "🎯")
with kc2:
    kpi_card("Net Impact", f"{abs(delta):,.0f}",
             "Uplift" if delta > 0 else "Decline", delta > 0, "⚡")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">⚠️ Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown(f'<p class="section-subtitle">Statistical outlier detection on {target_label} using rolling Z-score (2.5σ)</p>', unsafe_allow_html=True)

rolling_mean = daily[target_col].rolling(14, min_periods=1).mean()
rolling_std = daily[target_col].rolling(14, min_periods=1).std().fillna(0)
upper_bound = rolling_mean + 2.5 * rolling_std
lower_bound = rolling_mean - 2.5 * rolling_std
anomalies = daily[(daily[target_col] > upper_bound) | (daily[target_col] < lower_bound)]

fig_anom = go.Figure()
fig_anom.add_trace(go.Scatter(x=daily["date"], y=daily[target_col],
                              name=target_label, mode="lines",
                              line=dict(color="#6c63ff", width=1.5),
                              hovertemplate="%{y:,.0f}<extra></extra>"))
fig_anom.add_trace(go.Scatter(x=daily["date"], y=upper_bound,
                              name="Upper Bound (2.5σ)", mode="lines",
                              line=dict(color="#94a3b8", width=1, dash="dash")))
fig_anom.add_trace(go.Scatter(x=daily["date"], y=lower_bound,
                              name="Lower Bound (2.5σ)", mode="lines",
                              line=dict(color="#94a3b8", width=1, dash="dash")))
if len(anomalies) > 0:
    fig_anom.add_trace(go.Scatter(x=anomalies["date"], y=anomalies[target_col],
                                  name="Anomalies", mode="markers",
                                  marker=dict(color="#ef4444", size=8, symbol="diamond",
                                              line=dict(color="white", width=1)),
                                  hovertemplate="<b>Anomaly</b><br>%{y:,.0f}<extra></extra>"))
fig_anom.update_layout(**PLOTLY_LAYOUT, height=360, hovermode="x unified")
st.plotly_chart(fig_anom, use_container_width=True)

anom_pct = (len(anomalies) / len(daily) * 100) if len(daily) > 0 else 0
st.info(f"🔍 Detected **{len(anomalies)} anomalous days** out of {len(daily)} total ({anom_pct:.1f}%)")
