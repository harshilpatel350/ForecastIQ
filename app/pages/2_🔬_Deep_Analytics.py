"""
Page 2: Deep Analytics
======================
Drill-down charts, heatmaps, cohort analysis, weather & discount insights.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title="Deep Analytics — ForecastIQ", page_icon="🔬", layout="wide")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "app"))
from streamlit_app import inject_css, load_app_data, render_sidebar, kpi_card, PLOTLY_LAYOUT

inject_css()
df = load_app_data()
filtered = render_sidebar(df)

st.markdown('<p class="brand-header">🔬 Deep Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="brand-subtitle">Uncover hidden patterns through multi-dimensional drill-down analysis</p>', unsafe_allow_html=True)

# ── Drill-Down: City → Restaurant → Product ──────────────────────────────
st.markdown('<div class="section-title">🏙️ City → Restaurant → Product Drill-Down</div>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">Click through the hierarchy to explore revenue at every level</p>', unsafe_allow_html=True)

city_rev = filtered.groupby("city")["total_amount"].sum().sort_values(ascending=False).reset_index()
selected_drill_city = st.selectbox("Select a city to drill into:", ["— Select a city —"] + city_rev["city"].tolist(), key="drill_city")

if selected_drill_city != "— Select a city —":
    city_data = filtered[filtered["city"] == selected_drill_city]

    # City-level KPIs
    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        kpi_card("City Revenue", f"₹{city_data['total_amount'].sum():,.0f}", icon="💰")
    with kc2:
        kpi_card("Orders", f"{len(city_data):,}", icon="📦")
    with kc3:
        kpi_card("Avg Rating", f"{city_data['rating'].mean():.2f}", icon="⭐")
    with kc4:
        kpi_card("Cancel Rate", f"{city_data['cancellation_flag'].mean()*100:.1f}%", icon="🚫")
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        rest_rev = city_data.groupby("restaurant")["total_amount"].sum().sort_values(ascending=True).reset_index()
        fig_r = px.bar(rest_rev, x="total_amount", y="restaurant", orientation="h",
                       color="total_amount", color_continuous_scale=["#c7d2fe", "#6c63ff"],
                       title=f"Restaurant Revenue — {selected_drill_city}")
        fig_r.update_layout(**PLOTLY_LAYOUT, height=400, coloraxis_showscale=False)
        fig_r.update_traces(hovertemplate="₹%{x:,.0f}<extra></extra>")
        st.plotly_chart(fig_r, use_container_width=True)

    with col2:
        cat_rev = city_data.groupby("category")["total_amount"].sum().reset_index()
        fig_c = px.pie(cat_rev, values="total_amount", names="category", hole=0.48,
                       title=f"Category Split — {selected_drill_city}",
                       color_discrete_sequence=px.colors.qualitative.Set3)
        fig_c.update_traces(textposition="outside", textinfo="percent+label", textfont_size=11)
        fig_c.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
        st.plotly_chart(fig_c, use_container_width=True)

    # Restaurant → Product drill
    rest_list = city_data["restaurant"].unique().tolist()
    selected_restaurant = st.selectbox("Drill into a restaurant:", ["— Select —"] + sorted(rest_list), key="drill_rest")
    if selected_restaurant != "— Select —":
        rest_data = city_data[city_data["restaurant"] == selected_restaurant]
        prod_rev = rest_data.groupby("product")["total_amount"].sum().sort_values(ascending=True).reset_index()
        fig_p = px.bar(prod_rev, x="total_amount", y="product", orientation="h",
                       color="total_amount", color_continuous_scale=["#bfdbfe", "#3b82f6"],
                       title=f"Product Revenue — {selected_restaurant}")
        fig_p.update_layout(**PLOTLY_LAYOUT, height=380, coloraxis_showscale=False)
        fig_p.update_traces(hovertemplate="₹%{x:,.0f}<extra></extra>")
        st.plotly_chart(fig_p, use_container_width=True)

st.markdown("---")

# ── Heatmap: Hour × Weekday ──────────────────────────────────────────────
st.markdown('<div class="section-title">🕐 Hour × Weekday Heatmap</div>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">Identify peak demand windows to optimize staffing and promotions</p>', unsafe_allow_html=True)

heatmap_metric = st.radio("Select metric:", ["Orders", "Revenue", "Avg Rating"], horizontal=True)

if heatmap_metric == "Orders":
    heat_data = filtered.groupby(["weekday", "hour"])["order_id"].count().reset_index(name="value")
elif heatmap_metric == "Revenue":
    heat_data = filtered.groupby(["weekday", "hour"])["total_amount"].sum().reset_index(name="value")
else:
    heat_data = filtered.groupby(["weekday", "hour"])["rating"].mean().reset_index(name="value")

heat_pivot = heat_data.pivot(index="weekday", columns="hour", values="value").fillna(0)
weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

fig_heat = go.Figure(data=go.Heatmap(
    z=heat_pivot.values,
    x=[f"{h}:00" for h in heat_pivot.columns],
    y=[weekday_labels[i] if i < len(weekday_labels) else str(i) for i in heat_pivot.index],
    colorscale=[[0, "#f8f9fc"], [0.3, "#c7d2fe"], [0.6, "#6c63ff"], [1, "#312e81"]],
    hovertemplate="<b>%{y}, %{x}</b><br>Value: %{z:,.0f}<extra></extra>",
))
fig_heat.update_layout(**PLOTLY_LAYOUT, height=320)
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# ── Weather Impact ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🌦️ Weather Impact Analysis</div>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">How weather conditions affect order volume, revenue, and cancellations</p>', unsafe_allow_html=True)

col_w1, col_w2 = st.columns(2)
with col_w1:
    weather_rev = filtered.groupby("weather").agg(
        avg_revenue=("total_amount", "mean"),
        total_orders=("order_id", "count"),
    ).reset_index()
    fig_wt = px.bar(weather_rev, x="weather", y="avg_revenue", color="weather",
                    color_discrete_sequence=["#6c63ff", "#3b82f6", "#06b6d4", "#22c55e", "#f59e0b", "#ef4444"],
                    title="Avg Order Value by Weather")
    fig_wt.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False, xaxis_title="", yaxis_title="Avg Revenue (₹)")
    fig_wt.update_traces(hovertemplate="₹%{y:,.0f}<extra></extra>")
    st.plotly_chart(fig_wt, use_container_width=True)

with col_w2:
    weather_cancel = filtered.groupby("weather")["cancellation_flag"].mean().reset_index()
    weather_cancel["cancellation_flag"] *= 100
    fig_wc = px.bar(weather_cancel, x="weather", y="cancellation_flag", color="weather",
                    color_discrete_sequence=["#6c63ff", "#3b82f6", "#06b6d4", "#22c55e", "#f59e0b", "#ef4444"],
                    title="Cancellation Rate by Weather",
                    labels={"cancellation_flag": "Cancel Rate (%)"})
    fig_wc.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False, xaxis_title="", yaxis_title="Cancel Rate (%)")
    st.plotly_chart(fig_wc, use_container_width=True)

st.markdown("---")

# ── Monthly Revenue Cohort ────────────────────────────────────────────────
st.markdown('<div class="section-title">👥 Monthly Revenue by City</div>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">Track city-level revenue trends month-over-month</p>', unsafe_allow_html=True)

filtered_c = filtered.copy()
filtered_c["year_month"] = pd.to_datetime(filtered_c["date"]).dt.to_period("M").astype(str)
monthly = filtered_c.groupby(["year_month", "city"])["total_amount"].sum().reset_index()
fig_cohort = px.line(monthly, x="year_month", y="total_amount", color="city",
                     labels={"total_amount": "Revenue (₹)", "year_month": "Month"},
                     color_discrete_sequence=px.colors.qualitative.Plotly)
fig_cohort.update_layout(**PLOTLY_LAYOUT, height=400, hovermode="x unified",
                         legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig_cohort, use_container_width=True)

st.markdown("---")

# ── Discount Effectiveness ────────────────────────────────────────────────
st.markdown('<div class="section-title">💰 Discount Effectiveness</div>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">Does discounting drive higher order values or just cut margins?</p>', unsafe_allow_html=True)

disc_df = filtered.copy()
disc_df["discount_bucket"] = pd.cut(disc_df["discount_pct"], bins=[-1, 0, 5, 10, 15, 20, 100],
                                     labels=["0%", "1-5%", "6-10%", "11-15%", "16-20%", "20%+"])
disc_rev = disc_df.groupby("discount_bucket", observed=True).agg(
    avg_order=("total_amount", "mean"),
    count=("order_id", "count"),
).reset_index()

fig_disc = px.bar(disc_rev, x="discount_bucket", y="avg_order",
                  color="avg_order", color_continuous_scale=["#c7d2fe", "#6c63ff"],
                  hover_data=["count"],
                  title="Avg Order Value vs Discount Level")
fig_disc.update_layout(**PLOTLY_LAYOUT, height=350, coloraxis_showscale=False,
                       xaxis_title="Discount Bucket", yaxis_title="Avg Order Value (₹)")
fig_disc.update_traces(hovertemplate="Avg: ₹%{y:,.0f}<br>Orders: %{customdata[0]:,}<extra></extra>")
st.plotly_chart(fig_disc, use_container_width=True)
