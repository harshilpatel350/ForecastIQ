"""
Page 1: Overview Dashboard (Complete)
======================================
All KPIs, revenue/order trends, category/city breakdown, heatmap, payment & order type charts.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title="Overview — ForecastIQ", page_icon="📊", layout="wide")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "app"))
from streamlit_app import inject_css, load_market_data, load_daily_data, render_sidebar, kpi_card, PLOTLY_LAYOUT

inject_css()
df = load_market_data()
filtered = render_sidebar(df)

# ════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="brand-header">📊 Dashboard Overview</p>', unsafe_allow_html=True)
st.markdown('<p class="brand-subtitle">Real-time business intelligence — track revenue, orders, and performance across all cities and products</p>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ════════════════════════════════════════════════════════════════════════════
total_revenue = filtered["total_amount"].sum()
total_orders = len(filtered)
aov = filtered["total_amount"].mean() if total_orders > 0 else 0
avg_rating = filtered["rating"].mean() if total_orders > 0 else 0
cancel_rate = filtered["cancellation_flag"].mean() * 100 if total_orders > 0 else 0
unique_customers = filtered["customer_id"].nunique()
delivery_pct = (filtered["order_type"] == "Delivery").mean() * 100 if total_orders > 0 else 0

# Growth: compare second half vs first half
mid = len(filtered) // 2
if mid > 0:
    rev_first = filtered.iloc[:mid]["total_amount"].sum()
    rev_second = filtered.iloc[mid:]["total_amount"].sum()
    growth = ((rev_second - rev_first) / max(rev_first, 1)) * 100
else:
    growth = 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    kpi_card("Total Revenue", f"₹{total_revenue:,.0f}", f"{growth:+.1f}% growth", growth > 0, "💰")
with c2:
    kpi_card("Total Orders", f"{total_orders:,}", f"{total_orders/max(filtered['date'].nunique(),1):.0f}/day avg", True, "📦")
with c3:
    kpi_card("Avg Order Value", f"₹{aov:,.0f}", None, True, "🧾")
with c4:
    kpi_card("Customers", f"{unique_customers:,}", None, True, "👥")
with c5:
    kpi_card("Avg Rating", f"{avg_rating:.2f} ⭐", None, avg_rating >= 3.5, "⭐")
with c6:
    kpi_card("Cancel Rate", f"{cancel_rate:.1f}%", None, cancel_rate < 5, "🚫")

st.markdown("")

# ════════════════════════════════════════════════════════════════════════════
# REVENUE & ORDERS TABS
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📈 Revenue & Order Trends</div>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">Daily metrics with 7-day moving average overlay — switch between revenue and order volume</p>', unsafe_allow_html=True)

daily = load_daily_data(filtered)
daily["ma_7"] = daily["daily_revenue"].rolling(7, min_periods=1).mean()
daily["orders_ma7"] = daily["daily_orders"].rolling(7, min_periods=1).mean()

tab1, tab2 = st.tabs(["📈 Revenue Trend", "📦 Order Volume"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["daily_revenue"],
                             name="Daily Revenue", mode="lines",
                             line=dict(color="rgba(108,99,255,0.25)", width=1),
                             fill="tozeroy", fillcolor="rgba(108,99,255,0.05)",
                             hovertemplate="₹%{y:,.0f}<extra>Daily</extra>"))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["ma_7"],
                             name="7-Day Average", mode="lines",
                             line=dict(color="#6c63ff", width=2.5),
                             hovertemplate="₹%{y:,.0f}<extra>7d Avg</extra>"))
    fig.update_layout(**PLOTLY_LAYOUT, height=370, hovermode="x unified",
                      legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=daily["date"], y=daily["daily_orders"],
                          name="Daily Orders",
                          marker_color="rgba(59,130,246,0.4)",
                          hovertemplate="%{y:,}<extra>Daily</extra>"))
    fig2.add_trace(go.Scatter(x=daily["date"], y=daily["orders_ma7"],
                              name="7-Day Average", mode="lines",
                              line=dict(color="#3b82f6", width=2.5),
                              hovertemplate="%{y:,.0f}<extra>7d Avg</extra>"))
    fig2.update_layout(**PLOTLY_LAYOUT, height=370, hovermode="x unified",
                       legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"))
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# CATEGORY + CITY PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="section-title">🍕 Revenue by Category</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Proportional breakdown of revenue across food categories</p>', unsafe_allow_html=True)
    cat_rev = filtered.groupby("category")["total_amount"].sum().reset_index()
    fig_cat = px.pie(cat_rev, values="total_amount", names="category",
                     color_discrete_sequence=px.colors.qualitative.Set3, hole=0.48)
    fig_cat.update_traces(textposition="outside", textinfo="percent+label",
                          textfont_size=11, pull=[0.02]*len(cat_rev))
    fig_cat.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
    st.plotly_chart(fig_cat, use_container_width=True)

with col_right:
    st.markdown('<div class="section-title">🏙️ City Performance</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Revenue contribution ranked by city with percentage labels</p>', unsafe_allow_html=True)
    city_rev = filtered.groupby("city")["total_amount"].sum().sort_values(ascending=True).reset_index()
    total = city_rev["total_amount"].sum()
    city_rev["pct"] = (city_rev["total_amount"] / total * 100).round(1)
    city_rev["label"] = city_rev.apply(lambda r: f"{r['city']} ({r['pct']}%)", axis=1)
    fig_city = px.bar(city_rev, x="total_amount", y="label", orientation="h",
                      color="total_amount",
                      color_continuous_scale=["#c7d2fe", "#6c63ff", "#3b82f6"],
                      labels={"total_amount": "Revenue (₹)", "label": ""})
    fig_city.update_layout(**PLOTLY_LAYOUT, height=380, coloraxis_showscale=False)
    fig_city.update_traces(hovertemplate="Revenue: ₹%{x:,.0f}<extra></extra>")
    st.plotly_chart(fig_city, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# DEMAND HEATMAP
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🕐 Demand Heatmap</div>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">Order volume by hour of day and day of week — identify peak demand windows</p>', unsafe_allow_html=True)

heatmap_data = filtered.groupby(["weekday", "hour"])["order_id"].count().reset_index()
heatmap_pivot = heatmap_data.pivot(index="weekday", columns="hour", values="order_id").fillna(0)
weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

fig_heat = go.Figure(data=go.Heatmap(
    z=heatmap_pivot.values,
    x=[f"{h}:00" for h in heatmap_pivot.columns],
    y=[weekday_labels[i] for i in heatmap_pivot.index],
    colorscale=[[0, "#f8f9fc"], [0.3, "#c7d2fe"], [0.6, "#6c63ff"], [1, "#312e81"]],
    hovertemplate="<b>%{y}, %{x}</b><br>Orders: %{z:,.0f}<extra></extra>",
))
fig_heat.update_layout(**PLOTLY_LAYOUT, height=300)
st.plotly_chart(fig_heat, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAYMENT & ORDER TYPE
# ════════════════════════════════════════════════════════════════════════════
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-title">💳 Revenue by Payment Method</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Breakdown of customer payment preferences</p>', unsafe_allow_html=True)
    pay = filtered.groupby("payment_method")["total_amount"].sum().reset_index()
    fig_pay = px.pie(pay, values="total_amount", names="payment_method", hole=0.5,
                     color_discrete_sequence=["#6c63ff", "#3b82f6", "#06b6d4", "#22c55e", "#f59e0b"])
    fig_pay.update_traces(textposition="outside", textinfo="percent+label", textfont_size=11)
    fig_pay.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
    st.plotly_chart(fig_pay, use_container_width=True)

with col2:
    st.markdown('<div class="section-title">🚗 Revenue by Order Type</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Delivery vs Dine-In vs Takeaway revenue comparison</p>', unsafe_allow_html=True)
    ot = filtered.groupby("order_type")["total_amount"].sum().reset_index()
    fig_ot = px.bar(ot, x="order_type", y="total_amount",
                    color="order_type",
                    color_discrete_sequence=["#6c63ff", "#3b82f6", "#22d3ee"])
    fig_ot.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False,
                         xaxis_title="", yaxis_title="Revenue (₹)")
    fig_ot.update_traces(hovertemplate="₹%{y:,.0f}<extra></extra>")
    st.plotly_chart(fig_ot, use_container_width=True)
