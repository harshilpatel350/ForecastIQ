"""
Page 4: Product Intelligence
=============================
Top/underperforming products, demand patterns, treemap, AI recommendations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title="Product Intelligence — ForecastIQ", page_icon="🍕", layout="wide")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "app"))
from streamlit_app import inject_css, get_active_dataset, render_sidebar, kpi_card, PLOTLY_LAYOUT

inject_css()
df = get_active_dataset()
filtered = render_sidebar(df)
cols = filtered.columns.tolist()

st.markdown('<p class="brand-header">🍕 Product Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="brand-subtitle">Actionable insights to optimize your product portfolio — identify stars, underperformers, and growth opportunities</p>', unsafe_allow_html=True)

# Retrieve schema extracted by streamlit_app
schema = st.session_state.get("schema", {"date_col": None, "numeric_cols": [], "categorical_cols": []})
num_cols = schema["numeric_cols"]
cat_cols = schema["categorical_cols"]
date_col = schema["date_col"]

if len(cat_cols) == 0 or len(num_cols) == 0:
    st.warning("⚠️ This page requires at least one categorical and one numeric column.")
    st.stop()

# ── Product Stats ────────────────────────────────────────────────────────
target_cat = cat_cols[0]
target_num = num_cols[0]
cat_name = target_cat.replace("_", " ").title()
num_name = target_num.replace("_", " ").title()

product_stats = filtered.groupby(target_cat).agg(
    total_val=(target_num, "sum"),
    total_count=("order_id", "count"),
).reset_index().sort_values("total_val", ascending=False)

# ── Top 10 Products ──────────────────────────────────────────────────────
st.markdown(f'<div class="section-title">🏆 Top Performing {cat_name}s</div>', unsafe_allow_html=True)
st.markdown(f'<p class="section-subtitle">Highest {num_name}-generating {cat_name}s</p>', unsafe_allow_html=True)

top10 = product_stats.head(10)
fig_top = px.bar(top10, x="total_val", y=target_cat, orientation="h",
                 color="total_val", color_continuous_scale=["#c7d2fe", "#6c63ff"],
                 labels={"total_val": num_name, target_cat: ""})
fig_top.update_layout(**PLOTLY_LAYOUT, height=450, coloraxis_showscale=False)
st.plotly_chart(fig_top, use_container_width=True)

# Best seller KPIs
if len(top10) > 0:
    best = top10.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card(f"👑 Best {cat_name}", f"{best[target_cat]}", icon="🏆")
    with c2:
        kpi_card(f"Total {num_name}", f"₹{best['total_val']:,.0f}" if best['total_val'] > 1000 else f"{best['total_val']:.2f}", icon="💰")
    with c3:
        kpi_card("Total Recs", f"{best['total_count']:,}", icon="📦")
    with c4:
        share = (best['total_val'] / product_stats['total_val'].sum() * 100) if product_stats['total_val'].sum() > 0 else 0
        kpi_card("Share", f"{share:.1f}%", icon="📊")

st.markdown("---")

# ── Underperforming Entities ──────────────────────────────────────────────
st.markdown(f'<div class="section-title">⚠️ Underperforming {cat_name}s</div>', unsafe_allow_html=True)
bottom10 = product_stats.tail(10).sort_values("total_val", ascending=True)
fig_bot = px.bar(bottom10, x="total_val", y=target_cat, orientation="h",
                 color="total_val", color_continuous_scale=["#fca5a5", "#c7d2fe"],
                 labels={"total_val": num_name, target_cat: ""})
fig_bot.update_layout(**PLOTLY_LAYOUT, height=400, coloraxis_showscale=False)
st.plotly_chart(fig_bot, use_container_width=True)

st.markdown("---")

# ── Demand Patterns ──────────────────────────────────────────────────────
if date_col:
    st.markdown(f'<div class="section-title">📈 {cat_name} Demand Patterns</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-subtitle">Track weekly trends for any {cat_name}</p>', unsafe_allow_html=True)

    selected_entity = st.selectbox(f"Select a {cat_name} to analyze:", sorted(filtered[target_cat].dropna().unique().tolist()))
    ent_data = filtered[filtered[target_cat] == selected_entity].copy()
    ent_data["week"] = pd.to_datetime(ent_data[date_col], errors="coerce").dt.to_period("W").astype(str)

    weekly_demand = ent_data.groupby("week").agg(
        recs=("order_id", "count"),
        val=(target_num, "sum"),
    ).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig_pd = px.line(weekly_demand, x="week", y="recs",
                         title=f"Weekly Records — {selected_entity}",
                         color_discrete_sequence=["#6c63ff"])
        fig_pd.update_layout(**PLOTLY_LAYOUT, height=350, xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig_pd, use_container_width=True)

    with col2:
        fig_pr = px.area(weekly_demand, x="week", y="val",
                         title=f"Weekly {num_name} — {selected_entity}",
                         color_discrete_sequence=["#3b82f6"])
        fig_pr.update_layout(**PLOTLY_LAYOUT, height=350, xaxis_title="", yaxis_title=num_name)
        fig_pr.update_traces(fill="tozeroy", fillcolor="rgba(59,130,246,0.12)")
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("---")

# ── Dynamic Hierarchy Treemap ───────────────────────────────────────────
if len(cat_cols) > 1:
    st.markdown(f'<div class="section-title">🗂️ {num_name} Distribution Treemap</div>', unsafe_allow_html=True)
    tree_path = cat_cols[:2] if len(cat_cols) >= 2 else [cat_cols[0]]
    tree_data = filtered.groupby(tree_path)[target_num].sum().reset_index()
    fig_tree = px.treemap(tree_data, path=tree_path, values=target_num,
                          color=target_num, color_continuous_scale=["#e0e7ff", "#6c63ff"])
    fig_tree.update_layout(**PLOTLY_LAYOUT, height=500, coloraxis_showscale=False)
    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("---")

# ── Dynamic Insights ────────────────────────────────────────────────────
st.markdown('<div class="section-title">💡 Automated Insights</div>', unsafe_allow_html=True)
top_entity = product_stats.iloc[0]
low_entity = product_stats.iloc[-1]

st.info(f"🏆 **Top Performer**: '{top_entity[target_cat]}' leads with a total {num_name} of **{top_entity['total_val']:,.0f}**.")
st.warning(f"📉 **Lowest Performer**: '{low_entity[target_cat]}' has the lowest {num_name} volume in this dataset.")
