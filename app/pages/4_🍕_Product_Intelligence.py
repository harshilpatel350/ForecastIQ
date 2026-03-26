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

if "product" not in cols or "total_amount" not in cols:
    st.warning("⚠️ This page requires **product** and **total_amount** columns in your dataset.")
    st.stop()

# ── Product Stats ────────────────────────────────────────────────────────
agg_dict = {
    "total_revenue": ("total_amount", "sum"),
    "total_orders": ("order_id", "count"),
}
if "unit_price" in cols: agg_dict["avg_price"] = ("unit_price", "mean")
if "rating" in cols: agg_dict["avg_rating"] = ("rating", "mean")
if "cancellation_flag" in cols: agg_dict["cancel_rate"] = ("cancellation_flag", "mean")

product_stats = filtered.groupby("product").agg(**agg_dict).reset_index().sort_values("total_revenue", ascending=False)

# ── Top 10 Products ──────────────────────────────────────────────────────
st.markdown('<div class="section-title">🏆 Top Performing Products</div>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">Highest revenue-generating products</p>', unsafe_allow_html=True)

top10 = product_stats.head(10)
color_col = "avg_rating" if "avg_rating" in top10.columns else "total_revenue"
fig_top = px.bar(top10, x="total_revenue", y="product", orientation="h",
                 color=color_col, color_continuous_scale=["#fca5a5", "#fde68a", "#86efac"],
                 labels={"total_revenue": "Revenue (₹)", "product": ""})
fig_top.update_layout(**PLOTLY_LAYOUT, height=450, coloraxis_colorbar=dict(title="Rating" if color_col == "avg_rating" else "Revenue", thickness=12))
fig_top.update_traces(hovertemplate="<b>%{y}</b><br>Revenue: ₹%{x:,.0f}<extra></extra>")
st.plotly_chart(fig_top, use_container_width=True)

# Best seller KPIs
if len(top10) > 0:
    best = top10.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("👑 Best Seller", best["product"], icon="🏆")
    with c2:
        kpi_card("Revenue", f"₹{best['total_revenue']:,.0f}", icon="💰")
    with c3:
        kpi_card("Orders", f"{best['total_orders']:,}", icon="📦")
    with c4:
        if "avg_rating" in best.index:
            kpi_card("Avg Rating", f"{best['avg_rating']:.2f} ⭐", icon="⭐")
        else:
            kpi_card("Share", f"{best['total_revenue']/product_stats['total_revenue'].sum()*100:.1f}%", icon="📊")

st.markdown("---")

# ── Underperforming Products ──────────────────────────────────────────────
st.markdown('<div class="section-title">⚠️ Underperforming Products</div>', unsafe_allow_html=True)

bottom10 = product_stats.tail(10).sort_values("total_revenue", ascending=True)
color_col_b = "cancel_rate" if "cancel_rate" in bottom10.columns else "total_revenue"
fig_bot = px.bar(bottom10, x="total_revenue", y="product", orientation="h",
                 color=color_col_b,
                 color_continuous_scale=["#86efac", "#fde68a", "#fca5a5"] if color_col_b == "cancel_rate" else ["#c7d2fe", "#6c63ff"],
                 labels={"total_revenue": "Revenue (₹)", "product": ""})
fig_bot.update_layout(**PLOTLY_LAYOUT, height=400, coloraxis_colorbar=dict(title="Cancel %" if color_col_b == "cancel_rate" else "Revenue", thickness=12))
st.plotly_chart(fig_bot, use_container_width=True)

st.markdown("---")

# ── Product Demand Patterns ──────────────────────────────────────────────
if "date" in cols:
    st.markdown('<div class="section-title">📈 Product Demand Patterns</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Track weekly order and revenue trends for any product</p>', unsafe_allow_html=True)

    selected_product = st.selectbox("Select a product to analyze:", sorted(filtered["product"].unique().tolist()))
    prod_data = filtered[filtered["product"] == selected_product].copy()
    prod_data["week"] = pd.to_datetime(prod_data["date"], errors="coerce").dt.to_period("W").astype(str)

    weekly_demand = prod_data.groupby("week").agg(
        orders=("order_id", "count"),
        revenue=("total_amount", "sum"),
    ).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig_pd = px.line(weekly_demand, x="week", y="orders",
                         title=f"Weekly Orders — {selected_product}",
                         color_discrete_sequence=["#6c63ff"])
        fig_pd.update_layout(**PLOTLY_LAYOUT, height=350, xaxis_title="", yaxis_title="Orders")
        st.plotly_chart(fig_pd, use_container_width=True)

    with col2:
        fig_pr = px.area(weekly_demand, x="week", y="revenue",
                         title=f"Weekly Revenue — {selected_product}",
                         color_discrete_sequence=["#3b82f6"])
        fig_pr.update_layout(**PLOTLY_LAYOUT, height=350, xaxis_title="", yaxis_title="Revenue (₹)")
        fig_pr.update_traces(fill="tozeroy", fillcolor="rgba(59,130,246,0.12)")
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("---")

# ── Category × Product Treemap ───────────────────────────────────────────
if "category" in cols:
    st.markdown('<div class="section-title">🗂️ Revenue Distribution Treemap</div>', unsafe_allow_html=True)
    tree_data = filtered.groupby(["category", "product"])["total_amount"].sum().reset_index()
    fig_tree = px.treemap(tree_data, path=["category", "product"], values="total_amount",
                          color="total_amount",
                          color_continuous_scale=["#e0e7ff", "#a5b4fc", "#6c63ff", "#4338ca"])
    fig_tree.update_layout(**PLOTLY_LAYOUT, height=500, coloraxis_showscale=False)
    fig_tree.update_traces(hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<extra></extra>")
    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("---")

# ── AI Recommendations ───────────────────────────────────────────────────
if "rating" in cols:
    st.markdown('<div class="section-title">💡 AI-Powered Recommendations</div>', unsafe_allow_html=True)

    median_rating = product_stats["avg_rating"].median()

    star_products = product_stats[
        (product_stats["total_orders"] > product_stats["total_orders"].quantile(0.60)) &
        (product_stats["avg_rating"] >= median_rating)
    ].sort_values("total_revenue", ascending=False).head(5)

    if len(star_products) > 0:
        st.success("🌟 **Star Products** — High demand + above-average rating:")
        for _, row in star_products.iterrows():
            st.markdown(f"- **{row['product']}** — {row['total_orders']:,} orders · "
                        f"₹{row['total_revenue']:,.0f} revenue · "
                        f"⭐ {row['avg_rating']:.1f}")

    problem_products = product_stats[
        (product_stats["total_orders"] < product_stats["total_orders"].quantile(0.30)) &
        (product_stats["avg_rating"] < median_rating)
    ].sort_values("total_revenue", ascending=True).head(5)

    if len(problem_products) > 0:
        st.warning("⚠️ **Needs Attention** — Low demand + below-average rating:")
        for _, row in problem_products.iterrows():
            cancel_info = f" · Cancel: {row['cancel_rate']*100:.1f}%" if "cancel_rate" in row.index else ""
            st.markdown(f"- **{row['product']}** — {row['total_orders']:,} orders · "
                        f"⭐ {row['avg_rating']:.1f}{cancel_info}")

    if "cancel_rate" in product_stats.columns:
        cancel_threshold = max(product_stats["cancel_rate"].quantile(0.80), 0.05)
        high_cancel = product_stats[product_stats["cancel_rate"] > cancel_threshold].sort_values("cancel_rate", ascending=False).head(5)
        if len(high_cancel) > 0:
            st.error(f"🚨 **High Cancellation Alert** — Products with >{cancel_threshold*100:.0f}% cancellation rate:")
            for _, row in high_cancel.iterrows():
                st.markdown(f"- **{row['product']}** — Cancel rate: {row['cancel_rate']*100:.1f}% · "
                            f"{row['total_orders']:,} orders")
