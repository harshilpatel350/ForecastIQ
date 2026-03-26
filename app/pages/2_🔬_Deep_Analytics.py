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
from streamlit_app import inject_css, get_active_dataset, render_sidebar, kpi_card, PLOTLY_LAYOUT

inject_css()
df = get_active_dataset()
filtered = render_sidebar(df)
cols = filtered.columns.tolist()

# Retrieve schema extracted by streamlit_app
schema = st.session_state.get("schema", {"date_col": None, "numeric_cols": [], "categorical_cols": []})
num_cols = schema["numeric_cols"]
cat_cols = schema["categorical_cols"]
date_col = schema["date_col"]

st.markdown('<p class="brand-header">🔬 Deep Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="brand-subtitle">Uncover hidden patterns through multi-dimensional drill-down analysis</p>', unsafe_allow_html=True)

# ── Drill-Down: Hierarchy Analysis ──────────────────────────────────────────
if len(cat_cols) > 0 and len(num_cols) > 0:
    target_metric = num_cols[0]
    metric_name = target_metric.replace("_", " ").title()
    cat_primary = cat_cols[0]
    cat_label = cat_primary.replace("_", " ").title()

    st.markdown(f'<div class="section-title">🏙️ {cat_label} → Details Drill-Down</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Click through the hierarchy to explore metrics at every level</p>', unsafe_allow_html=True)

    agg_data = filtered.groupby(cat_primary)[target_metric].sum().sort_values(ascending=False).reset_index()
    selected_drill_val = st.selectbox(f"Select a {cat_label} to drill into:", ["— Select —"] + agg_data[cat_primary].tolist(), key="drill_val")

    if selected_drill_val != "— Select —":
        drill_data = filtered[filtered[cat_primary] == selected_drill_val]

        # City-level KPIs
        kc1, kc2, kc3, kc4 = st.columns(4)
        with kc1:
            kpi_card(f"Total {metric_name}", f"₹{drill_data[target_metric].sum():,.0f}", icon="💰")
        with kc2:
            kpi_card("Records", f"{len(drill_data):,}", icon="📦")
        with kc3:
            if "rating" in cols:
                kpi_card("Avg Rating", f"{city_data['rating'].mean():.2f}", icon="⭐")
        with kc4:
            if "cancellation_flag" in cols:
                kpi_card("Cancel Rate", f"{city_data['cancellation_flag'].mean()*100:.1f}%", icon="🚫")
        st.markdown("")

        col1, col2 = st.columns(2)
        if len(cat_cols) > 1:
            with col1:
                cat_sec = cat_cols[1]
                cat_sec_label = cat_sec.replace("_", " ").title()
                sec_rev = drill_data.groupby(cat_sec)[target_metric].sum().sort_values(ascending=True).tail(15).reset_index()
                fig_r = px.bar(sec_rev, x=target_metric, y=cat_sec, orientation="h",
                               color=target_metric, color_continuous_scale=["#c7d2fe", "#6c63ff"],
                               title=f"{metric_name} by {cat_sec_label}")
                fig_r.update_layout(**PLOTLY_LAYOUT, height=400, coloraxis_showscale=False)
                fig_r.update_traces(hovertemplate="₹%{x:,.0f}<extra></extra>")
                st.plotly_chart(fig_r, use_container_width=True)

        if len(cat_cols) > 2:
            with col2:
                cat_ter = cat_cols[2]
                cat_ter_label = cat_ter.replace("_", " ").title()
                ter_rev = drill_data.groupby(cat_ter)[target_metric].sum().reset_index()
                fig_c = px.pie(ter_rev, values=target_metric, names=cat_ter, hole=0.48,
                               title=f"Distribution by {cat_ter_label}",
                               color_discrete_sequence=px.colors.qualitative.Set3)
                fig_c.update_traces(textposition="outside", textinfo="percent+label", textfont_size=11)
                fig_c.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
                st.plotly_chart(fig_c, use_container_width=True)

        # Dynamic drill into next level
        if len(num_cols) > 0 and len(cat_cols) > 1:
            cat_sec = cat_cols[1]
            cat_sec_label = cat_sec.replace("_", " ").title()
            
            st.markdown(f"**Drill into {cat_sec_label} details:**")
            sec_list = drill_data[cat_sec].dropna().unique().tolist()
            selected_sec = st.selectbox(f"Select a {cat_sec_label}:", ["— Select —"] + sorted([str(x) for x in sec_list]), key="drill_sec")
            
            if selected_sec != "— Select —":
                sub_data = drill_data[drill_data[cat_sec].astype(str) == selected_sec]
                if len(num_cols) > 0:
                    summary_data = sub_data.mean(numeric_only=True)
                    st.write(f"Average {metric_name}: ₹{sub_data[target_metric].mean():,.0f}")
        
    st.markdown("---")

# ── Heatmap: Hour × Weekday ──────────────────────────────────────────────
if "weekday" in cols and "hour" in cols:
    st.markdown('<div class="section-title">🕐 Hour × Weekday Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Identify peak demand windows to optimize staffing and promotions</p>', unsafe_allow_html=True)

    metric_options = ["Orders"]
    if len(num_cols) > 0: metric_options.append(num_cols[0].replace("_", " ").title())
    if "rating" in cols: metric_options.append("Avg Rating")
    heatmap_metric = st.radio("Select metric:", metric_options, horizontal=True)

    if heatmap_metric == "Orders":
        heat_data = filtered.groupby(["weekday", "hour"])["order_id"].count().reset_index(name="value")
    elif len(num_cols) > 0 and heatmap_metric == num_cols[0].replace("_", " ").title():
        heat_data = filtered.groupby(["weekday", "hour"])[num_cols[0]].sum().reset_index(name="value")
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

# ── Environment / Group Impact ──────────────────────────────────────────────
impact_col = next((c for c in ["weather", "inventory_status"] if c in cols), 
                  cat_cols[0] if len(cat_cols) > 0 else None)
target_metric = num_cols[0] if len(num_cols) > 0 else None

if impact_col and target_metric:
    st.markdown(f'<div class="section-title">🌦️ {impact_col.title()} Impact Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-subtitle">How {impact_col} affects {target_metric.replace("_", " ")} and volume</p>', unsafe_allow_html=True)

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        impact_stats = filtered.groupby(impact_col).agg(
            avg_val=(target_metric, "mean"),
            total_count=("order_id", "count"),
        ).reset_index()
        fig_wt = px.bar(impact_stats, x=impact_col, y="avg_val", color=impact_col,
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        title=f"Avg {target_metric.replace('_', ' ').title()} by {impact_col.title()}")
        fig_wt.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False, xaxis_title="", yaxis_title="Avg Value")
        st.plotly_chart(fig_wt, use_container_width=True)

    if "cancellation_flag" in cols:
        with col_w2:
            cancel_stats = filtered.groupby(impact_col)["cancellation_flag"].mean().reset_index()
            cancel_stats["cancellation_flag"] *= 100
            fig_wc = px.bar(cancel_stats, x=impact_col, y="cancellation_flag", color=impact_col,
                            title="Cancellation Rate (%)", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_wc.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False)
            st.plotly_chart(fig_wc, use_container_width=True)
    elif len(num_cols) > 1:
        with col_w2:
            m2 = num_cols[1]
            m2_stats = filtered.groupby(impact_col)[m2].mean().reset_index()
            fig_m2 = px.bar(m2_stats, x=impact_col, y=m2, color=impact_col,
                            title=f"Avg {m2.replace('_', ' ').title()}", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_m2.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False)
            st.plotly_chart(fig_m2, use_container_width=True)

    st.markdown("---")

# ── Trend Metrics ─────────────────────────────────────────────────────────
if date_col and len(num_cols) > 0:
    target_metric = num_cols[0]
    metric_label = target_metric.replace("_", " ").title()
    label = f"Monthly {metric_label} by {cat_cols[0].title()}" if len(cat_cols) > 0 else f"{metric_label} Trend"
    
    st.markdown(f'<div class="section-title">👥 {label}</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Track trends over time months-over-month</p>', unsafe_allow_html=True)

    filtered_c = filtered.copy()
    filtered_c["year_month"] = pd.to_datetime(filtered_c[date_col], errors="coerce").dt.to_period("M").astype(str)

    if len(cat_cols) > 0:
        c1 = cat_cols[0]
        monthly = filtered_c.groupby(["year_month", c1])[target_metric].sum().reset_index()
        fig_cohort = px.line(monthly, x="year_month", y=target_metric, color=c1,
                             labels={target_metric: metric_label, "year_month": "Month"},
                             color_discrete_sequence=px.colors.qualitative.Plotly)
    else:
        monthly = filtered_c.groupby("year_month")[target_metric].sum().reset_index()
        fig_cohort = px.line(monthly, x="year_month", y=target_metric,
                             labels={target_metric: metric_label, "year_month": "Month"},
                             color_discrete_sequence=["#6c63ff"])

    fig_cohort.update_layout(**PLOTLY_LAYOUT, height=400, hovermode="x unified")
    st.plotly_chart(fig_cohort, use_container_width=True)

    st.markdown("---")

# ── Discount Effectiveness ────────────────────────────────────────────────
if "discount_pct" in cols and "total_amount" in cols:
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
