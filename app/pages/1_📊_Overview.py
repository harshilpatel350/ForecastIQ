"""
Page 1: Overview Dashboard (Auto-EDA)
======================================
Dynamically generates KPIs and charts based on the detected schema of the active dataset.
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
from streamlit_app import inject_css, get_active_dataset, render_sidebar, kpi_card, PLOTLY_LAYOUT

inject_css()
df = get_active_dataset()
filtered = render_sidebar(df)

# Retrieve schema extracted by streamlit_app
schema = st.session_state.get("schema", {"date_col": None, "numeric_cols": [], "categorical_cols": []})
num_cols = schema["numeric_cols"]
cat_cols = schema["categorical_cols"]
date_col = schema["date_col"]

# ════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="brand-header">📊 Dashboard Overview</p>', unsafe_allow_html=True)
st.markdown('<p class="brand-subtitle">Dynamic intelligence — automatically analyzing your uploaded dataset schema</p>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# DYNAMIC KPI ROW
# ════════════════════════════════════════════════════════════════════════════
total_rows = len(filtered)
kpi_cols = st.columns(6)

# KPI 1: Always row count
with kpi_cols[0]:
    kpi_card("Total Records", f"{total_rows:,}", None, True, "📑")

# KPI 2: Primary Metric Sum
if len(num_cols) > 0:
    m1 = num_cols[0].replace("_", " ").title()
    val = filtered[num_cols[0]].sum()
    with kpi_cols[1]:
        kpi_card(f"Total {m1}", f"{val:,.0f}" if val > 1000 else f"{val:,.2f}", None, True, "💰")
    
    avg_val = filtered[num_cols[0]].mean() if total_rows > 0 else 0
    with kpi_cols[2]:
        kpi_card(f"Avg {m1}", f"{avg_val:,.0f}" if avg_val > 1000 else f"{avg_val:,.2f}", None, True, "📈")
else:
    with kpi_cols[1]:
        kpi_card("Numeric Metrics", "0", "No metrics found", False, "⚠️")
    with kpi_cols[2]:
        kpi_card("Avg Metric", "N/A", None, True, "📈")

# KPI 4: Secondary Metric Sum
if len(num_cols) > 1:
    m2 = num_cols[1].replace("_", " ").title()
    val2 = filtered[num_cols[1]].sum()
    with kpi_cols[3]:
        kpi_card(f"Total {m2}", f"{val2:,.0f}" if val2 > 1000 else f"{val2:,.2f}", None, True, "⚡")
# Fallback to categorical count
elif len(cat_cols) > 0:
    c1 = cat_cols[0].replace("_", " ").title()
    with kpi_cols[3]:
        kpi_card(f"Unique {c1}", f"{filtered[cat_cols[0]].nunique():,}", None, True, "🏷️")
else:
    with kpi_cols[3]: kpi_card("Secondary Metric", "N/A", None, True, "⚡")

# KPI 5 & 6: Categorical counts
if len(cat_cols) > 1:
    c2 = cat_cols[1].replace("_", " ").title()
    with kpi_cols[4]:
        kpi_card(f"Unique {c2}", f"{filtered[cat_cols[1]].nunique():,}", None, True, "📋")
elif len(num_cols) > 2:
    m3 = num_cols[2].replace("_", " ").title()
    val3 = filtered[num_cols[2]].sum()
    with kpi_cols[4]:
        kpi_card(f"Total {m3}", f"{val3:,.0f}" if val3 > 1000 else f"{val3:,.2f}", None, True, "📉")
else:
    with kpi_cols[4]: kpi_card("Dimensions", f"{len(cat_cols)}", None, True, "📋")

if len(cat_cols) > 2:
    c3 = cat_cols[2].replace("_", " ").title()
    with kpi_cols[5]:
        kpi_card(f"Unique {c3}", f"{filtered[cat_cols[2]].nunique():,}", None, True, "🏷️")
else:
    with kpi_cols[5]: kpi_card("Total Columns", f"{len(cols)}", None, True, "📊")

st.markdown("")

# ════════════════════════════════════════════════════════════════════════════
# DYNAMIC TIME SERIES
# ════════════════════════════════════════════════════════════════════════════
if date_col and len(num_cols) > 0:
    target_metric = num_cols[0]
    metric_name = target_metric.replace("_", " ").title()

    st.markdown(f'<div class="section-title">📈 {metric_name} Over Time</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Daily aggregate trends overlayed with 7-day moving average</p>', unsafe_allow_html=True)

    # Group by date dynamically
    daily = filtered.groupby(date_col).agg(
        daily_val=(target_metric, "sum"),
        daily_count=(target_metric, "count")
    ).reset_index()
    
    # Needs to be sorted for moving average
    daily = daily.sort_values(date_col)
    daily["ma_7"] = daily["daily_val"].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily[date_col], y=daily["daily_val"],
                             name=f"Daily {metric_name}", mode="lines",
                             line=dict(color="rgba(108,99,255,0.25)", width=1),
                             fill="tozeroy", fillcolor="rgba(108,99,255,0.05)",
                             hovertemplate="%{y:,.0f}<extra>Daily</extra>"))
    fig.add_trace(go.Scatter(x=daily[date_col], y=daily["ma_7"],
                             name="7-Day Average", mode="lines",
                             line=dict(color="#6c63ff", width=2.5),
                             hovertemplate="%{y:,.0f}<extra>7d Avg</extra>"))
    fig.update_layout(**PLOTLY_LAYOUT, height=370, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# DYNAMIC CATEGORY BREAKDOWNS
# ════════════════════════════════════════════════════════════════════════════
if len(cat_cols) > 0 and len(num_cols) > 0:
    col_left, col_right = st.columns(2)
    
    target_metric = num_cols[0]
    metric_name = target_metric.replace("_", " ").title()

    # Breakdown 1: Primary Category
    with col_left:
        cat_primary = cat_cols[0]
        cat_name = cat_primary.replace("_", " ").title()
        st.markdown(f'<div class="section-title">🍕 {metric_name} by {cat_name}</div>', unsafe_allow_html=True)
        
        cat_data = filtered.groupby(cat_primary)[target_metric].sum().reset_index()
        fig_cat = px.pie(cat_data, values=target_metric, names=cat_primary,
                         color_discrete_sequence=px.colors.qualitative.Set3, hole=0.48)
        fig_cat.update_traces(textposition="outside", textinfo="percent+label",
                              textfont_size=11, pull=[0.02]*len(cat_data))
        fig_cat.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)

    # Breakdown 2: Secondary Category or Row Count if only 1 category exists
    with col_right:
        if len(cat_cols) > 1:
            cat_sec = cat_cols[1]
            cat_sec_name = cat_sec.replace("_", " ").title()
            st.markdown(f'<div class="section-title">🏙️ {metric_name} by {cat_sec_name}</div>', unsafe_allow_html=True)
            
            sec_data = filtered.groupby(cat_sec)[target_metric].sum().sort_values(ascending=True).reset_index()
            # keep top 15 if too many
            if len(sec_data) > 15: sec_data = sec_data.tail(15)
            
            total = sec_data[target_metric].sum()
            # Avoid division by zero
            sec_data["pct"] = (sec_data[target_metric] / max(total, 1) * 100).round(1)
            sec_data["label"] = sec_data.apply(lambda r: f"{r[cat_sec]} ({r['pct']}%)", axis=1)
            
            fig_sec = px.bar(sec_data, x=target_metric, y="label", orientation="h",
                             color=target_metric,
                             color_continuous_scale=["#c7d2fe", "#6c63ff", "#3b82f6"],
                             labels={target_metric: metric_name, "label": ""})
            fig_sec.update_layout(**PLOTLY_LAYOUT, height=380, coloraxis_showscale=False)
            fig_sec.update_traces(hovertemplate="%{x:,.0f}<extra></extra>")
            st.plotly_chart(fig_sec, use_container_width=True)
        else:
            # Fallback to row counts by primary category
            st.markdown(f'<div class="section-title">📋 Row Count by {cat_name}</div>', unsafe_allow_html=True)
            count_data = filtered[cat_primary].value_counts().reset_index()
            count_data.columns = [cat_primary, 'Count']
            count_data = count_data.sort_values(by='Count', ascending=True).tail(15)
            
            fig_count = px.bar(count_data, x='Count', y=cat_primary, orientation="h",
                               color='Count', color_continuous_scale=["#c7d2fe", "#6c63ff"])
            fig_count.update_layout(**PLOTLY_LAYOUT, height=380, coloraxis_showscale=False)
            st.plotly_chart(fig_count, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TERTIARY BREAKDOWNS (If more categories exist)
# ════════════════════════════════════════════════════════════════════════════
if len(cat_cols) > 2 and len(num_cols) > 0:
    col1, col2 = st.columns(2)
    target_metric = num_cols[0]
    metric_name = target_metric.replace("_", " ").title()

    with col1:
        c3 = cat_cols[2]
        c3_name = c3.replace("_", " ").title()
        st.markdown(f'<div class="section-title">💳 {metric_name} by {c3_name}</div>', unsafe_allow_html=True)
        c3_data = filtered.groupby(c3)[target_metric].sum().reset_index()
        fig_c3 = px.pie(c3_data, values=target_metric, names=c3, hole=0.5,
                        color_discrete_sequence=["#6c63ff", "#3b82f6", "#06b6d4", "#22c55e", "#f59e0b"])
        fig_c3.update_traces(textposition="outside", textinfo="percent+label", textfont_size=11)
        fig_c3.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
        st.plotly_chart(fig_c3, use_container_width=True)

    if len(cat_cols) > 3:
        with col2:
            c4 = cat_cols[3]
            c4_name = c4.replace("_", " ").title()
            st.markdown(f'<div class="section-title">🚗 {metric_name} by {c4_name}</div>', unsafe_allow_html=True)
            c4_data = filtered.groupby(c4)[target_metric].sum().sort_values(ascending=False).head(10).reset_index()
            fig_c4 = px.bar(c4_data, x=c4, y=target_metric,
                            color=c4,
                            color_discrete_sequence=["#6c63ff", "#3b82f6", "#22d3ee"])
            fig_c4.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False,
                                 xaxis_title="", yaxis_title=metric_name)
            fig_c4.update_traces(hovertemplate="%{y:,.0f}<extra></extra>")
            st.plotly_chart(fig_c4, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# RAW DATA PREVIEW
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("🔍 View Raw Dataset Sample"):
    st.dataframe(filtered.head(50), use_container_width=True)
