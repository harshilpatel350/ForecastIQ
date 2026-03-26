"""
Page 5: Data Explorer (Auto-EDA)
=================================
Interactive data table with dynamic filters, search, and visibility toggles based on schema.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title="Data Explorer — ForecastIQ", page_icon="📋", layout="wide")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "app"))
from streamlit_app import inject_css, get_active_dataset, render_sidebar, kpi_card, PLOTLY_LAYOUT

inject_css()
df = get_active_dataset()
filtered = render_sidebar(df)

# Retrieve schema
schema = st.session_state.get("schema", {"date_col": None, "numeric_cols": [], "categorical_cols": []})
num_cols = schema["numeric_cols"]
cat_cols = schema["categorical_cols"]
date_col = schema["date_col"]

st.markdown('<p class="brand-header">📋 Data Explorer</p>', unsafe_allow_html=True)
st.markdown('<p class="brand-subtitle">Browse, search, and export your complete dynamic dataset</p>', unsafe_allow_html=True)

# ── Data Quality KPIs ────────────────────────────────────────────────────
missing_pct = filtered.isna().sum().sum() / (len(filtered) * len(filtered.columns)) * 100

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    kpi_card("Records", f"{len(filtered):,}", icon="📦")
with c2:
    kpi_card("Columns", f"{len(filtered.columns)}", icon="📑")
with c3:
    kpi_card("Missing Data", f"{missing_pct:.2f}%", None, missing_pct < 1, "🔍")
with c4:
    kpi_card("Numeric Cols", f"{len(num_cols)}", icon="🔢")
with c5:
    if date_col and len(filtered) > 0:
        d_min = pd.to_datetime(filtered[date_col].min())
        d_max = pd.to_datetime(filtered[date_col].max())
        date_span = f"{d_min.strftime('%b %Y')} — {d_max.strftime('%b %Y')}"
        kpi_card("Date Span", date_span, icon="📅")
    else:
        kpi_card("Categories", f"{len(cat_cols)}", icon="📊")

st.markdown("")

# ── Dynamic Advanced Filters ─────────────────────────────────────────────
view_df = filtered.copy()

with st.expander("🔧 Advanced Column Filters", expanded=False):
    # Row 1: Top 3 categorical filters
    if len(cat_cols) > 0:
        filter_cols = st.columns(min(len(cat_cols), 3))
        for i, col in enumerate(cat_cols[:3]):
            with filter_cols[i]:
                selected = st.multiselect(f"🏷️ {col.replace('_', ' ').title()}",
                    sorted(filtered[col].dropna().unique().tolist()),
                    default=None, key=f"explorer_cat_{col}")
                if selected:
                    view_df = view_df[view_df[col].isin(selected)]

    # Row 2: Top numeric min/max filters
    if len(num_cols) > 0:
        filter_cols2 = st.columns(min(len(num_cols) * 2, 4))
        for i, col in enumerate(num_cols[:2]): 
            min_col_idx = i * 2
            max_col_idx = i * 2 + 1
            col_name = col.replace('_', ' ').title()
            
            with filter_cols2[min_col_idx]:
                col_min = float(filtered[col].min())
                min_val = st.number_input(f"Min {col_name}", value=col_min, key=f"min_{col}")
                if min_val > col_min:
                    view_df = view_df[view_df[col] >= min_val]
            with filter_cols2[max_col_idx]:
                col_max = float(filtered[col].max())
                max_val = st.number_input(f"Max {col_name}", value=col_max, key=f"max_{col}")
                if max_val < col_max:
                    view_df = view_df[view_df[col] <= max_val]

# ── Search ───────────────────────────────────────────────────────────────
search_term = st.text_input("🔍 Full-text search across all columns",
                            placeholder="Search across any column...")
if search_term:
    mask = view_df.astype(str).apply(
        lambda col: col.str.contains(search_term, case=False, na=False)
    ).any(axis=1)
    view_df = view_df[mask]

# ── Dynamic Column Visibility ────────────────────────────────────────────
all_cols = view_df.columns.tolist()
# Dynamic defaults based on schema
default_cols = []
if date_col: default_cols.append(date_col)
default_cols.extend(cat_cols[:4])
default_cols.extend(num_cols[:4])
# Only keep the ones that exist
default_cols = [c for c in default_cols if c in all_cols]

if not default_cols:
    default_cols = all_cols[:10]

with st.expander("📑 Column Visibility", expanded=False):
    selected_cols = st.multiselect("Select columns to display:", all_cols, default=default_cols, key="vis_cols")

if not selected_cols:
    selected_cols = default_cols

# ── Data Table ───────────────────────────────────────────────────────────
st.markdown(f'<div class="section-title">📊 Data Table</div>', unsafe_allow_html=True)
st.markdown(f'<p class="section-subtitle">Showing {len(view_df):,} records · {len(selected_cols)} columns</p>', unsafe_allow_html=True)

st.dataframe(
    view_df[selected_cols].head(1000),
    use_container_width=True,
    height=500,
)

if len(view_df) > 1000:
    st.caption(f"⚡ Displaying first 1,000 of {len(view_df):,} rows for performance. Use CSV export for the full dataset.")

# ── Download ─────────────────────────────────────────────────────────────
st.markdown("---")
col_dl1, col_dl2 = st.columns([1, 4])
with col_dl1:
    csv = view_df[selected_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        csv,
        "forecastiq_export.csv",
        "text/csv",
        key="download_csv",
    )
with col_dl2:
    st.caption(f"Export includes {len(view_df):,} rows × {len(selected_cols)} columns")

# ── Statistical Summary ─────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">📊 Statistical Summary</div>', unsafe_allow_html=True)
st.markdown('<p class="section-subtitle">Descriptive statistics for all numeric columns in the current view</p>', unsafe_allow_html=True)

with st.expander("View statistics", expanded=False):
    view_num_cols = view_df.select_dtypes(include=[np.number]).columns.tolist()
    if view_num_cols:
        st.dataframe(view_df[view_num_cols].describe().T.style.format("{:.2f}"),
                      use_container_width=True)
    else:
        st.info("No numeric columns in the current dataset.")
