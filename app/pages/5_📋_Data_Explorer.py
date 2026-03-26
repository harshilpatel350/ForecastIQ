"""
Page 5: Data Explorer
======================
Interactive data table with advanced filters, search, column visibility, CSV export, and statistics.
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
cols = filtered.columns.tolist()

st.markdown('<p class="brand-header">📋 Data Explorer</p>', unsafe_allow_html=True)
st.markdown('<p class="brand-subtitle">Browse, search, and export your complete dataset with advanced filtering</p>', unsafe_allow_html=True)

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
    if "customer_id" in cols:
        kpi_card("Unique Customers", f"{filtered['customer_id'].nunique():,}", icon="👥")
    else:
        kpi_card("Numeric Cols", f"{len(filtered.select_dtypes(include=[np.number]).columns)}", icon="🔢")
with c5:
    if "date" in cols:
        d_min = pd.to_datetime(filtered['date'].min())
        d_max = pd.to_datetime(filtered['date'].max())
        date_span = f"{d_min.strftime('%b %Y')} — {d_max.strftime('%b %Y')}"
        kpi_card("Date Span", date_span, icon="📅")
    else:
        kpi_card("Dtypes", f"{filtered.dtypes.nunique()}", icon="📊")

st.markdown("")

# ── Advanced Filters ─────────────────────────────────────────────────────
with st.expander("🔧 Advanced Column Filters", expanded=False):
    filter_cols = st.columns(3)

    view_df = filtered.copy()

    with filter_cols[0]:
        if "restaurant" in cols:
            selected_restaurants = st.multiselect("🏪 Restaurant",
                sorted(filtered["restaurant"].unique().tolist()),
                default=None, key="explorer_rest")
            if selected_restaurants:
                view_df = view_df[view_df["restaurant"].isin(selected_restaurants)]

    with filter_cols[1]:
        if "product" in cols:
            selected_products = st.multiselect("🍕 Product",
                sorted(filtered["product"].unique().tolist()),
                default=None, key="explorer_prod")
            if selected_products:
                view_df = view_df[view_df["product"].isin(selected_products)]

    with filter_cols[2]:
        if "order_type" in cols:
            order_types = st.multiselect("🚗 Order Type",
                sorted(filtered["order_type"].unique().tolist()),
                default=None, key="explorer_ot")
            if order_types:
                view_df = view_df[view_df["order_type"].isin(order_types)]

    filter_cols2 = st.columns(3)
    with filter_cols2[0]:
        if "total_amount" in cols:
            min_amount = st.number_input("Min Amount (₹)", min_value=0, value=0, key="min_amt")
            if min_amount > 0:
                view_df = view_df[view_df["total_amount"] >= min_amount]
    with filter_cols2[1]:
        if "total_amount" in cols:
            max_amount = st.number_input("Max Amount (₹)", min_value=0,
                                         value=int(filtered["total_amount"].max()) + 1, key="max_amt")
            if max_amount < filtered["total_amount"].max() + 1:
                view_df = view_df[view_df["total_amount"] <= max_amount]
    with filter_cols2[2]:
        if "cancellation_flag" in cols:
            cancel_filter = st.selectbox("Cancellation Status",
                                         ["All", "Not Cancelled", "Cancelled Only"], key="cancel_f")
            if cancel_filter == "Not Cancelled":
                view_df = view_df[view_df["cancellation_flag"] == 0]
            elif cancel_filter == "Cancelled Only":
                view_df = view_df[view_df["cancellation_flag"] == 1]

if "view_df" not in dir():
    view_df = filtered.copy()

# ── Search ───────────────────────────────────────────────────────────────
search_term = st.text_input("🔍 Full-text search across all columns",
                            placeholder="Search across any column...")
if search_term:
    mask = view_df.astype(str).apply(
        lambda col: col.str.contains(search_term, case=False, na=False)
    ).any(axis=1)
    view_df = view_df[mask]

# ── Column Visibility ────────────────────────────────────────────────────
all_cols = view_df.columns.tolist()
# Prefer showing common useful columns
preferred = ["order_id", "datetime", "date", "city", "restaurant", "product", "category",
             "quantity", "total_amount", "payment_method", "order_type", "rating",
             "weather", "cancellation_flag"]
default_cols = [c for c in preferred if c in all_cols]
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
    num_cols = view_df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        st.dataframe(view_df[num_cols].describe().T.style.format("{:.2f}"),
                      use_container_width=True)
    else:
        st.info("No numeric columns in the current selection.")
