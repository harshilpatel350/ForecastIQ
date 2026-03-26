"""
🚀 ForecastIQ — AI-Powered Sales Analytics Platform
====================================================
Main Streamlit entry point with industry-grade SaaS design.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os, sys, json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ForecastIQ — AI Sales Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Premium CSS Theme ──────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        /* ─── Global Reset ─── */
        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(180deg, #f5f0ff 0%, #ede7f6 50%, #f3e5f5 100%);
            color: #1a0a2e;
        }

        /* ─── Sidebar ─── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8f9fc 0%, #f1f5f9 100%);
            border-right: 1px solid #ddd6fe;
        }
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stMarkdown li {
            color: #334155;
            font-size: 13px;
        }
        section[data-testid="stSidebar"] .stMarkdown h5 {
            color: #1a0a2e;
        }
        section[data-testid="stSidebar"] .stCaption {
            color: #64748b !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: #ddd6fe !important;
        }

        /* ─── Hide default Streamlit chrome ─── */
        header[data-testid="stHeader"] { background: transparent; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        [data-testid="stStatusWidget"] { visibility: hidden; }
        .stDeployButton { display: none !important; }

        /* ─── Rename "streamlit app" → "🏠 Home" in sidebar nav ─── */
        [data-testid="stSidebarNav"] li:first-child a span {
            font-size: 0;
        }
        [data-testid="stSidebarNav"] li:first-child a span::after {
            content: "🏠 Home";
            font-size: 14px;
            font-weight: 600;
            letter-spacing: 0.2px;
        }

        /* ─── Brand Header ─── */
        .brand-header {
            background: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 28px;
            font-weight: 900;
            letter-spacing: -0.75px;
            line-height: 1.3;
            margin-bottom: 2px;
        }
        .brand-subtitle {
            font-size: 15px;
            color: #6b5b8a;
            font-weight: 400;
            margin-top: -4px;
            margin-bottom: 20px;
            line-height: 1.4;
        }

        /* ─── Section Titles ─── */
        .section-title {
            font-size: 18px;
            font-weight: 700;
            color: #1a0a2e;
            margin: 28px 0 8px 0;
            padding-bottom: 10px;
            border-bottom: 3px solid;
            border-image: linear-gradient(90deg, #7c3aed, #a855f7, transparent) 1;
            display: inline-block;
        }
        .section-subtitle {
            font-size: 13px;
            color: #8b7aaf;
            font-weight: 400;
            margin-top: -6px;
            margin-bottom: 16px;
        }

        /* ─── KPI Cards ─── */
        .kpi-card {
            background: rgba(255,255,255,0.85);
            border: 1px solid #ddd6fe;
            border-radius: 14px;
            padding: 22px 24px;
            text-align: left;
            position: relative;
            overflow: hidden;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 2px 8px rgba(124,58,237,0.06);
            backdrop-filter: blur(8px);
        }
        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #7c3aed, #a855f7, #c084fc);
            border-radius: 14px 14px 0 0;
        }
        .kpi-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 28px rgba(124,58,237,0.18);
            border-color: #a78bfa;
        }
        .kpi-icon {
            font-size: 20px;
            margin-bottom: 6px;
        }
        .kpi-title {
            font-size: 11px;
            font-weight: 600;
            color: #8b7aaf;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 6px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .kpi-value {
            font-size: 26px;
            font-weight: 800;
            color: #2d1b69;
            line-height: 1.2;
            letter-spacing: -0.5px;
            white-space: nowrap;
        }
        .kpi-delta {
            font-size: 12px;
            font-weight: 600;
            margin-top: 6px;
            display: inline-flex;
            align-items: center;
            gap: 3px;
            padding: 2px 8px;
            border-radius: 20px;
            white-space: nowrap;
        }
        .kpi-delta.positive {
            color: #7c3aed;
            background: #ede9fe;
        }
        .kpi-delta.negative {
            color: #dc2626;
            background: #fee2e2;
        }

        /* ─── Responsive Adjustments ─── */
        @media (max-width: 1200px) {
            .kpi-card { padding: 16px 14px; }
            .kpi-value { font-size: 20px; }
            .kpi-title { font-size: 10px; }
        }

        @media (max-width: 992px) {
            .kpi-card { padding: 12px 10px; }
            .kpi-value { font-size: 15px; }
            .kpi-title { font-size: 9px; letter-spacing: 0.5px; }
            .kpi-icon { font-size: 16px; margin-bottom: 2px;}
            .kpi-delta { font-size: 10px; padding: 2px 4px; }
            
            /* Constrain sidebar width on smaller laptops */
            section[data-testid="stSidebar"] { min-width: 260px !important; max-width: 260px !important; }
        }


        /* ─── Streamlit Native Widget Labels (replaces manual HTML labels) ─── */
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stMultiSelect label,
        section[data-testid="stSidebar"] .stDateInput label {
            font-size: 11px !important;
            font-weight: 700 !important;
            color: #6b5b8a !important;
            text-transform: uppercase !important;
            letter-spacing: 0.8px !important;
            margin-bottom: 2px !important;
        }

        /* ─── Plotly Charts ─── */
        .js-plotly-plot .plotly .main-svg {
            background: transparent !important;
        }

        /* ─── Streamlit Widgets ─── */
        .stSelectbox > div > div,
        .stMultiSelect > div > div {
            background: rgba(255,255,255,0.9);
            border: 1px solid #ddd6fe;
            border-radius: 10px;
            font-size: 13px;
        }
        .stDateInput > div > div {
            border-radius: 10px;
        }
        section[data-testid="stSidebar"] .stSelectbox > div > div,
        section[data-testid="stSidebar"] .stMultiSelect > div > div {
            background: rgba(255,255,255,0.9);
            border: 1px solid #ddd6fe;
            color: #1a0a2e;
        }
        section[data-testid="stSidebar"] .stDateInput > div > div > input {
            color: #1a0a2e;
        }

        /* ─── Buttons ─── */
        .stButton > button {
            background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            font-size: 13px;
            padding: 8px 24px;
            transition: all 0.2s;
            letter-spacing: 0.3px;
        }
        .stButton > button:hover {
            opacity: 0.9;
            box-shadow: 0 6px 16px rgba(124, 58, 237, 0.35);
        }

        /* ─── Tabs ─── */
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] {
            background: #ede9fe;
            border-radius: 10px;
            border: 1px solid #ddd6fe;
            padding: 8px 20px;
            font-weight: 600;
            font-size: 13px;
            color: #5b21b6;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
            color: white !important;
            border: none;
        }

        /* ─── DataFrames ─── */
        .stDataFrame { border: 1px solid #ddd6fe; border-radius: 10px; }

        /* ─── Metrics ─── */
        [data-testid="metric-container"] {
            background: rgba(255,255,255,0.85);
            border: 1px solid #ddd6fe;
            border-radius: 14px;
            padding: 16px;
            box-shadow: 0 2px 8px rgba(124,58,237,0.06);
        }

        /* ─── Expander ─── */
        .streamlit-expanderHeader {
            font-weight: 700;
            font-size: 14px;
            color: #1a0a2e;
        }

        /* ─── Sidebar Version Badge ─── */
        .version-badge {
            display: inline-block;
            background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
            color: white;
            font-size: 10px;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 20px;
            letter-spacing: 0.5px;
        }

        /* ─── Horizontal Rule ─── */
        hr {
            border: none;
            border-top: 1px solid #ddd6fe;
            margin: 16px 0;
        }

        /* ─── Info / Warning / Error boxes ─── */
        .stAlert { border-radius: 10px; }

        /* ─── Sidebar brand area ─── */
        .sidebar-brand {
            text-align: center;
            padding: 8px 0 16px 0;
        }
        .sidebar-brand-name {
            font-size: 22px;
            font-weight: 900;
            background: linear-gradient(135deg, #7c3aed, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }
        .sidebar-brand-tagline {
            font-size: 11px;
            color: #8b7aaf;
            font-weight: 500;
            letter-spacing: 0.3px;
        }
    </style>
    """, unsafe_allow_html=True)


# ── Plotly Defaults ────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color="#2d1b69"),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor="#ffffff", font_size=12, font_family="Inter", bordercolor="#ddd6fe"),
    legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
)


# ── Data Loading (v4: cache clear forced) ───────────────────────────────────
@st.cache_data(ttl=600)
def load_market_data():
    path = os.path.join(PROJECT_ROOT, "data", "sales_data.parquet")
    
    # Check for corrupted/empty files left over from previous crashes
    if os.path.exists(path) and os.path.getsize(path) == 0:
        os.remove(path)
        
    if not os.path.exists(path):
        with st.spinner("⚠️ Dataset not found. Generating a fresh AI dataset for deployment (this takes ~15 seconds)..."):
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
            from data_generation import generate_dataset
            # Generate 1 year of data instead of 3 to keep it lightweight for cloud
            df = generate_dataset(start_date="2024-01-01", end_date="2024-12-31", output_path="")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_parquet(path)
            st.success("Dataset generated successfully!")
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["date"] = df["datetime"].dt.normalize()
            return df
    
    try:
        df = pd.read_parquet(path)
        # Ensure datetime is a proper Timestamp
        df["datetime"] = pd.to_datetime(df["datetime"])
        # Set date as a normalized timestamp at midnight (keeps the .dt accessor alive)
        df["date"] = df["datetime"].dt.normalize()
        return df
    except Exception as e:
        # If still corrupt, delete and retry recursively once
        if os.path.exists(path): os.remove(path)
        st.error(f"⚠️ Data error: {str(e)}. Refreshing page.")
        return generate_dataset(start_date="2024-01-01", end_date="2024-12-31", output_path="")


@st.cache_data(ttl=600)
def load_daily_data(df):
    daily = df.groupby("date").agg(
        daily_revenue=("total_amount", "sum"),
        daily_orders=("order_id", "count"),
        avg_order_value=("total_amount", "mean"),
        avg_rating=("rating", "mean"),
        cancel_rate=("cancellation_flag", "mean"),
    ).reset_index()
    return daily


def load_model_results():
    path = os.path.join(PROJECT_ROOT, "data", "model_comparison.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# ── Custom Upload Helpers ──────────────────────────────────────────────────
def _normalize_uploaded_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize an uploaded CSV so it works with the dashboard.
    Auto-detects date columns, lowercases column names, and ensures
    the 'date' and 'datetime' columns are proper Timestamps."""
    # Lowercase all column names for case-insensitive matching
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Try to find / create a datetime column
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        # Try to find any column that looks like a date
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().sum() > len(df) * 0.5:  # >50% valid dates
                        df["datetime"] = parsed
                        break
                except Exception:
                    continue

    # Ensure 'date' column exists as normalized timestamp
    if "datetime" in df.columns:
        df["date"] = df["datetime"].dt.normalize()
        if "hour" not in df.columns:
            df["hour"] = df["datetime"].dt.hour
        if "weekday" not in df.columns:
            df["weekday"] = df["datetime"].dt.weekday
        if "month" not in df.columns:
            df["month"] = df["datetime"].dt.month
        if "year" not in df.columns:
            df["year"] = df["datetime"].dt.year
        if "day_of_year" not in df.columns:
            df["day_of_year"] = df["datetime"].dt.day_of_year

    # Generate a synthetic order_id if missing
    if "order_id" not in df.columns:
        df["order_id"] = [f"ORD-{i}" for i in range(1, len(df) + 1)]

    return df


def get_active_dataset() -> pd.DataFrame:
    """Return the uploaded dataset if available, otherwise the default."""
    if "uploaded_df" in st.session_state and st.session_state["uploaded_df"] is not None:
        return st.session_state["uploaded_df"]
    return load_market_data()


# ── KPI Card Helper ───────────────────────────────────────────────────────
def kpi_card(title: str, value: str, delta: str = None, delta_positive: bool = True, icon: str = ""):
    delta_html = ""
    if delta:
        cls = "positive" if delta_positive else "negative"
        arrow = "↑" if delta_positive else "↓"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    icon_html = f'<div class="kpi-icon">{icon}</div>' if icon else ""
    st.markdown(f"""
    <div class="kpi-card">
        {icon_html}
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────
def render_sidebar(df):
    cols = df.columns.tolist()

    with st.sidebar:
        # Brand header
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-name">📊 ForecastIQ</div>
            <div class="sidebar-brand-tagline">AI-Powered Sales Analytics</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        # ── Custom Dataset Upload ──
        st.markdown("##### 📂 Dataset")
        uploaded_file = st.file_uploader(
            "Upload your own CSV", type=["csv"],
            key="csv_uploader",
            help="Upload a CSV file to analyze your own data. The dashboard will adapt to your columns."
        )

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                normalized = _normalize_uploaded_df(raw_df)
                st.session_state["uploaded_df"] = normalized
                st.success(f"✅ Loaded **{len(normalized):,}** rows · **{len(normalized.columns)}** columns")
            except Exception as e:
                st.error(f"❌ Failed to parse CSV: {e}")

        # Show clear button if custom data is active
        if "uploaded_df" in st.session_state and st.session_state["uploaded_df"] is not None:
            st.caption("🔄 Using **uploaded dataset**")
            if st.button("❌ Clear Upload & Use Default", key="clear_upload"):
                st.session_state["uploaded_df"] = None
                st.rerun()
            # Refresh df to use uploaded data
            df = st.session_state["uploaded_df"]
            cols = df.columns.tolist()
        else:
            st.caption("📦 Using **default dataset**")

        st.markdown("---")

        # ── Date Range (only if 'date' column exists) ──
        if "date" in cols:
            try:
                raw_min = df["date"].min()
                raw_max = df["date"].max()
                min_date = pd.to_datetime(raw_min).date()
                max_date = pd.to_datetime(raw_max).date()
            except:
                min_date = datetime.now().date() - timedelta(days=30)
                max_date = datetime.now().date()
            date_range = st.date_input("📅 Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
        else:
            date_range = []

        # ── City (adaptive) ──
        selected_cities = []
        if "city" in cols:
            cities = sorted(df["city"].unique().tolist())
            selected_cities = st.multiselect("🏙️ City", cities, default=cities, key="filter_city")

        # ── Category (adaptive) ──
        selected_categories = []
        if "category" in cols:
            categories = sorted(df["category"].unique().tolist())
            selected_categories = st.multiselect("🍽️ Category", categories, default=categories, key="filter_category")

        # ── Apply base filters first to cascade ──
        filtered = df.copy()
        if "date" in cols and len(date_range) == 2:
            start_dt = pd.to_datetime(date_range[0])
            end_dt = pd.to_datetime(date_range[1])
            filtered = filtered[(filtered["date"] >= start_dt) & (filtered["date"] <= end_dt)]
        if selected_cities and "city" in cols:
            filtered = filtered[filtered["city"].isin(selected_cities)]
        if selected_categories and "category" in cols:
            filtered = filtered[filtered["category"].isin(selected_categories)]

        # ── Cascaded Filters (only show what exists) ──
        more_filters = []
        if "restaurant" in cols: more_filters.append("restaurant")
        if "product" in cols: more_filters.append("product")
        if "order_type" in cols: more_filters.append("order_type")
        if "payment_method" in cols: more_filters.append("payment_method")
        if "weather" in cols: more_filters.append("weather")

        selected_restaurants = selected_products = selected_order_types = selected_pay = selected_weather = []

        if more_filters:
            with st.expander("🔽 More Filters", expanded=False):
                if "restaurant" in cols:
                    avail_restaurants = sorted(filtered["restaurant"].unique().tolist())
                    selected_restaurants = st.multiselect("🏪 Restaurant", avail_restaurants, default=avail_restaurants, key="filter_rest")

                if "product" in cols:
                    avail_products = sorted(filtered["product"].unique().tolist())
                    selected_products = st.multiselect("🍕 Product", avail_products, default=avail_products, key="filter_prod")

                if "order_type" in cols:
                    order_types = sorted(filtered["order_type"].unique().tolist())
                    selected_order_types = st.multiselect("🚗 Order Type", order_types, default=order_types, key="filter_ot")

                if "payment_method" in cols:
                    pay_methods = sorted(filtered["payment_method"].unique().tolist())
                    selected_pay = st.multiselect("💳 Payment Method", pay_methods, default=pay_methods, key="filter_pay")

                if "weather" in cols:
                    weathers = sorted(filtered["weather"].unique().tolist())
                    selected_weather = st.multiselect("🌦️ Weather", weathers, default=weathers, key="filter_weather")

        # Apply remaining filters
        if selected_restaurants and "restaurant" in cols:
            filtered = filtered[filtered["restaurant"].isin(selected_restaurants)]
        if selected_products and "product" in cols:
            filtered = filtered[filtered["product"].isin(selected_products)]
        if selected_order_types and "order_type" in cols:
            filtered = filtered[filtered["order_type"].isin(selected_order_types)]
        if selected_pay and "payment_method" in cols:
            filtered = filtered[filtered["payment_method"].isin(selected_pay)]
        if selected_weather and "weather" in cols:
            filtered = filtered[filtered["weather"].isin(selected_weather)]

        st.markdown("---")

        # Quick stats (adaptive)
        st.markdown("##### ⚡ Dataset Overview")
        stat_cols_ui = st.columns(2)
        with stat_cols_ui[0]:
            st.caption(f"📦 **{len(filtered):,}** rows")
            if "city" in cols:
                st.caption(f"🏙️ **{filtered['city'].nunique()}** cities")
        with stat_cols_ui[1]:
            if "product" in cols:
                st.caption(f"🍕 **{filtered['product'].nunique()}** products")
            if "date" in cols:
                st.caption(f"📅 **{filtered['date'].nunique()}** days")

        st.markdown("---")
        st.markdown(
            '<div style="text-align:center;">'
            '<span class="version-badge">v1.1.0</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.session_state["filtered_df"] = filtered
    return filtered


# ── Main Entry Point (redirects to Overview) ──────────────────────────────
def main():
    inject_css()
    df = get_active_dataset()
    filtered = render_sidebar(df)

    # Show a welcome message — all analytics live on the Overview page
    st.markdown('<p class="brand-header">🏠 Welcome to ForecastIQ</p>', unsafe_allow_html=True)
    st.markdown('<p class="brand-subtitle">Navigate to any page using the sidebar to explore your sales data</p>', unsafe_allow_html=True)

    st.markdown("")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon">📊</div>
            <div class="kpi-title">Overview</div>
            <div style="font-size:13px; color:#64748b;">KPIs, revenue trends, city & category breakdown</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon">🔬</div>
            <div class="kpi-title">Deep Analytics</div>
            <div style="font-size:13px; color:#64748b;">Drill-downs, heatmaps, weather & discount analysis</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon">🔮</div>
            <div class="kpi-title">Forecasting</div>
            <div style="font-size:13px; color:#64748b;">AI predictions, scenario simulation, anomaly detection</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    col4, col5, _ = st.columns(3)
    with col4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon">🍕</div>
            <div class="kpi-title">Product Intelligence</div>
            <div style="font-size:13px; color:#64748b;">Top/bottom products, demand patterns, AI recommendations</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon">📋</div>
            <div class="kpi-title">Data Explorer</div>
            <div style="font-size:13px; color:#64748b;">Search, filter, and export your full dataset</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.info("👈 **Select a page from the sidebar** to get started with your analysis.")


if __name__ == "__main__":
    main()

