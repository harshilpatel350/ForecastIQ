"""
Advanced Feature Engineering for Time-Series Forecasting
========================================================
- Lag features (t-1, t-7, t-14, t-30)
- Rolling statistics (mean, std, EMA)
- Time-based features (cyclical hour, weekday, month)
- Fourier terms for seasonality
- Demand trend indicators
"""
import numpy as np
import pandas as pd
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.helpers import logger

# ── Lag Features ───────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, target: str = "daily_revenue",
                     lags: list = None) -> pd.DataFrame:
    if lags is None:
        lags = [1, 7, 14, 30]
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    logger.info(f"Added lag features: {lags}")
    return df


# ── Rolling Statistics ─────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame, target: str = "daily_revenue",
                         windows: list = None) -> pd.DataFrame:
    if windows is None:
        windows = [7, 14, 30]
    for w in windows:
        df[f"{target}_roll_mean_{w}"] = df[target].rolling(w, min_periods=1).mean()
        df[f"{target}_roll_std_{w}"] = df[target].rolling(w, min_periods=1).std().fillna(0)
        df[f"{target}_ema_{w}"] = df[target].ewm(span=w, min_periods=1).mean()
    logger.info(f"Added rolling features: windows={windows}")
    return df


# ── Time-Based Cyclical Features ──────────────────────────────────────────

def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    dt = pd.to_datetime(df[date_col])
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["month_num"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["day_of_year_feat"] = dt.dt.dayofyear

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df.get("hour", 12) / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.get("hour", 12) / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    logger.info("Added cyclical time features")
    return df


# ── Fourier Terms ──────────────────────────────────────────────────────────

def add_fourier_terms(df: pd.DataFrame, date_col: str = "date", n_terms: int = 4, period: int = 365) -> pd.DataFrame:
    dt = pd.to_datetime(df[date_col])
    day_of_year = dt.dt.dayofyear
    for k in range(1, n_terms + 1):
        df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * day_of_year / period)
        df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * day_of_year / period)
    logger.info(f"Added {n_terms} Fourier term pairs (period={period})")
    return df


# ── Demand Trend Indicators ───────────────────────────────────────────────

def add_trend_indicators(df: pd.DataFrame, target: str = "daily_revenue") -> pd.DataFrame:
    df["trend_diff_1"] = df[target].diff(1)
    df["trend_diff_7"] = df[target].diff(7)
    df["trend_pct_change_7"] = df[target].pct_change(7).replace([np.inf, -np.inf], 0).fillna(0)
    # Momentum: ratio of short-term vs long-term average
    short_ma = df[target].rolling(7, min_periods=1).mean()
    long_ma = df[target].rolling(30, min_periods=1).mean()
    df["momentum"] = (short_ma / long_ma.replace(0, 1)).fillna(1)
    logger.info("Added trend & momentum indicators")
    return df


# ── Aggregate to Daily Level ──────────────────────────────────────────────

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly order-level data to daily level for time-series modeling."""
    daily = df.groupby("date").agg(
        daily_revenue=("total_amount", "sum"),
        daily_orders=("order_id", "count"),
        avg_order_value=("total_amount", "mean"),
        avg_rating=("rating", "mean"),
        avg_temperature=("temperature", "mean"),
        avg_prep_time=("preparation_time", "mean"),
        avg_discount=("discount_pct", "mean"),
        avg_competitor_idx=("competitor_price_index", "mean"),
        avg_demand_idx=("demand_index", "mean"),
        cancel_rate=("cancellation_flag", "mean"),
        pct_delivery=("order_type", lambda x: (x == "Delivery").mean()),
        pct_new_customer=("is_new_customer", "mean"),
        is_holiday=("is_holiday", "max"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    logger.info(f"Aggregated to daily: {len(daily)} rows")
    return daily


# ── Full Feature Engineering Pipeline ──────────────────────────────────────

def run_feature_engineering(df: pd.DataFrame, aggregate: bool = True) -> pd.DataFrame:
    """Run the complete feature engineering pipeline."""
    logger.info("Starting feature engineering …")

    if aggregate:
        df = aggregate_daily(df)

    target = "daily_revenue"
    df = add_lag_features(df, target)
    df = add_rolling_features(df, target)
    df = add_time_features(df, "date")
    df = add_fourier_terms(df, "date")
    df = add_trend_indicators(df, target)

    # Drop rows with NaN from lag/rolling (head)
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Dropped {initial_len - len(df)} rows with NaN from lead-in period")
    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df
