"""
Advanced Data Preprocessing Pipeline
=====================================
- Smart missing-value imputation (KNN / median / mode by group)
- Outlier detection & capping (IQR method)
- Target encoding for high-cardinality categoricals
- Normalization for DL inputs
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from typing import Tuple, Dict, List
import joblib, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.helpers import logger, DATA_DIR

# ── Missing Value Imputation ───────────────────────────────────────────────

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Smart imputation: group-median for numerics, mode for categoricals."""
    df = df.copy()

    # Numeric columns – impute with city-category median
    num_cols = ["rating", "temperature", "preparation_time", "competitor_price_index", "discount_pct"]
    for col in num_cols:
        if col in df.columns and df[col].isna().any():
            medians = df.groupby(["city", "category"])[col].transform("median")
            df[col] = df[col].fillna(medians)
            # If still NaN (group had no data), use global median
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns
    cat_cols = ["weather", "delivery_partner", "inventory_status"]
    for col in cat_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    logger.info(f"Imputation done. Remaining NaNs: {df.isna().sum().sum()}")
    return df


# ── Outlier Handling ───────────────────────────────────────────────────────

def handle_outliers(df: pd.DataFrame, cols: List[str] = None, method: str = "iqr") -> pd.DataFrame:
    """Cap outliers using IQR fence method."""
    df = df.copy()
    if cols is None:
        cols = ["total_amount", "quantity", "preparation_time", "unit_price"]
    for col in cols:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        before = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        logger.info(f"Outliers capped in '{col}': {before} rows")
    return df


# ── Encoding ───────────────────────────────────────────────────────────────

def target_encode(df: pd.DataFrame, col: str, target: str = "total_amount", smoothing: int = 10) -> pd.DataFrame:
    """Target encoding with smoothing to prevent overfitting."""
    global_mean = df[target].mean()
    agg = df.groupby(col)[target].agg(["mean", "count"])
    smooth = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
    df[f"{col}_encoded"] = df[col].map(smooth)
    return df


def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode lower-cardinality categoricals; target-encode high-cardinality ones."""
    df = df.copy()
    encoders: Dict[str, LabelEncoder] = {}

    # Target encode high-cardinality
    for col in ["restaurant", "product", "customer_id"]:
        if col in df.columns:
            df = target_encode(df, col)

    # Label-encode low-cardinality
    label_cols = ["city", "category", "payment_method", "order_type", "delivery_partner",
                  "weather", "inventory_status"]
    for col in label_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_le"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    return df, encoders


# ── Normalization ──────────────────────────────────────────────────────────

def normalize_features(df: pd.DataFrame, feature_cols: List[str],
                       scaler_path: str = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """MinMax scale the numeric feature columns for DL."""
    df = df.copy()
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols].values)
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved → {scaler_path}")
    return df, scaler


# ── Full Pipeline ──────────────────────────────────────────────────────────

def run_preprocessing_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, MinMaxScaler]:
    """Run full preprocessing: impute → outlier → encode → normalize."""
    logger.info("Starting preprocessing pipeline …")
    df = impute_missing(df)
    df = handle_outliers(df)
    df, encoders = encode_categoricals(df)

    # Define numeric feature columns for scaling
    feature_cols = [
        "quantity", "unit_price", "discount_pct", "total_amount",
        "rating", "temperature", "competitor_price_index", "demand_index",
        "preparation_time", "holiday_multiplier",
    ]
    # Add encoded columns
    enc_cols = [c for c in df.columns if c.endswith("_encoded") or c.endswith("_le")]
    feature_cols.extend(enc_cols)
    feature_cols = [c for c in feature_cols if c in df.columns]

    scaler_path = os.path.join(DATA_DIR, "models", "scaler.pkl")
    df, scaler = normalize_features(df, feature_cols, scaler_path)

    logger.info(f"Preprocessing complete. Shape: {df.shape}")
    return df, encoders, scaler
