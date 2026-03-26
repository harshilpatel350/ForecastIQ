"""
Full ML Pipeline Runner
========================
Run this to execute the entire pipeline:
  1. Generate dataset (if not exists)
  2. Preprocess
  3. Feature engineering
  4. Train all models
  5. Evaluate and compare
"""
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_generation import generate_dataset
from src.preprocessing import run_preprocessing_pipeline
from src.feature_engineering import run_feature_engineering
from src.training.train import run_training_pipeline
from utils.helpers import logger, DATA_DIR
import pandas as pd


def main():
    # Step 1: Generate dataset
    data_path = os.path.join(DATA_DIR, "sales_data.csv")
    if not os.path.exists(data_path):
        logger.info("Step 1: Generating dataset …")
        df = generate_dataset(output_path=data_path)
    else:
        logger.info("Step 1: Loading existing dataset …")
        df = pd.read_csv(data_path, parse_dates=["datetime", "date"])

    logger.info(f"Dataset: {len(df):,} rows × {len(df.columns)} columns")

    # Step 2: Preprocess
    logger.info("Step 2: Preprocessing …")
    df_clean, encoders, scaler = run_preprocessing_pipeline(df)

    # Step 3: Feature Engineering
    logger.info("Step 3: Feature Engineering …")
    daily_features = run_feature_engineering(df_clean, aggregate=True)

    # Save feature-engineered data
    feat_path = os.path.join(DATA_DIR, "daily_features.csv")
    daily_features.to_csv(feat_path, index=False)
    logger.info(f"Features saved → {feat_path}")

    # Step 4+5: Train & Evaluate
    logger.info("Step 4: Training models …")
    feature_cols = [c for c in daily_features.columns if c not in ["date", "daily_revenue"]]
    results = run_training_pipeline(
        daily_features,
        feature_cols=feature_cols,
        target_col="daily_revenue",
        model_names=["lstm", "attention", "transformer"],
    )

    logger.info("✅ Pipeline complete!")
    for name, res in results.items():
        m = res["metrics"]
        logger.info(f"  {name:>15s}: RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  MAPE={m['MAPE']:.2f}%")


if __name__ == "__main__":
    main()
