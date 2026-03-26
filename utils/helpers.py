"""
Utility helpers used across the project.
├── metrics.py-like functions (RMSE, MAE, MAPE)
├── plotting helpers
├── file I/O
"""
import os, json, logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# ── Logging ────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("SalesForecast")

# ── Metrics ────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred), "MAPE": mape(y_true, y_pred)}

# ── Config ─────────────────────────────────────────────────────────────────
DEFAULT_CONFIG: Dict[str, Any] = {
    "sequence_length": 30,
    "forecast_horizon": 7,
    "batch_size": 64,
    "epochs": 50,
    "learning_rate": 1e-3,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "n_heads": 4,
    "d_model": 64,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
}

def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return DEFAULT_CONFIG.copy()

# ── I/O ────────────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = ensure_dir(os.path.join(PROJECT_ROOT, "data"))
MODEL_DIR = ensure_dir(os.path.join(PROJECT_ROOT, "data", "models"))
