"""
Training Pipeline
=================
- Dataset preparation (sliding windows)
- Time-series train/val/test split (no leakage)
- Training loop with early stopping
- Model saving
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.helpers import logger, compute_all_metrics, load_config, MODEL_DIR, DATA_DIR
from src.models.lstm import LSTMForecaster
from src.models.attention import AttentionLSTM
from src.models.transformer import TransformerForecaster


# ── Dataset ────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for sequence-to-value forecasting."""

    def __init__(self, data: np.ndarray, target_idx: int,
                 seq_len: int = 30, horizon: int = 1):
        self.seq_len = seq_len
        self.horizon = horizon
        self.target_idx = target_idx

        self.X, self.y = [], []
        for i in range(len(data) - seq_len - horizon + 1):
            self.X.append(data[i : i + seq_len])
            # Target: next `horizon` values of the target column
            self.y.append(data[i + seq_len : i + seq_len + horizon, target_idx])
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# ── Splits ─────────────────────────────────────────────────────────────────

def time_series_split(data: np.ndarray, train_r: float = 0.7,
                      val_r: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(data)
    t1 = int(n * train_r)
    t2 = int(n * (train_r + val_r))
    return data[:t1], data[t1:t2], data[t2:]


# ── Model Factory ─────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "lstm": LSTMForecaster,
    "attention": AttentionLSTM,
    "transformer": TransformerForecaster,
}


def build_model(name: str, input_size: int, cfg: Dict) -> nn.Module:
    if name == "lstm":
        return LSTMForecaster(input_size, cfg["hidden_size"], cfg["num_layers"],
                              cfg["dropout"], cfg["forecast_horizon"])
    elif name == "attention":
        return AttentionLSTM(input_size, cfg["hidden_size"], cfg["num_layers"],
                             cfg["dropout"], cfg["forecast_horizon"])
    elif name == "transformer":
        return TransformerForecaster(input_size, cfg["d_model"], cfg["n_heads"],
                                    cfg["num_layers"], cfg["d_model"] * 4,
                                    cfg["dropout"], cfg["forecast_horizon"])
    else:
        raise ValueError(f"Unknown model: {name}")


# ── Training Loop ─────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    cfg: Dict,
    device: torch.device,
) -> Dict:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 10
    history = {"train_loss": [], "val_loss": []}

    save_path = os.path.join(MODEL_DIR, f"{model_name}_best.pt")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_losses.append(criterion(pred, y_batch).item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        scheduler.step(avg_val)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"[{model_name}] Epoch {epoch:3d}/{cfg['epochs']} │ "
                        f"Train: {avg_train:.6f} │ Val: {avg_val:.6f} │ "
                        f"Time: {time.time()-t0:.1f}s")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"[{model_name}] Early stopping at epoch {epoch}")
                break

    # Load best weights
    model.load_state_dict(torch.load(save_path, weights_only=True))
    logger.info(f"[{model_name}] Best val loss: {best_val_loss:.6f} → {save_path}")
    return history


# ── Evaluation on test set ─────────────────────────────────────────────────

def evaluate_model(model: nn.Module, test_loader: DataLoader,
                   device: torch.device) -> Tuple[np.ndarray, np.ndarray, Dict]:
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_all_metrics(all_targets.flatten(), all_preds.flatten())
    return all_preds, all_targets, metrics


# ── Full Training Pipeline ────────────────────────────────────────────────

def run_training_pipeline(
    feature_df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "daily_revenue",
    model_names: list = None,
    cfg: Dict = None,
) -> Dict:
    """Train all specified models on the feature-engineered daily data."""
    if cfg is None:
        cfg = load_config()
    if model_names is None:
        model_names = ["lstm", "attention", "transformer"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Prepare data matrix
    all_cols = feature_cols + [target_col] if target_col not in feature_cols else feature_cols
    # Ensure target is last if not in feature_cols
    if target_col not in feature_cols:
        data_cols = feature_cols + [target_col]
    else:
        data_cols = feature_cols
    data = feature_df[data_cols].values.astype(np.float32)
    target_idx = data_cols.index(target_col)

    # Time-series split
    train_data, val_data, test_data = time_series_split(data, cfg["train_ratio"], cfg["val_ratio"])

    train_ds = TimeSeriesDataset(train_data, target_idx, cfg["sequence_length"], cfg["forecast_horizon"])
    val_ds = TimeSeriesDataset(val_data, target_idx, cfg["sequence_length"], cfg["forecast_horizon"])
    test_ds = TimeSeriesDataset(test_data, target_idx, cfg["sequence_length"], cfg["forecast_horizon"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"])

    input_size = data.shape[1]
    results = {}

    for name in model_names:
        logger.info(f"\n{'='*60}\nTraining {name.upper()} model\n{'='*60}")
        model = build_model(name, input_size, cfg)
        history = train_model(model, train_loader, val_loader, name, cfg, device)
        preds, targets, metrics = evaluate_model(model, test_loader, device)
        results[name] = {
            "history": history,
            "predictions": preds,
            "targets": targets,
            "metrics": metrics,
        }
        logger.info(f"[{name}] Test metrics: {metrics}")

    # Save results summary
    summary = {name: res["metrics"] for name, res in results.items()}
    summary_path = os.path.join(DATA_DIR, "model_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Model comparison saved → {summary_path}")

    # Auto-select best model
    best_model = min(summary, key=lambda k: summary[k]["RMSE"])
    logger.info(f"🏆 Best model: {best_model} (RMSE={summary[best_model]['RMSE']:.4f})")

    return results
