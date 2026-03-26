"""
Stacked LSTM Model for Time-Series Forecasting
===============================================
- Multi-layer LSTM with dropout
- Fully connected output head
"""
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """Stacked LSTM for sequence-to-value forecasting."""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2,
                 forecast_horizon: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            out: (batch, forecast_horizon)
        """
        lstm_out, _ = self.lstm(x)            # (B, T, H)
        last_hidden = lstm_out[:, -1, :]      # (B, H)
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)            # (B, forecast_horizon)
