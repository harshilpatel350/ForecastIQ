"""
Attention-Based LSTM for Time-Series Forecasting
=================================================
- Stacked LSTM encoder
- Bahdanau-style attention mechanism to weight important timesteps
- FC decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Bahdanau (additive) attention over LSTM hidden states."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_outputs: torch.Tensor) -> tuple:
        """
        Args:
            lstm_outputs: (B, T, H)
        Returns:
            context: (B, H)
            attn_weights: (B, T)
        """
        scores = self.V(torch.tanh(self.W(lstm_outputs)))  # (B, T, 1)
        attn_weights = F.softmax(scores, dim=1)             # (B, T, 1)
        context = (attn_weights * lstm_outputs).sum(dim=1)   # (B, H)
        return context, attn_weights.squeeze(-1)


class AttentionLSTM(nn.Module):
    """LSTM with Bahdanau attention for time-series forecasting."""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2,
                 forecast_horizon: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = Attention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)                    # (B, T, H)
        context, self.attn_weights = self.attention(lstm_out)  # (B, H)
        context = self.dropout(context)
        return self.fc(context)                        # (B, forecast_horizon)
