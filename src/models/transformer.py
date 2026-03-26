"""
Transformer Encoder for Time-Series Forecasting
================================================
- Positional encoding (sinusoidal)
- Multi-head self-attention encoder layers
- Global average pooling → FC head
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence data."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1]) if div_term.shape[0] > d_model // 2 else torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerForecaster(nn.Module):
    """Encoder-only Transformer for time-series forecasting."""

    def __init__(self, input_size: int, d_model: int = 64,
                 n_heads: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 256, dropout: float = 0.1,
                 forecast_horizon: int = 1):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            out: (batch, forecast_horizon)
        """
        x = self.input_projection(x)              # (B, T, d_model)
        x = self.pos_encoder(x)                    # (B, T, d_model)
        x = self.transformer_encoder(x)            # (B, T, d_model)
        x = x.mean(dim=1)                          # Global avg pool → (B, d_model)
        return self.fc(x)                           # (B, forecast_horizon)
