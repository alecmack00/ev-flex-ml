"""
Quantile Temporal Convolutional Network (Quantile TCN) for probabilistic time-series forecasting of EV fleet charging load.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def pinball_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    quantiles: List[float] = None,
    monotonicity_penalty: float = 0.05,
) -> torch.Tensor:
    """Computes multi-quantile Pinball Loss with composite non-crossing monotonicity regularization.

    Args:
        y_pred: Predicted quantiles tensor [batch_size, num_quantiles, horizon].
        y_true: Ground truth target tensor [batch_size, 1, horizon] or [batch_size, horizon].
        quantiles: List of target quantiles (e.g. [0.1, 0.5, 0.9]).
        monotonicity_penalty: Regularization weight on quantile crossing violations (q_k > q_{k+1}).

    Returns:
        torch.Tensor: Mean composite pinball loss scalar across batch and quantiles.
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    if y_true.dim() == 2:
        y_true = y_true.unsqueeze(1)  # [batch_size, 1, horizon]

    total_loss = 0.0
    for q_idx, q in enumerate(quantiles):
        pred_q = y_pred[:, q_idx:q_idx+1, :]
        error = y_true - pred_q
        loss_q = torch.maximum(q * error, (q - 1.0) * error)
        total_loss = total_loss + torch.mean(loss_q)

    base_pinball = total_loss / len(quantiles)

    # Monotonicity penalty across adjacent quantile pairs
    if monotonicity_penalty > 0.0 and len(quantiles) > 1:
        crossing_violations = 0.0
        for k in range(len(quantiles) - 1):
            # Penalize instances where lower quantile exceeds higher quantile
            crossing = F.relu(y_pred[:, k, :] - y_pred[:, k + 1, :])
            crossing_violations = crossing_violations + torch.mean(crossing)
        return base_pinball + monotonicity_penalty * crossing_violations

    return base_pinball


class TemporalBlock(nn.Module):
    """1D Dilated Causal Convolutional Block with Residual Connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.15,
    ) -> None:
        """Initializes Temporal Block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: 1D convolution kernel size.
            stride: Stride length.
            dilation: Dilation rate.
            padding: Padding size (causal trimming applied in forward).
            dropout: Dropout rate.
        """
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.padding = padding

        # 1x1 Convolution for residual matching if channel dimensions differ
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal trimming to maintain strict temporal causality.

        Args:
            x: Input tensor [batch_size, in_channels, sequence_length].

        Returns:
            torch.Tensor: Residual output tensor [batch_size, out_channels, sequence_length].
        """
        # First dilated conv + causal trim
        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second dilated conv + causal trim
        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)

        # Ensure temporal dimension alignment
        if out.shape[2] != res.shape[2]:
            min_len = min(out.shape[2], res.shape[2])
            out = out[:, :, :min_len]
            res = res[:, :, :min_len]

        return F.relu(out + res)


class QuantileTCN(nn.Module):
    """Temporal Convolutional Network producing multi-quantile probabilistic load forecasts."""

    def __init__(
        self,
        input_dim: int = 8,
        num_channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.15,
        quantiles: List[float] = None,
    ) -> None:
        """Initializes TCN architecture.

        Args:
            input_dim: Feature dimension per time step.
            num_channels: List of channel dimensions for stacked temporal blocks.
            kernel_size: Convolution kernel size.
            dropout: Dropout rate.
            quantiles: Target quantiles list (e.g. [0.1, 0.5, 0.9]).
        """
        super().__init__()
        if num_channels is None:
            num_channels = [32, 64, 64, 32]
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.quantile_head = nn.Conv1d(num_channels[-1], self.num_quantiles, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with guaranteed non-crossing monotonic quantile head activations.

        Args:
            x: Input tensor of shape [batch_size, input_dim, sequence_length].

        Returns:
            torch.Tensor: Quantile forecasts of shape [batch_size, num_quantiles, sequence_length].
        """
        tcn_out = self.tcn(x)
        raw_head = self.quantile_head(tcn_out)  # [batch_size, num_quantiles, sequence_length]
        
        if self.num_quantiles > 1:
            q_base = raw_head[:, 0:1, :]
            q_deltas = F.softplus(raw_head[:, 1:, :])
            q_ordered = torch.cat([q_base, q_base + torch.cumsum(q_deltas, dim=1)], dim=1)
            return q_ordered
            
        return raw_head
