"""Small, reusable ConvLSTM utilities for the DATA 255 spatial forecasting experiments.

The notebook owns experiment choices and Drive I/O.  This module deliberately contains only
model, masking, temporal-sampling, and metric helpers that can be tested without Colab.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class ConvLSTMCell(nn.Module):
    """One convolutional LSTM step using the gates in Shi et al. (2015)."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so map dimensions are preserved")
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        # One convolution produces input, forget, output, and candidate gate pre-activations.
        self.gates = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding)

    def forward(self, inputs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None):
        batch, _, height, width = inputs.shape
        if state is None:
            hidden = inputs.new_zeros(batch, self.hidden_channels, height, width)
            cell = inputs.new_zeros(batch, self.hidden_channels, height, width)
        else:
            hidden, cell = state
        input_gate, forget_gate, output_gate, candidate = self.gates(torch.cat([inputs, hidden], dim=1)).chunk(4, dim=1)
        input_gate, forget_gate, output_gate = torch.sigmoid(input_gate), torch.sigmoid(forget_gate), torch.sigmoid(output_gate)
        cell = forget_gate * cell + input_gate * torch.tanh(candidate)
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell


class ConvLSTMForecaster(nn.Module):
    """Process [batch, time, channels, height, width] and predict one next-month TWSA map."""

    def __init__(self, input_channels: int, hidden_channels: int = 16, kernel_size: int = 3):
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)
        self.output = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 5:
            raise ValueError("Expected sequence shaped [batch, time, channels, height, width]")
        state = None
        for step in range(sequence.shape[1]):
            state = self.cell(sequence[:, step], state)
        return self.output(state[0])


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean squared error over valid target cells only; reject an empty batch explicitly."""
    valid = valid.to(dtype=prediction.dtype)
    denominator = valid.sum()
    if denominator.item() == 0:
        raise ValueError("Masked loss received no valid target cells")
    return (((prediction - target) ** 2) * valid).sum() / denominator


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid = valid.to(dtype=prediction.dtype)
    denominator = valid.sum()
    if denominator.item() == 0:
        raise ValueError("Masked metric received no valid target cells")
    return (torch.abs(prediction - target) * valid).sum() / denominator


def split_target_indices(times, start: str, end: str, history_months: int, features: np.ndarray, target: np.ndarray, spatial_mask: np.ndarray) -> np.ndarray:
    """Return targets in a chronological split with complete preceding input histories.

    Feature histories must be finite at every eligible cell. Targets may have individual missing
    cells; those are handled by the loss/evaluation validity mask instead of being fabricated.
    """
    times = np.asarray(times, dtype="datetime64[ns]")
    start_time, end_time = np.datetime64(start), np.datetime64(end)
    eligible = np.flatnonzero(spatial_mask.ravel())
    indices = []
    for target_index, date in enumerate(times):
        if date < start_time or date > end_time or target_index < history_months:
            continue
        history = features[target_index - history_months : target_index].reshape(history_months, features.shape[1], -1)
        if not np.isfinite(history[..., eligible]).all():
            continue
        if np.isfinite(target[target_index].ravel()[eligible]).any():
            indices.append(target_index)
    return np.asarray(indices, dtype=int)


def patch_origins(spatial_mask: np.ndarray, patch_size: int, stride: int | None = None, excluded: np.ndarray | None = None):
    """Tile maps, including right/bottom boundaries, with eligible training patches."""
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    stride = patch_size if stride is None else stride
    allowed = spatial_mask.copy()
    height, width = allowed.shape
    if patch_size > height or patch_size > width:
        raise ValueError("patch_size cannot exceed map dimensions")
    row_starts = list(range(0, height - patch_size + 1, stride))
    col_starts = list(range(0, width - patch_size + 1, stride))
    if row_starts[-1] != height - patch_size:
        row_starts.append(height - patch_size)
    if col_starts[-1] != width - patch_size:
        col_starts.append(width - patch_size)
    origins = []
    for top in row_starts:
        for left in col_starts:
            if excluded is not None and excluded[top : top + patch_size, left : left + patch_size].any():
                continue
            if allowed[top : top + patch_size, left : left + patch_size].any():
                origins.append((top, left))
    if not origins:
        raise ValueError("No patches contain eligible training cells")
    return origins


def patch_coverage(shape: tuple[int, int], origins: list[tuple[int, int]], patch_size: int) -> np.ndarray:
    """Return locations touched by at least one prediction patch."""
    coverage = np.zeros(shape, dtype=bool)
    for top, left in origins:
        coverage[top : top + patch_size, left : left + patch_size] = True
    return coverage


def consecutive_months(times) -> bool:
    """True only when timestamps are consecutive calendar-month starts."""
    values = np.asarray(times, dtype="datetime64[M]")
    return len(values) > 0 and np.array_equal(values, np.arange(values[0], values[-1] + 1, dtype="datetime64[M]"))


@dataclass(frozen=True)
class PatchSample:
    target_index: int
    top: int
    left: int


class PatchSequenceDataset(Dataset):
    """Lazily slices normalized map sequences; invalid values are filled only after masking."""

    def __init__(self, features: np.ndarray, target: np.ndarray, valid_target: np.ndarray, samples: list[PatchSample], history_months: int, patch_size: int):
        self.features = np.nan_to_num(features, nan=0.0).astype(np.float32, copy=False)
        self.target = np.nan_to_num(target, nan=0.0).astype(np.float32, copy=False)
        self.valid_target = valid_target.astype(bool, copy=False)
        self.samples, self.history_months, self.patch_size = samples, history_months, patch_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        row = slice(sample.top, sample.top + self.patch_size)
        col = slice(sample.left, sample.left + self.patch_size)
        sequence = self.features[sample.target_index - self.history_months : sample.target_index, :, row, col]
        target = self.target[sample.target_index, row, col][None]
        valid = self.valid_target[sample.target_index, row, col][None]
        return torch.from_numpy(sequence), torch.from_numpy(target), torch.from_numpy(valid), sample.target_index, sample.top, sample.left


def fit_normalization(features: np.ndarray, training_time_indices: np.ndarray, spatial_mask: np.ndarray):
    """Fit channel statistics from training months and training-eligible cells only."""
    selected = features[training_time_indices].reshape(len(training_time_indices), features.shape[1], -1)[..., spatial_mask.ravel()]
    mean = np.nanmean(selected, axis=(0, 2), keepdims=True).reshape(1, -1, 1, 1)
    std = np.nanstd(selected, axis=(0, 2), keepdims=True).reshape(1, -1, 1, 1)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std == 0):
        raise ValueError("Training-only normalization statistics are invalid")
    return mean.astype(np.float32), std.astype(np.float32)


def area_weighted_metrics(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray, latitudes: np.ndarray, min_months: int = 12):
    """Area-weighted mean of per-cell temporal MAE/RMSE/Pearson r, matching repository style."""
    valid = valid & np.isfinite(prediction) & np.isfinite(target)
    count = valid.sum(axis=0)
    enough = count >= min_months
    # Apply the validity mask before differences so excluded predictions never enter a cell metric.
    error = np.where(valid, prediction - target, 0.0)
    mae = np.divide(np.abs(error).sum(axis=0), count, out=np.full(count.shape, np.nan), where=count > 0)
    rmse = np.sqrt(np.divide((error ** 2).sum(axis=0), count, out=np.full(count.shape, np.nan), where=count > 0))
    observed = np.where(valid, target, 0.0)
    forecast = np.where(valid, prediction, 0.0)
    observed_mean = np.divide(observed.sum(axis=0), count, out=np.zeros(count.shape), where=count > 0)
    forecast_mean = np.divide(forecast.sum(axis=0), count, out=np.zeros(count.shape), where=count > 0)
    centered_observed = np.where(valid, observed - observed_mean, 0.0)
    centered_forecast = np.where(valid, forecast - forecast_mean, 0.0)
    correlation = np.divide(
        (centered_observed * centered_forecast).sum(axis=0),
        np.sqrt((centered_observed ** 2).sum(axis=0) * (centered_forecast ** 2).sum(axis=0)),
        out=np.full(count.shape, np.nan),
        where=count > 1,
    )
    weights = np.broadcast_to(np.cos(np.deg2rad(latitudes))[:, None], mae.shape)

    def weighted(values):
        usable = enough & np.isfinite(values)
        return float(np.sum(values[usable] * weights[usable]) / np.sum(weights[usable])) if usable.any() else float("nan")

    return {
        "MAE_cm": weighted(mae),
        "RMSE_cm": weighted(rmse),
        "Pearson_r_area_weighted": weighted(correlation),
        "evaluated_months": int(valid.any(axis=(1, 2)).sum()),
        "evaluated_cells": int(enough.sum()),
        "valid_cell_months": int(valid.sum()),
    }
