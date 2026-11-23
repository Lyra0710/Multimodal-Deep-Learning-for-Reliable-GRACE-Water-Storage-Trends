"""Temporal Convolutional Network (TCN) for one-month-ahead GRACE TWS forecasting.

One global TCN is shared by every land cell. Each sample is one (target month t, cell) pair,
and its input is a 12-step monthly window. Step s carries what is known when predicting
TWS(s): TWS(s - 1) plus an "observed" flag (GRACE gaps are zero-filled), and, in the
multimodal run, precipitation and temperature anomalies for month s (nowcast timing, as in
the P/T baselines) or s - 1 (forecast timing). So to predict TWS(t) the network sees
TWS(t-12..t-1) and P/T(t-11..t). The target is the raw TWS anomaly in cm.

Fusion strategies compared:
  * GRACE only: TWS history alone.
  * Early fusion: TWS, P, and T stacked as input channels of one TCN.
  * Late fusion: a GRACE-only branch and a P/T-only branch are trained separately and their
    forecasts are combined by area-weighted linear stacking fit on 2023 validation predictions.

For each run a hyperparameter grid is trained on the 2019-2022 targets with early stopping on
2023 validation RMSE. The best configuration is refit on 2019-2023 for its best epoch count
(the baselines are also refit on train + validation) and scored once on the test period.
Scoring reuses the baseline evaluator on the same test cells as the saved baseline
predictions, so the TCN rows can be compared directly with the baseline table.

Example:
    python scripts/tcn_forecast.py --input data/processed/p_t_tws_aligned_0.5deg.nc \
        --config configs/tcn_config.yaml \
        --baseline-predictions data/processed/p_t_tws_baseline_predictions.nc \
        --metrics-dir results/metrics --figures-dir results/figures \
        --predictions-output data/processed/tcn_predictions.nc --model-dir models
"""

import argparse
import itertools
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import xarray as xr
import yaml

import p_t_tws_data as alignment
from grace_fo_processing import compressed_float_encoding
from p_t_tws_analysis import save_figure, shift
from p_t_tws_baselines import _variable_name, evaluate_predictions, evaluation_mask


SOURCES = ("tws", "precip", "t2m")


def load_tcn_config(config_path: Path, project_root: Path | None = None) -> dict:
    """TCN settings plus the P/T configuration (data, splits, evaluation) they build on."""
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    config["base"] = alignment.load_config(config_path.parent / config["base_config"], project_root)
    if config["data"]["timing"] not in ("nowcast", "forecast"):
        raise ValueError("data.timing must be 'nowcast' or 'forecast'")
    for name, sources in config["runs"].items():
        if not sources or set(sources) - set(SOURCES):
            raise ValueError(f"Run '{name}' must use a non-empty subset of {SOURCES}")
    fusion = config.get("late_fusion")
    if fusion and (len(fusion["branches"]) != 2 or set(fusion["branches"]) - set(config["runs"])):
        raise ValueError("late_fusion.branches must name two entries of 'runs'")
    return config


# ----------------------------------------------------------------------------- data


def build_channels(
    cube: alignment.PixelCube, train_rows: np.ndarray, sources: list[str], timing: str, calendar: bool
) -> tuple[np.ndarray, list[str], dict]:
    """(time, pixel, channel) float32 inputs. Row s holds what is known when predicting TWS(s).

    All scaling statistics come from the training rows only.
    """
    _, anomalies = alignment.training_anomalies(cube, train_rows)
    tws_mean = float(np.nanmean(cube.tws[train_rows]))
    tws_std = float(np.nanstd(cube.tws[train_rows]))
    channels = {}
    if "tws" in sources:
        previous_tws = shift(cube.tws, 1)
        channels["tws_prev"] = np.nan_to_num((previous_tws - tws_mean) / tws_std, nan=0.0)
        channels["tws_prev_observed"] = np.isfinite(previous_tws).astype(np.float64)
    scaling = {"tws_mean_cm": tws_mean, "tws_std_cm": tws_std}  # target scaling, used by every run
    driver_lag = 0 if timing == "nowcast" else 1
    for name in ("precip", "t2m"):
        if name in sources:
            std = float(np.nanstd(anomalies[name][train_rows]))
            channels[f"{name}_anom"] = np.nan_to_num(shift(anomalies[name], driver_lag) / std, nan=0.0)
            scaling[f"{name}_anom_std"] = std
    if calendar:
        angle = 2 * np.pi * cube.month[:, None] / 12 * np.ones((1, cube.n_pixels))
        channels["month_sin"], channels["month_cos"] = np.sin(angle), np.cos(angle)
    names = list(channels)
    features = np.stack([channels[name].astype(np.float32) for name in names], -1)
    return features, names, scaling


def sample_index(rows: np.ndarray, target: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """(time, pixel) indices of observed targets in the selected rows with a full input window."""
    usable = rows.copy()
    usable[: lookback - 1] = False
    time_index, pixel_index = np.nonzero(usable[:, None] & np.isfinite(target))
    return time_index.astype(np.int64), pixel_index.astype(np.int64)


class WindowData:
    """Gathers (batch, lookback, channel) windows from the (time, pixel, channel) feature cube."""

    def __init__(self, features: np.ndarray, target_scaled: np.ndarray, weights: np.ndarray, lookback: int):
        self.features = torch.from_numpy(features)
        self.target = torch.from_numpy(np.nan_to_num(target_scaled).astype(np.float32))
        self.weights = torch.from_numpy(weights.astype(np.float32))
        self.offsets = torch.arange(-lookback + 1, 1)

    def windows(self, time_index: torch.Tensor, pixel_index: torch.Tensor) -> torch.Tensor:
        return self.features[time_index[:, None] + self.offsets, pixel_index[:, None]]

    def batch(self, time_index: torch.Tensor, pixel_index: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (
            self.windows(time_index, pixel_index),
            self.target[time_index, pixel_index],
            self.weights[pixel_index],
        )


# ----------------------------------------------------------------------------- model


class CausalConv1d(nn.Conv1d):
    """Conv1d padded on the left only, so step s never sees steps after s."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation)
        self.left_padding = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(nn.functional.pad(x, (self.left_padding, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation), nn.ReLU(), nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation), nn.ReLU(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.net(x) + self.skip(x))


class TCN(nn.Module):
    """Stacked dilated causal residual blocks; the last step's features predict TWS(t)."""

    def __init__(self, n_inputs: int, channels: int, kernel_size: int, dilations: list[int], dropout: float):
        super().__init__()
        blocks, width = [], n_inputs
        for dilation in dilations:
            blocks.append(TemporalBlock(width, channels, kernel_size, dilation, dropout))
            width = channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, lookback, channel)
        hidden = self.blocks(x.transpose(1, 2))
        return self.head(hidden[:, :, -1]).squeeze(-1)


def receptive_field(kernel_size: int, dilations: list[int]) -> int:
    return 1 + 2 * (kernel_size - 1) * sum(dilations)


# ----------------------------------------------------------------------------- training


def predict(model: TCN, data: WindowData, time_index: np.ndarray, pixel_index: np.ndarray, batch: int = 16384) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(time_index), batch):
            t = torch.from_numpy(time_index[start:start + batch])
            p = torch.from_numpy(pixel_index[start:start + batch])
            outputs.append(model(data.windows(t, p)).numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, np.float32)


def weighted_rmse_cm(model: TCN, data: WindowData, index: tuple[np.ndarray, np.ndarray], tws_std: float) -> float:
    """Pooled area-weighted RMSE in cm (the same formula as area_rmse in the baselines)."""
    time_index, pixel_index = index
    predicted = predict(model, data, time_index, pixel_index)
    error = (predicted - data.target[time_index, pixel_index].numpy()) * tws_std
    weights = data.weights[pixel_index].numpy()
    return float(np.sqrt(np.sum(weights * error ** 2) / np.sum(weights)))


def train_tcn(
    data: WindowData,
    train_index: tuple[np.ndarray, np.ndarray],
    params: dict,
    config: dict,
    tws_std: float,
    validation_index: tuple[np.ndarray, np.ndarray] | None = None,
    epochs: int | None = None,
) -> tuple[TCN, list[dict]]:
    """Train with early stopping on validation RMSE, or for a fixed number of epochs when refitting."""
    settings, model_settings = config["training"], config["model"]
    torch.manual_seed(settings["seed"])
    rng = np.random.default_rng(settings["seed"])
    model = TCN(data.features.shape[-1], params["channels"], model_settings["kernel_size"],
                model_settings["dilations"], params["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=model_settings["weight_decay"])
    max_epochs = epochs if epochs is not None else settings["max_epochs"]
    n_train = len(train_index[0])
    per_epoch = min(settings["samples_per_epoch"], n_train)
    batch_size = model_settings["batch_size"]

    history, best_rmse, best_state, stale = [], np.inf, None, 0
    for epoch in range(1, max_epochs + 1):
        started = time.time()
        model.train()
        chosen = rng.choice(n_train, per_epoch, replace=False)
        total_loss, total_weight = 0.0, 0.0
        for start in range(0, per_epoch, batch_size):
            rows = chosen[start:start + batch_size]
            x, y, w = data.batch(torch.from_numpy(train_index[0][rows]), torch.from_numpy(train_index[1][rows]))
            loss = torch.sum(w * (model(x) - y) ** 2) / torch.sum(w)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * float(w.sum())
            total_weight += float(w.sum())
        record = {"epoch": epoch, "train_rmse_cm": float(np.sqrt(total_loss / total_weight)) * tws_std,
                  "seconds": round(time.time() - started, 1)}
        if validation_index is not None:
            record["val_rmse_cm"] = weighted_rmse_cm(model, data, validation_index, tws_std)
            if record["val_rmse_cm"] < best_rmse - 1e-4:
                best_rmse, stale = record["val_rmse_cm"], 0
                best_state = {key: value.clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
        history.append(record)
        print("    " + "  ".join(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
                                 for key, value in record.items()), flush=True)
        if validation_index is not None and stale >= settings["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def grid(config: dict) -> list[dict]:
    keys = list(config["grid"])
    return [dict(zip(keys, values)) for values in itertools.product(*(config["grid"][key] for key in keys))]


def run_experiment(cube: alignment.PixelCube, sources: list[str], config: dict, label: str) -> dict:
    """Grid search on validation, refit the best configuration on train + validation, predict every month."""
    base = config["base"]
    rows = cube.split_rows(base["splits"])
    train_validation = rows["train"] | rows["validation"]
    lookback = config["data"]["lookback_months"]

    def prepare(fit_rows: np.ndarray) -> tuple[WindowData, list[str], dict]:
        features, names, scaling = build_channels(cube, fit_rows, sources, config["data"]["timing"],
                                                  config["data"]["calendar_features"])
        target = (cube.tws - scaling["tws_mean_cm"]) / scaling["tws_std_cm"]
        return WindowData(features, target, cube.area_weight, lookback), names, scaling

    data, names, scaling = prepare(rows["train"])
    train_index = sample_index(rows["train"], cube.tws, lookback)
    validation_index = sample_index(rows["validation"], cube.tws, lookback)
    print(f"{label}: channels={names}, train samples={len(train_index[0]):,}, "
          f"validation samples={len(validation_index[0]):,}", flush=True)

    # Validation-period predictions of the best train-only model, used to fit late fusion without test data.
    validation_prediction = np.full(cube.tws.shape, np.nan)
    results = []
    for number, params in enumerate(grid(config), 1):
        print(f"  [{number}/{len(grid(config))}] {params}", flush=True)
        model, history = train_tcn(data, train_index, params, config, scaling["tws_std_cm"], validation_index)
        best = min(history, key=lambda record: record["val_rmse_cm"])
        if all(best["val_rmse_cm"] < result["val_rmse_cm"] for result in results):
            validation_prediction[validation_index] = (
                predict(model, data, *validation_index) * scaling["tws_std_cm"] + scaling["tws_mean_cm"]
            )
        results.append({"run": label, **params, "best_epoch": best["epoch"], "val_rmse_cm": best["val_rmse_cm"],
                        "epochs_trained": len(history), "seconds": sum(record["seconds"] for record in history),
                        "history": history})
    best = min(results, key=lambda result: result["val_rmse_cm"])
    params = {key: best[key] for key in config["grid"]}
    print(f"  best {params} at epoch {best['best_epoch']} (val RMSE {best['val_rmse_cm']:.3f} cm); "
          f"refitting on train + validation", flush=True)

    # Refit with scaling from train + validation, as the baselines are refit on train + validation.
    data, names, scaling = prepare(train_validation)
    refit_index = sample_index(train_validation, cube.tws, lookback)
    model, refit_history = train_tcn(data, refit_index, params, config, scaling["tws_std_cm"], epochs=best["best_epoch"])

    prediction = np.full(cube.tws.shape, np.nan)
    first_target = rows["train"].argmax()
    after_start = np.arange(len(cube.times)) >= first_target
    time_index, pixel_index = np.nonzero(np.broadcast_to(after_start[:, None], cube.tws.shape))
    time_index, pixel_index = time_index.astype(np.int64), pixel_index.astype(np.int64)
    prediction[time_index, pixel_index] = (
        predict(model, data, time_index, pixel_index) * scaling["tws_std_cm"] + scaling["tws_mean_cm"]
    )
    return {"label": label, "sources": sources, "channels": names, "scaling": scaling, "params": params,
            "best_epoch": best["best_epoch"], "grid": results, "refit_history": refit_history,
            "model": model, "prediction": prediction, "validation_prediction": validation_prediction}


# ----------------------------------------------------------------------------- late fusion


def fit_stacking(
    branch_predictions: list[np.ndarray], observed: np.ndarray, rows: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Area-weighted least squares y ~ a + sum_k w_k * branch_k on the selected rows; returns [a, w_1, ...]."""
    stacked = np.stack([prediction[rows] for prediction in branch_predictions], -1)  # (months, pixels, branches)
    target = observed[rows]
    valid = np.isfinite(target) & np.isfinite(stacked).all(-1)
    design = np.column_stack([np.ones(valid.sum()), stacked[valid]])
    root_weight = np.sqrt(np.broadcast_to(weights, target.shape)[valid])
    coefficients, *_ = np.linalg.lstsq(design * root_weight[:, None], target[valid] * root_weight, rcond=None)
    return coefficients


def apply_stacking(coefficients: np.ndarray, branch_predictions: list[np.ndarray]) -> np.ndarray:
    return coefficients[0] + sum(weight * prediction for weight, prediction in zip(coefficients[1:], branch_predictions))


def late_fusion(cube: alignment.PixelCube, experiments: list[dict], config: dict) -> dict:
    """Stack separately trained branches: weights fit on 2023 from the train-only branch models, then applied
    to the branches refit on train + validation (mirroring how every model is refit before testing)."""
    fusion = config["late_fusion"]
    branches = [next(e for e in experiments if e["label"] == name) for name in fusion["branches"]]
    rows = cube.split_rows(config["base"]["splits"])
    coefficients = fit_stacking([branch["validation_prediction"] for branch in branches], cube.tws,
                                rows["validation"], cube.area_weight)
    validation_fused = apply_stacking(coefficients, [branch["validation_prediction"] for branch in branches])
    error = (validation_fused - cube.tws)[rows["validation"]]
    weights = np.broadcast_to(cube.area_weight, error.shape)
    valid = np.isfinite(error)
    in_sample_rmse = float(np.sqrt(np.sum(weights[valid] * error[valid] ** 2) / np.sum(weights[valid])))
    summary = {"intercept_cm": float(coefficients[0]),
               **{f"weight[{branch['label']}]": float(w) for branch, w in zip(branches, coefficients[1:])},
               "val_rmse_cm_in_sample": in_sample_rmse}
    print(f"Late fusion {fusion['name']}: " + ", ".join(f"{k}={v:.3f}" for k, v in summary.items()), flush=True)
    return {"label": fusion["name"], "branches": fusion["branches"], "coefficients": summary,
            "prediction": apply_stacking(coefficients, [branch["prediction"] for branch in branches])}


# ----------------------------------------------------------------------------- evaluation and outputs


def load_baseline_predictions(
    path: Path, cube: alignment.PixelCube, model_names: list[str] | None = None
) -> tuple[dict, np.ndarray]:
    """Baseline predictions and the shared test mask from p_t_tws_baselines.py, as (time, pixel) arrays.

    The file stores variable-safe names (pred_Seasonal_persistence); model_names maps them back.
    """
    readable = {f"pred_{_variable_name(name)}": name for name in model_names or []}
    with xr.open_dataset(path) as saved:
        if not np.array_equal(saved.time.values, cube.times.values):
            raise ValueError(f"{path} has a different time axis from the aligned input")

        def flatten(name: str) -> np.ndarray:
            return saved[name].values.reshape(saved.sizes["time"], -1)[:, cube.pixels].astype(np.float64)

        predictions = {
            readable.get(name, name.removeprefix("pred_").replace("_", " ")): flatten(name)
            for name in saved.data_vars if name.startswith("pred_") and saved[name].dims == ("time", "lat", "lon")
        }
        test_mask = flatten("test_eval_mask") > 0
    return predictions, test_mask


def score_models(cube: alignment.PixelCube, predictions: dict, config: dict, shared_mask: np.ndarray | None) -> tuple[pd.DataFrame, np.ndarray]:
    base = config["base"]
    rows = cube.split_rows(base["splits"])
    climatology = alignment.monthly_climatology(cube.tws, rows["train"] | rows["validation"], cube.month)
    reference = climatology[cube.month - 1]
    mask = evaluation_mask(cube.tws, rows["test"], predictions)
    if shared_mask is not None:
        mask &= shared_mask
    print(f"Test cells: {int(mask.sum()):,} (months x pixels) in {int(mask.any(1).sum())} months", flush=True)
    metrics = {
        name: evaluate_predictions(cube.tws, prediction, mask, cube.area_weight, cube.month, climatology,
                                   reference=reference, min_months=base["baselines"]["min_evaluation_months"])[0]
        for name, prediction in predictions.items()
    }
    return pd.DataFrame(metrics).T.sort_values("MSESS_vs_clim", ascending=False).round(3), mask


def plot_comparison(metrics: pd.DataFrame, tcn_names: list[str]) -> plt.Figure:
    ordered = metrics.sort_values("MSESS_vs_clim")
    colors = ["tab:red" if name in tcn_names else "tab:grey" for name in ordered.index]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.45 * len(ordered) + 1.5), sharey=True)
    for ax, (column, title, fmt) in zip(axes, [("MSESS_vs_clim", "Skill vs climatology (MSESS, higher is better)", "%.2f"),
                                               ("RMSE_cm", "RMSE in cm (lower is better)", "%.2f")]):
        bars = ax.barh(ordered.index, ordered[column], color=colors)
        ax.bar_label(bars, labels=[fmt % value for value in ordered[column]], padding=3, fontsize=8)
        ax.set_title(title)
        ax.set_xlim(min(0, ordered[column].min()), ordered[column].max() * 1.2)
    fig.suptitle("TCN runs vs baselines on the shared test cells (red = TCN)")
    fig.tight_layout()
    return fig


def plot_learning_curves(experiments: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(1, len(experiments), figsize=(6 * len(experiments), 4), squeeze=False)
    for ax, experiment in zip(axes[0], experiments):
        best = min(experiment["grid"], key=lambda result: result["val_rmse_cm"])
        epochs = [record["epoch"] for record in best["history"]]
        ax.plot(epochs, [record["train_rmse_cm"] for record in best["history"]], label="train (2019-2022)")
        ax.plot(epochs, [record["val_rmse_cm"] for record in best["history"]], label="validation (2023)")
        ax.axvline(best["best_epoch"], color="grey", ls="--", lw=0.8, label=f"best epoch {best['best_epoch']}")
        ax.set(title=f"{experiment['label']}\n{experiment['params']}", xlabel="epoch", ylabel="area-weighted RMSE (cm)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def prediction_dataset(cube: alignment.PixelCube, predictions: dict, test_mask: np.ndarray, timing: str) -> xr.Dataset:
    variables = {}
    for label, prediction in predictions.items():
        variables[f"pred_{_variable_name(label)}"] = cube.to_cube(prediction).assign_attrs(model=label, units="cm")
    variables["tws_obs"] = cube.to_cube(cube.tws).assign_attrs(units="cm")
    variables["test_eval_mask"] = cube.to_cube(test_mask.astype(np.float32))
    return xr.Dataset(variables, attrs={"timing": timing, "description": "TCN one-month-ahead TWS predictions"})


def specification(experiment: dict, config: dict, input_path: Path) -> dict:
    model_settings = config["model"]
    return {
        "model": experiment["label"],
        "input": str(input_path),
        "sources": experiment["sources"],
        "input_channels": experiment["channels"],
        "timing": config["data"]["timing"],
        "lookback_months": config["data"]["lookback_months"],
        "kernel_size": model_settings["kernel_size"],
        "dilations": model_settings["dilations"],
        "receptive_field": receptive_field(model_settings["kernel_size"], model_settings["dilations"]),
        "weight_decay": model_settings["weight_decay"],
        "batch_size": model_settings["batch_size"],
        "selected_hyperparameters": experiment["params"],
        "refit_epochs": experiment["best_epoch"],
        "scaling_train_validation": experiment["scaling"],
        "splits": config["base"]["splits"],
        "parameters": sum(parameter.numel() for parameter in experiment["model"].parameters()),
    }


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Aligned TWS/P/T NetCDF from p_t_tws_data.py")
    parser.add_argument("--config", type=Path, default=Path("configs/tcn_config.yaml"), help="TCN configuration")
    parser.add_argument("--baseline-predictions", type=Path,
                        help="Baseline predictions NetCDF; its models are rescored on the same test cells")
    parser.add_argument("--metrics-dir", required=True, type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    parser.add_argument("--predictions-output", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input not found: {args.input}")
    config = load_tcn_config(args.config)
    torch.set_num_threads(config["training"]["threads"])
    for directory in (args.metrics_dir, args.figures_dir, args.predictions_output.parent, args.model_dir):
        directory.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(args.input) as aligned:
        cell_mask = config["data"]["cell_mask"]
        if cell_mask not in aligned:
            parser.error(f"{args.input} has no '{cell_mask}'; rebuild it with the current p_t_tws_data.py")
        cube = alignment.PixelCube.from_dataset(aligned.assign(mask=aligned[cell_mask]))
    print(f"Cells selected with '{cell_mask}': {cube.n_pixels:,}", flush=True)
    lookback, model_settings = config["data"]["lookback_months"], config["model"]
    if receptive_field(model_settings["kernel_size"], model_settings["dilations"]) < lookback:
        raise ValueError("The TCN receptive field is shorter than the lookback window")

    experiments, grid_rows = [], []
    for label, sources in config["runs"].items():
        experiment = run_experiment(cube, sources, config, label)
        experiments.append(experiment)
        grid_rows += [{key: value for key, value in result.items() if key != "history"} for result in experiment["grid"]]
        pd.DataFrame(grid_rows).to_csv(args.metrics_dir / "tcn_grid_search.csv", index=False)
        stem = "tcn_" + "_".join(sources)
        torch.save(experiment["model"].state_dict(), args.model_dir / f"{stem}.pt")
        (args.model_dir / f"{stem}.json").write_text(
            json.dumps(specification(experiment, config, args.input), indent=2, default=str) + "\n")

    predictions = {experiment["label"]: experiment["prediction"] for experiment in experiments}
    fusion = late_fusion(cube, experiments, config) if config.get("late_fusion") else None
    if fusion:
        predictions[fusion["label"]] = fusion["prediction"]
        (args.model_dir / "tcn_late_fusion.json").write_text(json.dumps(
            {"model": fusion["label"], "branches": fusion["branches"], "combiner": "area-weighted linear stacking",
             "fit_on": "validation predictions of the train-only branch models", **fusion["coefficients"]},
            indent=2) + "\n")
    tcn_predictions = dict(predictions)
    shared_mask = None
    if args.baseline_predictions:
        names_path = args.metrics_dir / "p_t_tws_baseline_metrics_gracefo.csv"
        names = list(pd.read_csv(names_path, index_col=0).index) if names_path.is_file() else None
        baselines, shared_mask = load_baseline_predictions(args.baseline_predictions, cube, names)
        predictions.update(baselines)
    metrics, test_mask = score_models(cube, predictions, config, shared_mask)
    metrics_path = args.metrics_dir / "tcn_test_metrics.csv"
    metrics.to_csv(metrics_path)

    dataset = prediction_dataset(cube, tcn_predictions, test_mask, config["data"]["timing"])
    dataset.to_netcdf(args.predictions_output, engine="netcdf4", encoding=compressed_float_encoding(dataset))
    tcn_names = list(tcn_predictions)
    for name, fig in {"01_comparison": plot_comparison(metrics, tcn_names),
                      "02_learning_curves": plot_learning_curves(experiments)}.items():
        save_figure(fig, args.figures_dir / f"tcn_{name}.png")
        plt.close(fig)

    print(metrics.to_string())
    print(f"Saved grid search: {args.metrics_dir / 'tcn_grid_search.csv'}")
    print(f"Saved test metrics: {metrics_path}")
    print(f"Saved predictions: {args.predictions_output}")
    print(f"Saved models and specifications to: {args.model_dir}")
    print(f"Saved figures to: {args.figures_dir}")


if __name__ == "__main__":
    main()
