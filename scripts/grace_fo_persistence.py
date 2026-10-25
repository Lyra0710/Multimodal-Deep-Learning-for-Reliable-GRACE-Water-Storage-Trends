"""Evaluate one-month-ahead GRACE-FO persistence forecasts.

For every target month t, the forecast is TWSA(t - 1). Metrics use only NASA
land-mask cells and valid, consecutive monthly observations. No model fitting,
normalization, or information from after the target month is used.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def load_config(config_path: Path) -> dict:
    """Load and minimally validate the shared chronological split definition."""
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "Loading the YAML split configuration requires PyYAML. Install it with: "
            "python -m pip install PyYAML"
        ) from error

    with config_path.open() as stream:
        config = yaml.safe_load(stream)

    for key in ("study_period", "forecast", "splits"):
        if key not in config:
            raise ValueError(f"Missing '{key}' in {config_path}")
    if config["forecast"].get("horizon_months") != 1:
        raise ValueError("The persistence baseline supports exactly a one-month horizon")

    split_bounds = []
    for name in ("train", "validation", "test"):
        if name not in config["splits"]:
            raise ValueError(f"Missing '{name}' split in {config_path}")
        split = config["splits"][name]
        start, end = pd.Timestamp(split["start"]), pd.Timestamp(split["end"])
        if start.day != 1 or end != end + pd.offsets.MonthEnd(0):
            raise ValueError(f"{name} split must start on a month boundary and end on month end")
        if start > end:
            raise ValueError(f"{name} split has start after end")
        split_bounds.append((name, start, end))

    for (_, _, previous_end), (name, current_start, _) in zip(split_bounds, split_bounds[1:]):
        if current_start.to_period("M") != previous_end.to_period("M") + 1:
            raise ValueError(f"{name} split is not immediately chronological after the previous split")
    study_start = pd.Timestamp(config["study_period"]["start"])
    study_end = pd.Timestamp(config["study_period"]["end"])
    if split_bounds[0][1] != study_start or split_bounds[-1][2] != study_end:
        raise ValueError("Chronological splits must cover the configured study period exactly")
    return config


def validate_processed_dataset(dataset: xr.Dataset, config: dict) -> None:
    """Validate the processed GRACE-FO master before generating forecasts."""
    target_variable = config["forecast"]["target_variable"]
    land_mask_variable = config["forecast"]["land_mask_variable"]
    if not {target_variable, land_mask_variable}.issubset(dataset.data_vars):
        raise ValueError("Processed dataset is missing the configured target or land-mask variable")
    if not {"time", "lat", "lon"}.issubset(dataset.coords):
        raise ValueError("Processed dataset must have time, lat, and lon coordinates")

    observed = pd.DatetimeIndex(dataset.time.values).to_period("M")
    expected = pd.period_range(
        config["study_period"]["start"][:7], config["study_period"]["end"][:7], freq="M"
    )
    if not observed.equals(expected):
        raise ValueError("Processed dataset does not contain the configured consecutive study months")
    if not dataset.lon.to_index().is_monotonic_increasing:
        raise ValueError("Processed longitude must be sorted")
    if float(dataset.lon.min()) < -180 or float(dataset.lon.max()) >= 180:
        raise ValueError("Processed longitude must use the [-180, 180) convention")


def land_modeling_view(dataset: xr.Dataset, target_variable: str, land_mask_variable: str) -> xr.DataArray:
    """Return a land-only TWSA view without altering the full master dataset."""
    if target_variable not in dataset or land_mask_variable not in dataset:
        raise ValueError(f"Dataset must contain {target_variable} and {land_mask_variable}")
    return dataset[target_variable].where(dataset[land_mask_variable] == 1)


def persistence_pairs(data: xr.DataArray, target_start: str, target_end: str) -> tuple[xr.DataArray, xr.DataArray]:
    """Align actual target months with prior-month observations, excluding time gaps."""
    periods = pd.DatetimeIndex(data.time.values).to_period("M")
    target_mask = (periods >= pd.Period(target_start[:7], freq="M")) & (
        periods <= pd.Period(target_end[:7], freq="M")
    )
    target_positions = np.flatnonzero(target_mask)
    target_positions = target_positions[target_positions > 0]
    consecutive = np.array(
        [periods[position] == periods[position - 1] + 1 for position in target_positions],
        dtype=bool,
    )
    target_positions = target_positions[consecutive]
    if not len(target_positions):
        raise ValueError(f"No consecutive target/previous-month pairs in {target_start} to {target_end}")

    actual = data.isel(time=target_positions)
    predicted = data.isel(time=target_positions - 1).assign_coords(time=actual.time)
    return actual, predicted


def latitude_weights(data: xr.DataArray) -> xr.DataArray:
    return xr.DataArray(
        np.cos(np.deg2rad(data.lat.values)), coords={"lat": data.lat}, dims=("lat",), name="area_weight"
    )


def temporal_correlation(actual: xr.DataArray, predicted: xr.DataArray, valid: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Return temporal Pearson r and its eligible-cell mask for each grid cell."""
    observed = actual.where(valid)
    forecast = predicted.where(valid)
    count = valid.sum(dim="time")
    observed_centered = observed - observed.mean(dim="time", skipna=True)
    forecast_centered = forecast - forecast.mean(dim="time", skipna=True)
    covariance = (observed_centered * forecast_centered).sum(dim="time", skipna=True)
    scale = np.sqrt(
        (observed_centered ** 2).sum(dim="time", skipna=True)
        * (forecast_centered ** 2).sum(dim="time", skipna=True)
    )
    correlation = covariance / scale
    eligible = (count >= 2) & np.isfinite(correlation)
    return correlation, eligible


def evaluate(actual: xr.DataArray, predicted: xr.DataArray) -> tuple[dict, xr.DataArray]:
    """Compute valid-land, latitude-area-weighted metrics in centimeters.

    MAE = sum(w * |actual - predicted|) / sum(w)
    RMSE = sqrt(sum(w * (actual - predicted)^2) / sum(w))
    where w = cos(latitude) for each valid land-cell/month observation.
    Pearson r is computed through time per grid cell, then averaged over
    eligible cells with the same latitude weights.
    """
    valid = np.isfinite(actual) & np.isfinite(predicted)
    weights = latitude_weights(actual).broadcast_like(actual)
    weighted_denominator = weights.where(valid).sum(skipna=True)
    if float(weighted_denominator) == 0:
        raise ValueError("No valid land-cell/month observations available for evaluation")

    error = actual - predicted
    mae = (weights * np.abs(error)).where(valid).sum(skipna=True) / weighted_denominator
    rmse = np.sqrt(((weights * error ** 2).where(valid).sum(skipna=True)) / weighted_denominator)
    correlation, eligible_correlation = temporal_correlation(actual, predicted, valid)
    cell_weights = latitude_weights(actual).broadcast_like(correlation)
    correlation_weight_sum = cell_weights.where(eligible_correlation).sum(skipna=True)
    weighted_correlation = (
        (cell_weights * correlation).where(eligible_correlation).sum(skipna=True) / correlation_weight_sum
        if float(correlation_weight_sum) > 0
        else xr.DataArray(np.nan)
    )

    valid_by_time = valid.any(dim=("lat", "lon"))
    valid_by_cell = valid.any(dim="time")
    metrics = {
        "mae_cm": float(mae),
        "rmse_cm": float(rmse),
        "area_weighted_mean_temporal_pearson_r": float(weighted_correlation),
        "evaluated_months": int(valid_by_time.sum()),
        "evaluated_land_cells": int(valid_by_cell.sum()),
        "correlation_eligible_land_cells": int(eligible_correlation.sum()),
        "valid_land_cell_month_observations": int(valid.sum()),
    }
    return metrics, valid


def plot_actual_vs_predicted(
    actual: xr.DataArray, predicted: xr.DataArray, valid: xr.DataArray, split_name: str, figure_path: Path
) -> None:
    weights = latitude_weights(actual).broadcast_like(actual)
    denominator = weights.where(valid).sum(dim=("lat", "lon"), skipna=True)
    actual_mean = (actual * weights).where(valid).sum(dim=("lat", "lon"), skipna=True) / denominator
    predicted_mean = (predicted * weights).where(valid).sum(dim=("lat", "lon"), skipna=True) / denominator

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(actual.time.values, actual_mean.values, marker="o", label="Actual TWSA")
    axis.plot(predicted.time.values, predicted_mean.values, marker="o", label="Persistence forecast")
    axis.set_title(f"GRACE-FO persistence baseline — {split_name.capitalize()} targets")
    axis.set_xlabel("Target month")
    axis.set_ylabel("Area-weighted land TWSA (cm)")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def split_manifest(time: xr.DataArray, config: dict) -> pd.DataFrame:
    """Create an explicit, seed-free target-month split manifest for handoff."""
    periods = pd.DatetimeIndex(time.values).to_period("M")
    rows = []
    for position, period in enumerate(periods):
        split_name = None
        for name, bounds in config["splits"].items():
            if pd.Period(bounds["start"][:7], freq="M") <= period <= pd.Period(bounds["end"][:7], freq="M"):
                split_name = name
                break
        if split_name is None:
            continue
        consecutive_input = position > 0 and periods[position - 1] == period - 1
        rows.append(
            {
                "target_month": period.to_timestamp().date().isoformat(),
                "split": split_name,
                "persistence_input_month": (period - 1).to_timestamp().date().isoformat(),
                "persistence_eligible": consecutive_input,
            }
        )
    return pd.DataFrame(rows)


def write_split_datasets(dataset: xr.Dataset, config: dict, output_dir: Path) -> dict[str, Path]:
    """Write direct, target-month GRACE-FO handoff datasets for each split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split_name, bounds in config["splits"].items():
        split_dataset = dataset.sel(time=slice(bounds["start"], bounds["end"])).copy()
        expected_months = len(pd.period_range(bounds["start"][:7], bounds["end"][:7], freq="M"))
        if split_dataset.sizes["time"] != expected_months:
            raise ValueError(f"{split_name} handoff dataset has an unexpected number of target months")
        split_dataset.attrs = dict(dataset.attrs)
        split_dataset.attrs.update(
            {
                "grace_fo_split": split_name,
                "target_month_start": bounds["start"],
                "target_month_end": bounds["end"],
                "split_note": "Chronological target-month handoff split; no random sampling.",
            }
        )
        output_path = output_dir / f"grace_fo_{split_name}.nc"
        encoding = {
            name: {"zlib": True, "complevel": 4}
            for name in split_dataset.data_vars
            if np.issubdtype(split_dataset[name].dtype, np.floating)
        }
        split_dataset.to_netcdf(output_path, engine="netcdf4", encoding=encoding)
        paths[split_name] = output_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Processed GRACE-FO NetCDF file")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/grace_fo_config.yaml"), help="Shared GRACE-FO configuration"
    )
    parser.add_argument("--metrics-dir", required=True, type=Path, help="Directory for GRACE-FO metric summaries")
    parser.add_argument("--figures-dir", required=True, type=Path, help="Directory for GRACE-FO figures")
    parser.add_argument(
        "--split-output-dir",
        required=True,
        type=Path,
        help="Directory for ready-to-load GRACE-FO train, validation, and test NetCDF files",
    )
    parser.add_argument(
        "--model-output",
        required=True,
        type=Path,
        help="JSON path for the GRACE-FO persistence model specification",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if not args.input.is_file():
        parser.error(f"Input not found: {args.input}")
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.split_output_dir.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)

    target_variable = config["forecast"]["target_variable"]
    land_mask_variable = config["forecast"]["land_mask_variable"]
    all_metrics = {
        "baseline": "one_month_persistence",
        "forecast_horizon_months": config["forecast"]["horizon_months"],
        "metric_units": {"mae_cm": "cm", "rmse_cm": "cm", "area_weighted_mean_temporal_pearson_r": "unitless"},
        "split_datasets": {},
        "splits": {},
    }

    with xr.open_dataset(args.input) as dataset:
        validate_processed_dataset(dataset, config)
        split_paths = write_split_datasets(dataset, config, args.split_output_dir)
        data = land_modeling_view(dataset, target_variable, land_mask_variable).load()

    all_metrics["split_datasets"] = {name: str(path) for name, path in split_paths.items()}

    split_manifest_path = args.metrics_dir / "grace_fo_chronological_splits.csv"
    split_manifest(data.time, config).to_csv(split_manifest_path, index=False)

    for split_name, split in config["splits"].items():
        actual, predicted = persistence_pairs(data, split["start"], split["end"])
        metrics, valid = evaluate(actual, predicted)
        metrics["target_start"] = split["start"]
        metrics["target_end"] = split["end"]
        all_metrics["splits"][split_name] = metrics
        plot_actual_vs_predicted(
            actual,
            predicted,
            valid,
            split_name,
            args.figures_dir / f"grace_fo_persistence_{split_name}_actual_vs_predicted.png",
        )

    metric_path = args.metrics_dir / "grace_fo_persistence_metrics.json"
    metric_path.write_text(json.dumps(all_metrics, indent=2) + "\n")
    model_specification = {
        "model_name": "grace_fo_one_month_persistence_baseline",
        "model_type": "deterministic_persistence",
        "equation": "predicted_twsa(t) = observed_twsa(t - 1)",
        "target_variable": target_variable,
        "modeling_mask": f"{land_mask_variable} == 1",
        "forecast_horizon_months": config["forecast"]["horizon_months"],
        "study_period": config["study_period"],
        "splits": config["splits"],
        "fitting": "None; persistence has no learned parameters.",
        "normalization": "None; metrics and forecasts remain in centimeters.",
        "processed_input": str(args.input),
        "split_datasets": {name: str(path) for name, path in split_paths.items()},
    }
    args.model_output.write_text(json.dumps(model_specification, indent=2) + "\n")
    print(f"Saved GRACE-FO persistence metrics: {metric_path}")
    print(f"Saved GRACE-FO persistence model specification: {args.model_output}")
    print(f"Saved GRACE-FO chronological split manifest: {split_manifest_path}")
    for split_name, split_path in split_paths.items():
        print(f"Saved GRACE-FO {split_name} handoff dataset: {split_path}")
    for split_name, metrics in all_metrics["splits"].items():
        print(
            f"{split_name}: MAE={metrics['mae_cm']:.3f} cm | "
            f"RMSE={metrics['rmse_cm']:.3f} cm | "
            f"r={metrics['area_weighted_mean_temporal_pearson_r']:.3f} | "
            f"months={metrics['evaluated_months']} | cells={metrics['evaluated_land_cells']}"
        )


if __name__ == "__main__":
    main()
