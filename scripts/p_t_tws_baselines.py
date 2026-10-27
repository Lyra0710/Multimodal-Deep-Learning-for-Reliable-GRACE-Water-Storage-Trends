"""Evaluate precipitation/temperature-driven TWS baselines on the shared GRACE-FO splits.

GRACE-FO temporal split: ridge strengths (and gradient-boosting iterations) are chosen
on the validation period by area-weighted RMSE, every model is refit on train +
validation, and all models are scored once on identical test cells, i.e. months where
TWS(t) and TWS(t - 1) are both observed and every model has a prediction. That keeps
one-step models (persistence, ARX) comparable with climate-driven ones.

Geographic split: pooled ridge and gradient boosting under spatial block
cross-validation (10 x 10 degree blocks); held-out pixels contribute no TWS to training.

Metrics are area-weighted (cos latitude) means over pixels with at least
min_evaluation_months scored months: MAE, RMSE, Pearson r, anomaly r after removing the
train + validation climatology, median NSE, and a pooled MSE skill score against
climatology (temporal) or a zero predictor (geographic).

Examples:
    python p_t_tws_baselines.py --input data/processed/p_t_tws_aligned_0.5deg.nc \
        --config configs/p_t_tws_config.yaml --metrics-dir results/metrics \
        --figures-dir results/figures \
        --predictions-output data/processed/p_t_tws_baseline_predictions.nc \
        --model-output models/p_t_tws_baselines.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import p_t_tws_data as alignment
from p_t_tws_analysis import map_axes, nan_corr, plot_map, rolling_sum, save_figure, shift
from grace_fo_processing import compressed_float_encoding


USES_PAST_TWS = ("Persistence", "Seasonal persistence", "ARX (P,T,trend,TWS t-1)")
SPATIAL_MODELS = ("Zero (time-mean)", "Pooled ridge (P,T)", "Pooled GBM (P,T)")


def build_features(cube: alignment.PixelCube, anomalies: dict, settings: dict) -> tuple[dict, dict, list[str]]:
    """Harmonic, trend, and driver features as (time, pixel) arrays plus the per-model feature sets.

    Drivers use the full P/T record, so lags and cumulative sums exist from the first GRACE month.
    """
    offset = 0 if settings["setup"] == "nowcast" else 1
    month = cube.month
    harmonics = np.stack(
        [np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
         np.sin(4 * np.pi * month / 12), np.cos(4 * np.pi * month / 12)], 1
    )
    shape = (len(cube.times), cube.n_pixels)
    features = {f"h{index}": np.broadcast_to(harmonics[:, index, None], shape) for index in range(4)}
    features["trend"] = np.broadcast_to(cube.years[:, None], shape)
    for lag in settings["precip_lags"]:
        features[f"p_lag{lag}"] = shift(anomalies["precip"], lag + offset)
    for window in settings["cumulative_windows"] + [settings["gbm_cumulative_window"]]:
        features[f"cumP{window}"] = shift(rolling_sum(anomalies["precip"], window), offset)
    for lag in settings["temperature_lags"]:
        features[f"t_lag{lag}"] = shift(anomalies["t2m"], lag + offset)
    features["tws_lag1"] = shift(cube.tws, 1)  # one-step models only

    harmonic_names = [f"h{index}" for index in range(4)]
    drivers = (
        [f"p_lag{lag}" for lag in settings["precip_lags"]]
        + [f"cumP{window}" for window in settings["cumulative_windows"]]
        + [f"t_lag{lag}" for lag in settings["temperature_lags"]]
    )
    feature_sets = {
        "Seasonal + trend": harmonic_names + ["trend"],
        "LR (P,T)": harmonic_names + drivers,
        "LR (P,T) + trend": harmonic_names + ["trend"] + drivers,
        "ARX (P,T,trend,TWS t-1)": harmonic_names + ["trend"] + drivers + ["tws_lag1"],
    }
    return features, feature_sets, drivers


def stack_features(features: dict, names: list[str]) -> np.ndarray:
    """(pixel, time, feature) float32 design array."""
    return np.stack([features[name].T for name in names], -1).astype(np.float32)


def ridge_fit_predict(
    features: np.ndarray, target: np.ndarray, fit_rows: np.ndarray, alpha: float, chunk: int = 1500
) -> np.ndarray:
    """Independent ridge regression per pixel on standardized features.

    features (pixel, time, feature), target (time, pixel), fit_rows (time,) bool. Returns (time, pixel)
    predictions; pixels with fewer than n_features + 6 fitting months get NaN.
    """
    n_pixels, n_times, n_features = features.shape
    target = target.T
    prediction = np.full((n_pixels, n_times), np.nan)
    features_valid = np.isfinite(features).all(-1)
    for start in range(0, n_pixels, chunk):
        block = features[start:start + chunk].astype(np.float64)
        valid = features_valid[start:start + chunk]
        weight = (fit_rows[None] & valid & np.isfinite(target[start:start + chunk])).astype(np.float64)
        block = np.where(valid[..., None], block, 0.0)
        observed = np.where(weight > 0, target[start:start + chunk], 0.0)
        count = np.maximum(weight.sum(1), 1.0)
        mean = (weight[..., None] * block).sum(1) / count[:, None]
        scale = np.sqrt((weight[..., None] * (block - mean[:, None]) ** 2).sum(1) / count[:, None])
        scale[scale < 1e-8] = 1.0
        standardized = (block - mean[:, None]) / scale[:, None]
        target_mean = (weight * observed).sum(1) / count
        gram = np.einsum("ntk,ntj->nkj", standardized * weight[..., None], standardized) + alpha * np.eye(n_features)
        moment = np.einsum("ntk,nt->nk", standardized * weight[..., None], (observed - target_mean[:, None]) * weight)
        coefficients = np.linalg.solve(gram, moment[..., None])[..., 0]
        block_prediction = np.einsum("ntk,nk->nt", standardized, coefficients) + target_mean[:, None]
        block_prediction[~valid] = np.nan
        block_prediction[count < n_features + 6] = np.nan
        prediction[start:start + chunk] = block_prediction
    return prediction.T


def area_rmse(observed: np.ndarray, predicted: np.ndarray, rows: np.ndarray, weights: np.ndarray) -> float:
    actual, forecast = observed[rows], predicted[rows]
    valid = np.isfinite(actual) & np.isfinite(forecast)
    squared_error = np.where(valid, (actual - forecast) ** 2, 0) * weights
    return float(np.sqrt(squared_error.sum() / (valid * weights).sum()))


def static_descriptors(cube: alignment.PixelCube, train_rows: np.ndarray, precip_climatology: np.ndarray) -> dict:
    """Per-pixel climate descriptors that let pooled models modulate the driver response."""
    return {
        "lat": cube.pixel_lat,
        "mean_p": np.nanmean(cube.precip[train_rows], 0),
        "mean_t": np.nanmean(cube.t2m[train_rows], 0),
        "p_season_amp": np.nanmax(precip_climatology, 0) - np.nanmin(precip_climatology, 0),
    }


def pooled_matrix(
    features: dict, static: dict, columns: list[str], time_index: np.ndarray, pixel_index: np.ndarray
) -> np.ndarray:
    """Time-major (time x pixel, feature) matrix of dynamic columns followed by static descriptors."""
    stacked = [features[name][np.ix_(time_index, pixel_index)].ravel() for name in columns]
    stacked += [np.tile(values[pixel_index], len(time_index)) for values in static.values()]
    return np.column_stack(stacked).astype(np.float32)


def _subsample(index: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(index, max_rows, replace=False) if len(index) > max_rows else index


def gbm_residual_model(
    cube: alignment.PixelCube,
    features: dict,
    static: dict,
    columns: list[str],
    fit_rows: np.ndarray,
    base_prediction: np.ndarray,
    settings: dict,
    rng: np.random.Generator,
    n_iterations: int | None = None,
    eval_rows: np.ndarray | None = None,
) -> tuple[np.ndarray, int | None, HistGradientBoostingRegressor]:
    """Pooled gradient boosting on the residual of a per-pixel base prediction.

    With eval_rows, the boosting iteration with the lowest residual MSE on those rows is returned.
    """
    pixels = np.arange(cube.n_pixels)
    residual = cube.tws - base_prediction
    fit_time = np.flatnonzero(fit_rows)
    fit_matrix = pooled_matrix(features, static, columns, fit_time, pixels)
    fit_target = residual[fit_time].ravel()
    usable = _subsample(
        np.flatnonzero(np.isfinite(fit_target) & np.isfinite(fit_matrix).all(1)), settings["gbm_max_rows"], rng
    )
    hyperparameters = dict(settings["gbm"])
    hyperparameters["max_iter"] = n_iterations or hyperparameters["max_iter"]
    model = HistGradientBoostingRegressor(**hyperparameters, early_stopping=False, random_state=settings["seed"])
    model.fit(fit_matrix[usable], fit_target[usable])

    best_iterations = n_iterations
    if eval_rows is not None:
        eval_time = np.flatnonzero(eval_rows)
        eval_matrix = pooled_matrix(features, static, columns, eval_time, pixels)
        eval_target = residual[eval_time].ravel()
        valid = np.isfinite(eval_target) & np.isfinite(eval_matrix).all(1)
        errors = [np.mean((eval_target[valid] - staged) ** 2) for staged in model.staged_predict(eval_matrix[valid])]
        best_iterations = int(np.argmin(errors)) + 1

    all_matrix = pooled_matrix(features, static, columns, np.arange(len(cube.times)), pixels)
    correction = model.predict(np.nan_to_num(all_matrix)).reshape(len(cube.times), cube.n_pixels)
    correction[~np.isfinite(all_matrix).all(1).reshape(len(cube.times), cube.n_pixels)] = np.nan
    return base_prediction + correction, best_iterations, model


def evaluation_mask(tws: np.ndarray, rows: np.ndarray, predictions: dict) -> np.ndarray:
    """Same cells for every model: TWS(t) and TWS(t - 1) observed, and every model has a prediction."""
    mask = rows[:, None] & np.isfinite(tws) & np.isfinite(shift(tws, 1))
    for prediction in predictions.values():
        mask &= np.isfinite(prediction)
    return mask


def evaluate_predictions(
    observed: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    month: np.ndarray,
    climatology: np.ndarray,
    reference: np.ndarray | None = None,
    min_months: int = 12,
) -> tuple[dict, dict]:
    """Area-weighted skill over cells with at least min_months scored months, plus per-pixel skill maps.

    climatology (12, pixel) is removed before the anomaly correlation, so anom_r measures non-seasonal skill.
    MSESS = 1 - sum(w * SSE_model) / sum(w * SSE_reference) when a reference prediction is given.
    """
    actual = np.where(mask, observed, np.nan)
    forecast = np.where(mask, predicted, np.nan)
    enough = mask.sum(0) >= min_months
    error = forecast - actual
    mae = np.nanmean(np.abs(error), 0)
    rmse = np.sqrt(np.nanmean(error ** 2, 0))
    nse = 1 - np.nansum(error ** 2, 0) / np.nansum((actual - np.nanmean(actual, 0)) ** 2, 0)
    correlation = nan_corr(actual, forecast)[0]
    seasonal = climatology[month - 1]
    anomaly_correlation = nan_corr(actual - seasonal, forecast - seasonal)[0]

    def weighted_mean(values: np.ndarray) -> float:
        valid = enough & np.isfinite(values)
        return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid])) if valid.any() else np.nan

    metrics = {
        "MAE_cm": weighted_mean(mae),
        "RMSE_cm": weighted_mean(rmse),
        "r_area_wtd": weighted_mean(correlation),
        "anom_r_area_wtd": weighted_mean(anomaly_correlation),
        "NSE_median": float(np.nanmedian(nse[enough])),
        "pct_NSE_pos": float(100 * np.nanmean(nse[enough] > 0)),
        "n_cells": int(enough.sum()),
    }
    if reference is not None:
        cell_weights = weights * enough
        reference_error = np.nansum((np.where(mask, reference, np.nan) - actual) ** 2, 0)
        metrics["MSESS_vs_clim"] = float(
            1 - np.nansum(np.nansum(error ** 2, 0) * cell_weights) / np.nansum(reference_error * cell_weights)
        )
    maps = {"mae": mae, "rmse": rmse, "nse": nse, "r": correlation, "r_anom": anomaly_correlation}
    return metrics, maps


def temporal_split_baselines(
    cube: alignment.PixelCube,
    features: dict,
    feature_sets: dict,
    drivers: list[str],
    climatologies: dict,
    config: dict,
    rng: np.random.Generator,
) -> dict:
    """Fit every baseline on the GRACE-FO temporal split and score it on the shared test cells."""
    settings = config["baselines"]
    rows = cube.split_rows(config["splits"])
    train_validation = rows["train"] | rows["validation"]
    weights = cube.area_weight
    tws_climatology = alignment.monthly_climatology(cube.tws, train_validation, cube.month)

    predictions = {"Climatology": tws_climatology[cube.month - 1]}
    alphas, validation_rmse = {}, {}
    for name, columns in feature_sets.items():
        design = stack_features(features, columns)
        candidates = [settings["seasonal_trend_alpha"]] if name == "Seasonal + trend" else settings["ridge_alphas"]
        scores = {
            alpha: area_rmse(cube.tws, ridge_fit_predict(design, cube.tws, rows["train"], alpha), rows["validation"], weights)
            for alpha in candidates
        }
        alphas[name] = min(scores, key=scores.get)
        validation_rmse[name] = scores[alphas[name]]
        predictions[name] = ridge_fit_predict(design, cube.tws, train_validation, alphas[name])
        print(f"{name:28s} alpha={alphas[name]:<6} val RMSE={validation_rmse[name]:.2f} cm")
    predictions["Persistence"] = shift(cube.tws, 1)
    predictions["Seasonal persistence"] = (
        shift(cube.tws, 1) + tws_climatology[cube.month - 1] - tws_climatology[(cube.month - 2) % 12]
    )

    # Pooled GBM on the seasonal-trend residual; iterations are chosen on validation, then refit on train + validation.
    static = static_descriptors(cube, rows["train"], climatologies["precip"])
    gbm_columns = drivers + [f"cumP{settings['gbm_cumulative_window']}", "h0", "h1"]
    seasonal_trend_train = ridge_fit_predict(
        stack_features(features, feature_sets["Seasonal + trend"]), cube.tws, rows["train"], alphas["Seasonal + trend"]
    )
    _, gbm_iterations, _ = gbm_residual_model(
        cube, features, static, gbm_columns, rows["train"], seasonal_trend_train, settings, rng, eval_rows=rows["validation"]
    )
    predictions["Seasonal-trend + GBM (P,T)"], _, _ = gbm_residual_model(
        cube, features, static, gbm_columns, train_validation, predictions["Seasonal + trend"], settings, rng,
        n_iterations=gbm_iterations,
    )
    print(f"GBM iterations chosen on validation: {gbm_iterations}")

    test_mask = evaluation_mask(cube.tws, rows["test"], predictions)
    print(f"Test cells: {int(test_mask.sum())} (months × pixels), {int(test_mask.any(1).sum())} months")
    metrics, maps = {}, {}
    for name, prediction in predictions.items():
        metrics[name], maps[name] = evaluate_predictions(
            cube.tws, prediction, test_mask, weights, cube.month, tws_climatology,
            reference=predictions["Climatology"], min_months=settings["min_evaluation_months"],
        )
    return {
        "predictions": predictions,
        "alphas": alphas,
        "validation_rmse": validation_rmse,
        "gbm_iterations": gbm_iterations,
        "tws_climatology": tws_climatology,
        "static": static,
        "test_mask": test_mask,
        "metrics": pd.DataFrame(metrics).T.sort_values("MSESS_vs_clim", ascending=False).round(3),
        "maps": maps,
    }


def spatial_block_folds(cube: alignment.PixelCube, block_deg: float, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Assign every pixel to a fold by randomly distributing lat/lon blocks across folds."""
    blocks = np.floor((cube.pixel_lat + 90) / block_deg) * 1000 + np.floor((cube.pixel_lon + 180) / block_deg)
    unique_blocks = np.unique(blocks)
    fold_of_block = dict(zip(unique_blocks, rng.permutation(len(unique_blocks)) % n_folds))
    return np.array([fold_of_block[block] for block in blocks])


def spatial_cv_baselines(
    cube: alignment.PixelCube,
    features: dict,
    drivers: list[str],
    climatologies: dict,
    static: dict,
    tws_climatology: np.ndarray,
    config: dict,
    rng: np.random.Generator,
) -> dict:
    """Pooled models predicting held-out spatial blocks from P/T features and static descriptors.

    Per-pixel models cannot transfer to unseen pixels, so only pooled models are used. The target is the
    raw TWS anomaly over all observed months; the reference is a zero predictor because GRACE anomalies
    are relative to a time mean.
    """
    settings = config["baselines"]
    folds = spatial_block_folds(cube, settings["spatial_block_deg"], settings["spatial_folds"], rng)
    spatial_features = dict(features)
    spatial_features["pclim_m"] = climatologies["precip"][cube.month - 1]
    spatial_features["tclim_m"] = climatologies["t2m"][cube.month - 1]
    columns = drivers + [f"cumP{settings['gbm_cumulative_window']}", "h0", "h1", "h2", "h3", "pclim_m", "tclim_m"]
    analysis_rows = cube.period(cube.tws_start)
    time_index = np.flatnonzero(analysis_rows)

    predictions = {name: np.full_like(cube.tws, np.nan) for name in SPATIAL_MODELS}
    for fold in range(settings["spatial_folds"]):
        train_pixels, test_pixels = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        train_matrix = pooled_matrix(spatial_features, static, columns, time_index, train_pixels)
        train_target = cube.tws[np.ix_(time_index, train_pixels)].ravel()
        usable = _subsample(
            np.flatnonzero(np.isfinite(train_target) & np.isfinite(train_matrix).all(1)), settings["gbm_max_rows"], rng
        )
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=settings["spatial_ridge_alpha"]))
        ridge.fit(train_matrix[usable], train_target[usable])
        hyperparameters = dict(settings["gbm"], max_iter=settings["spatial_gbm_max_iter"])
        gbm = HistGradientBoostingRegressor(**hyperparameters, random_state=settings["seed"])
        gbm.fit(train_matrix[usable], train_target[usable])

        test_matrix = pooled_matrix(spatial_features, static, columns, time_index, test_pixels)
        test_valid = np.isfinite(test_matrix).all(1)
        for name, model in (("Pooled ridge (P,T)", ridge), ("Pooled GBM (P,T)", gbm)):
            prediction = np.full(len(test_matrix), np.nan)
            prediction[test_valid] = model.predict(test_matrix[test_valid])
            predictions[name][np.ix_(time_index, test_pixels)] = prediction.reshape(len(time_index), len(test_pixels))
        predictions["Zero (time-mean)"][np.ix_(time_index, test_pixels)] = 0.0
        print(f"fold {fold}: train pixels {len(train_pixels)}, test pixels {len(test_pixels)}")

    mask = analysis_rows[:, None] & np.isfinite(cube.tws)
    for prediction in predictions.values():
        mask &= np.isfinite(prediction)
    metrics, maps = {}, {}
    for name, prediction in predictions.items():
        metrics[name], maps[name] = evaluate_predictions(
            cube.tws, prediction, mask, cube.area_weight, cube.month, tws_climatology,
            reference=predictions["Zero (time-mean)"], min_months=settings["min_evaluation_months"],
        )
    table = pd.DataFrame(metrics).T.rename(columns={"MSESS_vs_clim": "MSESS_vs_zero"}).round(3)
    return {"predictions": predictions, "folds": folds, "metrics": table, "maps": maps}


def best_climate_model(metrics: pd.DataFrame) -> str:
    """Highest-MSESS baseline that does not use past TWS observations."""
    return metrics.drop(index=[name for name in USES_PAST_TWS if name in metrics.index])["MSESS_vs_clim"].idxmax()


def plot_baseline_comparison(metrics: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    panels = [
        ("MSESS_vs_clim", "MSE skill score vs climatology (higher is better)", True),
        ("anom_r_area_wtd", "Anomaly r, area-weighted (higher is better)", True),
        ("MAE_cm", "MAE in cm, area-weighted (lower is better)", False),
    ]
    order = metrics["MSESS_vs_clim"].sort_values().index
    for axis, (column, label, higher_is_better) in zip(axes, panels):
        values = metrics.loc[order, column]
        colors = ["tab:green" if (value > 0 if higher_is_better else True) else "tab:grey" for value in values.values]
        axis.barh(values.index, values.values, color=colors)
        axis.axvline(0, c="k", lw=0.6)
        axis.set_title(label, fontsize=10)
        axis.grid(axis="x", alpha=0.3)
    for axis in axes[1:]:
        axis.set_yticklabels([])
    return fig


def plot_skill_maps(cube: alignment.PixelCube, temporal: dict) -> plt.Figure:
    maps = temporal["maps"]
    best_climate = best_climate_model(temporal["metrics"])
    nse = maps[best_climate]["nse"]
    fig, axes = map_axes(4)
    plot_map(axes[0], cube.to_map(nse), f"NSE, {best_climate}", "RdYlGn", -1, 1, "NSE")
    plot_map(axes[1], cube.to_map(nse - maps["Climatology"]["nse"]), "ΔNSE vs climatology", "PuOr", -1, 1, "ΔNSE")
    plot_map(axes[2], cube.to_map(maps[best_climate]["r_anom"]), "Anomaly correlation (non-seasonal)", "RdBu", -1, 1, "r")
    plot_map(axes[3], cube.to_map(maps["ARX (P,T,trend,TWS t-1)"]["nse"]), "NSE, ARX (one-step)", "RdYlGn", -1, 1, "NSE")
    fig.suptitle("GRACE-FO test period", y=1.01)
    return fig


def plot_regional_predictions(cube: alignment.PixelCube, predictions: dict, config: dict) -> plt.Figure:
    shown = ["Climatology", "Seasonal + trend", "LR (P,T) + trend", "Seasonal-trend + GBM (P,T)", "ARX (P,T,trend,TWS t-1)"]
    names = ["California Central Valley", "NW India", "Central Amazon", "Murray–Darling"]
    regions = {
        name: config["regions"][name]
        for name in names
        if name in config["regions"] and len(cube.region_index(config["regions"][name]))
    }
    fig, axes = plt.subplots(len(regions), 1, figsize=(13, 2.8 * len(regions)), sharex=True)
    axes = np.atleast_1d(axes)
    window = cube.times >= pd.Timestamp(config["splits"]["validation"]["start"])
    for axis, (name, bounds) in zip(axes, regions.items()):
        axis.plot(cube.times[window], cube.region_mean(cube.tws, bounds)[window], "k.-", lw=1.5, ms=4, label="GRACE/GRACE-FO")
        for model in shown:
            axis.plot(cube.times[window], cube.region_mean(predictions[model], bounds)[window], lw=1.1, label=model)
        axis.axvline(pd.Timestamp(config["splits"]["test"]["start"]), c="grey", ls="--", lw=0.8)
        axis.set_title(name, loc="left", fontsize=10)
        axis.set_ylabel("TWS (cm)")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=3)
    return fig


def plot_spatial_cv(cube: alignment.PixelCube, spatial: dict) -> plt.Figure:
    maps = spatial["maps"]["Pooled GBM (P,T)"]
    fig, axes = map_axes(2)
    plot_map(axes[0], cube.to_map(maps["r"]), "Held-out-block r, pooled GBM", "RdBu", -1, 1, "r")
    plot_map(axes[1], cube.to_map(maps["r_anom"]), "Held-out-block anomaly r, pooled GBM", "RdBu", -1, 1, "r")
    return fig


def _variable_name(model_name: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in model_name).strip("_")


def prediction_dataset(cube: alignment.PixelCube, temporal: dict, spatial: dict, config: dict) -> xr.Dataset:
    """Every baseline's prediction on the full grid and calendar plus the shared test mask and spatial folds.

    Deep-learning predictions can be scored with evaluate_predictions on exactly the same cells.
    """
    dataset = xr.Dataset({f"pred_{_variable_name(name)}": cube.to_cube(values) for name, values in temporal["predictions"].items()})
    dataset["tws_obs"] = cube.to_cube(cube.tws)
    dataset["test_eval_mask"] = cube.to_cube(temporal["test_mask"].astype(np.float32))
    dataset["spatial_fold"] = cube.to_map(spatial["folds"].astype(float))
    splits = config["splits"]
    dataset.attrs.update(
        train=f"{splits['train']['start'][:7]}–{splits['train']['end'][:7]}",
        val=f"{splits['validation']['start'][:7]}–{splits['validation']['end'][:7]}",
        test=f"{splits['test']['start'][:7]}–{cube.times[-1]:%Y-%m}",
        setup=config["baselines"]["setup"],
    )
    return dataset


def model_specification(cube: alignment.PixelCube, temporal: dict, feature_sets: dict, config: dict, input_path: Path) -> dict:
    settings = config["baselines"]
    return {
        "model_name": "p_t_tws_baselines",
        "model_type": "per_pixel_ridge_and_pooled_gradient_boosting",
        "target_variable": "tws (GRACE/GRACE-FO lwe_thickness, cm)",
        "setup": settings["setup"],
        "feature_sets": feature_sets,
        "ridge_alpha_selected": temporal["alphas"],
        "validation_rmse_cm": temporal["validation_rmse"],
        "gbm_iterations": temporal["gbm_iterations"],
        "gbm_hyperparameters": settings["gbm"],
        "splits": config["splits"],
        "last_month": f"{cube.times[-1]:%Y-%m}",
        "n_pixels": int(cube.n_pixels),
        "spatial_cv": {
            "block_deg": settings["spatial_block_deg"],
            "folds": settings["spatial_folds"],
            "ridge_alpha": settings["spatial_ridge_alpha"],
            "gbm_max_iter": settings["spatial_gbm_max_iter"],
        },
        "seed": settings["seed"],
        "fitting": "Per-pixel ridge models are refit on train + validation; fitted weights are not stored.",
        "normalization": "Ridge features are standardized per pixel on fitting months; metrics remain in centimeters.",
        "aligned_input": str(input_path),
    }


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Aligned TWS/P/T NetCDF from p_t_tws_data.py")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/p_t_tws_config.yaml"), help="Driver/baseline configuration"
    )
    parser.add_argument("--metrics-dir", required=True, type=Path, help="Directory for baseline metric tables")
    parser.add_argument("--figures-dir", required=True, type=Path, help="Directory for baseline figures")
    parser.add_argument(
        "--predictions-output", required=True, type=Path, help="NetCDF path for every baseline's gridded predictions"
    )
    parser.add_argument("--model-output", required=True, type=Path, help="JSON path for the baseline specification")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input not found: {args.input}")
    config = alignment.load_config(args.config)
    for directory in (args.metrics_dir, args.figures_dir, args.predictions_output.parent, args.model_output.parent):
        directory.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(args.input) as aligned:
        cube = alignment.PixelCube.from_dataset(aligned)
    rng = np.random.default_rng(config["baselines"]["seed"])
    train = config["splits"]["train"]
    climatologies, anomalies = alignment.training_anomalies(cube, cube.period(train["start"], train["end"]))
    features, feature_sets, drivers = build_features(cube, anomalies, config["baselines"])
    print(f"Setup: {config['baselines']['setup']}. Drivers: {drivers}")

    temporal = temporal_split_baselines(cube, features, feature_sets, drivers, climatologies, config, rng)
    spatial = spatial_cv_baselines(
        cube, features, drivers, climatologies, temporal["static"], temporal["tws_climatology"], config, rng
    )

    temporal_path = args.metrics_dir / "p_t_tws_baseline_metrics_gracefo.csv"
    spatial_path = args.metrics_dir / "p_t_tws_baseline_metrics_spatial_cv.csv"
    temporal["metrics"].to_csv(temporal_path)
    spatial["metrics"].to_csv(spatial_path)
    predictions = prediction_dataset(cube, temporal, spatial, config)
    predictions.to_netcdf(args.predictions_output, engine="netcdf4", encoding=compressed_float_encoding(predictions))
    specification = model_specification(cube, temporal, feature_sets, config, args.input)
    args.model_output.write_text(json.dumps(specification, indent=2, default=str) + "\n")

    figures = {
        "06_baseline_comparison": plot_baseline_comparison(temporal["metrics"]),
        "07_skill_maps": plot_skill_maps(cube, temporal),
        "08_regional_predictions": plot_regional_predictions(cube, temporal["predictions"], config),
        "09_spatial_cv": plot_spatial_cv(cube, spatial),
    }
    for name, fig in figures.items():
        save_figure(fig, args.figures_dir / f"p_t_tws_{name}.png")
        plt.close(fig)

    print(temporal["metrics"].to_string())
    print(spatial["metrics"].to_string())
    print(f"Saved GRACE-FO baseline metrics: {temporal_path}")
    print(f"Saved spatial-CV baseline metrics: {spatial_path}")
    print(f"Saved baseline predictions: {args.predictions_output}")
    print(f"Saved baseline specification: {args.model_output}")
    print(f"Saved baseline figures to: {args.figures_dir}")


if __name__ == "__main__":
    main()
