"""Analyze precipitation and temperature as drivers of GRACE terrestrial water storage.

Trends: Mann-Kendall test with the Hamed-Rao autocorrelation correction on
deseasonalized monthly series, Theil-Sen slopes per year, and Benjamini-Hochberg
FDR control across pixels.

Relationships: TWS against lagged and cumulative precipitation, lagged temperature,
the partial correlation with temperature after removing the best cumulative-P
signal, and a water-balance check (monthly TWS change vs precipitation). Series are
deseasonalized with the training climatology and linearly detrended over the
analysis period. p-values use the effective sample size
n_eff = n(1 - r1x * r1y) / (1 + r1x * r1y) (Bretherton et al., 1999), then BH-FDR.

Examples:
    python p_t_tws_analysis.py --input data/processed/p_t_tws_aligned_0.5deg.nc \
        --config configs/p_t_tws_config.yaml --metrics-dir results/metrics \
        --figures-dir results/figures --maps-dir data/processed
"""

import argparse
from functools import lru_cache
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr

import p_t_tws_data as alignment

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


def shift(values: np.ndarray, lag: int) -> np.ndarray:
    """Move a (time, pixel) array forward by lag months, padding the start with NaN."""
    if lag == 0:
        return values.copy()
    shifted = np.full_like(values, np.nan)
    shifted[lag:] = values[:-lag]
    return shifted


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing window-month sum; missing months count as zero anomaly."""
    cumulative = np.nancumsum(np.vstack([np.zeros((1, values.shape[1])), values]), 0)
    total = np.full_like(values, np.nan)
    total[window - 1:] = cumulative[window:] - cumulative[:-window]
    return total


def detrend(values: np.ndarray, rows: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Remove a per-pixel least-squares linear trend fitted on rows; other rows become NaN."""
    selected = np.where(rows[:, None], values, np.nan)
    valid = np.isfinite(selected)
    time_mean = np.nansum(np.where(valid, years[:, None], 0), 0) / valid.sum(0)
    value_mean = np.nanmean(selected, 0)
    centered_time = years[:, None] - time_mean
    slope = np.nansum(centered_time * (selected - value_mean), 0) / np.nansum(np.where(valid, centered_time ** 2, 0), 0)
    return selected - (value_mean + slope * centered_time)


def nan_corr(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel Pearson r over jointly valid months and the number of such months.

    Constant series return NaN rather than a spurious correlation.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.where(valid, x, np.nan)
    y = np.where(valid, y, np.nan)
    x = x - np.nanmean(x, 0)
    y = y - np.nanmean(y, 0)
    scale = np.sqrt(np.nansum(x ** 2, 0) * np.nansum(y ** 2, 0))
    correlation = np.nansum(x * y, 0) / np.where(scale > 1e-9, scale, np.nan)
    return correlation, valid.sum(0)


def lag1_autocorrelation(values: np.ndarray) -> np.ndarray:
    return nan_corr(values[1:], values[:-1])[0]


def correlation_test(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pearson r and two-sided p-value using an autocorrelation-adjusted effective sample size."""
    correlation, count = nan_corr(x, y)
    persistence = np.clip(lag1_autocorrelation(x) * lag1_autocorrelation(y), -0.99, 0.99)
    effective_n = np.clip(count * (1 - persistence) / (1 + persistence), 4, count)
    t_statistic = correlation * np.sqrt((effective_n - 2) / np.clip(1 - correlation ** 2, 1e-12, None))
    return correlation, 2 * stats.t.sf(np.abs(t_statistic), effective_n - 2)


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-pixel residual of y after least-squares regression on x."""
    valid = np.isfinite(x) & np.isfinite(y)
    x_centered = np.where(valid, x, np.nan)
    y_centered = np.where(valid, y, np.nan)
    x_centered = x_centered - np.nanmean(x_centered, 0)
    y_centered = y_centered - np.nanmean(y_centered, 0)
    slope = np.nansum(x_centered * y_centered, 0) / np.nansum(x_centered ** 2, 0)
    return y_centered - slope * x_centered


def bh_fdr(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg significance flags; non-finite p-values are never significant."""
    p_values = np.asarray(p_values, float)
    significant = np.zeros_like(p_values, bool)
    finite = np.isfinite(p_values)
    tested = p_values[finite]
    if tested.size == 0:
        return significant
    order = np.argsort(tested)
    passed = tested[order] <= q * np.arange(1, tested.size + 1) / tested.size
    rejections = np.max(np.flatnonzero(passed)) + 1 if passed.any() else 0
    tested_significant = np.zeros(tested.size, bool)
    tested_significant[order[:rejections]] = True
    significant[finite] = tested_significant
    return significant


def _pixel_trend(series: np.ndarray, years: np.ndarray, min_months: int) -> tuple[float, float]:
    import pymannkendall as mk

    valid = np.isfinite(series)
    if valid.sum() < min_months:
        return np.nan, np.nan
    slope = stats.theilslopes(series[valid], years[valid])[0]
    try:
        p_value = mk.hamed_rao_modification_test(series[valid]).p
    except Exception:  # noqa: BLE001 - degenerate series get no p-value
        p_value = np.nan
    return slope, p_value


def trend_map(
    values: np.ndarray, rows: np.ndarray, years: np.ndarray, settings: dict, n_jobs: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel Theil-Sen slope (per year), Hamed-Rao p-value, and FDR significance."""
    try:
        import pymannkendall  # noqa: F401
        from joblib import Parallel, delayed
    except ImportError as error:
        raise RuntimeError(
            "Trend analysis requires pymannkendall and joblib. Install them with: "
            "python -m pip install pymannkendall joblib"
        ) from error

    trend_years = years[rows]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_pixel_trend)(values[rows, pixel], trend_years, settings["min_trend_months"])
        for pixel in range(values.shape[1])
    )
    slope, p_value = map(np.array, zip(*results))
    return slope, p_value, bh_fdr(p_value, settings["fdr_q"])


def trend_analysis(cube: alignment.PixelCube, anomalies: dict, config: dict, rows: np.ndarray) -> dict:
    """Trend maps for deseasonalized P, T, and TWS over the selected months."""
    started = time.time()
    trends = {
        name: trend_map(anomalies[name], rows, cube.years, config["analysis"], config["data"]["n_jobs"])
        for name in ("precip", "t2m", "tws")
    }
    print(f"Trend maps in {time.time() - started:.0f}s")
    return trends


def trend_shares(trends: dict) -> pd.DataFrame:
    """Percentage of pixels with FDR-significant increasing and decreasing trends."""
    rows = [
        {
            "variable": name,
            "pct_sig_increasing": 100 * np.mean(significant & (slope > 0)),
            "pct_sig_decreasing": 100 * np.mean(significant & (slope < 0)),
        }
        for name, (slope, _, significant) in trends.items()
    ]
    return pd.DataFrame(rows).round(1)


def trend_dataset(cube: alignment.PixelCube, trends: dict) -> xr.Dataset:
    return xr.Dataset(
        {
            f"{name}_{label}": cube.to_map(values)
            for name, (slope, p_value, significant) in trends.items()
            for label, values in (
                ("sen_slope_per_yr", slope),
                ("p_hamed_rao", p_value),
                ("fdr_sig", significant.astype(float)),
            )
        }
    )


def _strongest(correlations: dict, keys: list[str], sign: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per pixel, the largest (sign=1) or most negative (sign=-1) r among keys, its p-value, and its index."""
    correlation = np.stack([correlations[key][0] for key in keys])
    p_value = np.stack([correlations[key][1] for key in keys])
    index = np.argmax(sign * np.where(np.isfinite(correlation), correlation, -np.inf * sign), 0)
    columns = np.arange(correlation.shape[1])
    return correlation[index, columns], p_value[index, columns], index


def driver_relationships(cube: alignment.PixelCube, anomalies: dict, config: dict) -> dict:
    """Correlate detrended TWS anomalies with precipitation and temperature drivers per pixel."""
    lags = config["analysis"]["correlation_lags"]
    windows = config["analysis"]["cumulative_windows"]
    rows = cube.period(config["data"]["tws_start"])
    years = cube.years
    tws = detrend(anomalies["tws"], rows, years)

    correlations = {}
    for lag in lags:
        correlations[f"P_lag{lag}"] = correlation_test(tws, detrend(shift(anomalies["precip"], lag), rows, years))
        correlations[f"T_lag{lag}"] = correlation_test(tws, detrend(shift(anomalies["t2m"], lag), rows, years))
    cumulative = {window: detrend(rolling_sum(anomalies["precip"], window), rows, years) for window in windows}
    for window in windows:
        correlations[f"cumP_{window}"] = correlation_test(tws, cumulative[window])

    best_lag_precip = _strongest(correlations, [f"P_lag{lag}" for lag in lags])
    best_cumulative = _strongest(correlations, [f"cumP_{window}" for window in windows])
    most_negative_temperature = _strongest(correlations, [f"T_lag{lag}" for lag in lags], sign=-1)

    best_cumulative_series = np.take_along_axis(
        np.stack([cumulative[window] for window in windows]), best_cumulative[2][None, None], 0
    )[0]
    partial_temperature = correlation_test(
        residualize(tws, best_cumulative_series),
        residualize(detrend(anomalies["t2m"], rows, years), best_cumulative_series),
    )
    storage_change = anomalies["tws"] - shift(anomalies["tws"], 1)  # NaN across gaps automatically
    water_balance = correlation_test(
        np.where(rows[:, None], storage_change, np.nan), np.where(rows[:, None], anomalies["precip"], np.nan)
    )
    return {
        "correlations": correlations,
        "best_lag_precip": best_lag_precip,
        "best_cumulative_precip": best_cumulative,
        "best_cumulative_window": np.array(windows)[best_cumulative[2]],
        "most_negative_temperature": most_negative_temperature,
        "most_negative_temperature_lag": np.array(lags)[most_negative_temperature[2]],
        "partial_temperature": partial_temperature,
        "water_balance": water_balance,
    }


def correlation_summary(relationships: dict, weights: np.ndarray, q: float) -> pd.DataFrame:
    """Median and area-weighted r plus the share of FDR-significant pixels per relationship."""
    tested = {
        "r(TWS, P) lag 0": relationships["correlations"]["P_lag0"],
        "r(TWS, P) best lag": relationships["best_lag_precip"][:2],
        "r(TWS, cumP) best window": relationships["best_cumulative_precip"][:2],
        "r(TWS, T) lag 0": relationships["correlations"]["T_lag0"],
        "r(TWS, T) most negative lag": relationships["most_negative_temperature"][:2],
        "partial r(TWS, T | cumP)": relationships["partial_temperature"],
        "r(ΔTWS, P)": relationships["water_balance"],
    }
    rows = []
    for name, (correlation, p_value) in tested.items():
        significant = bh_fdr(p_value, q)
        rows.append(
            {
                "relationship": name,
                "median_r": np.nanmedian(correlation),
                "area_wtd_mean_r": np.nansum(correlation * weights) / np.sum(weights * np.isfinite(correlation)),
                "pct_sig_pos": 100 * np.mean(significant & (correlation > 0)),
                "pct_sig_neg": 100 * np.mean(significant & (correlation < 0)),
            }
        )
    return pd.DataFrame(rows).round(3)


def correlation_dataset(cube: alignment.PixelCube, relationships: dict) -> xr.Dataset:
    return xr.Dataset(
        {
            "r_P_lag0": cube.to_map(relationships["correlations"]["P_lag0"][0]),
            "r_cumP_best": cube.to_map(relationships["best_cumulative_precip"][0]),
            "best_cumP_window": cube.to_map(relationships["best_cumulative_window"]),
            "r_T_best_neg": cube.to_map(relationships["most_negative_temperature"][0]),
            "partial_r_T_given_cumP": cube.to_map(relationships["partial_temperature"][0]),
            "r_dTWS_P": cube.to_map(relationships["water_balance"][0]),
        }
    )


def seasonal_phase_lag(climatologies: dict) -> np.ndarray:
    """Months from the wettest climatological month to the TWS maximum."""
    peak_precip = np.nanargmax(climatologies["precip"], 0) + 1
    peak_tws = np.nanargmax(climatologies["tws"], 0) + 1
    return (peak_tws - peak_precip) % 12


@lru_cache(maxsize=1)
def _has_coastlines() -> bool:
    """Coastlines need a one-time Natural Earth download; skip them when offline."""
    try:
        next(iter(cfeature.COASTLINE.geometries()))
        return True
    except Exception:  # noqa: BLE001
        return False


def map_axes(count: int, ncols: int = 2, width: float = 13) -> tuple[plt.Figure, np.ndarray]:
    nrows = int(np.ceil(count / ncols))
    projection = {"subplot_kw": {"projection": ccrs.PlateCarree()}} if HAS_CARTOPY else {}
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, 3.2 * nrows), **projection)
    axes = np.atleast_1d(axes).ravel()
    for axis in axes[count:]:
        axis.set_visible(False)
    return fig, axes[:count]


def plot_map(axis, data: xr.DataArray, title: str, cmap="RdBu_r", vmin=None, vmax=None, label="", discrete=None):
    transform = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    if discrete is not None:
        cmap = plt.get_cmap(cmap, discrete)
    image = axis.pcolormesh(data.lon, data.lat, data.values, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", **transform)
    lat_min, lat_max = float(data.lat.min()), float(data.lat.max())
    if HAS_CARTOPY:
        if _has_coastlines():
            axis.coastlines(linewidth=0.4)
        axis.set_extent([-180, 180, lat_min, lat_max], ccrs.PlateCarree())
    else:
        axis.set_ylim(lat_min, lat_max)
    axis.set_title(title, fontsize=10)
    plt.colorbar(image, ax=axis, orientation="horizontal", pad=0.05, fraction=0.05, label=label)
    return image


def shade_splits(axis, splits: dict, last_month: pd.Timestamp) -> None:
    for (name, bounds), color in zip(splits.items(), ("tab:blue", "tab:orange", "tab:green")):
        end = min(pd.Timestamp(bounds["end"]), last_month)
        axis.axvspan(pd.Timestamp(bounds["start"]), end, color=color, alpha=0.06)


def save_figure(fig: plt.Figure, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")


def plot_global_series(cube: alignment.PixelCube, anomalies: dict, splits: dict) -> plt.Figure:
    weights = cube.area_weight
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    panels = [
        (axes[0], cube.tws, "TWS anomaly (cm)", "k", False),
        (axes[1], anomalies["precip"], "P anomaly (mm/month)", "tab:blue", True),
        (axes[2], anomalies["t2m"], "T anomaly (°C)", "tab:red", True),
    ]
    for axis, values, label, color, smooth in panels:
        series = np.nansum(values * weights, 1) / np.sum(weights * np.isfinite(values), 1)
        axis.plot(cube.times, series, color=color, lw=1)
        if smooth:
            rolling = pd.Series(series).rolling(12, center=True).mean()
            axis.plot(cube.times, rolling, color=color, lw=2.2, alpha=0.7, label="12-mo mean")
            axis.legend(loc="upper left", fontsize=8)
        axis.set_ylabel(label)
        shade_splits(axis, splits, cube.times[-1])
        axis.grid(alpha=0.3)
    axes[0].set_title("Area-weighted land mean. Shading: train / validation / test (GRACE-FO)")
    return fig


def plot_climatology(cube: alignment.PixelCube, climatologies: dict, analysis_rows: np.ndarray) -> plt.Figure:
    annual_precip = np.nanmean(cube.precip[analysis_rows], 0) * 12
    mean_temperature = np.nanmean(cube.t2m[analysis_rows], 0)
    tws_amplitude = (np.nanmax(climatologies["tws"], 0) - np.nanmin(climatologies["tws"], 0)) / 2
    fig, axes = map_axes(4)
    plot_map(axes[0], cube.to_map(annual_precip), "Mean annual precipitation", "YlGnBu", 0,
             np.nanpercentile(annual_precip, 98), "mm/yr")
    plot_map(axes[1], cube.to_map(mean_temperature), "Mean 2 m air temperature", "RdYlBu_r", label="°C")
    plot_map(axes[2], cube.to_map(tws_amplitude), "TWS seasonal amplitude", "viridis", 0,
             np.nanpercentile(tws_amplitude, 98), "cm")
    plot_map(axes[3], cube.to_map(seasonal_phase_lag(climatologies)), "Lag from peak-P month to peak-TWS month",
             "twilight", 0, 11, "months", discrete=12)
    return fig


def plot_trends(cube: alignment.PixelCube, trends: dict) -> plt.Figure:
    fig, axes = map_axes(3, ncols=3, width=18)
    for axis, (name, unit) in zip(axes, [("precip", "mm/month per yr"), ("t2m", "°C/yr"), ("tws", "cm/yr")]):
        slope, _, significant = trends[name]
        limit = np.nanpercentile(np.abs(slope), 97)
        cmap = "RdBu_r" if name == "t2m" else "BrBG"
        plot_map(axis, cube.to_map(np.where(significant, slope, np.nan)), f"{name} Sen slope (FDR-significant only)",
                 cmap, -limit, limit, unit)
    return fig


def plot_driver_correlations(cube: alignment.PixelCube, relationships: dict, q: float) -> plt.Figure:
    def significant_only(correlation, p_value):
        return cube.to_map(np.where(bh_fdr(p_value, q), correlation, np.nan))

    best_cumulative = relationships["best_cumulative_precip"]
    fig, axes = map_axes(6)
    plot_map(axes[0], significant_only(*relationships["correlations"]["P_lag0"]), "r(TWS, P), lag 0", "RdBu", -1, 1,
             "r (FDR sig.)")
    plot_map(axes[1], significant_only(*best_cumulative[:2]), "r(TWS, cumulative P), best window", "RdBu", -1, 1,
             "r (FDR sig.)")
    plot_map(axes[2], cube.to_map(np.where(bh_fdr(best_cumulative[1], q), relationships["best_cumulative_window"], np.nan)),
             "Best cumulative-P window", "viridis", label="months")
    plot_map(axes[3], significant_only(*relationships["most_negative_temperature"][:2]), "r(TWS, T), most negative lag",
             "RdBu", -1, 1, "r (FDR sig.)")
    plot_map(axes[4], significant_only(*relationships["partial_temperature"]), "Partial r(TWS, T | cumulative P)",
             "RdBu", -0.8, 0.8, "r (FDR sig.)")
    plot_map(axes[5], significant_only(*relationships["water_balance"]), "r(ΔTWS, P)  (water-balance check)", "RdBu",
             -1, 1, "r (FDR sig.)")
    return fig


def plot_regional_drivers(cube: alignment.PixelCube, anomalies: dict, config: dict) -> plt.Figure:
    regions = {name: bounds for name, bounds in config["regions"].items() if len(cube.region_index(bounds))}
    rows = cube.period(config["data"]["tws_start"])
    years = cube.years
    fig, axes = plt.subplots(len(regions), 2, figsize=(14, 2.6 * len(regions)), gridspec_kw={"width_ratios": [3, 1]})
    axes = np.atleast_2d(axes)

    def zscore(values):
        return (values - np.nanmean(values)) / np.nanstd(values)

    def regional_corr(target, driver):
        return nan_corr(detrend(target[:, None], rows, years), detrend(driver[:, None], rows, years))[0][0]

    for row, (name, bounds) in zip(axes, regions.items()):
        tws = cube.region_mean(np.where(rows[:, None], anomalies["tws"], np.nan), bounds)
        precip = cube.region_mean(anomalies["precip"], bounds)
        temperature = cube.region_mean(anomalies["t2m"], bounds)
        precip_12 = cube.region_mean(rolling_sum(anomalies["precip"], 12), bounds)
        temperature_12 = pd.Series(temperature).rolling(12, center=True).mean().values

        axis = row[0]
        axis.plot(cube.times[rows], zscore(tws[rows]), "k", lw=1.4, label="TWS anomaly")
        axis.plot(cube.times[rows], zscore(precip_12[rows]), "tab:blue", lw=1, label="12-mo cumulative P anomaly")
        axis.plot(cube.times[rows], zscore(temperature_12[rows]), "tab:red", lw=1, label="T anomaly (12-mo mean)")
        axis.set_title(name, fontsize=10, loc="left")
        axis.set_ylabel("z-score")
        axis.grid(alpha=0.3)

        lags = np.arange(0, 13)
        lagged_precip = [regional_corr(tws, shift(precip[:, None], lag)[:, 0]) for lag in lags]
        cumulative_precip = [regional_corr(tws, rolling_sum(precip[:, None], max(lag, 1))[:, 0]) for lag in lags]
        lagged_temperature = [regional_corr(tws, shift(temperature[:, None], lag)[:, 0]) for lag in lags]
        row[1].plot(lags, lagged_precip, "o-", ms=3, c="tab:blue", label="P at lag")
        row[1].plot(lags, cumulative_precip, "s--", ms=3, c="navy", label="cum. P (window)")
        row[1].plot(lags, lagged_temperature, "o-", ms=3, c="tab:red", label="T at lag")
        row[1].axhline(0, c="grey", lw=0.6)
        row[1].set_ylim(-1, 1)
        row[1].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=3, loc="upper left")
    axes[0, 1].legend(fontsize=7)
    axes[-1, 1].set_xlabel("lag / window (months)")
    return fig


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Aligned TWS/P/T NetCDF from p_t_tws_data.py")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/p_t_tws_config.yaml"), help="Driver/baseline configuration"
    )
    parser.add_argument("--metrics-dir", required=True, type=Path, help="Directory for trend/correlation tables")
    parser.add_argument("--figures-dir", required=True, type=Path, help="Directory for driver-analysis figures")
    parser.add_argument("--maps-dir", required=True, type=Path, help="Directory for trend and correlation map NetCDFs")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input not found: {args.input}")
    config = alignment.load_config(args.config)
    for directory in (args.metrics_dir, args.figures_dir, args.maps_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(args.input) as aligned:
        cube = alignment.PixelCube.from_dataset(aligned)
    train = config["splits"]["train"]
    climatologies, anomalies = alignment.training_anomalies(cube, cube.period(train["start"], train["end"]))
    analysis_rows = cube.period(config["data"]["tws_start"])
    q = config["analysis"]["fdr_q"]

    figures = {
        "01_global_series": plot_global_series(cube, anomalies, config["splits"]),
        "02_climatology": plot_climatology(cube, climatologies, analysis_rows),
    }
    phase_lag = seasonal_phase_lag(climatologies)
    print(f"Median seasonal phase lag P→TWS: {np.nanmedian(phase_lag):.0f} months "
          f"(IQR {np.nanpercentile(phase_lag, 25):.0f}–{np.nanpercentile(phase_lag, 75):.0f})")

    trends = trend_analysis(cube, anomalies, config, analysis_rows)
    gracefo_trends = trend_analysis(cube, anomalies, config, cube.period(config["analysis"]["gracefo_trend_start"]))
    trend_table = pd.concat(
        [trend_shares(trends).assign(period="full record"), trend_shares(gracefo_trends).assign(period="GRACE-FO")]
    )
    trend_path = args.metrics_dir / "p_t_tws_trend_summary.csv"
    trend_table.to_csv(trend_path, index=False)
    trend_dataset(cube, trends).to_netcdf(args.maps_dir / "p_t_tws_trend_maps.nc")
    figures["03_trends"] = plot_trends(cube, trends)

    relationships = driver_relationships(cube, anomalies, config)
    summary = correlation_summary(relationships, cube.area_weight, q)
    summary_path = args.metrics_dir / "p_t_tws_correlation_summary.csv"
    summary.to_csv(summary_path, index=False)
    correlation_dataset(cube, relationships).to_netcdf(args.maps_dir / "p_t_tws_correlation_maps.nc")
    figures["04_tws_driver_correlations"] = plot_driver_correlations(cube, relationships, q)
    figures["05_regional_drivers"] = plot_regional_drivers(cube, anomalies, config)

    for name, fig in figures.items():
        save_figure(fig, args.figures_dir / f"p_t_tws_{name}.png")
        plt.close(fig)
    print(trend_table.to_string(index=False))
    print(summary.to_string(index=False))
    print(f"Saved trend summary: {trend_path}")
    print(f"Saved correlation summary: {summary_path}")
    print(f"Saved trend and correlation maps to: {args.maps_dir}")
    print(f"Saved driver-analysis figures to: {args.figures_dir}")


if __name__ == "__main__":
    main()
