"""Extra evaluation metrics for saved LSTM test predictions (no retraining).

Reads results/predictions/<prefix>_test_predictions.nc written by lstm_forecast.py, recomputes the
baselines on the same cells and test months, and scores every model with:

  MAE, per-cell RMSE (area-weighted), pooled RMSE, mean error (bias), temporal Pearson r,
  Nash–Sutcliffe efficiency per cell (area-weighted median and share of land area with NSE > 0),
  RMSE skill against persistence and MSE skill against climatology,
  CRPS (Gaussian predictive distribution for the LSTM; equals MAE for point forecasts),
  and, for the LSTM, the interval score of the calibrated 90% interval and the Gaussian NLL.

All averages use cos(latitude) area weights. Usage (repository root, same config as the run):
    python scripts/lstm_extra_metrics.py --config configs/lstm_aligned_combined.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm_forecast as lf  # noqa: E402


def read_predictions(path: Path) -> dict:
    """Gridded (time, lat, lon) prediction fields, via xarray or h5py."""
    names = ["prediction", "actual_tws", "sigma", "lower_90", "upper_90"]
    try:
        import xarray as xr
        with xr.open_dataset(path) as ds:
            out = {k: ds[k].values for k in names if k in ds}
            out["lat"], out["lon"] = ds.lat.values, ds.lon.values
            return out
    except (ImportError, ValueError, OSError):
        import h5py
        with h5py.File(path, "r") as f:
            out = {k: f[k][:] for k in names if k in f}
            out["lat"], out["lon"] = f["lat"][:], f["lon"][:]
            return out


def to_cells(grid_fields: dict, d: dict) -> dict:
    """Extract (time, cell) arrays for the cache's cells from the evaluator grid."""
    li = {round(float(v), 2): i for i, v in enumerate(grid_fields["lat"])}
    lj = {round(float(v), 2): j for j, v in enumerate(grid_fields["lon"])}
    lon180 = np.where(d["lon"] > 180, d["lon"] - 360, d["lon"])
    ii = np.array([li[round(float(v), 2)] for v in d["lat"]])
    jj = np.array([lj[round(float(v), 2)] for v in lon180])
    return {k: v[:, ii, jj].astype(float) for k, v in grid_fields.items() if k not in ("lat", "lon")}


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][np.searchsorted(cum, 0.5)])


def point_metrics(actual, pred, lat, persistence, climatology) -> dict:
    """Deterministic metrics on (time, cell) arrays with area weights."""
    w = lf.area_weights(lat)
    wt = np.broadcast_to(w, actual.shape)
    err = pred - actual
    base = lf.score(actual, pred, lat)
    sse = (err ** 2).sum(0)
    sst = ((actual - actual.mean(0)) ** 2).sum(0)
    ok = sst > 0
    nse = 1 - sse[ok] / sst[ok]
    mse = lambda e: float(((e ** 2) * wt).sum() / wt.sum())
    rmse_cell = lambda p: float((lf.per_cell_rmse(actual, p) * w).sum() / w.sum())
    return {
        "mae_cm": base["mae_cm"],
        "rmse_cm": base["rmse_cm"],
        "pooled_rmse_cm": float(np.sqrt(mse(err))),
        "bias_cm": float((err * wt).sum() / wt.sum()),
        "pearson_r": base["pearson_r"],
        "nse_median": weighted_median(nse, w[ok]),
        "nse_positive_area_pct": float(100 * (w[ok] * (nse > 0)).sum() / w[ok].sum()),
        "rmse_skill_vs_persistence": 1 - base["rmse_cm"] / rmse_cell(persistence),
        "mse_skill_vs_climatology": 1 - mse(err) / mse(climatology - actual),
        "crps_cm": base["mae_cm"],  # CRPS of a point forecast is its absolute error
    }


def probabilistic_metrics(actual, mean, sigma, lower, upper, lat, alpha=0.1) -> dict:
    """CRPS and NLL of the Gaussian predictive distribution, interval score of the calibrated interval."""
    wt = np.broadcast_to(lf.area_weights(lat), actual.shape)
    wmean = lambda x: float((x * wt).sum() / wt.sum())
    z = (actual - mean) / sigma
    crps = sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    nll = 0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * z ** 2
    interval = (upper - lower) + (2 / alpha) * ((lower - actual) * (actual < lower) + (actual - upper) * (actual > upper))
    return {"crps_cm": wmean(crps), "gaussian_nll": wmean(nll), "interval_score_90_cm": wmean(interval),
            "interval_width_90_cm": wmean(upper - lower),
            "interval_coverage_90_pct": 100 * wmean((actual >= lower) & (actual <= upper))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/lstm_config.yaml"))
    ap.add_argument("--root", type=Path, default=Path("."), help="repository root")
    ap.add_argument("--aligned-file", default=None, help="override data.aligned_file")
    args = ap.parse_args()

    root = args.root
    cfg = lf.load_config(root / args.config)
    if args.aligned_file:
        cfg["data"]["aligned_file"] = args.aligned_file
    shared = yaml.safe_load((root / args.config.parent / cfg["shared_split_config"]).read_text())
    cache = root / cfg["data"]["cache"]
    if not lf.cache_is_current(cfg, root, cache):
        lf.prepare_cache(cfg, shared, root)
    tag = cfg.get("output", {}).get("tag", "")
    prefix = f"lstm_{tag}" if tag else "lstm"

    d = lf.load_cache(cache)
    ds = lf.deseasonalize(d, cfg["data"]["min_anomaly_std_cm"])
    test_t = lf.target_index(d, 2, cfg["model"]["sequence_length"])
    actual = d["tws"][test_t]
    pred = to_cells(read_predictions(root / "results/predictions" / f"{prefix}_test_predictions.nc"), d)
    if not np.allclose(pred["actual_tws"], actual, atol=1e-3):
        raise ValueError("Saved observations do not match the configured data; check the config and data file")

    models = {"LSTM (ensemble mean)": pred["prediction"]}
    models.update(lf.baselines(d, ds, test_t, cfg["ar_baseline"]["max_order"]))
    rows = []
    for name, p in models.items():
        row = {"model": name, **point_metrics(actual, p, d["lat"], models["Persistence"], models["Climatology"])}
        if name.startswith("LSTM"):
            row.update(probabilistic_metrics(actual, pred["prediction"], pred["sigma"],
                                             pred["lower_90"], pred["upper_90"], d["lat"]))
        rows.append(row)
    table = pd.DataFrame(rows)
    out = root / "results/metrics" / f"{prefix}_extra_metrics.csv"
    table.to_csv(out, index=False)
    print(f"Test months {d['periods'][test_t[0]]} to {d['periods'][test_t[-1]]}, {actual.shape[1]} cells")
    print(table.round(3).to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
