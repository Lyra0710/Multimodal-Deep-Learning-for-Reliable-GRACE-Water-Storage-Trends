"""LSTM one-month-ahead GRACE-FO TWSA forecasts with ensemble prediction intervals.

One LSTM is shared by every land cell between 60°S and 60°N. For target month t it reads the
previous `sequence_length` months of that cell's deseasonalized TWSA anomaly plus the calendar
month, and predicts a Gaussian mean and variance for the anomaly at t. Five members trained with
different seeds form a deep ensemble; their spread is the model (epistemic) uncertainty and their
mean predicted variance is the data (aleatoric) uncertainty. Interval widths are then rescaled
with split-conformal calibration on the validation year so nominal coverage holds.

Leakage controls: every input month precedes the target month; the climatology, per-cell scale and
static feature normalization use training months only; the test split is touched once, at the end.
Metrics follow scripts/grace_fo_persistence.py: area-weighted MAE, area-weighted per-cell temporal
RMSE, and area-weighted mean temporal Pearson r, on land_mask == 1 cells within 60°S–60°N.

Data sources: `data.source: splits` (default) reads the three GRACE-FO split files;
`data.source: aligned` reads the team's aligned GRACE/IMERG/MERRA-2 file, its cell mask and optional
earlier history months, and `data.inputs` can add past precipitation and temperature (strict forecast:
only months before the target). `forecast.test_end` shortens the test period, and `output.tag` keeps
each configuration's outputs separate. Test predictions are saved in the team evaluator format.

Usage (repository root):
    python scripts/lstm_forecast.py --config configs/lstm_config.yaml
    python scripts/lstm_forecast.py --config configs/lstm_aligned_combined.yaml --aligned-file PATH
    python scripts/lstm_forecast.py --prepare-only   # build the cache without PyTorch
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

SPLITS = ("train", "validation", "test")


# ----------------------------------------------------------------------------- data
def read_split(path: Path, target: str, mask_name: str) -> dict:
    """Read one split file with xarray, falling back to h5py (NetCDF-4 is HDF5)."""
    try:
        import xarray as xr
        with xr.open_dataset(path) as ds:
            return {
                "tws": ds[target].transpose("time", "lat", "lon").values.astype("float32"),
                "land": ds[mask_name].values,
                "lat": ds.lat.values, "lon": ds.lon.values,
                "months": pd.DatetimeIndex(ds.time.values).to_period("M"),
            }
    except (ImportError, ValueError, OSError):
        import h5py
        with h5py.File(path, "r") as f:
            units = f["time"].attrs["units"].decode()
            origin = pd.Timestamp(units.split("since")[1].strip())
            months = pd.DatetimeIndex(origin + pd.to_timedelta(f["time"][:], unit="D")).to_period("M")
            return {"tws": f[target][:].astype("float32"), "land": f[mask_name][:],
                    "lat": f["lat"][:], "lon": f["lon"][:], "months": months}


def load_config(path: Path) -> dict:
    """Read a YAML config; `extends: other.yaml` starts from another config and overrides its keys."""
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop("extends", None)
    if not parent:
        return cfg

    def merge(base, new):
        for k, v in new.items():
            base[k] = merge(base.get(k, {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
        return base
    return merge(load_config(path.parent / parent), cfg)


def split_ids(months: pd.PeriodIndex, shared: dict) -> np.ndarray:
    """0/1/2 for train/validation/test target months, -1 for history-only months."""
    ids = np.full(len(months), -1)
    for i, s in enumerate(SPLITS):
        b = shared["splits"][s]
        ids[(months >= pd.Period(b["start"][:7], "M")) & (months <= pd.Period(b["end"][:7], "M"))] = i
    return ids


def read_aligned(path: Path, variables: list, mask_variable: str) -> dict:
    """Read the aligned multimodal NetCDF (time, lat, lon) with xarray, falling back to h5py."""
    try:
        import xarray as xr
        with xr.open_dataset(path) as ds:
            name = mask_variable if mask_variable in ds else "mask"
            return {"data": {v: ds[v].transpose("time", "lat", "lon").values.astype("float32") for v in variables},
                    "mask": ds[name].values, "mask_name": name, "lat": ds.lat.values, "lon": ds.lon.values,
                    "months": pd.DatetimeIndex(ds.time.values).to_period("M")}
    except (ImportError, ValueError, OSError):
        import h5py
        with h5py.File(path, "r") as f:
            units = f["time"].attrs["units"]
            units = units.decode() if isinstance(units, bytes) else units
            origin = pd.Timestamp(units.split("since")[1].strip())
            name = mask_variable if mask_variable in f else "mask"
            return {"data": {v: f[v][:].astype("float32") for v in variables}, "mask": f[name][:],
                    "mask_name": name, "lat": f["lat"][:], "lon": f["lon"][:],
                    "months": pd.DatetimeIndex(origin + pd.to_timedelta(f["time"][:], unit="D")).to_period("M")}


def prepare_cache(cfg: dict, shared: dict, root: Path) -> Path:
    if cfg["data"].get("source", "splits") == "aligned":
        return prepare_aligned_cache(cfg, shared, root)
    return prepare_split_cache(cfg, shared, root)


def test_end(cfg: dict, shared: dict) -> pd.Period:
    return pd.Period((cfg.get("forecast", {}).get("test_end") or shared["splits"]["test"]["end"])[:7], "M")


def source_id(path: Path) -> str:
    """Name and size of the aligned file, so a cache built from a different file is rebuilt."""
    return f"{path.name}:{path.stat().st_size}"


def cache_is_current(cfg: dict, root: Path, cache: Path) -> bool:
    if not cache.exists():
        return False
    if cfg["data"].get("source", "splits") != "aligned":
        return True
    with np.load(cache) as c:
        return "source" in c and str(c["source"]) == source_id(root / cfg["data"]["aligned_file"])


def prepare_aligned_cache(cfg: dict, shared: dict, root: Path) -> Path:
    """Aligned TWS, precipitation and temperature for the file's mask, history_start through test_end."""
    dc = cfg["data"]
    a = read_aligned(root / dc["aligned_file"], ["tws", "precip", "t2m"], dc.get("mask_variable", "training_mask"))
    keep = (np.nan_to_num(a["mask"]) > 0) & (np.abs(a["lat"])[:, None] <= dc["latitude_limit_deg"])
    sel = (a["months"] >= pd.Period(dc.get("history_start", shared["splits"]["train"]["start"])[:7], "M")) & \
          (a["months"] <= test_end(cfg, shared))
    months = a["months"][sel]
    if not months.equals(pd.period_range(months[0], months[-1], freq="M")):
        raise ValueError("Aligned file months are not consecutive")
    tws = a["data"]["tws"][sel][:, keep]
    extra = np.stack([a["data"][v][sel][:, keep] for v in ("precip", "t2m")], axis=-1)
    lat_idx, lon_idx = np.nonzero(keep)
    out = root / dc["cache"]
    np.savez_compressed(out, tws=tws, extra=extra, extra_names=np.array(["precip", "t2m"]),
                        source=np.array(source_id(root / dc["aligned_file"])), mask_name=np.array(a["mask_name"]),
                        lat=a["lat"][lat_idx], lon=a["lon"][lon_idx], lat_idx=lat_idx, lon_idx=lon_idx,
                        grid_lat=a["lat"], grid_lon=a["lon"], month=months.month.values, year=months.year.values,
                        split=split_ids(months, shared))
    print(f"Cached {tws.shape[0]} months ({months[0]}–{months[-1]}) x {tws.shape[1]} cells "
          f"from {dc['aligned_file']} using mask '{a['mask_name']}' -> {out}")
    return out


def prepare_split_cache(cfg: dict, shared: dict, root: Path) -> Path:
    """Stack the three chronological splits into (time, cell) arrays for evaluation cells."""
    target = shared["forecast"]["target_variable"]
    mask_name = shared["forecast"]["land_mask_variable"]
    parts = {s: read_split(root / cfg["data"]["split_files"][s], target, mask_name) for s in SPLITS}
    lat, lon, land = parts["train"]["lat"], parts["train"]["lon"], parts["train"]["land"]
    keep = (land == 1) & (np.abs(lat)[:, None] <= cfg["data"]["latitude_limit_deg"])
    months = parts["train"]["months"].append([parts[s]["months"] for s in SPLITS[1:]])
    expected = pd.period_range(shared["study_period"]["start"][:7], shared["study_period"]["end"][:7], freq="M")
    if not months.equals(expected):
        raise ValueError("Split files do not form the consecutive configured study period")
    tws = np.concatenate([parts[s]["tws"][:, keep] for s in SPLITS])
    lat_idx, lon_idx = np.nonzero(keep)
    split_id = np.concatenate([np.full(len(parts[s]["months"]), i) for i, s in enumerate(SPLITS)])
    upto = months <= test_end(cfg, shared)
    tws, months, split_id = tws[upto], months[upto], split_id[upto]
    out = root / cfg["data"]["cache"]
    np.savez_compressed(out, tws=tws, lat=lat[lat_idx], lon=lon[lon_idx], lat_idx=lat_idx, lon_idx=lon_idx,
                        grid_lat=lat, grid_lon=lon, month=months.month.values, year=months.year.values,
                        split=split_id)
    print(f"Cached {tws.shape[0]} months x {tws.shape[1]} cells -> {out}")
    return out


def load_cache(path: Path, inputs=("tws",)) -> dict:
    d = dict(np.load(path))
    d["periods"] = pd.PeriodIndex([pd.Period(year=y, month=m, freq="M") for y, m in zip(d["year"], d["month"])])
    extra = [v for v in inputs if v != "tws"]
    if extra:
        names = list(d.get("extra_names", []))
        missing = [v for v in extra if v not in names]
        if missing:
            raise ValueError(f"Inputs {missing} need data.source: aligned")
        d["extra"] = d["extra"][..., [names.index(v) for v in extra]]
        d["extra_names"] = np.array(extra)
    else:
        d.pop("extra", None)
        d.pop("extra_names", None)
    return d


def deseasonalize(d: dict, min_std: float) -> dict:
    """Training-only statistics for the model inputs and the baselines.

    Model: z(t) = (TWSA(t) - TWSA(t-1)) / scale, the per-cell standardized month-to-month change;
    the forecast is TWSA(t-1) + scale * z_hat(t). Baselines: monthly climatology and anomalies.
    """
    train = d["split"] == 0
    tws, month = d["tws"], d["month"]
    clim = np.stack([np.nanmean(tws[train & (month == m)], axis=0) for m in range(1, 13)])  # (12, cell)
    seasonal = clim[month - 1]                                                               # (time, cell)
    change = np.vstack([np.full((1, tws.shape[1]), np.nan, tws.dtype), np.diff(tws, axis=0)])
    scale = np.maximum(np.nanstd(change[train], axis=0), min_std)
    base = np.vstack([np.full((1, tws.shape[1]), np.nan, tws.dtype), tws[:-1]])              # TWSA(t-1)
    log_scale = np.log(scale)
    static = (log_scale - log_scale.mean()) / log_scale.std()
    out = {"seasonal": seasonal, "anomaly": tws - seasonal, "base": base, "scale": scale,
           "z": change / scale, "static": static}
    if "extra" in d:  # past precipitation/temperature: anomaly from training climatology / per-cell std
        x = d["extra"]
        xclim = np.stack([np.nanmean(x[train & (month == m)], axis=0) for m in range(1, 13)])
        xa = x - xclim[month - 1]
        out["extra_z"] = xa / np.maximum(np.nanstd(xa[train], axis=0), 1e-6)
    return out


def valid_targets(ds: dict, seq_len: int) -> np.ndarray:
    """(time, cell) True where the target change and every input month t-seq_len..t-1 are finite."""
    ok = np.isfinite(ds["z"])
    if "extra_z" in ds:
        ok_in = ok & np.isfinite(ds["extra_z"]).all(axis=-1)
    else:
        ok_in = ok
    csum = np.vstack([np.zeros((1, ok.shape[1]), int), np.cumsum(ok_in, axis=0)])
    valid = np.zeros_like(ok)
    t = np.arange(seq_len, ok.shape[0])
    valid[t] = ok[t] & ((csum[t] - csum[t - seq_len]) == seq_len)
    return valid


def target_index(d: dict, split: int, seq_len: int) -> np.ndarray:
    t = np.flatnonzero(d["split"] == split)
    return t[t >= seq_len + 1]  # input changes at t-seq_len..t-1 need TWSA from t-seq_len-1; all precede t


# ----------------------------------------------------------------------------- metrics
def area_weights(lat: np.ndarray) -> np.ndarray:
    return np.cos(np.deg2rad(lat))


def score(actual: np.ndarray, pred: np.ndarray, lat: np.ndarray) -> dict:
    """Team metrics on (time, cell) arrays; mirrors grace_fo_persistence.evaluate()."""
    w = area_weights(lat)
    err = pred - actual
    mae = (np.abs(err) * w).sum() / (w.sum() * err.shape[0])
    rmse_cell = np.sqrt((err ** 2).mean(axis=0))
    a, p = actual - actual.mean(0), pred - pred.mean(0)
    r_cell = (a * p).sum(0) / np.sqrt((a ** 2).sum(0) * (p ** 2).sum(0))
    ok = np.isfinite(r_cell)
    return {"mae_cm": float(mae), "rmse_cm": float((rmse_cell * w).sum() / w.sum()),
            "pearson_r": float((r_cell[ok] * w[ok]).sum() / w[ok].sum()),
            "months": int(err.shape[0]), "cells": int(err.shape[1])}


def per_cell_rmse(actual, pred):
    return np.sqrt(((pred - actual) ** 2).mean(axis=0))


# ----------------------------------------------------------------------------- baselines
def baselines(d: dict, ds: dict, t_idx: np.ndarray, max_order: int) -> dict:
    tws, seasonal, anomaly = d["tws"], ds["seasonal"], ds["anomaly"]
    out = {
        "Persistence": tws[t_idx - 1],
        "Seasonal persistence": tws[t_idx - 1] - seasonal[t_idx - 1] + seasonal[t_idx],
        "Climatology": seasonal[t_idx],
    }
    # AR(p) on anomalies, p <= max_order by AIC, fitted per cell on training targets only.
    train_t = np.flatnonzero(d["split"] == 0)
    train_t = np.array([t for t in train_t if t >= max_order and np.isfinite(anomaly[t - max_order:t + 1]).all()])
    y = anomaly[train_t]
    lags = np.stack([anomaly[train_t - k] for k in range(1, max_order + 1)], axis=-1)  # (n, cell, P)
    n = len(train_t)
    best_aic = np.full(tws.shape[1], np.inf)
    best_pred = np.zeros((len(t_idx), tws.shape[1]), dtype="float32")
    for p in range(1, max_order + 1):
        X = np.concatenate([np.ones((n, tws.shape[1], 1)), lags[..., :p]], axis=-1).transpose(1, 0, 2)
        XtX = np.einsum("cnp,cnq->cpq", X, X)
        Xty = np.einsum("cnp,nc->cp", X, y)
        beta = np.linalg.solve(XtX + 1e-6 * np.eye(p + 1), Xty[..., None])[..., 0]  # (cell, p+1)
        rss = ((y - np.einsum("cnp,cp->nc", X, beta)) ** 2).sum(0)
        aic = n * np.log(rss / n) + 2 * (p + 1)
        Xt = np.concatenate([np.ones((len(t_idx), tws.shape[1], 1)),
                             np.stack([anomaly[t_idx - k] for k in range(1, p + 1)], -1)], -1)
        pred = np.einsum("ncp,cp->nc", Xt, beta) + seasonal[t_idx]
        better = aic < best_aic
        best_aic[better] = aic[better]
        best_pred[:, better] = pred[:, better]
    out["AR (p≤3, anomalies)"] = best_pred
    return out


# ----------------------------------------------------------------------------- model
def build_model(cfg: dict):
    import torch
    from torch import nn

    class LSTMForecaster(nn.Module):
        """Sequence of [scaled change, sin month, cos month, static] -> Gaussian mean, log-variance of z(t)."""

        def __init__(self, n_in=4, hidden=64, layers=2, dropout=0.1):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
            self.head = nn.Sequential(nn.Linear(hidden + 2, 32), nn.ReLU(), nn.Linear(32, 2))

        def forward(self, x, target_month_feat):
            h, _ = self.lstm(x)
            out = self.head(torch.cat([h[:, -1], target_month_feat], dim=1))
            return out[:, 0], out[:, 1].clamp(-8.0, 5.0)

    m = cfg["model"]
    n_in = 4 + len([v for v in cfg["data"].get("inputs", ["tws"]) if v != "tws"])
    return LSTMForecaster(n_in, m["hidden_size"], m["num_layers"], m["dropout"])


class Batcher:
    """Builds (B, L, 4) input windows on the device from (target month, cell) index pairs."""

    def __init__(self, d, ds, seq_len, device):
        import torch
        self.torch, self.L, self.device = torch, seq_len, device
        self.z = torch.tensor(ds["z"], dtype=torch.float32, device=device)
        ang = 2 * np.pi * (d["month"] - 1) / 12
        self.mon = torch.tensor(np.stack([np.sin(ang), np.cos(ang)], 1), dtype=torch.float32, device=device)
        self.static = torch.tensor(ds["static"], dtype=torch.float32, device=device)
        self.extra = (torch.tensor(np.nan_to_num(ds["extra_z"]), dtype=torch.float32, device=device)
                      if "extra_z" in ds else None)
        w = area_weights(d["lat"])
        self.w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
        self.offsets = torch.arange(-seq_len, 0, device=device)

    def __call__(self, t, c):
        torch = self.torch
        tt = t[:, None] + self.offsets                          # (B, L), all < t
        x = torch.stack([self.z[tt, c[:, None]],
                         self.mon[tt, 0], self.mon[tt, 1],
                         self.static[c][:, None].expand(-1, self.L)], dim=-1)
        if self.extra is not None:                              # past precipitation/temperature only
            x = torch.cat([x, self.extra[tt, c[:, None]]], dim=-1)
        return x, self.mon[t], self.z[t, c], self.w[c]


def gaussian_nll(mu, log_var, y, w):
    return (w * 0.5 * (log_var + (y - mu) ** 2 / log_var.exp())).mean()


def predict(model, batcher, t_idx, n_cells, batch=8192):
    import torch
    model.eval()
    mu = np.zeros((len(t_idx), n_cells), dtype="float32")
    var = np.zeros_like(mu)
    with torch.no_grad():
        for i, t in enumerate(t_idx):
            for s in range(0, n_cells, batch):
                c = torch.arange(s, min(s + batch, n_cells), device=batcher.device)
                tt = torch.full_like(c, int(t))
                x, mf, _, _ = batcher(tt, c)
                m, lv = model(x, mf)
                mu[i, s:s + len(c)] = m.cpu().numpy()
                var[i, s:s + len(c)] = lv.exp().cpu().numpy()
    return mu, var


def train_member(cfg, d, ds, seed, device, log):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, L = cfg["training"], cfg["model"]["sequence_length"]
    batcher = Batcher(d, ds, L, device)
    n_cells = d["tws"].shape[1]
    train_t, val_t = target_index(d, 0, L), target_index(d, 1, L)
    rows, cells = np.nonzero(valid_targets(ds, L)[train_t])
    pairs = torch.tensor(np.stack([train_t[rows], cells], -1), device=device)
    if seed == cfg["training"]["seeds"][0]:
        print(f"  training cell-months with complete history: {len(pairs):,}")
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr["max_epochs"])
    best, best_state, wait = np.inf, None, 0
    zv = ds["z"][val_t]
    wv = area_weights(d["lat"]) / area_weights(d["lat"]).mean()
    for epoch in range(1, tr["max_epochs"] + 1):
        model.train()
        t0, total, nb = time.time(), 0.0, 0
        perm = torch.randperm(len(pairs), device=device)
        for s in range(0, len(pairs), tr["batch_size"]):
            b = pairs[perm[s:s + tr["batch_size"]]]
            x, mf, y, w = batcher(b[:, 0], b[:, 1])
            mu, lv = model(x, mf)
            loss = gaussian_nll(mu, lv, y, w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            opt.step()
            total, nb = total + loss.item(), nb + 1
        sched.step()
        mu_v, var_v = predict(model, batcher, val_t, n_cells)
        val_nll = float((wv * 0.5 * (np.log(var_v) + (zv - mu_v) ** 2 / var_v)).mean())
        val_rmse = score(d["tws"][val_t], ds["base"][val_t] + mu_v * ds["scale"], d["lat"])["rmse_cm"]
        log.append({"seed": seed, "epoch": epoch, "train_nll": total / nb, "val_nll": val_nll,
                    "val_rmse_cm": val_rmse, "seconds": round(time.time() - t0, 1)})
        print(f"  seed {seed} epoch {epoch:2d} train NLL {total / nb:.4f} val NLL {val_nll:.4f} "
              f"val RMSE {val_rmse:.3f} cm ({time.time() - t0:.0f}s)", flush=True)
        if val_nll < best - 1e-4:
            best, wait = val_nll, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= tr["patience"]:
                break
    model.load_state_dict(best_state)
    return model, batcher


# ----------------------------------------------------------------------------- uncertainty
def ensemble(preds, ds, t_idx):
    """Combine member (mu, var) in z units into cm mean, aleatoric and epistemic variances."""
    mus = np.stack([p[0] for p in preds])
    vars_ = np.stack([p[1] for p in preds])
    scale2 = ds["scale"] ** 2
    return {"mean": ds["base"][t_idx] + mus.mean(0) * ds["scale"],
            "aleatoric_var": vars_.mean(0) * scale2,
            "epistemic_var": mus.var(0) * scale2}


def conformal_factors(actual, mean, sigma, levels, lat):
    """Area-weighted split-conformal quantile of |error| / sigma for each nominal level."""
    r = (np.abs(actual - mean) / sigma).ravel()
    w = np.broadcast_to(area_weights(lat), actual.shape).ravel()
    order = np.argsort(r)
    cum = np.cumsum(w[order]) / w.sum()
    adj = lambda q: min(1.0, q * (r.size + 1) / r.size)  # finite-sample conformal correction
    return {q: float(r[order][min(np.searchsorted(cum, adj(q)), r.size - 1)]) for q in levels}


def coverage_table(actual, mean, sigma, levels, factors, lat):
    w = np.broadcast_to(area_weights(lat), actual.shape)
    rows = []
    for q in levels:
        for kind, k in (("raw Gaussian", stats.norm.ppf(0.5 + q / 2)), ("conformal-calibrated", factors[q])):
            half = k * sigma
            inside = np.abs(actual - mean) <= half
            rows.append({"nominal": q, "interval": kind, "multiplier": float(k),
                         "empirical_coverage": float((inside * w).sum() / w.sum()),
                         "mean_width_cm": float((2 * half * w).sum() / w.sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- predictions
def write_predictions(path: Path, d: dict, t_idx, fields: dict) -> Path:
    """Gridded (time, lat, lon) predictions in cm for the team evaluator; CSV if no NetCDF engine."""
    lon180 = np.where(d["grid_lon"] > 180, d["grid_lon"] - 360, d["grid_lon"])
    order = np.argsort(lon180)
    col = np.argsort(order)[d["lon_idx"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    times = d["periods"][t_idx].to_timestamp()
    grids = {}
    for k, v in fields.items():
        g = np.full((len(t_idx), len(d["grid_lat"]), len(lon180)), np.nan, dtype="float32")
        g[:, d["lat_idx"], col] = v
        grids[k] = g
    try:
        import xarray as xr
        ds = xr.Dataset({k: (("time", "lat", "lon"), g, {"units": "cm"}) for k, g in grids.items()},
                        coords={"time": times, "lat": d["grid_lat"], "lon": lon180[order]})
        ds.to_netcdf(path.with_suffix(".nc"), encoding={k: {"zlib": True} for k in grids})
        return path.with_suffix(".nc")
    except (ImportError, ValueError) as error:
        print(f"NetCDF writer unavailable ({error}); writing CSV instead")
        frame = pd.DataFrame({"time": np.repeat(times.strftime("%Y-%m-%d"), len(d["lat"])),
                              "lat": np.tile(d["lat"], len(t_idx)),
                              "lon": np.tile(np.where(d["lon"] > 180, d["lon"] - 360, d["lon"]), len(t_idx))})
        for k, v in fields.items():
            frame[k] = np.asarray(v, dtype="float32").ravel()
        frame.to_csv(path.with_suffix(".csv"), index=False, float_format="%.4f")
        return path.with_suffix(".csv")


# ----------------------------------------------------------------------------- figures
def to_grid(values, d):
    grid = np.full((len(d["grid_lat"]), len(d["grid_lon"])), np.nan)
    grid[d["lat_idx"], d["lon_idx"]] = values
    lon = np.where(d["grid_lon"] > 180, d["grid_lon"] - 360, d["grid_lon"])
    order = np.argsort(lon)
    return grid[:, order], lon[order]


def wmean_series(x, lat):
    w = area_weights(lat)
    return (x * w).sum(1) / w.sum()


def make_figures(d, ds, t_idx, actual, models, ens, sigma_cal, cov, fig_dir, prefix="lstm",
                 example_cell=(28.25, 77.25)):
    fig_dir.mkdir(parents=True, exist_ok=True)
    lat = d["lat"]
    dates = d["periods"][t_idx].to_timestamp()

    # 1. (a) global land mean; (b) one example cell with its calibrated 90% interval
    q90 = cov.query("nominal == 0.9 and interval == 'conformal-calibrated'")["multiplier"].item()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ax = axes[0]
    ax.plot(dates, wmean_series(actual, lat), "o-", color="k", label="GRACE-FO actual")
    ax.plot(dates, wmean_series(ens["mean"], lat), "o-", label="LSTM ensemble mean")
    ax.plot(dates, wmean_series(models["Persistence"], lat), "--", alpha=0.7, label="Persistence")
    ax.set(title="(a) Area-weighted land mean, 60°S–60°N", ylabel="TWSA (cm)")
    lon180 = np.where(d["lon"] > 180, d["lon"] - 360, d["lon"])
    cell = int(np.argmin((d["lat"] - example_cell[0]) ** 2 + (lon180 - example_cell[1]) ** 2))
    ax = axes[1]
    m, h = ens["mean"][:, cell], q90 * sigma_cal[:, cell]
    ax.fill_between(dates, m - h, m + h, alpha=0.25, label="LSTM 90% interval (calibrated)")
    ax.plot(dates, actual[:, cell], "o-", color="k", label="GRACE-FO actual")
    ax.plot(dates, m, "o-", label="LSTM ensemble mean")
    ax.set(title=f"(b) Single grid cell {d['lat'][cell]:.2f}°N, {lon180[cell]:.2f}°E (Northern India)",
           ylabel="TWSA (cm)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"LSTM one-month-ahead forecasts, test period {d['periods'][t_idx[0]]} to {d['periods'][t_idx[-1]]}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_test_actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    # 2. RMSE by test month for every model
    fig, ax = plt.subplots(figsize=(10, 4))
    w = area_weights(lat)
    for name, pred in models.items():
        if name == "Climatology" or "member" in name:
            continue
        ax.plot(dates, np.sqrt((((pred - actual) ** 2) * w).sum(1) / w.sum()), "o-", label=name, ms=3)
    ax.set(title="Error by test month (area-weighted RMSE across cells)", ylabel="RMSE (cm)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_rmse_by_test_month.png", dpi=150)
    plt.close(fig)

    # 3. Skill maps vs persistence and AR
    r_lstm = per_cell_rmse(actual, models["LSTM (ensemble mean)"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
    for ax, ref in zip(axes, ("Persistence", "AR (p≤3, anomalies)")):
        skill = 1 - r_lstm / per_cell_rmse(actual, models[ref])
        grid, glon = to_grid(skill, d)
        im = ax.imshow(grid, origin="lower", cmap="RdBu", vmin=-0.6, vmax=0.6, aspect="auto",
                       extent=[glon[0], glon[-1], d["grid_lat"][0], d["grid_lat"][-1]])
        ax.set_ylim(-60, 60)
        better = (skill > 0) * w
        ax.set_title(f"LSTM skill vs {ref}: 1 − RMSE ratio (better in {better.sum() / w.sum():.0%} of land area)",
                     fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.8, label="skill (blue = LSTM better)")
    fig.savefig(fig_dir / f"{prefix}_skill_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Reliability: nominal vs empirical coverage, raw and calibrated
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for kind, mk in (("raw Gaussian", "s--"), ("conformal-calibrated", "o-")):
        sub = cov[cov.interval == kind]
        axes[0].plot(sub.nominal, sub.empirical_coverage, mk, label=kind)
    axes[0].plot([0.4, 1], [0.4, 1], "k:", label="perfect calibration")
    axes[0].set(xlabel="nominal coverage", ylabel="empirical test coverage", title="Interval reliability (test)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    # 5. Spread–skill: does predicted sigma track actual error?
    err = np.abs(actual - ens["mean"]).ravel()
    sig = sigma_cal.ravel()
    bins = np.quantile(sig, np.linspace(0, 1, 11))
    b = np.clip(np.digitize(sig, bins[1:-1]), 0, 9)
    ms = [sig[b == i].mean() for i in range(10)]
    rm = [np.sqrt((err[b == i] ** 2).mean()) for i in range(10)]
    axes[1].plot(ms, rm, "o-", label="LSTM (decile bins)")
    lim = max(max(ms), max(rm)) * 1.05
    axes[1].plot([0, lim], [0, lim], "k:", label="RMSE = σ")
    rho = stats.spearmanr(sig, err).statistic
    axes[1].set(xlabel="predicted σ (cm, calibrated)", ylabel="actual RMSE (cm)",
                title=f"Spread–skill (Spearman ρ between σ and |error| = {rho:.2f})")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_interval_reliability.png", dpi=150)
    plt.close(fig)
    return float(rho)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/lstm_config.yaml"))
    ap.add_argument("--root", type=Path, default=Path("."), help="repository root")
    ap.add_argument("--prepare-only", action="store_true", help="only build the data cache")
    ap.add_argument("--device", default=None, help="cpu, cuda or mps (default: best available)")
    ap.add_argument("--aligned-file", default=None, help="override data.aligned_file")
    args = ap.parse_args()

    root = args.root
    cfg = load_config(root / args.config)
    if args.aligned_file:
        cfg["data"]["aligned_file"] = args.aligned_file
    shared = yaml.safe_load((root / args.config.parent / cfg["shared_split_config"]).read_text())
    cache = root / cfg["data"]["cache"]
    if args.prepare_only or not cache_is_current(cfg, root, cache):
        prepare_cache(cfg, shared, root)
        if args.prepare_only:
            return

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    tag = cfg.get("output", {}).get("tag", "")
    prefix = f"lstm_{tag}" if tag else "lstm"
    metrics_dir, fig_dir, model_dir = root / "results/metrics", root / "results/figures", root / "models" / prefix
    for p in (metrics_dir, fig_dir, model_dir):
        p.mkdir(parents=True, exist_ok=True)

    inputs = cfg["data"].get("inputs", ["tws"])
    d = load_cache(cache, inputs)
    ds = deseasonalize(d, cfg["data"]["min_anomaly_std_cm"])
    L = cfg["model"]["sequence_length"]
    val_t, test_t = target_index(d, 1, L), target_index(d, 2, L)
    n_cells = d["tws"].shape[1]
    valid = valid_targets(ds, L)
    if not (valid[val_t].all() and valid[test_t].all()):
        raise ValueError("Validation/test targets need complete histories in every cell")
    print(f"Device {device}; inputs {inputs}; {n_cells} cells; train target months {len(target_index(d, 0, L))}, "
          f"validation {len(val_t)}, test {len(test_t)} ({d['periods'][test_t[0]]} to {d['periods'][test_t[-1]]})")

    log, val_preds, test_preds = [], [], []
    for seed in cfg["training"]["seeds"][: cfg["training"]["ensemble_size"]]:
        model, batcher = train_member(cfg, d, ds, seed, device, log)
        torch.save(model.state_dict(), model_dir / f"{prefix}_member_seed{seed}.pt")
        val_preds.append(predict(model, batcher, val_t, n_cells))
        test_preds.append(predict(model, batcher, test_t, n_cells))
    pd.DataFrame(log).to_csv(metrics_dir / f"{prefix}_training_history.csv", index=False)

    # Ensemble + conformal calibration on validation (2023) only
    levels = cfg["uncertainty"]["nominal_levels"]
    ens_v, ens = ensemble(val_preds, ds, val_t), ensemble(test_preds, ds, test_t)
    sig_v = np.sqrt(ens_v["aleatoric_var"] + ens_v["epistemic_var"])
    sig = np.sqrt(ens["aleatoric_var"] + ens["epistemic_var"])
    factors = conformal_factors(d["tws"][val_t], ens_v["mean"], sig_v, levels, d["lat"])

    # Test evaluation (single pass)
    actual = d["tws"][test_t]
    models = {"LSTM (ensemble mean)": ens["mean"]}
    for k, (mu, _) in enumerate(test_preds):
        models[f"LSTM member {k}"] = ds["base"][test_t] + mu * ds["scale"]
    models.update(baselines(d, ds, test_t, cfg["ar_baseline"]["max_order"]))
    rows = [{"model": n, **score(actual, p, d["lat"])} for n, p in models.items()]
    table = pd.DataFrame(rows)
    table.to_csv(metrics_dir / f"{prefix}_test_metrics.csv", index=False)

    w = area_weights(d["lat"])
    by_month = pd.DataFrame({"target_month": d["periods"][test_t].astype(str)})
    for n, p in models.items():
        if "member" not in n:
            by_month[n] = np.sqrt((((p - actual) ** 2) * w).sum(1) / w.sum())
    by_month.to_csv(metrics_dir / f"{prefix}_rmse_by_test_month.csv", index=False)
    years = d["periods"][test_t].year
    by_year = pd.DataFrame([{"year": y, "model": n, **score(actual[years == y], p[years == y], d["lat"])}
                            for y in np.unique(years) for n, p in models.items() if "member" not in n])
    by_year.to_csv(metrics_dir / f"{prefix}_test_metrics_by_year.csv", index=False)

    lon180 = np.where(d["lon"] > 180, d["lon"] - 360, d["lon"])
    reg_rows = []
    for name, (la0, la1, lo0, lo1) in cfg["regions"].items():
        sel = (d["lat"] >= la0) & (d["lat"] <= la1) & (lon180 >= lo0) & (lon180 <= lo1)
        for n, p in models.items():
            if "member" not in n:
                reg_rows.append({"region": name, "model": n, **score(actual[:, sel], p[:, sel], d["lat"][sel])})
    pd.DataFrame(reg_rows).to_csv(metrics_dir / f"{prefix}_region_metrics.csv", index=False)

    cov = coverage_table(actual, ens["mean"], sig, levels, factors, d["lat"])
    cov.to_csv(metrics_dir / f"{prefix}_interval_coverage.csv", index=False)
    rho = make_figures(d, ds, test_t, actual, models, ens, sig, cov, fig_dir, prefix)
    q90 = factors[0.9] if 0.9 in factors else None
    fields = {"prediction": ens["mean"], "actual_tws": actual, "sigma": sig}
    if q90 is not None:
        fields.update(lower_90=ens["mean"] - q90 * sig, upper_90=ens["mean"] + q90 * sig)
    pred_path = write_predictions(metrics_dir.parent / "predictions" / f"{prefix}_test_predictions", d, test_t, fields)
    unc = {
        "aleatoric_share_of_total_variance": float(((ens["aleatoric_var"] / sig ** 2) * w).sum()
                                                   / (w.sum() * len(test_t))),
        "mean_sigma_cm": float((sig * w).sum() / (w.sum() * len(test_t))),
        "spearman_sigma_vs_abs_error": rho,
        "conformal_factors_from_validation": factors,
    }
    (metrics_dir / f"{prefix}_uncertainty_summary.json").write_text(json.dumps(unc, indent=2) + "\n")
    spec = {"model_name": "lstm_one_month_forecaster", "config": cfg, "device": device,
            "inputs": "previous sequence_length per-cell standardized month-to-month TWSA changes, "
                      "sin/cos calendar month, log anomaly scale (static), plus standardized past "
                      "precipitation/temperature anomalies when configured; target-month sin/cos in head",
            "target": "standardized change TWSA(t) - TWSA(t-1); forecast cm = TWSA(t-1) + scale * z",
            "normalization": "per-cell change scale (and baseline climatology) fitted on training months only",
            "selection": "early stopping on validation Gaussian NLL", "test_months": len(test_t),
            "cells": n_cells}
    spec.update(input_variables=inputs, test_period=[str(d["periods"][test_t[0]]), str(d["periods"][test_t[-1]])],
                predictions_file=str(pred_path))
    (model_dir / f"{prefix}_model_spec.json").write_text(json.dumps(spec, indent=2, default=str) + "\n")
    print(f"Saved test predictions: {pred_path}")
    print(table.round(3).to_string(index=False))
    print(cov.round(3).to_string(index=False))
    print(json.dumps(unc, indent=2))


if __name__ == "__main__":
    main()
