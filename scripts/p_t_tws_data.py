"""Align GRACE/GRACE-FO TWS, GPM IMERG precipitation, and MERRA-2 temperature on one grid.

GRACE CRI mascons stay on their native 0.5-degree grid. IMERG (0.1 degree) is
block-averaged and MERRA-2 (0.5 x 0.625 degree) is bilinearly interpolated to
that grid. Every product gets a calendar-month time label; precipitation is
converted to mm/month and temperature to degrees Celsius. GRACE gain factors are
not applied. Anomalies use a monthly climatology fitted on the shared GRACE-FO
training split only, so no validation or test information reaches the features.

Examples:
    # Reuse files in data/raw/, otherwise download them with the credentials in .env.
    python p_t_tws_data.py --config configs/p_t_tws_config.yaml \
        --output data/processed/p_t_tws_aligned_0.5deg.nc --env-file .env
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import xarray as xr

from grace_fo_persistence import load_config as load_split_config
from grace_fo_processing import compressed_float_encoding, download_source


REQUIRED_SECTIONS = ("shared_split_config", "data", "products", "analysis", "baselines", "regions")
GRACE_PATTERN = "GRCTellus*CRI*.nc"


def load_config(config_path: Path, project_root: Path | None = None) -> dict:
    """Load the driver/baseline configuration and attach the shared chronological splits.

    Data directories are resolved against project_root (default: the working directory).
    """
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "Loading the YAML configuration requires PyYAML. Install it with: "
            "python -m pip install PyYAML"
        ) from error

    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    for key in REQUIRED_SECTIONS:
        if key not in config:
            raise ValueError(f"Missing '{key}' in {config_path}")

    config["splits"] = load_split_config(config_path.parent / config["shared_split_config"])["splits"]
    data = config["data"]
    driver_start, tws_start = pd.Period(data["driver_start"], freq="M"), pd.Period(data["tws_start"], freq="M")
    if not driver_start <= tws_start <= pd.Period(config["splits"]["train"]["start"][:7], freq="M"):
        raise ValueError("Expected driver_start <= tws_start <= start of the training split")
    if config["baselines"]["setup"] not in ("nowcast", "forecast"):
        raise ValueError("baselines.setup must be 'nowcast' or 'forecast'")
    if 0 not in config["analysis"]["correlation_lags"]:
        raise ValueError("analysis.correlation_lags must include lag 0")

    root = Path.cwd() if project_root is None else project_root
    for key in ("raw_dir", "cache_dir"):
        data[key] = root / data[key]
    return config


def target_axes(resolution: float) -> tuple[np.ndarray, np.ndarray]:
    latitude = np.arange(-90 + resolution / 2, 90, resolution)
    longitude = np.arange(-180 + resolution / 2, 180, resolution)
    return latitude, longitude


def to_month_start(values) -> pd.DatetimeIndex:
    try:
        index = pd.DatetimeIndex(values)
    except TypeError:
        index = pd.DatetimeIndex([pd.Timestamp(str(value)) for value in values])
    return index.to_period("M").to_timestamp()


def standardize_coords(data: xr.DataArray) -> xr.DataArray:
    """Rename latitude/longitude dimensions, use -180..180 longitude, and sort both axes."""
    aliases = {"latitude": "lat", "longitude": "lon", "Latitude": "lat", "Longitude": "lon", "nlat": "lat", "nlon": "lon"}
    data = data.rename({name: alias for name, alias in aliases.items() if name in data.dims})
    if float(data["lon"].max()) > 180:
        data = data.assign_coords(lon=((data["lon"] + 180) % 360) - 180)
    return data.sortby("lat").sortby("lon")


def to_target_grid(data: xr.DataArray, resolution: float) -> xr.DataArray:
    """Block-average when the source grid nests exactly in the target grid, otherwise interpolate bilinearly."""
    data = standardize_coords(data)
    target_lat, target_lon = target_axes(resolution)
    lat, lon = data["lat"].values.astype(float), data["lon"].values.astype(float)
    lat_step, lon_step = np.diff(lat).mean(), np.diff(lon).mean()
    lat_factor, lon_factor = resolution / lat_step, resolution / lon_step
    nests = (
        abs(lat_factor - round(lat_factor)) < 1e-3
        and abs(lon_factor - round(lon_factor)) < 1e-3
        and abs(lat[0] - lat_step / 2 + 90) < 1e-3
        and abs(lon[0] - lon_step / 2 + 180) < 1e-3
        and len(lat) == round(180 / lat_step)
        and len(lon) == round(360 / lon_step)
    )
    if nests:
        lat_block, lon_block = int(round(lat_factor)), int(round(lon_factor))
        coarse = data.coarsen(lat=lat_block, lon=lon_block).mean() if lat_block > 1 or lon_block > 1 else data
        return coarse.assign_coords(lat=target_lat, lon=target_lon)
    # Pad one column on each side so interpolation wraps across the antimeridian.
    left = data.isel(lon=[-1]).assign_coords(lon=data["lon"].values[-1:] - 360)
    right = data.isel(lon=[0]).assign_coords(lon=data["lon"].values[:1] + 360)
    return xr.concat([left, data, right], "lon").interp(lat=target_lat, lon=target_lon)


def grace_month_labels(dataset: xr.Dataset) -> pd.DatetimeIndex:
    """Label each GRACE solution by the calendar month of its time_bounds midpoint."""
    if "time_bounds" not in dataset:
        return to_month_start(dataset["time"].values)
    bounds = dataset["time_bounds"]
    bound_dim = [dim for dim in bounds.dims if dim != "time"][0]
    if np.issubdtype(bounds.dtype, np.datetime64):
        start, end = bounds.isel({bound_dim: 0}), bounds.isel({bound_dim: 1})
        return to_month_start((start + (end - start) / 2).values)
    return to_month_start(pd.Timestamp("2002-01-01") + pd.to_timedelta(bounds.mean(bound_dim).values, "D"))


def grace_on_target_grid(dataset: xr.Dataset, resolution: float) -> tuple[xr.DataArray, xr.DataArray]:
    """Return monthly TWS anomalies (cm) and the NASA land mask on the target grid.

    The rare months with two solutions are averaged; missing months are left out
    and become NaN once the products are aligned.
    """
    tws = dataset["lwe_thickness"].assign_coords(time=grace_month_labels(dataset)).groupby("time").mean()
    tws = to_target_grid(tws.astype("float32"), resolution).rename("tws")
    tws.attrs = {"units": "cm", "long_name": "TWS anomaly (equivalent water height)"}
    land = to_target_grid(dataset["land_mask"].astype("float32"), resolution) >= 0.5
    return tws, land.rename("land")


def grace_source(config: dict, env_file: Path) -> Path:
    """Reuse the GRACE CRI NetCDF in data/raw/, otherwise download it with grace_fo_processing."""
    raw_dir = config["data"]["raw_dir"]
    files = sorted(raw_dir.glob(GRACE_PATTERN))
    if files:
        return files[0]
    start = f"{config['data']['tws_start']}-01"
    end = str((pd.Period(config["data"]["end"], freq="M")).to_timestamp(how="end").date())
    return download_source(raw_dir, env_file, start, end)


def earthaccess_login(env_file: Path) -> None:
    """Log in to NASA Earthdata with the same .env credentials used by grace_fo_processing."""
    try:
        import earthaccess
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError(
            "Downloading IMERG/MERRA-2 requires earthaccess and python-dotenv. Install them with: "
            "python -m pip install earthaccess python-dotenv"
        ) from error

    if env_file.is_file():
        load_dotenv(env_file)
    if not os.getenv("EARTHDATA_USERNAME") or not os.getenv("EARTHDATA_PASSWORD"):
        raise RuntimeError(
            f"Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD in {env_file}. "
            "GES DISC products also require authorizing the NASA GESDISC DATA ARCHIVE application in your profile."
        )
    earthaccess.login(strategy="environment")


def driver_sources(key: str, patterns: tuple[str, ...], config: dict, env_file: Path) -> list:
    """Local IMERG/MERRA-2 files if present, else streamed file objects (or downloaded paths)."""
    directory = config["data"]["raw_dir"] / key
    local = [path for pattern in patterns for path in sorted(directory.rglob(pattern))]
    if local:
        return local

    import earthaccess

    earthaccess_login(env_file)
    product = config["products"][key]
    search = {
        "short_name": product["short_name"],
        "temporal": (f"{config['data']['driver_start']}-01", f"{config['data']['end']}-28"),
    }
    if product.get("version"):
        search["version"] = product["version"]
    results = earthaccess.search_data(**search)
    print(f"{key}: {len(results)} granules found")
    if not config["data"]["stream"]:
        directory.mkdir(parents=True, exist_ok=True)
        earthaccess.download(results, local_path=str(directory))
        return [path for pattern in patterns for path in sorted(directory.rglob(pattern))]
    return earthaccess.open(results)


def _open_any(source, group: str | None = None) -> xr.Dataset:
    # netCDF4 only reads local paths; for streamed file objects it would just mask the h5netcdf error.
    candidates = ("h5netcdf", "netcdf4") if isinstance(source, (str, Path)) else ("h5netcdf",)
    engines = [engine for engine in candidates if engine in xr.backends.list_engines()] or ["h5netcdf"]
    last_error = None
    for engine in engines:
        try:
            return xr.open_dataset(source, group=group, engine=engine)
        except Exception as error:  # noqa: BLE001 - try the next engine
            last_error = error
    raise last_error


def reduce_imerg(source, resolution: float) -> xr.DataArray:
    """Read one IMERG monthly file and return precipitation in mm/month on the target grid."""
    dataset = _open_any(source, group="Grid")
    rate = standardize_coords(dataset["precipitation"].load()).transpose("time", "lat", "lon")
    rate = rate.where(rate >= 0)
    months = to_month_start(rate["time"].values)
    hours = xr.DataArray(24.0 * months.days_in_month.values, dims="time", coords={"time": months})
    result = to_target_grid(rate.assign_coords(time=months) * hours, resolution).astype("float32")
    dataset.close()
    return result


def reduce_merra2(source, resolution: float) -> xr.DataArray:
    """Read one MERRA-2 monthly file and return 2 m air temperature in degrees C on the target grid."""
    dataset = _open_any(source)
    temperature = dataset["T2M"].load() - 273.15
    temperature = temperature.assign_coords(time=to_month_start(temperature["time"].values))
    result = to_target_grid(temperature, resolution).astype("float32")
    dataset.close()
    return result


DRIVER_PRODUCTS = {
    "imerg": {"patterns": ("*.HDF5", "*.h5"), "reducer": reduce_imerg, "name": "precip", "units": "mm/month"},
    "merra2": {"patterns": ("*.nc4",), "reducer": reduce_merra2, "name": "t2m", "units": "degC"},
}


def _reduce_with_retries(source, reducer, resolution: float, attempts: int = 4) -> xr.DataArray:
    """Streamed reads occasionally fail on transient network errors; retry before giving up on the whole run."""
    for attempt in range(1, attempts + 1):
        try:
            return reducer(source, resolution=resolution)
        except Exception as error:  # noqa: BLE001 - re-raised after the last attempt
            if attempt == attempts:
                raise RuntimeError(f"Failed to read {source} after {attempts} attempts") from error
            print(f"Retrying {getattr(source, 'path', source)} ({attempt}/{attempts - 1}): {error}")
            if hasattr(source, "seek"):
                source.seek(0)
            time.sleep(5 * attempt)


def load_driver_cube(key: str, config: dict, env_file: Path) -> xr.DataArray:
    """Reduce every monthly file of one driver product to the target grid, caching the result."""
    product = DRIVER_PRODUCTS[key]
    resolution = config["data"]["resolution_deg"]
    cache = config["data"]["cache_dir"] / f"{key}_{resolution}deg.nc"
    if cache.exists():
        return xr.open_dataarray(cache).load()

    sources = driver_sources(key, product["patterns"], config, env_file)
    started = time.time()
    with ThreadPoolExecutor(config["data"]["n_jobs"]) as executor:
        reduce = partial(_reduce_with_retries, reducer=product["reducer"], resolution=resolution)
        parts = list(executor.map(reduce, sources))
    cube = xr.concat(parts, "time").sortby("time").groupby("time").mean().rename(product["name"])
    cube.attrs["units"] = product["units"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cube.to_netcdf(cache)
    print(f"{key}: {cube.sizes['time']} months reduced in {time.time() - started:.0f}s -> {cache}")
    return cube


def monthly_climatology(values: np.ndarray, rows: np.ndarray, month: np.ndarray) -> np.ndarray:
    """(12, pixel) monthly means computed only on the selected time rows."""
    return np.stack([np.nanmean(values[rows & (month == number)], 0) for number in range(1, 13)])


def deseasonalize(values: np.ndarray, climatology: np.ndarray, month: np.ndarray) -> np.ndarray:
    return values - climatology[month - 1]


@dataclass
class PixelCube:
    """Valid land pixels flattened to (time, pixel) arrays for vectorized per-pixel work."""

    times: pd.DatetimeIndex
    lat: np.ndarray
    lon: np.ndarray
    pixels: np.ndarray
    tws: np.ndarray
    precip: np.ndarray
    t2m: np.ndarray
    tws_start: str

    @classmethod
    def from_dataset(cls, aligned: xr.Dataset) -> "PixelCube":
        pixels = np.flatnonzero(aligned["mask"].values.astype(bool).ravel())

        def flatten(name: str) -> np.ndarray:
            return aligned[name].values.reshape(aligned.sizes["time"], -1)[:, pixels].astype(np.float64)

        return cls(
            times=pd.DatetimeIndex(aligned.time.values),
            lat=aligned.lat.values,
            lon=aligned.lon.values,
            pixels=pixels,
            tws=flatten("tws"),
            precip=flatten("precip"),
            t2m=flatten("t2m"),
            tws_start=aligned.attrs["tws_start"],
        )

    @property
    def n_pixels(self) -> int:
        return len(self.pixels)

    @property
    def month(self) -> np.ndarray:
        return self.times.month.values

    @property
    def years(self) -> np.ndarray:
        """Decimal years since the first GRACE month, used for trends."""
        return ((self.times - pd.Timestamp(self.tws_start)).days / 365.25).values

    @property
    def pixel_lat(self) -> np.ndarray:
        return np.repeat(self.lat, len(self.lon))[self.pixels]

    @property
    def pixel_lon(self) -> np.ndarray:
        return np.tile(self.lon, len(self.lat))[self.pixels]

    @property
    def area_weight(self) -> np.ndarray:
        return np.cos(np.deg2rad(self.pixel_lat))

    def period(self, start: str, end: str | None = None) -> np.ndarray:
        """Boolean time mask for start..end inclusive (end defaults to the last aligned month)."""
        last = pd.Timestamp(end) if end else self.times[-1]
        return np.asarray((self.times >= pd.Timestamp(start)) & (self.times <= last))

    def split_rows(self, splits: dict) -> dict[str, np.ndarray]:
        return {name: self.period(bounds["start"], bounds["end"]) for name, bounds in splits.items()}

    def to_map(self, values: np.ndarray, name: str | None = None) -> xr.DataArray:
        full = np.full(len(self.lat) * len(self.lon), np.nan)
        full[self.pixels] = values
        return xr.DataArray(
            full.reshape(len(self.lat), len(self.lon)), coords={"lat": self.lat, "lon": self.lon}, dims=("lat", "lon"), name=name
        )

    def to_cube(self, values: np.ndarray) -> xr.DataArray:
        full = np.full((len(self.times), len(self.lat) * len(self.lon)), np.nan, np.float32)
        full[:, self.pixels] = values
        return xr.DataArray(
            full.reshape(len(self.times), len(self.lat), len(self.lon)),
            coords={"time": self.times, "lat": self.lat, "lon": self.lon},
            dims=("time", "lat", "lon"),
        )

    def region_index(self, bounds: dict) -> np.ndarray:
        lat, lon = self.pixel_lat, self.pixel_lon
        inside = (lat >= bounds["lat"][0]) & (lat <= bounds["lat"][1]) & (lon >= bounds["lon"][0]) & (lon <= bounds["lon"][1])
        return np.flatnonzero(inside)

    def region_mean(self, values: np.ndarray, bounds: dict) -> np.ndarray:
        """Area-weighted regional mean series; NaN if the region holds no valid pixels."""
        index = self.region_index(bounds)
        if len(index) == 0:
            return np.full(values.shape[0], np.nan)
        weights = self.area_weight[index]
        return np.nansum(values[:, index] * weights, 1) / np.sum(weights * np.isfinite(values[:, index]), 1)


def training_anomalies(cube: PixelCube, train_rows: np.ndarray) -> tuple[dict, dict]:
    """Training-period monthly climatologies and the anomalies they define for TWS, P, and T."""
    climatologies, anomalies = {}, {}
    for name in ("tws", "precip", "t2m"):
        values = getattr(cube, name)
        climatologies[name] = monthly_climatology(values, train_rows, cube.month)
        anomalies[name] = deseasonalize(values, climatologies[name], cube.month)
    return climatologies, anomalies


def align_products(
    tws: xr.DataArray, land: xr.DataArray, precip: xr.DataArray, t2m: xr.DataArray, config: dict
) -> xr.Dataset:
    """Put TWS, P, and T on one calendar and mask, and add training-climatology anomalies.

    Mask: land, |lat| <= latitude limit, finite P and T in every month, and at least
    min_tws_fraction of GRACE months observed. TWS stays NaN before tws_start and in
    GRACE gaps, including the GRACE -> GRACE-FO gap.
    """
    data = config["data"]
    last = min(pd.Timestamp(series.time.values[-1]) for series in (tws, precip, t2m))
    last = min(last, pd.Timestamp(data["end"]))
    times = pd.date_range(data["driver_start"], last, freq="MS")
    limit = data["latitude_limit_deg"]
    aligned = xr.Dataset(
        {"tws": tws.reindex(time=times), "precip": precip.reindex(time=times), "t2m": t2m.reindex(time=times)}
    ).sel(lat=slice(-limit, limit))
    aligned["tws"] = aligned["tws"].where(aligned.time >= pd.Timestamp(data["tws_start"]))
    tws_fraction = aligned["tws"].sel(time=slice(data["tws_start"], None)).notnull().mean("time")
    aligned["mask"] = (
        land.sel(lat=aligned.lat, lon=aligned.lon)
        & aligned["precip"].notnull().all("time")
        & aligned["t2m"].notnull().all("time")
        & (tws_fraction >= data["min_tws_fraction"])
    )
    train = config["splits"]["train"]
    aligned.attrs.update(
        description="Group 5 aligned GRACE/GRACE-FO TWS, IMERG precipitation, MERRA-2 T2M",
        anomalies=f"monthly climatology over training period {train['start'][:7]}–{train['end'][:7]}",
        grid_deg=data["resolution_deg"],
        tws_start=data["tws_start"],
    )

    cube = PixelCube.from_dataset(aligned)
    _, anomalies = training_anomalies(cube, cube.period(train["start"], train["end"]))
    for name in ("precip", "t2m", "tws"):
        aligned[f"{name}_anom"] = cube.to_cube(anomalies[name])
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/p_t_tws_config.yaml"), help="Driver/baseline configuration"
    )
    parser.add_argument("--output", required=True, type=Path, help="Aligned TWS/P/T NetCDF path")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Earthdata credentials file, used only when sources must be downloaded (default: .env)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    resolution = config["data"]["resolution_deg"]
    with xr.open_dataset(grace_source(config, args.env_file)) as source:
        tws, land = grace_on_target_grid(source, resolution)
        tws, land = tws.load(), land.load()
    precip = load_driver_cube("imerg", config, args.env_file)
    t2m = load_driver_cube("merra2", config, args.env_file)
    for name, series in (("GRACE", tws), ("IMERG", precip), ("MERRA-2", t2m)):
        print(f"{name}: {pd.Timestamp(series.time.values[0]):%Y-%m} … {pd.Timestamp(series.time.values[-1]):%Y-%m}")

    aligned = align_products(tws, land, precip, t2m, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_netcdf(args.output, engine="netcdf4", encoding=compressed_float_encoding(aligned))
    times = pd.DatetimeIndex(aligned.time.values)
    print(f"Months: {times[0]:%Y-%m} … {times[-1]:%Y-%m} ({len(times)}) | valid land pixels: {int(aligned['mask'].sum())}")
    print(f"Saved aligned dataset: {args.output}")


if __name__ == "__main__":
    main()
