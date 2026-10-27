"""Prepare NASA JPL GRACE/GRACE-FO mascon data for multimodal integration.

The source NetCDF is read-only. This script does not fit scalers, remove ocean
cells, interpolate measurements, or construct future-dependent model features.

Examples:
    # Process an already-downloaded NASA source file.
    python grace_fo_processing.py --input /path/to/original.nc \
        --output /path/to/grace_fo_processed.nc

    # Download a source copy using credentials in .env, then process it.
    python grace_fo_processing.py --download-dir /path/to/data/raw \
        --output /path/to/data/processed/grace_fo_processed.nc
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import xarray as xr


REQUIRED = {
    "lwe_thickness",
    "uncertainty",
    "land_mask",
    "mascon_ID",
    "scale_factor",
    "GAD",
}
COLLECTION = "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4"


def download_source(download_dir: Path, env_file: Path, start: str, end: str) -> Path:
    """Download the source NetCDF without changing an existing source file."""
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError(
            "Downloading requires python-dotenv. Install it with: "
            "python -m pip install python-dotenv"
        ) from error

    if not env_file.is_file():
        raise FileNotFoundError(
            f"Credentials file not found: {env_file}. "
            "Copy .env.example to .env and set EARTHDATA_USERNAME and EARTHDATA_PASSWORD."
        )

    load_dotenv(env_file)
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD in the credentials file."
        )

    downloader = shutil.which("podaac-data-downloader")
    if downloader is None:
        raise RuntimeError(
            "podaac-data-downloader is not installed. Install it with: "
            "python -m pip install podaac-data-subscriber"
        )

    download_dir.mkdir(parents=True, exist_ok=True)
    command = [
        downloader,
        "-c", COLLECTION,
        "-d", str(download_dir),
        "--start-date", f"{start}T00:00:00Z",
        "--end-date", f"{end}T23:59:59Z",
        "-e", ".nc",
    ]
    # Give the downloader an isolated, permissions-restricted credentials file.
    with tempfile.TemporaryDirectory(prefix="earthdata_") as home:
        netrc_path = Path(home) / ".netrc"
        netrc_path.write_text(
            "machine urs.earthdata.nasa.gov\n"
            f"login {username}\n"
            f"password {password}\n"
        )
        netrc_path.chmod(0o600)
        environment = os.environ.copy()
        environment["HOME"] = home
        environment["NETRC"] = str(netrc_path)
        subprocess.run(command, check=True, env=environment)

    files = sorted(download_dir.glob("*.nc"))
    if len(files) != 1:
        raise RuntimeError(
            f"Expected exactly one downloaded NetCDF file in {download_dir}, found {len(files)}."
        )
    return files[0]


def prepare(source: xr.Dataset, start: str, end: str) -> xr.Dataset:
    missing_vars = REQUIRED - set(source.data_vars)
    if missing_vars:
        raise ValueError(f"Missing required NASA variables: {sorted(missing_vars)}")
    if not {"time", "lat", "lon"}.issubset(source.coords):
        raise ValueError("Expected time, lat, lon coordinates")

    result = source.sel(time=slice(start, end)).copy()
    if result.sizes["time"] == 0:
        raise ValueError("No GRACE observations in the requested period")

    original_times = pd.DatetimeIndex(result.time.values)
    months = original_times.to_period("M")
    if months.has_duplicates:
        raise ValueError("Multiple GRACE observations occur in the same calendar month")
    expected = pd.period_range(pd.Period(start[:7], freq="M"), pd.Period(end[:7], freq="M"), freq="M")
    gaps = expected.difference(months)
    if len(gaps):
        raise ValueError(f"Missing months: {list(gaps.astype(str))}. Choose a continuous interval or handle gaps explicitly.")

    # Source bounds remain intact; only the time coordinate used for joins is relabeled.
    result = result.assign_coords(time=months.to_timestamp())
    result["original_time"] = xr.DataArray(original_times.values, dims=("time",), coords={"time": result.time})
    result["original_time"].attrs["description"] = "Original monthly observation time from NASA source NetCDF"

    # Re-label and sort longitude, without altering any measurement values.
    longitude = ((result.lon + 180) % 360) - 180
    result = result.assign_coords(lon=longitude).sortby("lon")

    result.attrs = dict(source.attrs)
    result.attrs.update({
        "preprocessing_note": "Calendar month labels and -180..180 sorted longitude; no imputation, scaling, spatial interpolation, or filtering.",
        "source_time_coverage_selected": f"{start} to {end}",
        "modeling_note": "Use land_mask for selecting terrestrial cells. Fit scaling only on training data. GRACE grid is not independent 0.5-degree measurements.",
    })
    return result


def land_modeling_view(dataset: xr.Dataset) -> xr.Dataset:
    """Derive, but do not substitute for, the full processed GRACE-FO dataset."""
    land = dataset["land_mask"] == 1
    view = dataset[["lwe_thickness", "uncertainty", "land_mask", "mascon_ID", "GAD", "scale_factor"]].copy()
    for variable in ("lwe_thickness", "uncertainty", "GAD", "scale_factor"):
        view[variable] = view[variable].where(land)
    view.attrs = dict(dataset.attrs)
    view.attrs["modeling_view_note"] = (
        "Land-only view derived with NASA land_mask == 1; the full master dataset is unchanged."
    )
    return view


def data_quality_summary(dataset: xr.Dataset) -> dict[str, dict[str, int]]:
    """Count missing and infinite values without altering the scientific data."""
    summary = {}
    for name, variable in dataset.data_vars.items():
        if np.issubdtype(variable.dtype, np.number):
            values = variable.values
            summary[name] = {
                "missing": int(np.isnan(values).sum()),
                "infinite": int(np.isinf(values).sum()),
            }
    return summary


def compressed_float_encoding(dataset: xr.Dataset) -> dict[str, dict[str, int]]:
    return {
        name: {"zlib": True, "complevel": 4}
        for name in dataset.data_vars
        if np.issubdtype(dataset[name].dtype, np.floating)
    }


def verify(result: xr.Dataset, source: xr.Dataset, start: str, end: str) -> None:
    expected_months = len(pd.period_range(start[:7], end[:7], freq="M"))
    assert result.sizes["time"] == expected_months
    assert result.time.to_index().is_unique
    assert result.time.to_index().is_monotonic_increasing
    assert result.lon.to_index().is_monotonic_increasing
    assert float(result.lon.min()) >= -180 and float(result.lon.max()) < 180
    assert result["lwe_thickness"].attrs.get("units") == "cm"
    assert result["lwe_thickness"].dims == ("time", "lat", "lon")
    original_selected = source.sel(time=slice(start, end))
    # Compare arrays after the same spatial longitude permutation.
    original_sorted = original_selected.assign_coords(
        lon=((original_selected.lon + 180) % 360) - 180
    ).sortby("lon")
    assert set(source.data_vars).issubset(set(result.data_vars))
    for name in source.data_vars:
        np.testing.assert_array_equal(result[name].values, original_sorted[name].values)
    np.testing.assert_array_equal(result["original_time"].values, original_selected.time.values)
    land = result["land_mask"] == 1
    missing_land = int((result["lwe_thickness"].isnull() & land).sum().item())
    if missing_land:
        print(f"WARNING: {missing_land} missing TWSA land observations; no imputation performed")
    for name, counts in data_quality_summary(result).items():
        print(f"Quality {name}: missing={counts['missing']} | infinite={counts['infinite']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Existing original NASA GRACE NetCDF path")
    source.add_argument(
        "--download-dir",
        type=Path,
        help="Directory where the original NASA NetCDF will be downloaded",
    )
    parser.add_argument("--output", required=True, type=Path, help="New integration-ready NetCDF path")
    parser.add_argument(
        "--land-view-output",
        type=Path,
        help="Optional separate land-only GRACE-FO modeling-view NetCDF path",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Earthdata credentials file used only with --download-dir (default: .env)",
    )
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    if args.download_dir:
        args.input = download_source(args.download_dir, args.env_file, args.start, args.end)
    elif not args.input.is_file():
        parser.error(f"Input not found: {args.input}")

    if args.input.resolve() == args.output.resolve():
        parser.error("Output must be different from the original NASA input")
    if args.land_view_output and args.land_view_output.resolve() in {
        args.input.resolve(), args.output.resolve()
    }:
        parser.error("Land-view output must differ from both input and full processed output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.land_view_output:
        args.land_view_output.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(args.input) as source:
        ready = prepare(source, args.start, args.end)
        verify(ready, source, args.start, args.end)
        ready.to_netcdf(args.output, engine="netcdf4", encoding=compressed_float_encoding(ready))
        if args.land_view_output:
            land_view = land_modeling_view(ready)
            land_view.to_netcdf(
                args.land_view_output,
                engine="netcdf4",
                encoding=compressed_float_encoding(land_view),
            )

    with xr.open_dataset(args.output) as saved:
        assert saved.sizes["time"] == ready.sizes["time"]
        assert set(saved.data_vars) == set(ready.data_vars)
        assert saved.time.to_index().equals(ready.time.to_index())
        assert saved.lon.to_index().equals(ready.lon.to_index())
    print(f"Validated: {args.output}")
    print(f"Months: {ready.sizes['time']} | Grid: {ready.sizes['lat']} x {ready.sizes['lon']}")
    print(f"Variables: {list(ready.data_vars)}")
    print(f"File size: {args.output.stat().st_size / (1024 ** 2):.1f} MiB")
    if args.land_view_output:
        print(f"Land-only modeling view: {args.land_view_output}")


if __name__ == "__main__":
    main()
