"""Score gridded NetCDF and long CSV predictions on one common set of targets."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

# Use the same metric implementation as the published Ridge baseline.
RIDGE = Path(__file__).resolve().parents[1] / 'combined_dataset_analysis_ridge' / 'scripts'
sys.path.insert(0, str(RIDGE))
from combined_ridge import evaluate, evaluate_per_cell


def monthly(array):
    array = array.transpose('time', 'lat', 'lon')
    dates = pd.DatetimeIndex(array.time.values).to_period('M').to_timestamp()
    array = array.assign_coords(time=dates)
    for dim in array.dims:
        if not array.get_index(dim).is_unique:
            raise ValueError(f'Duplicate {dim} coordinates')
    return array


def read_prediction(path, variable, reference):
    if path.suffix.lower() == '.csv':
        keys = ['time', 'lat', 'lon']
        frame = pd.read_csv(path, usecols=keys + [variable])
        frame['time'] = pd.to_datetime(frame.time).dt.to_period('M').dt.to_timestamp()
        frame = frame.set_index(keys)
        if frame.index.has_duplicates:
            raise ValueError(f'Duplicate cell-month in {path}')
        array = frame[variable].to_xarray()
    else:
        with xr.open_dataset(path) as ds:
            array = ds[variable].load()
    array = monthly(array)
    # Coordinates are labels. Extra dates or a wider domain are allowed.
    for dim in array.dims:
        if not len(array.get_index(dim).intersection(reference.get_index(dim))):
            raise ValueError(f'{path} has no matching {dim} coordinates')
    return array.reindex_like(reference).values


def score(reference_file, specs):
    with xr.open_dataset(reference_file) as ds:
        actual = monthly(ds.actual_tws).load()
        allowed = monthly(ds.test_eval_mask).fillna(0).values > 0 if 'test_eval_mask' in ds else np.ones(actual.shape, bool)
    allowed &= (abs(actual.lat.values) <= 60)[None, :, None]
    mask = allowed & np.isfinite(actual.values)
    predictions = {}
    for name, path, var in specs:
        if name in predictions:
            raise ValueError(f'Duplicate model name {name}')
        predictions[name] = read_prediction(Path(path), var, actual)
        mask &= np.isfinite(predictions[name])
    valid_cells = mask.any(axis=0)
    if not valid_cells.any():
        raise ValueError('No common cell-months across the supplied models')
    weights = np.broadcast_to(np.cos(np.deg2rad(actual.lat.values))[:, None], valid_cells.shape)[valid_cells]
    truth = actual.values[:, valid_cells].astype(float)
    rows = []
    for name, cube in predictions.items():
        pred = cube[:, valid_cells].astype(float)
        common = mask[:, valid_cells]
        rows.append(dict(model=name, **evaluate_per_cell(truth, pred, common, weights),
                         **evaluate(truth, pred, common, weights)))
    coverage = pd.DataFrame({'target_month': actual.time.values, 'reference_cells': (allowed & np.isfinite(actual)).sum(axis=(1, 2)),
                             'common_cells': mask.sum(axis=(1, 2))})
    return pd.DataFrame(rows), coverage


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path, help='Test-period NetCDF with actual_tws in cm over land')
    parser.add_argument('output', type=Path)
    parser.add_argument('--model', nargs=3, action='append', required=True, metavar=('NAME', 'FILE', 'VARIABLE'))
    args = parser.parse_args()
    metrics, coverage = score(args.reference, args.model)
    args.output.mkdir(exist_ok=True, parents=True)
    metrics.to_csv(args.output/'metrics.csv', index=False)
    coverage.to_csv(args.output/'coverage.csv', index=False)
    print(metrics.to_string(index=False))
