"""Recompute the group evaluation convention from archived common-mask predictions."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from combined_ridge import evaluate_per_cell


def score(root):
    root = Path(root)
    with xr.open_dataset(root / 'test_predictions.nc') as ds:
        common = ds.test_eval_mask.transpose('time', 'lat', 'lon').values.astype(bool)
        common &= (np.abs(ds.lat.values) <= 60)[None, :, None]
        common = common.reshape(ds.sizes['time'], -1)
        y = ds.actual_tws.transpose('time', 'lat', 'lon').values.reshape(common.shape).astype(float)
        weights = np.repeat(np.cos(np.deg2rad(ds.lat.values)), ds.sizes['lon'])
        rows = []
        for name in ['ridge', 'persistence', 'seasonal_persistence', 'climatology']:
            pred = ds[name].transpose('time', 'lat', 'lon').values.reshape(common.shape).astype(float)
            if not np.isfinite(y[common]).all() or not np.isfinite(pred[common]).all():
                raise ValueError('Non-finite observation or prediction within the evaluation mask')
            rows.append(dict(model=name, **evaluate_per_cell(y, pred, common, weights)))
    frame = pd.DataFrame(rows)
    frame.to_csv(root / 'per_cell_test_metrics.csv', index=False)
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parents[1])
    print(score(parser.parse_args().results_dir).to_string(index=False))
