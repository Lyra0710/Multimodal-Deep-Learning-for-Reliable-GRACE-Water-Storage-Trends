"""Check coordinate alignment, missing pairs, and metric aggregation."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from evaluate import score


class ComparisonTest(unittest.TestCase):
    def test_csv_and_reordered_grid(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            coords = dict(time=pd.date_range('2024-01-01', periods=3, freq='MS'), lat=[0., 60.], lon=[5.])
            truth = xr.DataArray(np.arange(6.).reshape(3, 2, 1), coords=coords, dims=list(coords))
            pred = truth.copy()
            pred.values[:, 0, 0] += [0., 0., 3.]
            pred.values[-1, 1, 0] = np.nan
            truth.to_dataset(name='actual_tws').to_netcdf(root/'truth.nc', engine='h5netcdf')
            pred.isel(lat=[1, 0]).to_dataset(name='pred').to_netcdf(root/'model.nc', engine='h5netcdf')
            pred.to_dataframe(name='pred').dropna().to_csv(root/'model.csv')
            rows, coverage = score(root/'truth.nc', [(ext, root/f'model.{ext}', 'pred') for ext in ['nc','csv']])
            np.testing.assert_allclose(rows.area_weighted_mean_cell_rmse_cm, np.sqrt(3)/1.5)
            np.testing.assert_allclose(rows.rmse_cm, 1.5)
            np.testing.assert_array_equal(coverage.common_cells, [2, 2, 1])


if __name__ == '__main__':
    unittest.main()
