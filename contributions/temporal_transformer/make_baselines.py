"""Generate persistence and climatology baselines on the shared training mask."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def make_baselines(source, output):
    with xr.open_dataset(source) as ds:
        if ds.tws.attrs.get('units') != 'cm':
            raise ValueError('Expected tws target in cm')
        land = (ds.training_mask.fillna(0) > 0) & (abs(ds.lat) <= 60)
        tws = ds.tws.where(land).sel(time=slice('2019-01', '2025-12')).load()
    months = pd.DatetimeIndex(tws.time.values).to_period('M')
    if months.has_duplicates or not months.is_monotonic_increasing:
        raise ValueError('Expected ordered, unique calendar months')
    complete_calendar = pd.period_range('2019-01', months[-1], freq='M').to_timestamp()
    tws = tws.assign_coords(time=months.to_timestamp()).reindex(time=complete_calendar)
    seasonal_cycle = tws.sel(time=slice('2019-01', '2022-12')).groupby('time.month').mean('time')
    seasonal = seasonal_cycle.sel(month=tws.time.dt.month).drop_vars('month')
    previous = tws.shift(time=1)
    fields = xr.Dataset({
        'actual_tws': tws,
        'persistence': previous,
        'seasonal_persistence': previous - seasonal.shift(time=1) + seasonal,
        'climatology': seasonal,
    }).sel(time=slice('2024-01', '2025-12'))
    fields['test_eval_mask'] = np.isfinite(fields.actual_tws)
    for name in ('actual_tws', 'persistence', 'seasonal_persistence', 'climatology'):
        fields[name].attrs = {'units': 'cm'}
    fields.attrs = {'selection_mask': 'training_mask', 'climatology_fit': '2019-01/2022-12'}
    output.parent.mkdir(parents=True, exist_ok=True)
    fields.to_netcdf(output, engine='h5netcdf', encoding={name: {'zlib': True} for name in fields})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    make_baselines(args.data, args.output)
