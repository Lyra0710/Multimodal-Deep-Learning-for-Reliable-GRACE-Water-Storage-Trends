"""Train a shared temporal self-attention model for one-month-ahead TWS."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import xarray as xr


HISTORY = 12
SEED = 42
BATCH = 512
WIDTH = 24


class TemporalTransformer(nn.Module):
    """Attend over a year of monthly tokens and predict a change in TWS."""
    def __init__(self, features):
        super().__init__()
        self.project = nn.Linear(features, WIDTH)
        self.position = nn.Parameter(torch.randn(HISTORY, WIDTH) * 0.02)
        self.attention = nn.MultiheadAttention(WIDTH, 4, batch_first=True)
        self.norm = nn.LayerNorm(WIDTH)
        self.feedforward = nn.Sequential(nn.Linear(WIDTH, 48), nn.GELU(), nn.Linear(48, WIDTH))
        self.readout = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, 1))

    def forward(self, x):
        tokens = self.project(x) + self.position
        last = tokens[:, -1:]
        context, _ = self.attention(last, tokens, tokens, need_weights=False)
        state = self.norm((last + context).squeeze(1))
        return x[:, -1, 0] + self.readout(state + self.feedforward(state)).squeeze(-1)


def load_data(path, combined):
    variables = ['tws', 'precip', 't2m'] if combined else ['tws']
    with xr.open_dataset(path) as ds:
        dates = pd.DatetimeIndex(ds.time.values).to_period('M')
        if not dates.is_unique or not dates.is_monotonic_increasing:
            raise ValueError('Expected ordered, unique calendar months')
        dates = pd.period_range('2018-01', min(dates[-1], pd.Period('2025-12', 'M')), freq='M').to_timestamp()
        ds = ds.assign_coords(time=pd.DatetimeIndex(ds.time.values).to_period('M').to_timestamp()).reindex(time=dates)
        land = (ds['mask'].fillna(0) > 0) & (abs(ds.lat) <= 60)
        land = land.transpose('lat', 'lon').values
        values = np.stack([ds[v].transpose('time', 'lat', 'lon').values[:, land] for v in variables], axis=-1)
        coords = dict(lat=ds.lat.values, lon=ds.lon.values)
    train = (dates.year >= 2019) & (dates.year <= 2022)
    mean = np.nanmean(values[train], axis=(0, 1), dtype=np.float64).astype('float32')
    scale = np.nanstd(values[train], axis=(0, 1), dtype=np.float64).astype('float32')
    if not np.all(np.isfinite(mean) & np.isfinite(scale) & (scale > 0)):
        raise ValueError('Training variables need finite, nonzero scales')
    z = (values - mean) / scale
    return dates, land, coords, variables, z, mean, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--combined', action='store_true', help='Add historical precipitation and temperature')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rng = np.random.default_rng(SEED)
    dates, land, coords, variables, z, mean, scale = load_data(args.data, args.combined)
    month_angle = 2 * np.pi * (dates.month.to_numpy() - 1) / 12
    calendar = np.stack([np.sin(month_angle), np.cos(month_angle)], axis=1).astype('float32')
    offsets = np.arange(-HISTORY, 0)
    valid_history = np.zeros(z.shape[:2], dtype=bool)
    for t in range(HISTORY, len(z)):
        valid_history[t] = np.isfinite(z[t-HISTORY:t]).all(axis=(0, 2))

    def examples(t, cell):
        history = t[:, None] + offsets
        inputs = np.concatenate([z[history, cell[:, None]], calendar[history]], axis=-1)
        return torch.from_numpy(inputs)

    def split(year_min, year_max):
        eligible = valid_history & np.isfinite(z[:, :, 0])
        eligible &= ((dates.year >= year_min) & (dates.year <= year_max))[:, None]
        pairs = np.column_stack(np.where(eligible))
        if len(pairs) == 0:
            raise ValueError(f'No eligible targets in {year_min}–{year_max}')
        return pairs

    train = split(2019, 2022)
    validation = split(2023, 2023)
    validation = validation[np.random.default_rng(SEED).choice(len(validation), min(16384, len(validation)), replace=False)]
    model = TemporalTransformer(len(variables) + 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    log = []
    best = np.inf
    for epoch in range(1, 13):
        model.train()
        selected = rng.choice(len(train), min(32768, len(train)), replace=False)
        losses = []
        for start in range(0, len(selected), BATCH):
            t, c = train[selected[start:start+BATCH]].T
            prediction = model(examples(t, c))
            loss = nn.functional.mse_loss(prediction, torch.from_numpy(z[t, c, 0]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.extend([loss.item()] * len(t))
        model.eval()
        errors = []
        with torch.no_grad():
            for start in range(0, len(validation), BATCH):
                t, c = validation[start:start+BATCH].T
                errors.extend((model(examples(t, c)).numpy() - z[t, c, 0]) ** 2)
        score = float(np.sqrt(np.mean(errors)) * scale[0])
        log.append(dict(epoch=epoch, train_rmse_cm=float(np.sqrt(np.mean(losses))*scale[0]), validation_rmse_cm=score))
        if score < best:
            best = score
            best_epoch = epoch
            torch.save(model.state_dict(), out / 'model.pt')
    model.load_state_dict(torch.load(out / 'model.pt', weights_only=True))
    test = np.flatnonzero(dates.year >= 2024)
    if not len(test):
        raise ValueError('No test dates available')
    pred = np.full((len(test), *land.shape), np.nan, dtype='float32')
    truth = np.full_like(pred, np.nan)
    with torch.no_grad():
        for row, t in enumerate(test):
            cells = np.flatnonzero(valid_history[t])
            estimates = np.full(z.shape[1], np.nan, dtype='float32')
            for start in range(0, len(cells), BATCH):
                c = cells[start:start+BATCH]
                estimates[c] = model(examples(np.full(len(c), t), c)).numpy()*scale[0] + mean[0]
            pred[row, land] = estimates
            truth[row, land] = z[t, :, 0]*scale[0] + mean[0]
    dims = ('time', 'lat', 'lon')
    result = xr.Dataset({k: (dims, a, {'units': 'cm'}) for k, a in [('prediction', pred), ('actual_tws', truth)]},
                        coords=dict(time=dates[test], **coords))
    result.to_netcdf(out/'predictions.nc', engine='h5netcdf', encoding={k: {'zlib': True} for k in result.data_vars})
    pd.DataFrame(log).to_csv(out/'learning_curve.csv', index=False)
    config = dict(seed=SEED, history_months=HISTORY, variables=variables, calendar=['sin_month','cos_month'],
                  train='2019-01/2022-12', validation='2023-01/2023-12', test=[str(dates[test[0]].date()), str(dates[test[-1]].date())],
                  training_pairs=len(train), sampled_per_epoch=len(selected), validation_sample=len(validation),
                  selected_epoch=best_epoch, mean=mean.tolist(), scale=scale.tolist(), torch_version=str(torch.__version__))
    (out/'run.json').write_text(json.dumps(config, indent=2)+'\n')


if __name__ == '__main__':
    main()
