"""Revanth's combined-data EDA and retrospective one-month Ridge baseline.

Input: Rachan's aligned_0.5deg.nc (tws [cm], precip [mm/month], t2m [degC], mask).
No downloads, fabricated observations, random temporal splits, or gap interpolation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['tws_lag1', 'precip_lag1', 'precip_sum3_lag1',
            'precip_sum12_lag1', 'temperature_lag1', 'month_sin', 'month_cos', 'time_years']
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


def prepare(ds):
    """Validate the handoff and restore calendar gaps before constructing lags."""
    for name in ['tws', 'precip', 't2m', 'mask']:
        if name not in ds:
            raise ValueError(f'Missing {name}; supply the aligned combined dataset, not GRACE alone.')
    accepted = {'tws': {'cm'}, 'precip': {'mm/month', 'mm month-1'},
                't2m': {'degC', '°C', 'C', 'degrees_Celsius'}}
    for name, units in accepted.items():
        if set(ds[name].dims) != {'time', 'lat', 'lon'}:
            raise ValueError(f'{name} must have time, lat, lon dimensions')
        if ds[name].attrs.get('units') not in units:
            raise ValueError(f'{name}: expected units {units}, got {ds[name].attrs.get("units")}')
    if set(ds['mask'].dims) != {'lat', 'lon'}:
        raise ValueError('mask must have lat, lon dimensions')
    for coord in ['lat', 'lon']:
        v = ds[coord].values
        if len(np.unique(v)) != len(v) or not np.isfinite(v).all():
            raise ValueError(f'{coord} must contain unique finite coordinates')
        if len(v) > 2 and not np.allclose(np.diff(np.sort(v)), np.diff(np.sort(v))[0]):
            raise ValueError('Area weights require a regular latitude/longitude grid')
    if np.any(np.abs(ds.lat.values) >= 90):
        raise ValueError('Latitude cell centers must lie between -90 and 90')
    periods = pd.DatetimeIndex(ds.time.values).to_period('M')
    if periods.has_duplicates:
        raise ValueError('Duplicate calendar months; resolve them in the upstream handoff')
    ds = ds[['tws', 'precip', 't2m', 'mask']].assign_coords(time=periods.to_timestamp()).sortby('time')
    static_mask = ds['mask'].fillna(0) > 0
    full = pd.date_range(ds.time.values[0], ds.time.values[-1], freq='MS')
    ds = ds.reindex(time=full)
    ds['mask'] = static_mask
    return ds


def split_masks(time, config):
    dates = pd.DatetimeIndex(time)
    masks = {}
    for name in ['train', 'validation', 'test']:
        b = config['splits'][name]
        masks[name] = np.asarray((dates >= b['start']) & (dates <= b['end']))
        if not masks[name].any():
            raise ValueError(f'Input has no {name} months')
    if np.any(sum(masks.values()) > 1):
        raise ValueError('Split boundaries overlap')
    if not dates[masks['train']][-1] < dates[masks['validation']][0] < dates[masks['test']][0]:
        raise ValueError('Splits must be chronological')
    if dates[masks['validation']][-1] >= dates[masks['test']][0]:
        raise ValueError('Validation must end before testing')
    return masks


def features(tws, precip, temp, time):
    """All observed features end at t-1; calendar features are known in advance."""
    def lag(a):
        return np.concatenate([np.full(1, np.nan), a[:-1]])
    previous_p = pd.Series(lag(precip))
    month = pd.DatetimeIndex(time).month.to_numpy()
    years = (pd.DatetimeIndex(time) - pd.Timestamp('2019-01-01')).days.to_numpy() / 365.25
    return np.column_stack([lag(tws), previous_p,
        previous_p.rolling(3, min_periods=3).sum(),
        previous_p.rolling(12, min_periods=12).sum(), lag(temp),
        np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12), years])


def fit_model(X, y, rows, alpha):
    use = rows & np.isfinite(y) & np.isfinite(X).all(axis=1)
    if use.sum() < 24:
        return None
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(X[use], y[use])
    return model


def predict(model, X):
    out = np.full(len(X), np.nan)
    good = np.isfinite(X).all(axis=1)
    if model is not None and good.any():
        out[good] = model.predict(X[good])
    return out


def climatology(y, month, rows):
    result = np.full(12, np.nan)
    for m in range(1, 13):
        v = y[rows & (month == m)]
        v = v[np.isfinite(v)]
        if len(v):
            result[m-1] = v.mean()
    return result


def correlation(a, b):
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3 or np.std(a[good]) == 0 or np.std(b[good]) == 0:
        return np.nan
    return float(np.corrcoef(a[good], b[good])[0, 1])


def evaluate(y, pred, common, weights):
    """Pooled area-weighted errors; area-weighted mean within-cell temporal r."""
    w = common * weights[None, :]
    denom = w.sum()
    if not denom:
        raise ValueError('No common evaluation observations')
    err = np.where(common, pred-y, 0)
    r = np.array([correlation(np.where(common[:, j], y[:, j], np.nan),
                              np.where(common[:, j], pred[:, j], np.nan))
                  for j in range(y.shape[1])])
    ok = np.isfinite(r)
    return dict(mae_cm=float((np.abs(err)*w).sum()/denom),
                rmse_cm=float(np.sqrt((err**2*w).sum()/denom)),
                area_weighted_mean_temporal_pearson_r=float(np.average(r[ok], weights=weights[ok])) if ok.any() else np.nan,
                evaluated_months=int(common.any(axis=1).sum()),
                evaluated_land_cells=int(common.any(axis=0).sum()),
                valid_land_cell_month_observations=int(common.sum()))


def weighted_quantile(values, weights, probs):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    return np.interp(probs, (np.cumsum(w)-0.5*w)/w.sum(), v)


def eda(arrays, time, masks, weights, out):
    """Descriptive correlations use training only; split distributions expose shift."""
    month = pd.DatetimeIndex(time).month.to_numpy()
    anomaly = {}
    for name, a in arrays.items():
        clim = np.column_stack([climatology(a[:, j], month, masks['train']) for j in range(a.shape[1])])
        anomaly[name] = a - clim[month-1]
    rows, histograms = [], []
    for name, a in arrays.items():
        for split, select in masks.items():
            v = a[select].ravel()
            w = np.broadcast_to(weights, a[select].shape).ravel()
            good = np.isfinite(v)
            v, w = v[good], w[good]
            if not len(v):
                continue
            q = weighted_quantile(v, w, [0.05, 0.25, 0.5, 0.75, 0.95])
            mean = np.average(v, weights=w)
            rows.append(dict(variable=name, split=split, observations=len(v),
                missing_fraction=float(1-good.mean()), area_weighted_mean=mean,
                area_weighted_std=np.sqrt(np.average((v-mean)**2, weights=w)),
                **dict(zip(['p05', 'p25', 'median', 'p75', 'p95'], q))))
        valid_all = a[np.isfinite(a)]
        if len(valid_all):
            # Shared edges across displayed splits are visualization only, never model inputs.
            edges = np.histogram_bin_edges(valid_all, bins=60)
            for split, select in masks.items():
                v = a[select].ravel(); w = np.broadcast_to(weights, a[select].shape).ravel()
                good = np.isfinite(v)
                h, _ = np.histogram(v[good], bins=edges, weights=w[good])
                total = h.sum()
                for left, right, count in zip(edges[:-1], edges[1:], h):
                    histograms.append(dict(variable=name, split=split, left=left, right=right,
                                           area_weighted_fraction=count/total if total else np.nan))
    pd.DataFrame(rows).to_csv(out/'distributions.csv', index=False)
    pd.DataFrame(histograms).to_csv(out/'histograms.csv', index=False)
    corrs, summary = {}, []
    for kind, source in [('raw', arrays), ('seasonal_anomaly', anomaly)]:
        for a, b in [('tws','precip'), ('tws','t2m'), ('precip','t2m')]:
            r = np.array([correlation(source[a][masks['train'], j], source[b][masks['train'], j])
                          for j in range(len(weights))])
            key = f'{kind}_{a}_{b}'
            corrs[key] = r
            good = np.isfinite(r)
            summary.append(dict(relationship=key, period='train', eligible_cells=int(good.sum()),
                area_weighted_mean_r=np.average(r[good], weights=weights[good]) if good.any() else np.nan))
    pd.DataFrame(summary).to_csv(out/'correlations.csv', index=False)
    return corrs


def run(input_path, config_path, output_dir, alphas=ALPHAS):
    input_path, config_path, out = Path(input_path), Path(config_path), Path(output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f'{input_path} is missing. Obtain aligned_0.5deg.nc from Rachan; '
                                'the supplied notebook ZIP does not contain that output.')
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError('Ridge alphas must be finite and positive')
    config = yaml.safe_load(config_path.read_text())
    with xr.open_dataset(input_path) as source:
        ds = prepare(source).load()
    time = pd.DatetimeIndex(ds.time.values)
    masks = split_masks(time, config)
    # Keep only dates relevant to this experiment plus 12 months of feature history.
    keep = (time >= pd.Timestamp(config['splits']['train']['start'])-pd.DateOffset(months=12)) & (time <= config['splits']['test']['end'])
    ds = ds.isel(time=np.flatnonzero(keep)); time = pd.DatetimeIndex(ds.time.values)
    masks = split_masks(time, config)
    mask = ds['mask'].transpose('lat','lon').values
    pixels = np.flatnonzero(mask.ravel())
    if not len(pixels):
        raise ValueError('No valid cells in supplied mask')
    arrays = {n: ds[n].transpose('time','lat','lon').values.reshape(len(time), -1)[:, pixels].astype(float)
              for n in ['tws','precip','t2m']}
    lat = np.repeat(ds.lat.values, ds.sizes['lon'])[pixels]
    weights = np.cos(np.deg2rad(lat))
    y = arrays['tws']; month = time.month.to_numpy()
    out.mkdir(parents=True, exist_ok=True)
    print(f'Analyzing {len(pixels)} retained cells and {len(time)} calendar months', flush=True)
    corrs = eda(arrays, time, masks, weights, out)
    scores = np.zeros(len(alphas)); denominators = np.zeros(len(alphas))
    # Tune ONE global alpha on 2023; each location gets its own coefficients.
    for j in range(len(pixels)):
        X = features(y[:, j], arrays['precip'][:, j], arrays['t2m'][:, j], time)
        candidates = [predict(fit_model(X, y[:, j], masks['train'], a), X) for a in alphas]
        common = masks['validation'] & np.isfinite(y[:, j]) & np.isfinite(candidates).all(axis=0)
        for k, p in enumerate(candidates):
            scores[k] += weights[j]*np.sum((p[common]-y[common,j])**2)
            denominators[k] += weights[j]*common.sum()
    if np.any(denominators == 0):
        raise ValueError('Insufficient training/validation observations for Ridge selection')
    rmses = np.sqrt(scores/denominators)
    alpha = float(alphas[int(np.argmin(rmses))])
    pd.DataFrame({'alpha':alphas, 'validation_rmse_cm':rmses}).to_csv(out/'alpha_selection.csv', index=False)
    predictions = {n: np.full_like(y, np.nan) for n in ['ridge', 'persistence', 'seasonal_persistence', 'climatology']}
    coef = np.full((len(pixels), len(FEATURES)), np.nan)
    scaler_mean = coef.copy(); scaler_scale = coef.copy(); intercept = np.full(len(pixels), np.nan)
    fit_rows = masks['train'] | masks['validation']
    for j in range(len(pixels)):
        X = features(y[:,j], arrays['precip'][:,j], arrays['t2m'][:,j], time)
        model = fit_model(X, y[:,j], fit_rows, alpha)
        predictions['ridge'][:,j] = predict(model, X)
        predictions['persistence'][:,j] = X[:,0]
        clim = climatology(y[:,j], month, fit_rows)
        c = clim[month-1]
        predictions['climatology'][:,j] = c
        predictions['seasonal_persistence'][1:,j] = y[:-1,j]-c[:-1]+c[1:]
        if model is not None:
            scaler, ridge = model.steps[0][1], model.steps[1][1]
            coef[j] = ridge.coef_; intercept[j] = ridge.intercept_
            scaler_mean[j] = scaler.mean_; scaler_scale[j] = scaler.scale_
    common = masks['test'][:,None] & np.isfinite(y)
    for pred in predictions.values():
        common &= np.isfinite(pred)
    metrics = [dict(model=n, **evaluate(y,p,common,weights)) for n,p in predictions.items()]
    pd.DataFrame(metrics).to_csv(out/'test_metrics.csv',index=False)
    coords = {'time':time, 'lat':ds.lat.values, 'lon':ds.lon.values}
    def cube(a):
        v = np.full((len(time), ds.sizes['lat']*ds.sizes['lon']),np.nan,dtype='float32')
        v[:,pixels] = a
        return v.reshape(len(time),ds.sizes['lat'],ds.sizes['lon'])
    data = {'actual_tws':(('time','lat','lon'),cube(y),{'units':'cm'}),
            'test_eval_mask':(('time','lat','lon'),np.nan_to_num(cube(common),nan=0).astype('int8'))}
    for n,p in predictions.items():
        data[n] = (('time','lat','lon'),cube(np.where(masks['test'][:,None],p,np.nan)),{'units':'cm'})
    result = xr.Dataset(data,coords=coords,attrs={'description':'Revanth: common-mask retrospective one-month forecasts', 'selected_alpha':alpha})
    result = result.sel(time=slice(config['splits']['test']['start'],config['splits']['test']['end']))
    encoding = {n:{'zlib':True,'complevel':4} for n in result.data_vars}
    result.to_netcdf(out/'test_predictions.nc',engine='h5netcdf',encoding=encoding)
    # A non-pickle model archive allows exact reconstruction of final predictions.
    np.savez_compressed(out/'ridge_model.npz', pixel_flat_index=pixels, coefficient=coef,
        intercept=intercept, scaler_mean=scaler_mean, scaler_scale=scaler_scale,
        features=np.array(FEATURES), alpha=alpha, lat=ds.lat.values, lon=ds.lon.values)
    maps = {}
    for n,r in corrs.items():
        m = np.full(ds.sizes['lat']*ds.sizes['lon'],np.nan); m[pixels] = r
        maps[n] = (('lat','lon'),m.reshape(ds.sizes['lat'],ds.sizes['lon']))
    xr.Dataset(maps, coords={k:coords[k] for k in ['lat','lon']}).to_netcdf(out/'training_correlations.nc',engine='h5netcdf')
    split_names = np.full(len(time),'history',dtype=object)
    for n,rows in masks.items(): split_names[rows] = n
    pd.DataFrame({'month':time.strftime('%Y-%m'),'split':split_names,
                  'common_test_cells':common.sum(axis=1)}).to_csv(out/'split_manifest.csv',index=False)
    digest = hashlib.sha256()
    with input_path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): digest.update(chunk)
    metadata = dict(status='executed', executed_utc=datetime.now(timezone.utc).isoformat(),
        package_versions={n:version(n) for n in ['numpy','pandas','xarray','scikit-learn','scipy','h5netcdf']},
        input_file=input_path.name, input_sha256=digest.hexdigest(),
        input_attributes=dict(ds.attrs), configuration=config, features=FEATURES, alpha_candidates=alphas,
        selected_alpha=alpha, refit='train + validation after alpha selection',
        forecast='retrospective one-step; uses observations through prior calendar month',
        source_mask='upstream mask; may use full-record coverage, so spatial selection is not strictly prospective',
        actual_test_start=str(time[common.any(axis=1)][0].date()), actual_test_end=str(time[common.any(axis=1)][-1].date()),
        missing_requested_test_months=sorted(set(pd.period_range(config['splits']['test']['start'],config['splits']['test']['end'],freq='M').astype(str))-set(time[common.any(axis=1)].to_period('M').astype(str))),
        limitations=['No operational product-latency simulation','No geographic generalization test',
                     'Correlations are descriptive, not causal or significance tests',
                     'Fine grid cells are not independent GRACE measurements'])
    (out/'run_manifest.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
    plots(out, time, y, predictions, common, weights)
    return pd.DataFrame(metrics)


def plots(out, time, y, predictions, common, weights):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir = out/'figures'; figdir.mkdir(exist_ok=True)
    hist = pd.read_csv(out/'histograms.csv')
    fig, axes = plt.subplots(1,3,figsize=(13,4))
    for ax,name,label in zip(axes,['tws','precip','t2m'],['Water storage (cm)','Rainfall (mm/month)','Temperature (°C)']):
        for split in ['train','validation','test']:
            h = hist[(hist.variable==name)&(hist.split==split)]
            ax.plot((h.left+h.right)/2,h.area_weighted_fraction,label=split)
        ax.set(xlabel=label,ylabel='Area-weighted fraction'); ax.legend()
    fig.suptitle('Combined-data distributions by chronological split'); fig.tight_layout()
    fig.savefig(figdir/'distributions.png',dpi=160); plt.close(fig)
    fig,ax = plt.subplots(figsize=(10,4))
    used = common.any(axis=1); w=common*weights
    def mean(a):
        return (np.where(common,a,0)*w).sum(axis=1)[used]/w.sum(axis=1)[used]
    ax.plot(time[used],mean(y),label='Observed',color='black',linewidth=2)
    for name in ['ridge','persistence','seasonal_persistence']:
        ax.plot(time[used],mean(predictions[name]),label=name.replace('_',' '))
    ax.set(ylabel='Water-storage anomaly (cm)',title='Test period: area-weighted means on the common evaluation mask')
    ax.legend(); fig.tight_layout(); fig.savefig(figdir/'test_predictions.png',dpi=160); plt.close(fig)
    s=pd.read_csv(out/'correlations.csv')
    fig,ax=plt.subplots(figsize=(9,5)); ax.barh(s.relationship,s.area_weighted_mean_r)
    ax.set(xlim=(-1,1),xlabel='Area-weighted mean within-cell Pearson r',title='Training-period relationships (descriptive)')
    ax.axvline(0,color='gray',linewidth=1); fig.tight_layout(); fig.savefig(figdir/'correlations.png',dpi=160); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('configs/grace_fo_config.yaml'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/revanth_combined_ridge'))
    args=parser.parse_args()
    print(run(args.input,args.config,args.output_dir).to_string(index=False))

if __name__=='__main__':
    main()
