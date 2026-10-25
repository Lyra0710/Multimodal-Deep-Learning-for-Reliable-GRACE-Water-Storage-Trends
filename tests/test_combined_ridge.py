"""Synthetic data here tests software behavior; it is not a scientific result."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import xarray as xr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import combined_ridge as cr

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    time=pd.date_range('2018-01-01','2025-09-01',freq='MS')
    rng=np.random.default_rng(16); shape=(len(time),2,2)
    p=rng.gamma(2,30,shape)
    t=15+8*np.sin(2*np.pi*np.arange(len(time))[:,None,None]/12)+rng.normal(size=shape)
    y=np.zeros(shape)
    for i in range(1,len(time)):
        y[i]=.6*y[i-1]+.03*p[i-1]-.1*t[i-1]+rng.normal(0,.2,(2,2))
    return xr.Dataset({'tws':(('time','lat','lon'),y,{'units':'cm'}),
        'precip':(('time','lat','lon'),p,{'units':'mm/month'}),
        't2m':(('time','lat','lon'),t,{'units':'degC'}),
        'mask':(('lat','lon'),[[1,1],[1,0]])},
        coords={'time':time,'lat':[-30,30],'lon':[0,1]},
        attrs={'description':'SYNTHETIC TEST FIXTURE - NOT PROJECT DATA'})


class CombinedTests(unittest.TestCase):
    def test_calendar_gap_is_restored_and_not_used_as_previous_month(self):
        d=fixture().isel(time=[i for i in range(93) if i!=20])
        d=cr.prepare(d)
        self.assertEqual(len(d.time),93)
        X=cr.features(d.tws[:,0,0].values,d.precip[:,0,0].values,d.t2m[:,0,0].values,d.time.values)
        self.assertTrue(np.isnan(X[21,0]))
        self.assertTrue(np.isnan(X[25,3]))

    def test_rejects_duplicate_months_and_wrong_units(self):
        d=fixture(); d=d.assign_coords(time=[d.time.values[0]]*len(d.time))
        with self.assertRaisesRegex(ValueError,'Duplicate'): cr.prepare(d)
        d=fixture(); d.tws.attrs['units']='mm'
        with self.assertRaisesRegex(ValueError,'expected units'): cr.prepare(d)

    def test_inputs_do_not_use_target_month(self):
        d=fixture(); a=[d[n][:,0,0].values.copy() for n in ['tws','precip','t2m']]
        X=cr.features(*a,d.time.values)
        for v in a: v[40:]+=99999
        X2=cr.features(*a,d.time.values)
        np.testing.assert_allclose(X[:41],X2[:41],equal_nan=True)

    def test_scaler_uses_only_fit_rows(self):
        X=np.arange(400,dtype=float).reshape(50,8); y=np.arange(50,dtype=float)
        rows=np.arange(50)<30
        m=cr.fit_model(X,y,rows,1)
        X[30:]+=100000; y[30:]+=100000
        n=cr.fit_model(X,y,rows,1)
        np.testing.assert_array_equal(m[0].mean_,n[0].mean_)
        np.testing.assert_array_equal(m[1].coef_,n[1].coef_)

    def test_metrics_match_hand_calculation_and_mask(self):
        y=np.array([[1.,2.],[3.,4.],[5.,6.]])
        p=y+np.array([1.,3.]); mask=np.ones_like(y,dtype=bool); mask[0,1]=False
        p[0,1]=9999
        m=cr.evaluate(y,p,mask,np.array([1.,.5]))
        self.assertAlmostEqual(m['mae_cm'],1.5)
        self.assertAlmostEqual(m['rmse_cm'],np.sqrt(3))
        self.assertAlmostEqual(m['area_weighted_mean_temporal_pearson_r'],1)
        self.assertEqual(m['valid_land_cell_month_observations'],5)

    def test_constant_series_correlation_is_undefined(self):
        self.assertTrue(np.isnan(cr.correlation(np.ones(10),np.arange(10))))

    def test_missing_input_has_actionable_message(self):
        with self.assertRaisesRegex(FileNotFoundError,'Obtain aligned'):
            cr.run('/not/a/file.nc',ROOT/'configs/grace_fo_config.yaml','unused')

    def test_end_to_end_and_no_test_target_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inp=root/'synthetic.nc'; d=fixture()
            d.to_netcdf(inp,engine='h5netcdf')
            first=cr.run(inp,ROOT/'configs/grace_fo_config.yaml',root/'first')
            self.assertEqual(set(first.evaluated_months),{21})
            self.assertEqual(set(first.evaluated_land_cells),{3})
            self.assertEqual(first.valid_land_cell_month_observations.nunique(),1)
            with xr.open_dataset(root/'first/test_predictions.nc') as pred:
                self.assertEqual(pred.sizes['time'],21)
                self.assertEqual(int(pred.test_eval_mask.sum()),63)
                self.assertTrue(pred.ridge[:,1,1].isnull().all())
            manifest=json.loads((root/'first/run_manifest.json').read_text())
            self.assertEqual(manifest['missing_requested_test_months'],['2025-10','2025-11','2025-12'])
            d['tws'].loc[dict(time=slice('2024-01-01',None))]+=1000
            d.to_netcdf(inp,engine='h5netcdf')
            cr.run(inp,ROOT/'configs/grace_fo_config.yaml',root/'second')
            pd.testing.assert_frame_equal(pd.read_csv(root/'first/alpha_selection.csv'),pd.read_csv(root/'second/alpha_selection.csv'))
            with np.load(root/'first/ridge_model.npz') as a,np.load(root/'second/ridge_model.npz') as b:
                for key in ['coefficient','intercept','scaler_mean','scaler_scale','alpha']:
                    np.testing.assert_array_equal(a[key],b[key])
            # Observed prior test-month TWS is allowed for subsequent one-step forecasts.
            self.assertTrue((root/'first/figures/distributions.png').is_file())

if __name__=='__main__': unittest.main()
