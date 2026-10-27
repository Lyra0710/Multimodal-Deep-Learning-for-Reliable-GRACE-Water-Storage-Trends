import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import p_t_tws_analysis as analysis
import p_t_tws_baselines as baselines
import p_t_tws_data as alignment


def synthetic_config() -> dict:
    config = alignment.load_config(PROJECT_ROOT / "configs" / "p_t_tws_config.yaml", project_root=PROJECT_ROOT)
    config["data"].update(driver_start="2017-01", tws_start="2017-01", end="2025-12", min_tws_fraction=0.7)
    return config


def synthetic_aligned(config: dict, seed: int = 0) -> xr.Dataset:
    """Coarse global grid where TWS responds to accumulated precipitation and temperature."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2017-01-01", "2025-12-01", freq="MS")
    lat, lon = np.array([-25.0, 5.0, 35.0]), np.array([-120.0, -60.0, 0.0, 60.0])
    shape = (len(times), len(lat), len(lon))
    month = times.month.values[:, None, None]
    precip = 80 + 40 * np.sin(2 * np.pi * month / 12) + rng.normal(0, 15, shape)
    t2m = 15 + 8 * np.cos(2 * np.pi * month / 12) + rng.normal(0, 1, shape)
    precip_anomaly = precip - precip.mean(0)
    tws = 0.05 * np.cumsum(precip_anomaly, 0) - 0.5 * (t2m - t2m.mean(0)) + rng.normal(0, 0.5, shape)
    tws[times.get_loc(pd.Timestamp("2018-03-01"))] = np.nan  # one GRACE gap month
    land = xr.DataArray(np.ones((len(lat), len(lon)), bool), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    coords = {"time": times, "lat": lat, "lon": lon}
    tws = xr.DataArray(tws.astype("float32"), dims=("time", "lat", "lon"), coords=coords)
    precip = xr.DataArray(precip.astype("float32"), dims=("time", "lat", "lon"), coords=coords)
    t2m = xr.DataArray(t2m.astype("float32"), dims=("time", "lat", "lon"), coords=coords)
    return alignment.align_products(tws, land, precip, t2m, config)


class PTTwsAlignmentTests(unittest.TestCase):
    def test_config_reuses_shared_grace_fo_splits(self):
        config = alignment.load_config(PROJECT_ROOT / "configs" / "p_t_tws_config.yaml")
        shared = alignment.load_split_config(PROJECT_ROOT / "configs" / "grace_fo_config.yaml")
        self.assertEqual(config["splits"], shared["splits"])

    def test_target_grid_block_averages_nested_grid(self):
        lat = np.arange(-89.5, 90, 1.0)
        lon = np.arange(0.5, 360, 1.0)
        data = xr.DataArray(
            np.tile(np.arange(len(lon), dtype=float), (len(lat), 1)), dims=("lat", "lon"), coords={"lat": lat, "lon": lon}
        )
        result = alignment.to_target_grid(data, 2.0)
        target_lat, target_lon = alignment.target_axes(2.0)
        np.testing.assert_array_equal(result.lat.values, target_lat)
        np.testing.assert_array_equal(result.lon.values, target_lon)
        # Longitude 180.5 (index 180) becomes -179.5 after wrapping, so the first block averages columns 180 and 181.
        self.assertAlmostEqual(float(result.isel(lat=0, lon=0)), 180.5)

    def test_grace_months_use_time_bounds_midpoint_and_average_duplicates(self):
        starts = pd.to_datetime(["2019-01-01", "2019-01-20", "2019-03-01"])
        ends = pd.to_datetime(["2019-01-31", "2019-02-10", "2019-03-31"])
        dataset = xr.Dataset(
            {
                "lwe_thickness": (("time", "lat", "lon"), np.array([1.0, 3.0, 5.0])[:, None, None], {"units": "cm"}),
                "land_mask": (("lat", "lon"), np.array([[1.0]])),
                "time_bounds": (("time", "bounds"), np.stack([starts, ends], 1)),
            },
            coords={"time": starts, "lat": [0.25], "lon": [0.25]},
        )
        labels = alignment.grace_month_labels(dataset)
        self.assertEqual(list(labels.strftime("%Y-%m")), ["2019-01", "2019-01", "2019-03"])
        tws = dataset["lwe_thickness"].assign_coords(time=labels).groupby("time").mean()
        np.testing.assert_array_equal(tws.values.ravel(), [2.0, 5.0])

    def test_anomalies_use_training_months_only(self):
        config = synthetic_config()
        aligned = synthetic_aligned(config)
        cube = alignment.PixelCube.from_dataset(aligned)
        train = cube.period(config["splits"]["train"]["start"], config["splits"]["train"]["end"])
        climatology, _ = alignment.training_anomalies(cube, train)

        cube.tws[~train] += 1000.0
        changed, _ = alignment.training_anomalies(cube, train)
        np.testing.assert_allclose(climatology["tws"], changed["tws"])
        self.assertTrue(np.isnan(cube.tws[cube.times.get_loc(pd.Timestamp("2018-03-01"))]).all())

    def test_pixel_cube_round_trips_maps(self):
        aligned = synthetic_aligned(synthetic_config())
        cube = alignment.PixelCube.from_dataset(aligned)
        self.assertEqual(cube.n_pixels, 12)
        restored = cube.to_cube(cube.precip)
        np.testing.assert_allclose(restored.values, aligned["precip"].values, rtol=1e-6)


class PTTwsAnalysisTests(unittest.TestCase):
    def test_shift_and_rolling_sum(self):
        values = np.arange(1.0, 6.0)[:, None]
        np.testing.assert_array_equal(analysis.shift(values, 2).ravel(), [np.nan, np.nan, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(analysis.rolling_sum(values, 3).ravel(), [np.nan, np.nan, 6.0, 9.0, 12.0])

    def test_nan_corr_ignores_missing_months_and_constant_series(self):
        x = np.array([[1.0, 1.0], [2.0, 1.0], [np.nan, 1.0], [4.0, 1.0]])
        y = np.array([[2.0, 3.0], [4.0, 4.0], [9.0, 5.0], [8.0, 6.0]])
        correlation, count = analysis.nan_corr(x, y)
        self.assertAlmostEqual(correlation[0], 1.0)
        self.assertTrue(np.isnan(correlation[1]))
        np.testing.assert_array_equal(count, [3, 4])

    def test_bh_fdr_matches_step_up_rule(self):
        p_values = np.array([0.001, 0.008, 0.039, 0.041, 0.2, np.nan])
        np.testing.assert_array_equal(analysis.bh_fdr(p_values, 0.05), [True, True, False, False, False, False])

    def test_detrend_removes_linear_trend_inside_rows(self):
        years = np.arange(10, dtype=float)
        rows = years >= 2
        values = (3.0 + 0.5 * years)[:, None]
        result = analysis.detrend(values, rows, years)
        self.assertTrue(np.isnan(result[:2]).all())
        np.testing.assert_allclose(result[2:], 0.0, atol=1e-12)

    def test_driver_relationships_detect_cumulative_precipitation(self):
        config = synthetic_config()
        cube = alignment.PixelCube.from_dataset(synthetic_aligned(config))
        train = cube.period(config["splits"]["train"]["start"], config["splits"]["train"]["end"])
        _, anomalies = alignment.training_anomalies(cube, train)
        relationships = analysis.driver_relationships(cube, anomalies, config)
        summary = analysis.correlation_summary(relationships, cube.area_weight, config["analysis"]["fdr_q"]).set_index(
            "relationship"
        )
        self.assertGreater(summary.loc["r(TWS, cumP) best window", "median_r"], summary.loc["r(TWS, P) lag 0", "median_r"])
        self.assertLess(summary.loc["r(TWS, T) lag 0", "median_r"], 0)


class PTTwsBaselineTests(unittest.TestCase):
    def test_ridge_recovers_per_pixel_linear_relationship(self):
        rng = np.random.default_rng(1)
        features = rng.normal(size=(2, 60, 2)).astype(np.float32)
        target = (np.array([[1.0, -2.0], [0.5, 3.0]])[:, None, :] * features).sum(-1).T + 4.0
        fit_rows = np.arange(60) < 40
        prediction = baselines.ridge_fit_predict(features, target, fit_rows, alpha=1e-8)
        np.testing.assert_allclose(prediction[~fit_rows], target[~fit_rows], atol=1e-4)

    def test_evaluation_mask_requires_previous_month_and_all_predictions(self):
        tws = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0]])
        rows = np.array([False, True, True, True, True])
        predictions = {"a": np.ones_like(tws), "b": np.array([[1.0], [1.0], [1.0], [1.0], [np.nan]])}
        mask = baselines.evaluation_mask(tws, rows, predictions)
        np.testing.assert_array_equal(mask.ravel(), [False, True, False, False, False])

    def test_evaluate_perfect_prediction(self):
        rng = np.random.default_rng(2)
        observed = rng.normal(size=(24, 3))
        month = np.tile(np.arange(1, 13), 2)
        mask = np.ones_like(observed, bool)
        metrics, _ = baselines.evaluate_predictions(
            observed, observed, mask, np.ones(3), month, np.zeros((12, 3)), reference=np.zeros_like(observed)
        )
        self.assertAlmostEqual(metrics["RMSE_cm"], 0.0)
        self.assertAlmostEqual(metrics["r_area_wtd"], 1.0)
        self.assertAlmostEqual(metrics["MSESS_vs_clim"], 1.0)
        self.assertEqual(metrics["n_cells"], 3)

    def test_spatial_folds_keep_blocks_together(self):
        cube = alignment.PixelCube.from_dataset(synthetic_aligned(synthetic_config()))
        folds = baselines.spatial_block_folds(cube, 30, 3, np.random.default_rng(0))
        blocks = np.floor((cube.pixel_lat + 90) / 30) * 1000 + np.floor((cube.pixel_lon + 180) / 30)
        for block in np.unique(blocks):
            self.assertEqual(len(np.unique(folds[blocks == block])), 1)

    def test_temporal_baselines_score_identical_test_cells(self):
        config = synthetic_config()
        config["baselines"].update(gbm_max_rows=5000, gbm={**config["baselines"]["gbm"], "max_iter": 20, "min_samples_leaf": 5})
        cube = alignment.PixelCube.from_dataset(synthetic_aligned(config))
        train = cube.period(config["splits"]["train"]["start"], config["splits"]["train"]["end"])
        climatologies, anomalies = alignment.training_anomalies(cube, train)
        features, feature_sets, drivers = baselines.build_features(cube, anomalies, config["baselines"])
        temporal = baselines.temporal_split_baselines(
            cube, features, feature_sets, drivers, climatologies, config, np.random.default_rng(0)
        )
        test_rows = cube.period(config["splits"]["test"]["start"], config["splits"]["test"]["end"])
        self.assertFalse(temporal["test_mask"][~test_rows].any())
        self.assertEqual(set(temporal["metrics"]["n_cells"]), {cube.n_pixels})
        self.assertGreater(temporal["metrics"].loc["ARX (P,T,trend,TWS t-1)", "MSESS_vs_clim"], 0)


if __name__ == "__main__":
    unittest.main()
