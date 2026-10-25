import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import grace_fo_persistence as persistence
import grace_fo_processing as processing


def synthetic_source(months: int = 84) -> xr.Dataset:
    time = pd.to_datetime(
        [f"{month}-16 12:00:00" for month in pd.period_range("2019-01", periods=months, freq="M")]
    )
    lat = np.array([-0.25, 0.25])
    lon = np.array([0.25, 359.75])
    shape = (months, len(lat), len(lon))
    values = np.arange(np.prod(shape), dtype=float).reshape(shape)
    land_mask = xr.DataArray([[1, 0], [1, 1]], dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    return xr.Dataset(
        {
            "lwe_thickness": (("time", "lat", "lon"), values, {"units": "cm"}),
            "uncertainty": (("time", "lat", "lon"), np.ones(shape), {"units": "cm"}),
            "land_mask": land_mask,
            "mascon_ID": (("lat", "lon"), np.array([[1, 2], [3, 4]], dtype=float)),
            "scale_factor": (("lat", "lon"), np.ones((2, 2))),
            "GAD": (("time", "lat", "lon"), np.zeros(shape)),
            "lat_bounds": (("lat", "bounds"), np.array([[-0.5, 0], [0, 0.5]])),
            "lon_bounds": (("lon", "bounds"), np.array([[0, 0.5], [359.5, 360]])),
        },
        coords={"time": time, "lat": lat, "lon": lon, "bounds": [0, 1]},
        attrs={"title": "synthetic NASA source"},
    )


class GraceFoPreprocessingTests(unittest.TestCase):
    def test_prepare_preserves_values_and_standardizes_coordinates(self):
        source = synthetic_source()
        source_before = source.copy(deep=True)
        result = processing.prepare(source, "2019-01-01", "2025-12-31")
        processing.verify(result, source, "2019-01-01", "2025-12-31")

        self.assertEqual(result.sizes["time"], 84)
        self.assertEqual(str(result.time.values[0])[:10], "2019-01-01")
        self.assertEqual(str(result.time.values[-1])[:10], "2025-12-01")
        self.assertTrue(np.array_equal(result.lon.values, [-0.25, 0.25]))
        self.assertTrue(np.array_equal(result.lat.values, source.lat.values))
        self.assertTrue(np.array_equal(result.original_time.values, source.time.values))
        self.assertEqual(result.attrs["title"], source.attrs["title"])
        for name in source.data_vars:
            np.testing.assert_array_equal(source[name].values, source_before[name].values)

    def test_prepare_rejects_missing_month(self):
        source = synthetic_source().isel(time=[index for index in range(84) if index != 20])
        with self.assertRaisesRegex(ValueError, "Missing months"):
            processing.prepare(source, "2019-01-01", "2025-12-31")

    def test_land_view_masks_ocean_without_changing_master(self):
        source = synthetic_source()
        result = processing.prepare(source, "2019-01-01", "2025-12-31")
        view = processing.land_modeling_view(result)
        self.assertTrue(np.isnan(view["lwe_thickness"].isel(time=0, lat=0, lon=0)).item())
        self.assertFalse(np.isnan(result["lwe_thickness"].isel(time=0, lat=0, lon=0)).item())


class GraceFoPersistenceTests(unittest.TestCase):
    def test_persistence_uses_previous_consecutive_month_only(self):
        time = pd.date_range("2022-12-01", periods=3, freq="MS")
        data = xr.DataArray(
            np.array([[[10.0]], [[20.0]], [[30.0]]]),
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": [0.0], "lon": [0.0]},
        )
        actual, predicted = persistence.persistence_pairs(data, "2023-01-01", "2023-02-28")
        self.assertTrue(np.array_equal(actual.values.ravel(), [20.0, 30.0]))
        self.assertTrue(np.array_equal(predicted.values.ravel(), [10.0, 20.0]))
        self.assertTrue(np.array_equal(actual.time.values, predicted.time.values))

    def test_persistence_excludes_missing_month_gap(self):
        data = xr.DataArray(
            np.array([[[10.0]], [[30.0]]]),
            dims=("time", "lat", "lon"),
            coords={"time": pd.to_datetime(["2023-01-01", "2023-03-01"]), "lat": [0.0], "lon": [0.0]},
        )
        with self.assertRaisesRegex(ValueError, "No consecutive"):
            persistence.persistence_pairs(data, "2023-03-01", "2023-03-31")

    def test_area_weighted_metrics_and_temporal_correlation(self):
        actual = xr.DataArray(
            np.array([[[2.0], [4.0]], [[4.0], [6.0]], [[6.0], [8.0]]]),
            dims=("time", "lat", "lon"),
            coords={"time": pd.date_range("2023-01-01", periods=3, freq="MS"), "lat": [0.0, 60.0], "lon": [0.0]},
        )
        predicted = actual - 1.0
        metrics, _ = persistence.evaluate(actual, predicted)
        self.assertAlmostEqual(metrics["mae_cm"], 1.0)
        self.assertAlmostEqual(metrics["rmse_cm"], 1.0)
        self.assertAlmostEqual(metrics["area_weighted_mean_temporal_pearson_r"], 1.0)
        self.assertEqual(metrics["evaluated_months"], 3)
        self.assertEqual(metrics["evaluated_land_cells"], 2)

    def test_shared_split_configuration_is_chronological(self):
        config = persistence.load_config(PROJECT_ROOT / "configs" / "grace_fo_config.yaml")
        self.assertEqual(config["splits"]["train"]["end"], "2022-12-31")
        self.assertEqual(config["splits"]["validation"]["start"], "2023-01-01")
        self.assertEqual(config["splits"]["test"]["start"], "2024-01-01")

        manifest = persistence.split_manifest(
            xr.DataArray(pd.date_range("2019-01-01", periods=84, freq="MS"), dims=("time",)), config
        )
        self.assertEqual(manifest["split"].value_counts().to_dict(), {"train": 48, "test": 24, "validation": 12})
        self.assertFalse(manifest.iloc[0]["persistence_eligible"])
        self.assertTrue(manifest.iloc[1]["persistence_eligible"])


if __name__ == "__main__":
    unittest.main()
