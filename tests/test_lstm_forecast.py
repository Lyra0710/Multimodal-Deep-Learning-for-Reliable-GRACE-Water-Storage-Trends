import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import lstm_forecast as lf


def synthetic_cache(months: int = 84, cells: int = 3) -> dict:
    periods = pd.period_range("2019-01", periods=months, freq="M")
    rng = np.random.default_rng(0)
    tws = rng.normal(size=(months, cells)).astype("float32")
    split = np.where(periods.year <= 2022, 0, np.where(periods.year == 2023, 1, 2))
    return {"tws": tws, "lat": np.array([0.25, 30.25, -45.25]), "month": periods.month.values,
            "year": periods.year.values, "split": split, "periods": periods}


class LSTMForecastTests(unittest.TestCase):
    def test_inputs_precede_target_and_splits_do_not_mix(self):
        d = synthetic_cache()
        for split, year in ((0, (2019, 2022)), (1, (2023, 2023)), (2, (2024, 2025))):
            t = lf.target_index(d, split, 12)
            self.assertTrue(((d["year"][t] >= year[0]) & (d["year"][t] <= year[1])).all())
            self.assertTrue((t - 13 >= 0).all())  # changes at t-12..t-1 need TWSA(t-13); target t excluded
        self.assertEqual(len(lf.target_index(d, 0, 12)), 35)
        self.assertEqual(len(lf.target_index(d, 2, 12)), 24)

    def test_normalization_uses_training_months_only(self):
        d = synthetic_cache()
        before = lf.deseasonalize(d, 0.5)
        d["tws"][d["split"] > 0] += 1000.0  # change every validation/test value
        after = lf.deseasonalize(d, 0.5)
        np.testing.assert_allclose(before["scale"], after["scale"])
        np.testing.assert_allclose(before["seasonal"], after["seasonal"])

    def test_score_matches_hand_computation(self):
        lat = np.array([0.0, 60.0])
        actual = np.array([[0.0, 0.0], [2.0, 4.0]])
        pred = np.array([[1.0, 0.0], [1.0, 2.0]])
        m = lf.score(actual, pred, lat)
        w = np.cos(np.deg2rad(lat))
        self.assertAlmostEqual(m["mae_cm"], ((1 + 1) / 2 * w[0] + (0 + 2) / 2 * w[1]) / w.sum())
        self.assertAlmostEqual(m["rmse_cm"], (1.0 * w[0] + np.sqrt(2.0) * w[1]) / w.sum())

    def test_conformal_factor_reaches_nominal_coverage(self):
        rng = np.random.default_rng(1)
        actual = rng.normal(scale=2.0, size=(200, 50))
        mean, sigma = np.zeros_like(actual), np.ones_like(actual)  # sigma deliberately too small
        lat = np.zeros(50)
        f = lf.conformal_factors(actual, mean, sigma, [0.9], lat)
        self.assertAlmostEqual(np.mean(np.abs(actual) <= f[0.9]), 0.9, delta=0.01)


if __name__ == "__main__":
    unittest.main()
