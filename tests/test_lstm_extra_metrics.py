import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import lstm_extra_metrics as em


class ExtraMetricTests(unittest.TestCase):
    def test_gaussian_crps_matches_monte_carlo(self):
        rng = np.random.default_rng(0)
        y, mu, sigma = np.array([[1.3]]), np.array([[0.2]]), np.array([[0.8]])
        draws = rng.normal(mu[0, 0], sigma[0, 0], size=(400_000, 2))
        mc = np.abs(draws[:, 0] - y[0, 0]).mean() - 0.5 * np.abs(draws[:, 0] - draws[:, 1]).mean()
        m = em.probabilistic_metrics(y, mu, sigma, mu - 1, mu + 1, np.zeros(1))
        self.assertAlmostEqual(m["crps_cm"], mc, places=2)

    def test_interval_score_penalizes_misses(self):
        lat = np.zeros(1)
        inside = em.probabilistic_metrics(np.array([[0.0]]), np.zeros((1, 1)), np.ones((1, 1)),
                                          np.array([[-1.0]]), np.array([[1.0]]), lat)
        outside = em.probabilistic_metrics(np.array([[2.0]]), np.zeros((1, 1)), np.ones((1, 1)),
                                           np.array([[-1.0]]), np.array([[1.0]]), lat)
        self.assertAlmostEqual(inside["interval_score_90_cm"], 2.0)          # width only
        self.assertAlmostEqual(outside["interval_score_90_cm"], 2.0 + 20.0)  # width + (2/0.1) * miss

    def test_nse_and_skill_for_perfect_forecast(self):
        rng = np.random.default_rng(1)
        actual = rng.normal(size=(12, 5))
        m = em.point_metrics(actual, actual.copy(), np.zeros(5), actual + 1.0, actual * 0.0)
        self.assertAlmostEqual(m["nse_median"], 1.0)
        self.assertAlmostEqual(m["rmse_skill_vs_persistence"], 1.0)
        self.assertAlmostEqual(m["mse_skill_vs_climatology"], 1.0)


if __name__ == "__main__":
    unittest.main()
