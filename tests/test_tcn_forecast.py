"""Tests for the TCN forecaster. Run with an environment that has PyTorch (e.g. the grace-tcn env)."""

import sys
from pathlib import Path
import unittest

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

import p_t_tws_data as alignment
import tcn_forecast as tcn
from test_p_t_tws_pipeline import synthetic_aligned


def tiny_config() -> dict:
    config = tcn.load_tcn_config(PROJECT_ROOT / "configs" / "tcn_config.yaml", project_root=PROJECT_ROOT)
    config["base"]["data"].update(driver_start="2017-01", tws_start="2017-01", end="2025-12", min_tws_fraction=0.7)
    config["training"].update(max_epochs=3, patience=2, samples_per_epoch=10_000, threads=1)
    config["model"]["batch_size"] = 64
    config["grid"] = {"channels": [8], "dropout": [0.0], "learning_rate": [3e-3]}
    return config


class TCNForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = tiny_config()
        cls.cube = alignment.PixelCube.from_dataset(synthetic_aligned(cls.config["base"]))
        cls.rows = cls.cube.split_rows(cls.config["base"]["splits"])

    def test_network_is_causal(self):
        model = tcn.TCN(n_inputs=3, channels=8, kernel_size=3, dilations=[1, 2, 4], dropout=0.0).eval()
        x = torch.randn(4, 12, 3)
        changed = x.clone()
        changed[:, 6:] += 10.0  # perturb only steps after step 5
        with torch.no_grad():
            hidden, hidden_changed = model.blocks(x.transpose(1, 2)), model.blocks(changed.transpose(1, 2))
        torch.testing.assert_close(hidden[:, :, :6], hidden_changed[:, :, :6])
        self.assertFalse(torch.allclose(hidden[:, :, -1], hidden_changed[:, :, -1]))

    def test_receptive_field_covers_lookback(self):
        model = self.config["model"]
        self.assertGreaterEqual(tcn.receptive_field(model["kernel_size"], model["dilations"]),
                                self.config["data"]["lookback_months"])

    def test_nowcast_window_holds_previous_tws_and_current_drivers(self):
        features, names, scaling = tcn.build_channels(self.cube, self.rows["train"], ["tws", "precip", "t2m"], "nowcast", True)
        _, anomalies = alignment.training_anomalies(self.cube, self.rows["train"])
        data = tcn.WindowData(features, self.cube.tws, self.cube.area_weight, lookback=12)
        t = int(np.flatnonzero(self.rows["test"])[0])
        window = data.windows(torch.tensor([t]), torch.tensor([0]))[0].numpy()  # (12, channel)
        self.assertEqual(window.shape, (12, len(names)))
        # Last step: TWS(t-1) and P(t); first step: TWS(t-12).
        expected_tws = (self.cube.tws[t - 1, 0] - scaling["tws_mean_cm"]) / scaling["tws_std_cm"]
        self.assertAlmostEqual(window[-1, names.index("tws_prev")], expected_tws, places=4)
        self.assertAlmostEqual(window[-1, names.index("precip_anom")],
                               anomalies["precip"][t, 0] / scaling["precip_anom_std"], places=4)
        expected_first = (self.cube.tws[t - 12, 0] - scaling["tws_mean_cm"]) / scaling["tws_std_cm"]
        self.assertAlmostEqual(window[0, names.index("tws_prev")], expected_first, places=4)

    def test_forecast_timing_lags_drivers_by_one_month(self):
        nowcast, names, _ = tcn.build_channels(self.cube, self.rows["train"], ["tws", "precip"], "nowcast", False)
        forecast, _, _ = tcn.build_channels(self.cube, self.rows["train"], ["tws", "precip"], "forecast", False)
        column = names.index("precip_anom")
        np.testing.assert_allclose(forecast[1:, :, column], nowcast[:-1, :, column])

    def test_grace_gap_is_zero_filled_and_flagged(self):
        features, names, _ = tcn.build_channels(self.cube, self.rows["train"], ["tws"], "nowcast", False)
        gap = int(np.flatnonzero(~np.isfinite(self.cube.tws).any(1) & (np.arange(len(self.cube.times)) > 0))[0])
        self.assertTrue(np.all(features[gap + 1, :, names.index("tws_prev")] == 0))
        self.assertTrue(np.all(features[gap + 1, :, names.index("tws_prev_observed")] == 0))
        self.assertTrue(np.isfinite(features).all())

    def test_grace_only_run_has_no_driver_channels(self):
        _, names, _ = tcn.build_channels(self.cube, self.rows["train"], ["tws"], "nowcast", True)
        self.assertEqual(names, ["tws_prev", "tws_prev_observed", "month_sin", "month_cos"])

    def test_driver_only_run_has_no_tws_channels(self):
        _, names, scaling = tcn.build_channels(self.cube, self.rows["train"], ["precip", "t2m"], "nowcast", True)
        self.assertEqual(names, ["precip_anom", "t2m_anom", "month_sin", "month_cos"])
        self.assertIn("tws_std_cm", scaling)  # target scaling is still defined

    def test_stacking_recovers_known_weights_and_ignores_other_rows(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=(30, 5)), rng.normal(size=(30, 5))
        observed = 2.0 + 0.7 * a + 0.2 * b
        rows = np.zeros(30, bool)
        rows[10:20] = True
        observed[~rows] += 100.0  # rows outside the fitting period must not matter
        coefficients = tcn.fit_stacking([a, b], observed, rows, np.ones(5))
        np.testing.assert_allclose(coefficients, [2.0, 0.7, 0.2], atol=1e-8)
        np.testing.assert_allclose(tcn.apply_stacking(coefficients, [a, b])[rows], observed[rows])

    def test_late_fusion_uses_validation_predictions_only(self):
        config = tiny_config()
        config["runs"] = {"G": ["tws"], "PT": ["precip", "t2m"]}
        config["late_fusion"] = {"name": "Late", "branches": ["G", "PT"]}
        experiments = [tcn.run_experiment(self.cube, sources, config, label) for label, sources in config["runs"].items()]
        for experiment in experiments:
            validation = experiment["validation_prediction"]
            self.assertTrue(np.isfinite(validation[self.rows["validation"]]).all())
            self.assertFalse(np.isfinite(validation[~self.rows["validation"]]).any())
        fusion = tcn.late_fusion(self.cube, experiments, config)
        self.assertEqual(fusion["label"], "Late")
        self.assertTrue(np.isfinite(fusion["prediction"][self.rows["test"]]).all())
        self.assertEqual(set(fusion["coefficients"]), {"intercept_cm", "weight[G]", "weight[PT]", "val_rmse_cm_in_sample"})

    def test_scaling_uses_training_rows_only(self):
        tws = self.cube.tws.copy()
        self.cube.tws[self.rows["test"]] += 1000.0
        try:
            _, _, scaling = tcn.build_channels(self.cube, self.rows["train"], ["tws"], "nowcast", False)
        finally:
            self.cube.tws = tws
        self.assertAlmostEqual(scaling["tws_mean_cm"], float(np.nanmean(tws[self.rows["train"]])), places=6)

    def test_experiment_learns_and_scores_on_shared_cells(self):
        experiment = tcn.run_experiment(self.cube, ["tws", "precip", "t2m"], self.config, "TCN test")
        prediction = experiment["prediction"]
        self.assertTrue(np.isfinite(prediction[self.rows["test"]]).all())
        self.assertFalse(np.isfinite(prediction[~(self.rows["train"] | self.rows["validation"] | self.rows["test"])]).any())
        climatology = alignment.monthly_climatology(
            self.cube.tws, self.rows["train"] | self.rows["validation"], self.cube.month)
        metrics, mask = tcn.score_models(
            self.cube, {"TCN test": prediction, "Climatology": climatology[self.cube.month - 1]}, self.config, None)
        self.assertTrue(mask[self.rows["test"]].any() and not mask[~self.rows["test"]].any())
        self.assertGreater(metrics.loc["TCN test", "MSESS_vs_clim"], 0)  # beats climatology on learnable data

    def test_integrated_gradients_is_exact_for_a_linear_model(self):
        class Linear(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                return (x * self.weight).sum((1, 2)) + 0.7

        weight = torch.randn(12, 3)
        x, baseline = torch.randn(5, 12, 3), torch.randn(5, 12, 3)
        attributions, gaps = tcn.integrated_gradients(Linear(weight), x, baseline, steps=4)
        torch.testing.assert_close(attributions, weight * (x - baseline))
        self.assertLess(gaps.abs().max().item(), 1e-4)

    def test_integrated_gradients_completeness_and_context_channels_on_tcn(self):
        torch.manual_seed(0)
        names = ["tws_prev", "tws_prev_observed", "precip_anom", "t2m_anom", "month_sin", "month_cos"]
        model = tcn.TCN(n_inputs=len(names), channels=8, kernel_size=3, dilations=[1, 2, 4, 8], dropout=0.3)
        x = torch.randn(16, 12, len(names))
        baseline = tcn.average_conditions_baseline(x, names)
        attributions, gaps = tcn.integrated_gradients(model, x, baseline, steps=128)
        with torch.no_grad():
            spread = (model(x) - model(baseline)).abs().mean().item()
        self.assertLess(gaps.abs().mean().item(), 0.02 * spread + 1e-4)  # sum of attributions ≈ F(x) − F(baseline)
        for index, name in enumerate(names):
            if name not in tcn.ATTRIBUTED_CHANNELS:  # flag and calendar keep real values -> zero attribution
                self.assertTrue(torch.all(attributions[..., index] == 0), name)
        self.assertTrue(torch.all(baseline[..., names.index("tws_prev")] == 0))

    def test_attribution_lags_follow_nowcast_alignment(self):
        lags = tcn.attribution_lags(["tws_prev", "precip_anom"], lookback=12)
        self.assertEqual((lags["tws_prev"][0], lags["tws_prev"][-1]), (12, 1))    # TWS(t-12) .. TWS(t-1)
        self.assertEqual((lags["precip_anom"][0], lags["precip_anom"][-1]), (11, 0))  # P(t-11) .. P(t)


if __name__ == "__main__":
    unittest.main()
