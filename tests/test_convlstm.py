import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from convlstm import (
    ConvLSTMForecaster,
    PatchSample,
    PatchSequenceDataset,
    area_weighted_metrics,
    consecutive_months,
    masked_mse,
    patch_coverage,
    patch_origins,
    split_target_indices,
)


class ConvLSTMTests(unittest.TestCase):
    def test_forward_and_backward_preserve_map_shape(self):
        model = ConvLSTMForecaster(input_channels=3, hidden_channels=4)
        sequence = torch.randn(2, 12, 3, 8, 10)
        target = torch.randn(2, 1, 8, 10)
        valid = torch.ones_like(target, dtype=torch.bool)
        prediction = model(sequence)
        self.assertEqual(tuple(prediction.shape), (2, 1, 8, 10))
        masked_mse(prediction, target, valid).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_masked_loss_ignores_invalid_cells(self):
        prediction = torch.tensor([[[[2.0, 100.0]]]])
        target = torch.tensor([[[[1.0, 0.0]]]])
        valid = torch.tensor([[[[True, False]]]])
        self.assertEqual(float(masked_mse(prediction, target, valid)), 1.0)

    def test_history_precedes_target_and_skips_missing_input(self):
        times = np.arange("2019-01", "2021-01", dtype="datetime64[M]").astype("datetime64[ns]")
        features = np.ones((len(times), 1, 2, 2), dtype=np.float32)
        target = np.ones((len(times), 2, 2), dtype=np.float32)
        features[3] = np.nan
        indices = split_target_indices(times, "2019-01-01", "2020-12-31", 3, features, target, np.ones((2, 2), bool))
        self.assertNotIn(4, indices)  # target index 4 would include the missing index 3 in its history.
        self.assertTrue(np.all(indices >= 3))

    def test_patch_dataset_returns_12_prior_steps(self):
        features = np.ones((13, 1, 4, 4), dtype=np.float32)
        target = np.ones((13, 4, 4), dtype=np.float32)
        valid = np.ones((13, 4, 4), dtype=bool)
        dataset = PatchSequenceDataset(features, target, valid, [PatchSample(12, 0, 0)], 12, 4)
        sequence, one_target, mask, target_index, _, _ = dataset[0]
        self.assertEqual(tuple(sequence.shape), (12, 1, 4, 4))
        self.assertEqual(tuple(one_target.shape), (1, 4, 4))
        self.assertTrue(bool(mask.all()))
        self.assertEqual(target_index, 12)

    def test_excluded_cells_reject_entire_intersecting_patch(self):
        mask = np.ones((4, 4), bool)
        excluded = np.zeros_like(mask)
        excluded[1, 1] = True
        self.assertEqual(patch_origins(mask, patch_size=2, stride=2, excluded=excluded), [(0, 2), (2, 0), (2, 2)])

    def test_boundary_patches_cover_all_eligible_cells(self):
        mask = np.ones((5, 5), bool)
        origins = patch_origins(mask, patch_size=3, stride=3)
        self.assertIn((2, 2), origins)
        self.assertTrue(patch_coverage(mask.shape, origins, 3)[mask].all())

    def test_area_weighted_metrics_on_synthetic_data(self):
        target = np.arange(12, dtype=float)[:, None, None] * np.ones((1, 2, 2))
        prediction = target + 2.0
        valid = np.ones_like(target, dtype=bool)
        metrics = area_weighted_metrics(prediction, target, valid, np.array([-30.0, 30.0]))
        self.assertAlmostEqual(metrics["MAE_cm"], 2.0)
        self.assertAlmostEqual(metrics["RMSE_cm"], 2.0)
        self.assertAlmostEqual(metrics["Pearson_r_area_weighted"], 1.0)
        self.assertEqual(metrics["evaluated_cells"], 4)

    def test_invalid_predictions_do_not_affect_metrics(self):
        target = np.arange(12, dtype=float)[:, None, None] * np.ones((1, 2, 2))
        prediction = target + 1.0
        valid = np.ones_like(target, dtype=bool)
        prediction[:, 0, 0] = 1_000_000.0
        valid[:, 0, 0] = False
        metrics = area_weighted_metrics(prediction, target, valid, np.array([-20.0, 20.0]))
        self.assertAlmostEqual(metrics["MAE_cm"], 1.0)
        self.assertAlmostEqual(metrics["RMSE_cm"], 1.0)
        self.assertAlmostEqual(metrics["Pearson_r_area_weighted"], 1.0)

    def test_minimal_training_step_updates_parameters(self):
        model = ConvLSTMForecaster(input_channels=1, hidden_channels=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        sequence = torch.randn(1, 12, 1, 4, 4)
        target = torch.randn(1, 1, 4, 4)
        valid = torch.ones_like(target, dtype=torch.bool)
        before = model.output.weight.detach().clone()
        optimizer.zero_grad()
        masked_mse(model(sequence), target, valid).backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.output.weight.detach()))

    def test_calendar_months_and_checkpoint_loading(self):
        self.assertTrue(consecutive_months(np.arange("2020-01", "2021-01", dtype="datetime64[M]")))
        self.assertFalse(consecutive_months(np.array(["2020-01", "2020-03"], dtype="datetime64[M]")))
        model = ConvLSTMForecaster(input_channels=1, hidden_channels=2)
        sequence = torch.randn(1, 12, 1, 4, 4)
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            torch.save({"model_state": model.state_dict()}, checkpoint.name)
            reloaded = ConvLSTMForecaster(input_channels=1, hidden_channels=2)
            reloaded.load_state_dict(torch.load(checkpoint.name, weights_only=True)["model_state"])
        torch.testing.assert_close(model(sequence), reloaded(sequence))


if __name__ == "__main__":
    unittest.main()
