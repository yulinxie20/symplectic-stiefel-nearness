"""论文后验验收的 CPU 测试，不调用 GPU 求解器或 CUDA。"""
import sys
import unittest
from importlib.metadata import version
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from paper_cases import inspect_output, left_j, make_target


class PaperValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        print(
            f"\nPython: {sys.executable}; {sys.version.split()[0]}; "
            f"PyTorch: {torch.__version__}; NumPy: {version('numpy')}; "
            "device=cpu; dtype=float64; threads=1"
        )

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_stationary_saddle_does_not_pass_global_quality(self):
        x = torch.eye(2, dtype=torch.float64).unsqueeze(0)
        omega = -2 * left_j(x)
        record = inspect_output(-x, x, omega)[0]
        self.assertTrue(record["accepted"])
        self.assertEqual(record["objective"], 8.0)
        self.assertAlmostEqual(record["lambda_min_k"], -1.0)
        self.assertFalse(record["global_accuracy_pass"])
        self.assertFalse(record["quality_pass"])

    def test_known_solution_passes_with_independent_projection_multiplier(self):
        for mu in (0.5, 1e-6):
            target, known = make_target(
                dict(family="known", n=8, k=2, mu=mu), 1701
            )
            for returned in (known["omega"], torch.zeros_like(known["omega"]), None,
                             torch.full_like(known["omega"], float("nan"))):
                with self.subTest(mu=mu, missing=returned is None):
                    record = inspect_output(target, known["x"], returned, known)[0]
                    self.assertTrue(record["accepted"])
                    self.assertTrue(record["global_accuracy_pass"])
                    self.assertTrue(record["quality_pass"])
                    self.assertAlmostEqual(record["lambda_min_k"], mu, places=13)
                    if returned is None or not bool(torch.isfinite(returned).all()):
                        self.assertFalse(record["returned_multiplier_available"])
                        self.assertIsNone(record["returned_multiplier_stationarity"])

    def test_unknown_optimum_uses_projected_multiplier_and_complex_dual(self):
        target, known = make_target(dict(family="known", n=8, k=2, mu=0.5), 1701)
        record = inspect_output(target, known["x"], torch.zeros_like(known["omega"]))[0]
        self.assertTrue(record["quality_pass"])
        self.assertAlmostEqual(record["lambda_min_k"], 0.5, places=13)
        self.assertAlmostEqual(record["computed_dual_value"], known["objective"], places=12)
        self.assertGreater(record["returned_multiplier_stationarity"], 0.1)

    def test_known_zero_target_passes_at_singular_dual_boundary(self):
        target, known = make_target(dict(family="zero", n=1, k=1), 1701)
        x = torch.eye(2, dtype=torch.float64).unsqueeze(0)
        record = inspect_output(target, x, None, known)[0]
        self.assertTrue(record["quality_pass"])
        self.assertAlmostEqual(record["lambda_min_k"], 0.0)
        self.assertEqual(record["dual_validation_status"], "not_positive_definite")

    def test_known_objective_gate_uses_absolute_relative_difference(self):
        target, known = make_target(dict(family="known", n=8, k=2, mu=0.5), 1701)
        known = {**known, "objective": known["objective"] + 1.0}
        record = inspect_output(target, known["x"], None, known)[0]
        self.assertTrue(record["accepted"])
        self.assertLess(record["relative_objective_error"], -1e-8)
        self.assertFalse(record["global_accuracy_pass"])
        self.assertFalse(record["quality_pass"])

    def test_strict_gap_global_accuracy_remains_unknown(self):
        target, known = make_target(dict(family="strict_gap", n=2, k=2), 1701)
        x = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        record = inspect_output(target, x, None, known)[0]
        self.assertIsNone(record["global_accuracy_pass"])
        self.assertFalse(record["quality_pass"])
        self.assertEqual(record["known_dual_value"], 18.0)
        self.assertIn("difference_from_known_dual", record)

    def test_projection_failure_keeps_other_batch_items(self):
        target = torch.eye(2, dtype=torch.float64).repeat(2, 1, 1)
        x = target.clone()
        x[0].zero_()
        records = inspect_output(target, x, None)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["reason"], "projection_failed")
        self.assertIn("objective", records[0])
        self.assertIn("feasibility", records[0])
        self.assertFalse(records[0]["quality_pass"])
        self.assertTrue(records[1]["quality_pass"])


if __name__ == "__main__":
    unittest.main()
