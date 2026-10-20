"""CPU 几何与残差验证；不调用 GPU 求解器或 CUDA。

运行：python -m unittest discover -s tests -p test_riemannian_baselines.py -v
"""
import math
import sys
import unittest
from importlib.metadata import version

import torch

from symplectic_projection.baselines import SolverOptions, solve_riemannian
from symplectic_projection.baselines.common import inner, left_j, skew, standard_j
from symplectic_projection.baselines.geometry import (
    orthosymplectic_initial,
    prepare_cayley,
    prepare_projection,
)
from symplectic_projection.baselines.riemannian import (
    EuclideanGeometry,
    objective_increment,
    truncated_cg,
)


class RiemannianBaselineTests(unittest.TestCase):
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

    def setUp(self):
        self.generator = torch.Generator(device="cpu").manual_seed(1701)

    def random(self, *shape):
        return torch.randn(
            *shape, generator=self.generator, dtype=torch.float64, device="cpu"
        )

    def feasible_point(self, n=9, k=2):
        x = torch.zeros((2 * n, 2 * k), dtype=torch.float64, device="cpu")
        x[:k, :k] = torch.eye(k, dtype=x.dtype)
        x[n:n + k, k:] = torch.eye(k, dtype=x.dtype)
        z = prepare_projection(x).project(self.random(2 * n, 2 * k))
        return prepare_cayley(x, z)(0.15)

    def known_solution(self):
        x = self.feasible_point()
        omega = skew(self.random(4, 4))
        omega *= 0.35 / torch.linalg.matrix_norm(omega, ord=2)
        # K=I+i*omega 正定；T=X-JX*omega 满足原始驻点与全局证书。
        target = x - left_j(x) @ omega
        return x, target, omega

    def test_initial_is_orthosymplectic_for_rank_deficient_targets(self):
        targets = (
            self.random(18, 4),
            torch.zeros((18, 4), dtype=torch.float64),
            self.random(18, 1) @ self.random(1, 4),
        )
        for target in targets:
            with self.subTest(rank=int(torch.linalg.matrix_rank(target))):
                x = orthosymplectic_initial(target)
                self.assertEqual(x.device.type, "cpu")
                self.assertEqual(x.dtype, torch.float64)
                self.assertLess(
                    float(torch.linalg.vector_norm(x.mT @ left_j(x) - standard_j(2))),
                    1e-12,
                )
                self.assertLess(
                    float(torch.linalg.vector_norm(x.mT @ x - torch.eye(4))), 1e-12
                )

    def test_projection_is_tangent_idempotent_and_self_adjoint(self):
        x = self.feasible_point()
        cache = prepare_projection(x)
        a, b = self.random(*x.shape), self.random(*x.shape)
        z, multiplier = cache.project_with_multiplier(a)
        self.assertLess(
            float(torch.linalg.vector_norm(z.mT @ left_j(x) + x.mT @ left_j(z))),
            1e-11,
        )
        self.assertLess(float(torch.linalg.vector_norm(cache.project(z) - z)), 1e-11)
        self.assertAlmostEqual(
            float(inner(cache.project(a), b)), float(inner(a, cache.project(b))), places=11
        )
        gram = x.mT @ x
        coupling = left_j(x).mT @ a
        sylvester_residual = (
            gram @ multiplier + multiplier @ gram - coupling + coupling.mT
        )
        self.assertLess(float(torch.linalg.vector_norm(sylvester_residual)), 1e-11)
        self.assertLess(float(torch.linalg.vector_norm(multiplier + multiplier.mT)), 1e-13)

    def test_cayley_derivative_and_feasibility(self):
        x = self.feasible_point()
        z = prepare_projection(x).project(self.random(*x.shape))
        retraction = prepare_cayley(x, z)
        for step in (0.001, 0.1, 1.0):
            with self.subTest(step=step):
                trial = retraction(step)
                self.assertLess(
                    float(torch.linalg.vector_norm(
                        trial.mT @ left_j(trial) - standard_j(2)
                    )),
                    2e-10,
                )
        epsilon = 1e-5
        derivative = (retraction(epsilon) - retraction(-epsilon)) / (2 * epsilon)
        self.assertLess(float(torch.linalg.vector_norm(derivative - z)), 2e-8)

    def test_hessian_finite_difference_and_symmetry(self):
        x = self.feasible_point()
        target = self.random(*x.shape)
        geometry = EuclideanGeometry(x, target)
        z = geometry.project(self.random(*x.shape))
        z /= torch.linalg.vector_norm(z)
        w = geometry.project(self.random(*x.shape))
        retraction = prepare_cayley(x, z)
        epsilon = 1e-5
        plus = EuclideanGeometry(retraction(epsilon), target).gradient
        minus = EuclideanGeometry(retraction(-epsilon), target).gradient
        derivative = geometry.project((plus - minus) / (2 * epsilon))
        self.assertLess(
            float(torch.linalg.vector_norm(derivative - geometry.hessian(z))), 2e-8
        )
        self.assertAlmostEqual(
            float(inner(w, geometry.hessian(z))),
            float(inner(z, geometry.hessian(w))),
            places=10,
        )

    def test_quadratic_objective_has_no_half_factor(self):
        x = self.feasible_point()
        target = self.random(*x.shape)
        geometry = EuclideanGeometry(x, target)
        z = geometry.project(self.random(*x.shape))
        z /= torch.linalg.vector_norm(z)
        epsilon = 1e-5
        retraction = prepare_cayley(x, z)
        plus = float(torch.sum((retraction(epsilon) - target) ** 2))
        minus = float(torch.sum((retraction(-epsilon) - target) ** 2))
        derivative = (plus - minus) / (2 * epsilon)
        self.assertAlmostEqual(derivative, float(inner(geometry.gradient, z)), places=7)

    def test_stable_objective_increment(self):
        residual = torch.full((2, 2), 1e8, dtype=torch.float64)
        delta = torch.full((2, 2), -1e-10, dtype=torch.float64)
        stable = objective_increment(residual, delta)
        direct = float(inner(residual + delta, residual + delta) - inner(residual, residual))
        self.assertAlmostEqual(stable, -0.08, places=14)
        self.assertEqual(direct, 0.0)

    def test_truncated_cg_true_residual_and_negative_curvature(self):
        gradient = self.random(3, 2)
        eta, h_eta, details = truncated_cg(
            gradient, lambda z: 2 * z, 100.0, relative_tolerance=0.01, max_iter=10
        )
        self.assertTrue(details["residual_satisfied"])
        self.assertLess(float(torch.linalg.vector_norm(eta + 0.5 * gradient)), 1e-12)
        relative_residual = float(
            torch.linalg.vector_norm(gradient + h_eta) / torch.linalg.vector_norm(gradient)
        )
        self.assertAlmostEqual(details["relative_residual"], relative_residual, places=14)
        eta, _, details = truncated_cg(
            gradient, lambda z: -z, 0.2, relative_tolerance=0.01, max_iter=10
        )
        self.assertTrue(details["nonpositive_curvature"])
        self.assertAlmostEqual(float(torch.linalg.vector_norm(eta)), 0.2, places=12)
        self.assertGreater(details["predicted_decrease"], 0)

    def test_both_methods_recover_known_global_solution_and_residuals(self):
        expected, target, expected_omega = self.known_solution()
        options = SolverOptions(
            max_iter=500, grad_tol=1e-10, feas_tol=1e-10,
            cg_rtol=0.05, time_limit=60.0,
        )
        scale = max(1.0, float(torch.linalg.vector_norm(target)))
        original = target.clone()
        for method in ("abb", "rtr"):
            with self.subTest(method=method):
                result = solve_riemannian(target, options, method=method)
                self.assertEqual(result.status, "converged", msg=str(result.stats))
                self.assertTrue(torch.equal(target, original))
                self.assertEqual(result.x.device.type, "cpu")
                self.assertEqual(result.x.dtype, torch.float64)
                self.assertLess(float(torch.linalg.vector_norm(result.x - expected)), 2e-8)
                self.assertLess(
                    float(torch.linalg.vector_norm(result.omega - expected_omega)), 2e-8
                )
                feasibility = float(torch.linalg.vector_norm(
                    result.x.mT @ left_j(result.x) - standard_j(2)
                )) / math.sqrt(target.shape[1])
                stationarity = float(torch.linalg.vector_norm(
                    result.x - target - left_j(result.x) @ result.omega
                )) / scale
                self.assertLessEqual(feasibility, options.feas_tol)
                self.assertLessEqual(stationarity, options.grad_tol)
                self.assertAlmostEqual(
                    result.stats["gradient_relative"], stationarity, places=13
                )
                self.assertAlmostEqual(
                    result.stats["objective"], float(torch.sum((result.x - target) ** 2)),
                    places=13,
                )
                self.assertGreater(result.stats["counts"]["accepted_steps"], 0)

    def test_explicit_failure_and_input_validation(self):
        _, target, _ = self.known_solution()
        for method in ("abb", "rtr"):
            result = solve_riemannian(
                target, SolverOptions(max_iter=0), method=method
            )
            self.assertEqual(result.status, "max_iter")
            self.assertFalse(result.success)
        with self.assertRaises(ValueError):
            solve_riemannian(target, method="unsupported")
        with self.assertRaises((ValueError, TypeError)):
            solve_riemannian(target.float())
        with self.assertRaises(ValueError):
            solve_riemannian(target, initial=torch.zeros_like(target))


if __name__ == "__main__":
    unittest.main()
