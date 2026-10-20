"""GPU 正确性回归；运行：python -m unittest discover -s tests -v。"""
import math
import sys
import unittest

import numpy
import torch

from symplectic_projection import solve
import cupy


def J(matrix):
    n = matrix.shape[-2] // 2
    return torch.cat((matrix[..., n:, :], -matrix[..., :n, :]), dim=-2)


def known_case(n=128, k=4):
    """构造满足约束及驻点方程的 X、内部 omega 和 T。"""
    generator = torch.Generator().manual_seed(20261019)
    basis = torch.linalg.qr(
        torch.randn(n, 2 * k, dtype=torch.float64, generator=generator)
    ).Q
    left, right = basis[:, :k], basis[:, k:]
    A = torch.cat((math.sqrt(1.5) * left, math.sqrt(0.5) * right), dim=-1)
    D = torch.cat((math.sqrt(0.5) * right, math.sqrt(1.5) * left), dim=-1)
    X = torch.cat((A, D), dim=0).unsqueeze(0)
    omega = 0.3 * J(torch.eye(2 * k, dtype=torch.float64)).unsqueeze(0)
    return X - J(X) @ omega, X, omega


@unittest.skipUnless(torch.cuda.is_available(), "需要 NVIDIA GPU 和 CUDA 版 PyTorch")
class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(
            f"\nPython: {sys.executable}; {sys.version.split()[0]}; "
            f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}; "
            f"NumPy: {numpy.__version__}; CuPy: {cupy.__version__}; "
            f"GPU: {torch.cuda.get_device_name()}"
        )

    def assert_residuals(self, T, X, omega):
        m = T.shape[-1]
        identity_j = J(torch.eye(m, dtype=T.dtype))
        feasibility = torch.linalg.vector_norm(
            X.mT @ J(X) - identity_j, dim=(-2, -1)
        ) / math.sqrt(m)
        stationarity = torch.linalg.vector_norm(
            X - T - J(X) @ omega, dim=(-2, -1)
        ) / torch.linalg.vector_norm(T, dim=(-2, -1)).clamp_min(1)
        self.assertTrue(torch.isfinite(X).all().item())
        self.assertLess(feasibility.max().item(), 1e-8)
        self.assertLess(stationarity.max().item(), 1e-9)
        self.assertEqual(X.device.type, "cpu")
        self.assertEqual(omega.device.type, "cpu")
        self.assertEqual(X.dtype, torch.float64)
        self.assertEqual(omega.dtype, torch.float64)

    def test_known_solution(self):
        T, expected_X, expected_omega = known_case()
        original = T.clone()
        X, omega, iterations = solve(T)
        self.assert_residuals(T, X, omega)
        torch.testing.assert_close(X, expected_X, rtol=1e-7, atol=1e-8)
        torch.testing.assert_close(omega, expected_omega, rtol=1e-7, atol=1e-8)
        self.assertTrue(torch.equal(T, original))
        self.assertGreater(iterations[0].item(), 0)

    def test_initially_converged(self):
        _, T, _ = known_case()
        X, omega, iterations = solve(T)
        self.assert_residuals(T, X, omega)
        self.assertEqual(iterations.tolist(), [0])
        self.assertEqual(torch.count_nonzero(omega).item(), 0)
        torch.testing.assert_close(X, T, rtol=1e-13, atol=1e-13)

    def test_mixed_activity(self):
        T, feasible, _ = known_case()
        batch = torch.cat((feasible, T))
        X, omega, iterations = solve(batch)
        self.assert_residuals(batch, X, omega)
        self.assertEqual(iterations[0].item(), 0)
        self.assertGreater(iterations[1].item(), 0)
        torch.testing.assert_close(X[0], feasible[0], rtol=1e-13, atol=1e-13)

    def test_noncontiguous_and_tail_size(self):
        T, _, _ = known_case(n=131, k=5)
        storage = torch.empty(*T.shape[:-1], 2 * T.shape[-1], dtype=T.dtype)
        view = storage[..., ::2]
        view.copy_(T)
        self.assertFalse(view.is_contiguous())
        result = solve(view)
        reference = solve(T)
        self.assert_residuals(view, result[0], result[1])
        for actual, expected in zip(result, reference):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_inverse_paths(self):
        # 64 阶采用库回退；128 阶采用专用两核求逆。
        for k in (32, 64):
            with self.subTest(k=k):
                n = 512
                T = torch.randn(
                    2, 2 * n, 2 * k, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(20261019),
                ) / math.sqrt(2 * n)
                X, omega, iterations = solve(T)
                self.assert_residuals(T, X, omega)
                self.assertTrue(torch.all(iterations > 0).item())


if __name__ == "__main__":
    unittest.main()
