# SPDX-License-Identifier: GPL-3.0-or-later
# Original MATLAB implementation: Rasmus Jensen, symplecticStiefelfactory.m.
# Source: https://github.com/JensenRasmus/secondorder_spst
# Commit: c121d28f2116da6a566b8eff44642a42d8971bd0
# Python/PyTorch adaptation by the symplectic-stiefel-nearness contributors.
"""Jensen--Zimmermann 的右不变几何与 type 1 回缩。

保留作者的度量、梯度、投影和论文采用的 R-TR2 投影梯度导数。
所有环境阶矩阵乘积均以瘦矩阵作用实现，只分解列数相关的小矩阵。
输入是单个实二维 Tensor，缓存基点后不得原位修改它。

来源中的 ``ehesstorhess2`` 对应投影后的梯度导数，即论文的 R--TR2。
许可全文及第三方来源见根目录 THIRD_PARTY_NOTICES.txt。
"""
from __future__ import annotations

import torch


SOURCE = {
    "repository": "https://github.com/JensenRasmus/secondorder_spst",
    "commit": "c121d28f2116da6a566b8eff44642a42d8971bd0",
    "file": "symplecticStiefelfactory.m",
    "author": "Rasmus Jensen",
    "license": "GPL-3.0-or-later",
    "metric": "Bendokat--Zimmermann right-invariant metric",
    "retraction_type": 1,
}


def symplectic_plus(a: torch.Tensor) -> torch.Tensor:
    """A^+ = -J A^T J，按源文件四块转置实现，不创建稠密 J。"""
    if a.ndim != 2 or a.shape[0] % 2 or a.shape[1] % 2:
        raise ValueError("symplectic_plus requires an even-by-even matrix")
    r, c = a.shape[0] // 2, a.shape[1] // 2
    return torch.cat((torch.cat((a[r:, c:].mT, -a[:r, c:].mT), dim=1),
                      torch.cat((-a[r:, :c].mT, a[:r, :c].mT), dim=1)), dim=0)


def _right_solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """MATLAB 的 a / b，即 a b^{-1}；不假设 b 对称。"""
    return torch.linalg.solve(b.mT, a.mT).mT


class JensenGeometry:
    """冻结基点的右不变几何；基点须满足 X^+ X = I。

    不在构造时修正可行性或改变 dtype/device。float64 可在 CPU/CUDA
    使用；若 Gram 矩阵奇异，由 torch.linalg 报错。
    """

    def __init__(self, x: torch.Tensor):
        if (x.ndim != 2 or x.shape[0] % 2 or x.shape[1] % 2
                or x.shape[1] == 0 or x.shape[0] < x.shape[1]):
            raise ValueError("x must have shape (2*n, 2*k), n >= k >= 1")
        if x.is_complex() or not x.is_floating_point():
            raise TypeError("the Jensen geometry requires a real floating tensor")
        self.x = x
        self.n, self.k = x.shape[0] // 2, x.shape[1] // 2
        self.gram = x.mT @ x
        self.gram_inv = torch.linalg.inv(self.gram)
        self.x_plus = symplectic_plus(x)
        self._xt_plus = self.x_plus.mT
        self._gram_inv_plus = symplectic_plus(self.gram_inv)
        self._identity = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)

    def inner(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        correction = b - 0.5 * (self._xt_plus @ (
            self._gram_inv_plus @ (self.x_plus @ b)))
        # tr(a^T correction G^{-1}) = <a, correction G^{-1}>_F.
        return torch.sum(a * (correction @ self.gram_inv))

    def norm(self, v: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.inner(v, v))

    def project(self, v: torch.Tensor) -> torch.Tensor:
        coupling = symplectic_plus(v) @ self.x
        normal = coupling + symplectic_plus(coupling)
        if self.n == self.k:
            # 原 factory 的辛群端点分支。
            return v - 0.5 * (self.x @ normal)
        return v - 0.5 * (self._xt_plus @ (self._gram_inv_plus @ normal))

    def gradient(self, egrad: torch.Tensor) -> torch.Tensor:
        return egrad @ self.gram - self._xt_plus @ (
            symplectic_plus(egrad) @ self.x)

    def _gradient_derivative(self, egrad: torch.Tensor, ehess: torch.Tensor,
                             v: torch.Tensor) -> torch.Tensor:
        """源 FYY 的重结合；ehess 是环境 Hessian 在 v 上的作用。

        使用 (AB)^+ = B^+ A^+，例如
        (egrad v^T)^+ X = (v^+)^T (egrad^+ X)。
        """
        vp, ep = symplectic_plus(v), symplectic_plus(egrad)
        vtx = v.mT @ self.x
        return (ehess @ self.gram + egrad @ (vtx + vtx.mT)
                - vp.mT @ (ep @ self.x)
                - self._xt_plus @ (symplectic_plus(ehess) @ self.x + ep @ v))

    def hessian(self, egrad: torch.Tensor, ehess: torch.Tensor,
                v: torch.Tensor) -> torch.Tensor:
        """R-TR2：作者的投影梯度导数近似。

        传入的 ehess 必须已经是环境 Hessian 在 v 上的作用。
        """
        derivative = self._gradient_derivative(egrad, ehess, v)
        return self.project(derivative)

    def retract(self, v: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        """作者的 type 1 右不变 Cayley 回缩，最大求解阶数为 8k。

        以 (I-X X^+)Z=Z-X(X^+Z) 消去环境阶投影矩阵。
        最后 w=I+t hatX (I-t hatY^T hatX/2)^{-1} hatY^T，
        直接计算 w*cay_right，避免显式形成 w 或 w-I。
        """
        x, xp, gi, gip = self.x, self.x_plus, self.gram_inv, self._gram_inv_plus
        vp_t = symplectic_plus(v).mT
        z = vp_t @ gip
        a = (-xp @ z + gi @ (v.mT @ x)
             - gi @ (v.mT @ (self._xt_plus @ gip)))
        h = -z + x @ (xp @ z)
        bar_delta = x @ a + h
        bp = symplectic_plus(bar_delta)
        xx = torch.cat((bar_delta - 0.5 * (x @ (xp @ bar_delta)), -x), dim=1)
        yy = torch.cat((xp.mT, (bp - 0.5 * ((bp @ x) @ xp)).mT), dim=1)
        hat_x, hat_y = torch.cat((yy, -xx), dim=1), torch.cat((xx, yy), dim=1)
        theta = self._identity - 0.5 * t * a + 0.25 * t**2 * (symplectic_plus(h) @ h)
        cay_right = -x + _right_solve(t * h + 2 * x, theta)
        small = torch.eye(hat_x.shape[1], dtype=x.dtype, device=x.device)
        small = small - 0.5 * t * (hat_y.mT @ hat_x)
        return cay_right + t * (hat_x @ torch.linalg.solve(small, hat_y.mT @ cay_right))
