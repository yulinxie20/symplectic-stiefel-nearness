# 来源：2026-10-13_a800_symplectic_dual/symplectic_gpu/geometry.py
"""常欧氏辛 Stiefel 几何；仅形成 2k 阶方阵，支持 CPU 和 CUDA。"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .common import left_j, skew, sym


@dataclass
class ProjectionCache:
    """冻结在同一基点的投影谱分解，避免每次 Hessian 作用重复分解。"""

    x: torch.Tensor
    jx: torch.Tensor
    values: torch.Tensor
    vectors: torch.Tensor
    denominator: torch.Tensor

    def project_with_multiplier(self, value: torch.Tensor):
        coupling = self.jx.mT @ value
        rhs = coupling - coupling.mT
        rotated = self.vectors.mT @ rhs @ self.vectors
        multiplier = skew(self.vectors @ (rotated / self.denominator) @ self.vectors.mT)
        return value - self.jx @ multiplier, multiplier

    def project(self, value: torch.Tensor) -> torch.Tensor:
        return self.project_with_multiplier(value)[0]


def prepare_projection(x: torch.Tensor) -> ProjectionCache:
    """求解 (XᵀX)Λ+Λ(XᵀX)=(JX)ᵀA-AᵀJX。"""
    values, vectors = torch.linalg.eigh(sym(x.mT @ x))
    if not bool(torch.isfinite(values).all()) or float(values.min()) <= 0:
        raise FloatingPointError("辛基点的 Gram 矩阵不是正定矩阵")
    return ProjectionCache(x, left_j(x), values, vectors,
                           values[:, None] + values[None, :])


def orthosymplectic_initial(target: torch.Tensor) -> torch.Tensor:
    """以复极分解构造正交辛初值，含 thin QR 与 k 阶 SVD。

    B=T_top+iT_bottom，C=B_q-iB_p，Q=polar(C)，U=[Q,iQ]。
    即使 C 秩亏，QR/SVD 仍提供一个可行的正交补全。
    """
    n, k = target.shape[0] // 2, target.shape[1] // 2
    # 直接组装 C，避免额外分配一份完整 n×2k 复数目标矩阵。
    c = torch.complex(target[:n, :k] + target[n:, k:],
                      target[n:, :k] - target[:n, k:])
    q, r = torch.linalg.qr(c, mode="reduced")
    u, _, vh = torch.linalg.svd(r, full_matrices=False)
    q = q @ (u @ vh)
    return torch.cat((torch.cat((q.real, -q.imag), dim=1),
                      torch.cat((q.imag, q.real), dim=1)), dim=0).contiguous()


@dataclass
class CayleyRetraction:
    """Gao Cayley 的 2k 阶等价式；普通回缩不声称欧氏二阶性质。"""

    x: torch.Tensor
    z: torch.Tensor
    k_matrix: torch.Tensor
    c_matrix: torch.Tensor
    xc: torch.Tensor
    identity: torch.Tensor

    def increment(self, step: float = 1.0) -> torch.Tensor:
        denominator = self.identity - (0.5 * step) * self.k_matrix + (0.25 * step**2) * self.c_matrix
        rhs = step * (self.z - (0.5 * step) * self.xc)
        # 返回解析增量，避免 trial-X 在接近驻点时的消去误差。
        return torch.linalg.solve(denominator.mT, rhs.mT).mT

    def __call__(self, step: float = 1.0) -> torch.Tensor:
        return self.x + self.increment(step)


def prepare_cayley(x: torch.Tensor, tangent: torch.Tensor) -> CayleyRetraction:
    """只解 2k 阶系统，不构造 2n 阶 Hamiltonian 提升矩阵。"""
    k_matrix = -left_j(x.mT @ left_j(tangent))
    horizontal = tangent - x @ k_matrix
    c_matrix = -left_j(horizontal.mT @ left_j(horizontal))
    return CayleyRetraction(x, tangent, k_matrix, c_matrix, x @ c_matrix,
                            torch.eye(x.shape[1], dtype=x.dtype, device=x.device))
