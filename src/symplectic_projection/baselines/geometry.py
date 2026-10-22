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
