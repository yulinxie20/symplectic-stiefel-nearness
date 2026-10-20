# 来源：2026-10-13_a800_symplectic_dual/symplectic_gpu/common.py
"""公共数值约定：目标 ||X-T||²，J=[[0,I],[-I,0]]，float64。"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch


@dataclass
class SolverOptions:
    max_iter: int = 300
    grad_tol: float = 1e-8
    feas_tol: float = 1e-8
    max_backtracks: int = 40
    cg_max_iter: int = 100
    cg_rtol: float = 0.1
    time_limit: float = 120.0
    history: bool = True

    def __post_init__(self) -> None:
        if self.max_iter < 0 or self.max_backtracks < 1 or self.cg_max_iter < 1:
            raise ValueError("迭代预算无效")
        if not all(math.isfinite(x) and x > 0 for x in (self.grad_tol, self.feas_tol, self.time_limit)):
            raise ValueError("容差与时间预算必须有限且为正")
        if not 0 < self.cg_rtol < 1:
            raise ValueError("cg_rtol 必须在 (0,1) 中")


@dataclass
class SolveResult:
    x: torch.Tensor | None
    omega: torch.Tensor | None
    status: str
    iterations: int
    history: list[dict] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status in {"converged", "success"}


def standard_j(k: int, device=None, dtype=torch.float64) -> torch.Tensor:
    if k < 1:
        raise ValueError("k 必须为正")
    j = torch.zeros((2 * k, 2 * k), dtype=dtype, device=device)
    eye = torch.eye(k, dtype=dtype, device=device)
    j[:k, k:] = eye
    j[k:, :k] = -eye
    return j


def left_j(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or x.shape[0] % 2:
        raise ValueError("left_j 需要偶数行的二维矩阵")
    n = x.shape[0] // 2
    return torch.cat((x[n:], -x[:n]), dim=0)


def skew(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x - x.mT)


def sym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.mT)


def inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum((a.conj() * b).real)


def synchronize(device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_target(target: torch.Tensor) -> None:
    if not isinstance(target, torch.Tensor) or target.dtype != torch.float64:
        raise ValueError("目标必须为 torch.float64 实矩阵；不隐式转换精度")
    if target.ndim != 2 or min(target.shape) == 0 or any(s % 2 for s in target.shape):
        raise ValueError("目标形状须为 (2n,2k)，n>=k>=1")
    if target.shape[0] < target.shape[1]:
        raise ValueError("需要 n>=k")
    if target.device.type not in {"cpu", "cuda"}:
        raise ValueError("仅支持 CPU/CUDA")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("目标包含非有限值")
