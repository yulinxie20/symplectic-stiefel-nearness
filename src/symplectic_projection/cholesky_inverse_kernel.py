"""128 阶 FP64 Cholesky 因子的批量求逆核。

每个 warp 共享 L 读取并计算 R=L^-1 的相邻两列；
随后用 128 线程块计算 P=R.T@R 并对称写回。
适用于有效的下三角 Cholesky 因子；输入形状 [batch,128,128]，
允许非连续步长，输出连续；推荐求解器在该阶数使用此核。
实验开始前统一初始化核函数，计算时直接复用已保存的句柄。
"""
from pathlib import Path

import numpy as np
import torch
from .cuda_kernels import compile_source, launch

_module = None  # 保留 RawModule，供已保存的核句柄复用。
_kernels = {}


def initialize_kernels():
    global _module, _kernels
    _module, _kernels = compile_source(
        Path(__file__).with_suffix(".cu").read_text(encoding="utf-8"),
        {name: name for name in ("triangular_inverse", "inverse_product")})


@torch.no_grad()
def inverse_from_cholesky(L):
    """返回 (L@L.T)^-1；L 必须是 CUDA float64 [batch,128,128] 的有效下三角因子。"""
    if L.ndim != 3 or L.shape[-2:] != (128, 128) or not L.is_cuda or L.dtype != torch.float64:
        raise ValueError("要求 CUDA float64 张量，形状为 [batch,128,128]")
    batch = L.shape[0]
    P = torch.empty(L.shape, dtype=L.dtype, device=L.device)
    if batch == 0:
        return P
    R = torch.empty_strided(L.shape, (16384, 1, 128), dtype=L.dtype, device=L.device)
    launch(_kernels["triangular_inverse"], (16, 1, batch), (128, 1, 1),
           (L, R), tuple(np.int64(s) for s in L.stride()))
    launch(_kernels["inverse_product"], (36, 1, batch), (128, 1, 1), (R, P))
    return P
