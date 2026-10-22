"""整数反射对角核；在默认流上累加 Gram 矩阵。"""
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
        {name: "gram_" + name for name in ("packed", "offdiag", "diagonal")})


def gram_add_(A, D, Cr, Ci, split_diagonal=True):
    """Cr += A.T@A + D.T@D；Ci += A.T@D - D.T@A。输入输出不别名。"""
    batch, rows, m = A.shape
    assert D.shape == A.shape and Cr.shape == Ci.shape == (batch, m, m)
    assert all(x.is_cuda and x.dtype == torch.float64 and x.device == A.device
               and x.is_contiguous() for x in (A, D, Cr, Ci))
    tensors = (A, D, Cr, Ci)
    scalars = (np.int32(rows), np.int32(m))
    tiles = (m + 15) // 16
    if not split_diagonal:
        launch(_kernels["packed"], (tiles*(tiles+1)//2, 1, batch),
               (16, 16, 1), tensors, scalars)
    else:
        if tiles > 1:
            launch(_kernels["offdiag"], (tiles*(tiles-1)//2, 1, batch),
                   (16, 16, 1), tensors, scalars)
        launch(_kernels["diagonal"], (tiles, 1, batch),
               (128, 1, 1), tensors, scalars)
