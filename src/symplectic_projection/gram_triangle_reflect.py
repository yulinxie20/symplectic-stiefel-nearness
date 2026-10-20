"""整数反射对角核；保持原 Gram 接口和当前 CUDA 流。"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from .cuda_kernels import compile_source, launch


@lru_cache(maxsize=1)
def compile_kernel():
    return compile_source(
        Path(__file__).with_suffix(".cu").read_text(encoding="utf-8"),
        {name: "gram_" + name for name in ("packed", "offdiag", "diagonal")})


def gram_add_(A, D, Cr, Ci, split_diagonal=True):
    """Cr += A.T@A + D.T@D；Ci += A.T@D - D.T@A。输入输出不别名。"""
    batch, rows, m = A.shape
    assert D.shape == A.shape and Cr.shape == Ci.shape == (batch, m, m)
    assert all(x.is_cuda and x.dtype == torch.float64 and x.device == A.device
               and x.is_contiguous() for x in (A, D, Cr, Ci))
    _, kernels, _, _ = compile_kernel()
    tensors = (A, D, Cr, Ci)
    scalars = (np.int32(rows), np.int32(m))
    tiles = (m + 15) // 16
    if not split_diagonal:
        launch(kernels["packed"], (tiles*(tiles+1)//2, 1, batch),
               (16, 16, 1), tensors, scalars)
    else:
        if tiles > 1:
            launch(kernels["offdiag"], (tiles*(tiles-1)//2, 1, batch),
                   (16, 16, 1), tensors, scalars)
        launch(kernels["diagonal"], (tiles, 1, batch),
               (128, 1, 1), tensors, scalars)
