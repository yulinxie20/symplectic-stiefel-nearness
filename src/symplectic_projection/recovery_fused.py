"""CuPy 编译恢复核，零复制使用 PyTorch 张量并在默认流启动。"""
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
        {"recover": "recover_fused"})


def recover_block(A, D, P, Q):
    """输入 float64 GPU 张量；A/D 为连续 [batch,rows,m]，P/Q 为 [batch,m,m]。"""
    assert all(x.is_cuda and x.dtype == torch.float64 and x.device == A.device for x in (A, D, P, Q))
    batch, rows, m = A.shape
    assert D.shape == A.shape and A.is_contiguous() and D.is_contiguous()
    assert P.shape == Q.shape == (batch, m, m)
    Ur, Ui = torch.empty_like(A), torch.empty_like(A)
    scalars = (np.int32(rows), np.int32(m))
    scalars += tuple(np.int64(s) for x in (P, Q) for s in x.stride())
    launch(_kernels["recover"], ((m+15)//16, (rows+15)//16, batch), (16, 16, 1),
           (A, D, P, Q, Ur, Ui), scalars)
    return Ur, Ui
