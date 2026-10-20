"""CuPy 编译恢复核，零复制使用 PyTorch 张量并在其当前 CUDA 流启动。"""
import ctypes as ct
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from .cuda_kernels import compile_source, launch


def bind(library, name, arguments):
    """保留历史实验的 ctypes 绑定接口；当前恢复核使用 CuPy。"""
    function = getattr(library, name)
    function.argtypes, function.restype = arguments, ct.c_int
    return function


def check(code):
    if code:
        raise RuntimeError(f"CUDA/NVRTC 调用失败，错误码 {code}")


@lru_cache(maxsize=1)
def compile_kernel():
    """每进程一次；保留 (module, kernel, launch, timing) 返回结构。"""
    module, kernels, launcher, timing = compile_source(
        Path(__file__).with_suffix(".cu").read_text(encoding="utf-8"),
        {"recover": "recover_fused"})
    return module, kernels["recover"], launcher, timing


def recover_block(A, D, P, Q):
    """输入 float64 GPU 张量；A/D 为连续 [batch,rows,m]，P/Q 为 [batch,m,m]。"""
    assert all(x.is_cuda and x.dtype == torch.float64 and x.device == A.device for x in (A, D, P, Q))
    batch, rows, m = A.shape
    assert D.shape == A.shape and A.is_contiguous() and D.is_contiguous()
    assert P.shape == Q.shape == (batch, m, m)
    Ur, Ui = torch.empty_like(A), torch.empty_like(A)
    _, kernel, _, _ = compile_kernel()
    scalars = (np.int32(rows), np.int32(m))
    scalars += tuple(np.int64(s) for x in (P, Q) for s in x.stride())
    launch(kernel, ((m+15)//16, (rows+15)//16, batch), (16, 16, 1),
           (A, D, P, Q, Ur, Ui), scalars)
    return Ur, Ui
