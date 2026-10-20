"""HVP、线搜索及预条件/CG 的 FP64 融合核；CuPy 编译，在 PyTorch 当前流启动。"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from .cuda_kernels import compile_source


@lru_cache(maxsize=1)
def compile_kernel():
    """每进程编译一次；基准前调用以排除编译和模块加载时间。"""
    names = ("inner", "outer", "sandwich_front", "outer_off16", "outer_diag16",
             "line_partial", "line_finish", "precondition", "cg_update", "cg_next")
    return compile_source(Path(__file__).with_suffix(".cu").read_text(encoding="utf-8"),
                          {name: "hvp_" + name for name in names})


def inner_products(E, Sr, Si):
    """返回 U=E@Sr、V=E@Si；输入可非连续，输出连续。"""
    batch, m, columns = E.shape
    assert columns == m and Sr.shape == Si.shape == E.shape
    assert all(x.is_cuda and x.dtype == torch.float64 and x.device == E.device
               for x in (E, Sr, Si))
    U = torch.empty(E.shape, dtype=E.dtype, device=E.device)
    V = torch.empty_like(U)
    _, kernels, launch, _ = compile_kernel()
    scalars = [np.int32(m)]
    scalars += [np.int64(s) for x in (E, Sr, Si) for s in x.stride()]
    tiles = (m + 15) // 16
    launch(kernels["inner"], (tiles, tiles, batch), (16, 16, 1), (E, Sr, Si, U, V), scalars)
    return U, V


def outer_skew(P, Q, U, V):
    """返回 Z-Z.T，其中 Z=P@U-Q@V；直接写入反对称结果。"""
    batch, m, columns = P.shape
    assert columns == m and Q.shape == U.shape == V.shape == P.shape
    assert all(x.is_cuda and x.dtype == torch.float64 and x.device == P.device
               for x in (P, Q, U, V))
    H = torch.empty(P.shape, dtype=P.dtype, device=P.device)
    _, kernels, launch, _ = compile_kernel()
    tensors = (P, Q, U, V, H)
    if m == 128 and all(x.is_contiguous() for x in (P, Q, U, V)):
        launch(kernels["outer_off16"], (28, 1, batch), (16, 16, 1), tensors)
        launch(kernels["outer_diag16"], (8, 1, batch), (128, 1, 1), tensors)
        return H
    scalars = [np.int32(m)]
    scalars += [np.int64(s) for x in (P, Q, U, V) for s in x.stride()]
    tiles = (m + 15) // 16
    launch(kernels["outer"], (tiles*(tiles+1)//2, 1, batch), (16, 16, 1), tensors, scalars)
    return H


def sandwich_front(values, Ci, P, Q):
    """返回 P*values-Q@Ci、P@Ci+Q*values；按列缩放，输入可非连续。"""
    batch, m, columns = P.shape
    assert columns == m and Ci.shape == Q.shape == P.shape and values.shape == (batch, m)
    assert all(x.is_cuda and x.dtype == torch.float64 and x.device == P.device
               for x in (values, Ci, P, Q))
    Yr = torch.empty(P.shape, dtype=P.dtype, device=P.device)
    Yi = torch.empty_like(Yr)
    _, kernels, launch, _ = compile_kernel()
    scalars = [np.int32(m)]
    scalars += [np.int64(s) for x in (values, Ci, P, Q) for s in x.stride()]
    tiles = (m + 15) // 16
    launch(kernels["sandwich_front"], (tiles, tiles, batch), (16, 16, 1),
           (values, Ci, P, Q, Yr, Yi), scalars)
    return Yr, Yi


def line_change(gradient, delta, V, delta_Sr, trial, delta_Si):
    """两项各自固定树归约后相加，返回每题线搜索 change。"""
    inputs = (gradient, delta, V, delta_Sr, trial, delta_Si)
    batch, m, columns = V.shape
    assert batch > 0 and m > 0 and columns == m
    assert all(x.shape == V.shape and x.is_cuda and x.dtype == torch.float64 and x.device == V.device
               for x in inputs)
    tiles = (m + 15) // 16
    partial = torch.empty((batch, 2, tiles * tiles), dtype=V.dtype, device=V.device)
    output = torch.empty((batch,), dtype=V.dtype, device=V.device)
    _, kernels, launch, _ = compile_kernel()
    scalars = [np.int32(m)]
    scalars += [np.int64(stride) for x in inputs for stride in x.stride()]
    launch(kernels["line_partial"], (tiles, tiles, batch), (16, 16, 1), inputs + (partial,), scalars)
    launch(kernels["line_finish"], (batch, 1, 1), (256, 1, 1),
           (partial, output), (np.int32(tiles * tiles),))
    return output


def _check_vector_matrices(*matrices):
    first = matrices[0]
    assert first.ndim == 3
    batch, m, columns = first.shape
    assert batch > 0 and m > 0 and columns == m
    assert all(x.shape == first.shape and x.is_cuda and x.dtype == torch.float64
               and x.device == first.device for x in matrices)
    return batch, m


def _check_batch_vector(vector, matrix, dtype):
    assert vector.shape == (matrix.shape[0],) and vector.is_cuda
    assert vector.device == matrix.device and vector.dtype == dtype


def _launch_vector(name, reference, tensors, scalars):
    batch, m, _ = reference.shape
    _, kernels, launch, _ = compile_kernel()
    launch(kernels[name], ((m*m + 255)//256, batch, 1), (256, 1, 1), tensors, scalars)


def fused_precondition(R, denominator):
    """按完整两个位置计算 skew(R/denominator)，不依赖输入对称性。"""
    _, m = _check_vector_matrices(R, denominator)
    output = torch.empty(R.shape, dtype=R.dtype, device=R.device)
    scalars = [np.int32(m)]
    scalars += [np.int64(s) for x in (R, denominator) for s in x.stride()]
    _launch_vector("precondition", R, (R, denominator, output), scalars)
    return output


def cg_update(direction, r, p, Hp, alpha):
    """融合两项更新；输出均为新张量，原输入及 initial_z 别名保持不变。"""
    _, m = _check_vector_matrices(direction, r, p, Hp)
    _check_batch_vector(alpha, direction, torch.float64)
    new_direction = torch.empty(direction.shape, dtype=direction.dtype, device=direction.device)
    new_r = torch.empty_like(new_direction)
    scalars = [np.int32(m)]
    scalars += [np.int64(s) for x in (direction, r, p, Hp) for s in x.stride()]
    scalars += [np.int64(alpha.stride(0))]
    _launch_vector("cg_update", direction, (direction, r, p, Hp, alpha, new_direction, new_r), scalars)
    return new_direction, new_r


def cg_next(z, p, beta, cg_active):
    """融合 beta*p、加 z、最终活动掩码选择，不计算或更新掩码本身。"""
    _, m = _check_vector_matrices(z, p)
    _check_batch_vector(beta, z, torch.float64)
    _check_batch_vector(cg_active, z, torch.bool)
    output = torch.empty(z.shape, dtype=z.dtype, device=z.device)
    scalars = [np.int32(m)]
    scalars += [np.int64(s) for x in (z, p) for s in x.stride()]
    scalars += [np.int64(beta.stride(0)), np.int64(cg_active.stride(0))]
    _launch_vector("cg_next", z, (z, p, beta, cg_active, output), scalars)
    return output
