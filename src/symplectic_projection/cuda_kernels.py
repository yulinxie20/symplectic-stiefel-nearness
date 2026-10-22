"""在单 GPU 的默认流上编译和启动 CuPy 核，张量存储由 PyTorch 管理。"""
import numpy as np
import cupy as cp


def compile_source(source, names):
    """一次编译整组核函数；调用方保存模块及全部入口句柄。"""
    module = cp.RawModule(
        code=source,
        options=("--std=c++17",),
        backend="nvrtc",
    )
    kernels = {key: module.get_function(name) for key, name in names.items()}
    return module, kernels


def launch(kernel, grid, block, tensors, scalars=()):
    """零复制指针传参；PyTorch 与 CuPy 在同一默认流上按调用顺序执行。"""
    arguments = tuple(np.uint64(tensor.data_ptr()) for tensor in tensors) + tuple(scalars)
    kernel(grid, block, arguments)
