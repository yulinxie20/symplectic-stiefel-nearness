"""共用 CuPy RawModule 编译及 PyTorch 当前流启动，张量存储保持由 PyTorch 管理。"""
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch


_DLL_HANDLES = []
if sys.platform == "win32":
    torch_lib = Path(torch.__file__).parent / "lib"
    if torch_lib.is_dir():
        # 句柄需存活到进程结束，供 CuPy 延迟加载 NVRTC 及其 builtins。
        _DLL_HANDLES.append(os.add_dll_directory(str(torch_lib)))
    if not os.environ.get("CUDA_PATH"):
        cuda_root = Path(sys.prefix) / "Library"
        if ((cuda_root / "bin").is_dir()
                and all((cuda_root / "include" / name).is_file()
                        for name in ("cuda_runtime.h", "cuda_fp16.h", "vector_types.h"))):
            os.environ["CUDA_PATH"] = str(cuda_root)

import cupy as cp


def compile_source(source, names):
    """编译并取得各入口；调用方缓存四元组，CuPy 另按设备缓存内核。"""
    stream = torch.cuda.current_stream()
    start = time.perf_counter()
    options = ("--std=c++17",)
    with cp.cuda.Device(stream.device.index):
        module = cp.RawModule(code=source, options=options, backend="nvrtc")
        kernels = {key: module.get_function(name) for key, name in names.items()}
    timing = dict(
        nvrtc_ms=None,
        compile_and_load_ms=(time.perf_counter() - start) * 1000,
        options=list(options), log="",
        timing_note="CuPy 公共接口合并编译、缓存读取与模块加载，不单独报告 NVRTC 时间。",
    )
    return module, kernels, launch, timing


def launch(kernel, grid, block, tensors, scalars=()):
    """零复制指针传参；调用方提供有明确位宽的标量，并负责跨流数据依赖。"""
    tensors = tuple(tensors)
    device_id = tensors[0].device.index
    stream = torch.cuda.current_stream(device_id)
    arguments = tuple(np.uint64(tensor.data_ptr()) for tensor in tensors) + tuple(scalars)
    with cp.cuda.Device(device_id):
        external_stream = cp.cuda.ExternalStream(stream.cuda_stream, device_id=device_id)
        for tensor in tensors:
            tensor.record_stream(stream)
        kernel(grid, block, arguments, stream=external_stream)
