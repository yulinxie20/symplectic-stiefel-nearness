"""最近实辛 Stiefel 矩阵的批量 GPU 求解器。"""
from . import cholesky_inverse_kernel, gram_triangle_reflect, hvp_fused, recovery_fused
from .dual_recommended_hvp import solve

__all__ = ["initialize_kernels", "solve"]

_initialized = False


def initialize_kernels():
    """每个进程在求解或正式计时前，统一编译各组核并保存句柄。"""
    global _initialized
    if _initialized:
        return
    gram_triangle_reflect.initialize_kernels()
    hvp_fused.initialize_kernels()
    recovery_fused.initialize_kernels()
    cholesky_inverse_kernel.initialize_kernels()
    _initialized = True
