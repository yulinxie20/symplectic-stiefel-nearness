"""128 阶 FP64 Cholesky 因子的批量求逆核。

每个 warp 共享 L 读取并计算 R=L^-1 的相邻两列；
随后用 128 线程块计算 P=R.T@R 并对称写回。
适用于有效的下三角 Cholesky 因子；输入形状 [batch,128,128]，
允许非连续步长，输出连续；推荐求解器在该阶数使用此核。
首次调用包含 CuPy 编译/缓存加载；计时前调用 compile_kernel() 预热。
"""
from functools import lru_cache

import numpy as np
import torch
from .cuda_kernels import compile_source, launch

_CUDA_SOURCE = r"""
extern "C" __global__ void triangular_inverse(
    const double* L, double* R, long long s0, long long s1, long long s2)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int col0 = 2 * (blockIdx.x * (blockDim.x / 32) + warp);
    const int col1 = col0 + 1, batch = blockIdx.z;
    __shared__ double diagonal_inverse[128];
    for (int j = threadIdx.x; j < 128; j += blockDim.x)
        diagonal_inverse[j] = 1.0 / L[batch*s0 + j*s1 + j*s2];
    __syncthreads();

    double v0[4], v1[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        v0[t] = lane + 32*t == col0 ? 1.0 : 0.0;
        v1[t] = lane + 32*t == col1 ? 1.0 : 0.0;
    }
    for (int j = col0; j < 128; ++j) {
        // 固定下标选择有助于保持v0/v1为寄存器；两列各自仍按j递增。
        const double own0 = j < 32 ? v0[0] : j < 64 ? v0[1] : j < 96 ? v0[2] : v0[3];
        const double reciprocal = diagonal_inverse[j];
        const double pivot0 = __dmul_rn(__shfl_sync(0xffffffff, own0, j & 31), reciprocal);
        const bool second = j >= col1; // warp一致；第一轮不更新第二列。
        double pivot1 = 0.0;
        if (second) {
            const double own1 = j < 32 ? v1[0] : j < 64 ? v1[1] : j < 96 ? v1[2] : v1[3];
            pivot1 = __dmul_rn(__shfl_sync(0xffffffff, own1, j & 31), reciprocal);
        }
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int row = lane + 32*t;
            if (row > j) {
                // 同一个L[row,j]供相邻两列复用；各列保留原独立FMA链。
                const double lij = L[batch*s0 + row*s1 + j*s2];
                v0[t] = fma(-lij, pivot0, v0[t]);
                if (second) v1[t] = fma(-lij, pivot1, v1[t]);
            }
            if (row == j) {
                v0[t] = pivot0;
                if (second) v1[t] = pivot1;
            }
        }
    }
    const long long base = (long long)batch * 16384;
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int row = lane + 32*t;
        R[base + row + col0*128] = v0[t];
        R[base + row + col1*128] = v1[t];
    }
}

extern "C" __global__ void inverse_product(const double* R, double* P)
{
    const int packed = blockIdx.x;
    // 与当前永久实现相同的整数三角块定位；不恢复旧的浮点sqrt映射。
    const int bj = packed < 10
        ? (packed < 3 ? (packed < 1 ? 0 : 1) : (packed < 6 ? 2 : 3))
        : (packed < 21 ? (packed < 15 ? 4 : 5) : (packed < 28 ? 6 : 7));
    const int bi = packed - bj*(bj+1)/2;
    const int tid = threadIdx.x, row = tid >> 4, col = tid & 15;
    const bool diagonal = bi == bj;
    // 非对角：每线程同一列的上下两个输出；对角：128点反射+后8个对角点。
    const int u0 = diagonal && col < row ? 15-row : row;
    const int v0 = diagonal && col < row ? 15-col : col;
    const int u1 = diagonal ? tid+8 : row+8;
    const int v1 = diagonal ? tid+8 : col;
    const bool second = !diagonal || tid < 8;
    const long long base = (long long)blockIdx.z * 16384;
    __shared__ double a[16][17], b[16][17];
    double acc0 = 0.0, acc1 = 0.0;

    // 保持原核的块列K起点，包含原来实际执行的结构零FMA。
    for (int start = bj*16; start < 128; start += 16) {
        #pragma unroll
        for (int index = tid; index < 256; index += 128) {
            const int k = index >> 4, c = index & 15;
            const double left = R[base + start+k + (bi*16+c)*128];
            a[k][c] = left;
            b[k][c] = diagonal ? left : R[base + start+k + (bj*16+c)*128];
        }
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            const double right = b[t][v0];
            acc0 = fma(a[t][u0], right, acc0);
            if (second) {
                const double right1 = diagonal ? b[t][v1] : right;
                acc1 = fma(a[t][u1], right1, acc1);
            }
        }
        __syncthreads();
    }
    const int i0 = bi*16+u0, j0 = bj*16+v0;
    P[base + i0*128+j0] = acc0;
    if (i0 != j0) P[base + j0*128+i0] = acc0;
    if (second) {
        const int i1 = bi*16+u1, j1 = bj*16+v1;
        P[base + i1*128+j1] = acc1;
        if (i1 != j1) P[base + j1*128+i1] = acc1;
    }
}
"""


@lru_cache(maxsize=1)
def compile_kernel():
    """在当前 CUDA 设备编译，保留原三元组返回结构。"""
    module, functions, launcher, _ = compile_source(
        _CUDA_SOURCE, {name: name for name in ("triangular_inverse", "inverse_product")})
    return module, functions, launcher


@torch.no_grad()
def inverse_from_cholesky(L):
    """返回 (L@L.T)^-1；L 必须是 CUDA float64 [batch,128,128] 的有效下三角因子。"""
    if L.ndim != 3 or L.shape[-2:] != (128, 128) or not L.is_cuda or L.dtype != torch.float64:
        raise ValueError("要求 CUDA float64 张量，形状为 [batch,128,128]")
    batch = L.shape[0]
    P = torch.empty(L.shape, dtype=L.dtype, device=L.device)
    if batch == 0:
        return P
    # 单 GPU 实验模块与现有融合核使用相同的当前设备约定。
    if L.device.index != torch.cuda.current_device():
        raise ValueError("输入必须位于当前 CUDA 设备")
    R = torch.empty_strided(L.shape, (16384, 1, 128), dtype=L.dtype, device=L.device)
    _, functions, _ = compile_kernel()
    launch(functions["triangular_inverse"], (16, 1, batch), (128, 1, 1),
           (L, R), tuple(np.int64(s) for s in L.stride()))
    launch(functions["inverse_product"], (36, 1, batch), (128, 1, 1), (R, P))
    return P
