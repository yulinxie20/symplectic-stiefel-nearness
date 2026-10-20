// 每对对称位置只有一个写入者，保持原核的四条点积累加顺序。
__device__ __forceinline__ void write_pair(
    double* Cr, double* Ci, long long base, int m, int i, int j,
    double aa, double dd, double ad, double da)
{
    const double real = aa + dd, imag = ad - da;
    Cr[base + i * m + j] += real;
    if (i != j) {
        Cr[base + j * m + i] += real;
        Ci[base + i * m + j] += imag;
        Ci[base + j * m + i] -= imag;
    }
}

template<bool STRICT>
__device__ __forceinline__ void gram_tiles(
    const double* A, const double* D, double* Cr, double* Ci, int rows, int m)
{
    const int q = blockIdx.x;
    const int column = (int)((sqrt(8.0 * q + 1.0) - 1.0) * 0.5);
    const int bi = q - column * (column + 1) / 2;
    const int bj = column + (STRICT ? 1 : 0);
    const int x = threadIdx.x, y = threadIdx.y;
    const int i = bi * 16 + y, j = bj * 16 + x;
    const long long input = (long long)blockIdx.z * rows * m;
    const long long output = (long long)blockIdx.z * m * m;
    __shared__ double ai[16][17], di[16][17], aj[16][17], dj[16][17];
    double aa = 0.0, dd = 0.0, ad = 0.0, da = 0.0;
    const bool valid = (STRICT || i <= j) && j < m;
    for (int start = 0; start < rows; start += 16) {
        const int row = start + y, left = bi * 16 + x, right = bj * 16 + x;
        ai[y][x] = row < rows && left < m ? A[input + row * m + left] : 0.0;
        di[y][x] = row < rows && left < m ? D[input + row * m + left] : 0.0;
        aj[y][x] = row < rows && right < m ? A[input + row * m + right] : 0.0;
        dj[y][x] = row < rows && right < m ? D[input + row * m + right] : 0.0;
        __syncthreads();
        if (valid) {
            #pragma unroll
            for (int t = 0; t < 16; ++t) {
                aa = fma(ai[t][y], aj[t][x], aa);
                dd = fma(di[t][y], dj[t][x], dd);
                ad = fma(ai[t][y], dj[t][x], ad);
                da = fma(di[t][y], aj[t][x], da);
            }
        }
        __syncthreads();
    }
    if (valid) write_pair(Cr, Ci, output, m, i, j, aa, dd, ad, da);
}

extern "C" __global__ void gram_packed(
    const double* A, const double* D, double* Cr, double* Ci, int rows, int m)
{
    gram_tiles<false>(A, D, Cr, Ci, rows, m);
}

extern "C" __global__ void gram_offdiag(
    const double* A, const double* D, double* Cr, double* Ci, int rows, int m)
{
    gram_tiles<true>(A, D, Cr, Ci, rows, m);
}

extern "C" __global__ void gram_diagonal(
    const double* A, const double* D, double* Cr, double* Ci, int rows, int m)
{
    const int tid = threadIdx.x, tile = blockIdx.x * 16;
    // 8x16 矩形的下三角反射到右下角：120 个非对角点 + 前 8 个对角点。
    // 第二点仅取后 8 个对角点；展开 s 后，编译器消去不用的 ad[1]/da[1]。
    const int r = tid >> 4, c = tid & 15;
    const int u[2] = {c < r ? 15-r : r, tid+8};
    const int v[2] = {c < r ? 15-c : c, tid+8};
    const bool valid[2] = {tile + v[0] < m, tid < 8 && tile + v[1] < m};
    const long long input = (long long)blockIdx.z * rows * m;
    const long long output = (long long)blockIdx.z * m * m;
    __shared__ double a[16][17], d[16][17];
    double aa[2] = {}, dd[2] = {}, ad[2] = {}, da[2] = {};
    for (int start = 0; start < rows; start += 16) {
        #pragma unroll
        for (int p = tid; p < 256; p += 128) {
            const int r = p / 16, c = p % 16;
            const int row = start + r, col = tile + c;
            a[r][c] = row < rows && col < m ? A[input + row * m + col] : 0.0;
            d[r][c] = row < rows && col < m ? D[input + row * m + col] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            if (valid[s]) {
                #pragma unroll
                for (int t = 0; t < 16; ++t) {
                    aa[s] = fma(a[t][u[s]], a[t][v[s]], aa[s]);
                    dd[s] = fma(d[t][u[s]], d[t][v[s]], dd[s]);
                    ad[s] = fma(a[t][u[s]], d[t][v[s]], ad[s]);
                    da[s] = fma(d[t][u[s]], a[t][v[s]], da[s]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int s = 0; s < 2; ++s)
        if (valid[s]) write_pair(Cr, Ci, output, m, tile + u[s], tile + v[s],
                                aa[s], dd[s], ad[s], da[s]);
}
