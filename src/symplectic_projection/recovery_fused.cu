// 每线程计算一个位置的实部和虚部；A/D 连续，P/Q 保留实际步长。
extern "C" __global__ void recover_fused(
    const double* A, const double* D, const double* P, const double* Q,
    double* Ur, double* Ui, int rows, int m,
    long long pb, long long pr, long long pc,
    long long qb, long long qr, long long qc)
{
    constexpr int TILE = 16;
    const int x = threadIdx.x, y = threadIdx.y;
    const int row = blockIdx.y * TILE + y;
    const int col = blockIdx.x * TILE + x;
    const int batch = blockIdx.z;
    const long long base = (long long)batch * rows * m;
    __shared__ double a[TILE][TILE + 1], d[TILE][TILE + 1];
    __shared__ double p[TILE][TILE + 1], q[TILE][TILE + 1];
    double real = 0.0, imag = 0.0;

    for (int start = 0; start < m; start += TILE) {
        const int k = start + x;
        const int j = blockIdx.x * TILE + y;
        a[y][x] = row < rows && k < m ? A[base + row * m + k] : 0.0;
        d[y][x] = row < rows && k < m ? D[base + row * m + k] : 0.0;
        p[y][x] = j < m && k < m ? P[batch * pb + j * pr + k * pc] : 0.0;
        q[y][x] = j < m && k < m ? Q[batch * qb + j * qr + k * qc] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < TILE; ++t) {
            const double av = a[y][t], dv = d[y][t];
            const double pv = p[x][t], qv = q[x][t];
            real = fma(av, pv, real);
            real = fma(dv, qv, real);
            imag = fma(dv, pv, imag);
            imag = fma(-av, qv, imag);
        }
        __syncthreads();
    }
    if (row < rows && col < m) {
        Ur[base + row * m + col] = real;
        Ui[base + row * m + col] = imag;
    }
}
