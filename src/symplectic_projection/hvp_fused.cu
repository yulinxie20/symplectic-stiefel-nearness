// FP64 HVP：内层共享 E 分块；外层保留两条点积分别累加后相减。
extern "C" __global__ void hvp_inner(
    const double* E, const double* Sr, const double* Si,
    double* U, double* V, int m,
    long long eb, long long er, long long ec,
    long long rb, long long rr, long long rc,
    long long ib, long long ir, long long ic)
{
    constexpr int TILE = 16;
    const int x = threadIdx.x, y = threadIdx.y;
    const int row = blockIdx.y * TILE + y;
    const int col = blockIdx.x * TILE + x;
    const int batch = blockIdx.z;
    __shared__ double e[TILE][TILE + 1];
    __shared__ double sr[TILE][TILE + 1], si[TILE][TILE + 1];
    double u = 0.0, v = 0.0;

    for (int start = 0; start < m; start += TILE) {
        const int kx = start + x, ky = start + y;
        e[y][x] = row < m && kx < m ? E[batch * eb + row * er + kx * ec] : 0.0;
        sr[y][x] = ky < m && col < m ? Sr[batch * rb + ky * rr + col * rc] : 0.0;
        si[y][x] = ky < m && col < m ? Si[batch * ib + ky * ir + col * ic] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < TILE; ++t) {
            const double ev = e[y][t];
            u = fma(ev, sr[t][x], u);
            v = fma(ev, si[t][x], v);
        }
        __syncthreads();
    }
    if (row < m && col < m) {
        const long long out = ((long long)batch * m + row) * m + col;
        U[out] = u;
        V[out] = v;
    }
}

// 特征基中的 sandwich 前半：共享 Ci 分块，两个点积各自按 k 递增累加。
// 列缩放与最终加减分别舍入，避免编译器将它们合并成额外的 FMA。
extern "C" __global__ void hvp_sandwich_front(
    const double* values, const double* Ci, const double* P, const double* Q,
    double* Yr, double* Yi, int m,
    long long vb, long long vc,
    long long cb, long long cr, long long cc,
    long long pb, long long pr, long long pc,
    long long qb, long long qr, long long qc)
{
    constexpr int TILE = 16;
    const int x = threadIdx.x, y = threadIdx.y;
    const int row = blockIdx.y * TILE + y;
    const int col = blockIdx.x * TILE + x;
    const int batch = blockIdx.z;
    __shared__ double p[TILE][TILE + 1], q[TILE][TILE + 1];
    __shared__ double ci[TILE][TILE + 1];
    double pci = 0.0, qci = 0.0;

    for (int start = 0; start < m; start += TILE) {
        const int kx = start + x, ky = start + y;
        p[y][x] = row < m && kx < m ? P[batch * pb + row * pr + kx * pc] : 0.0;
        q[y][x] = row < m && kx < m ? Q[batch * qb + row * qr + kx * qc] : 0.0;
        ci[y][x] = ky < m && col < m ? Ci[batch * cb + ky * cr + col * cc] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < TILE; ++t) {
            const double cv = ci[t][x];
            pci = fma(p[y][t], cv, pci);
            qci = fma(q[y][t], cv, qci);
        }
        __syncthreads();
    }
    if (row < m && col < m) {
        const double d = values[batch * vb + col * vc];
        const double pd = __dmul_rn(P[batch * pb + row * pr + col * pc], d);
        const double qd = __dmul_rn(Q[batch * qb + row * qr + col * qc], d);
        const long long out = ((long long)batch * m + row) * m + col;
        Yr[out] = __dsub_rn(pd, qci);
        Yi[out] = __dadd_rn(pci, qd);
    }
}

// 一个上三角块同时计算 Z_ij/Z_ji；不假设浮点 P/Q/U/V 的对称性。
extern "C" __global__ void hvp_outer(
    const double* P, const double* Q, const double* U, const double* V,
    double* H, int m,
    long long pb, long long pr, long long pc,
    long long qb, long long qr, long long qc,
    long long ub, long long ur, long long uc,
    long long vb, long long vr, long long vc)
{
    constexpr int TILE = 16;
    const int x = threadIdx.x, y = threadIdx.y;
    const int batch = blockIdx.z;
    // 列块 j 对应 j+1 个行块：0; 0,1; 0,1,2; ...。
    int block_col = 0, block_row = blockIdx.x;
    while (block_row > block_col) {
        block_row -= block_col + 1;
        ++block_col;
    }
    const int row = block_row * TILE + y;
    const int col = block_col * TILE + x;
    const bool active = row < m && col < m &&
                        (block_row != block_col || y <= x);
    __shared__ double pi[TILE][TILE + 1], qi[TILE][TILE + 1];
    __shared__ double pj[TILE][TILE + 1], qj[TILE][TILE + 1];
    __shared__ double ui[TILE][TILE + 1], vi[TILE][TILE + 1];
    __shared__ double uj[TILE][TILE + 1], vj[TILE][TILE + 1];
    double pu = 0.0, qv = 0.0, pu_transpose = 0.0, qv_transpose = 0.0;

    for (int start = 0; start < m; start += TILE) {
        const int kx = start + x, ky = start + y;
        const int irow = block_row * TILE + y;
        const int jrow = block_col * TILE + y;
        const int icol = block_row * TILE + x;
        const int jcol = block_col * TILE + x;
        pi[y][x] = irow < m && kx < m ? P[batch * pb + irow * pr + kx * pc] : 0.0;
        qi[y][x] = irow < m && kx < m ? Q[batch * qb + irow * qr + kx * qc] : 0.0;
        pj[y][x] = jrow < m && kx < m ? P[batch * pb + jrow * pr + kx * pc] : 0.0;
        qj[y][x] = jrow < m && kx < m ? Q[batch * qb + jrow * qr + kx * qc] : 0.0;
        ui[y][x] = ky < m && icol < m ? U[batch * ub + ky * ur + icol * uc] : 0.0;
        vi[y][x] = ky < m && icol < m ? V[batch * vb + ky * vr + icol * vc] : 0.0;
        uj[y][x] = ky < m && jcol < m ? U[batch * ub + ky * ur + jcol * uc] : 0.0;
        vj[y][x] = ky < m && jcol < m ? V[batch * vb + ky * vr + jcol * vc] : 0.0;
        __syncthreads();
        if (active && row != col) {
            #pragma unroll
            for (int t = 0; t < TILE; ++t) {
                pu = fma(pi[y][t], uj[t][x], pu);
                qv = fma(qi[y][t], vj[t][x], qv);
                pu_transpose = fma(pj[x][t], ui[t][y], pu_transpose);
                qv_transpose = fma(qj[x][t], vi[t][y], qv_transpose);
            }
        }
        __syncthreads();
    }
    if (active) {
        const long long base = (long long)batch * m * m;
        if (row == col) {
            H[base + row * m + col] = 0.0;
        } else {
            const double z = pu - qv;
            const double z_transpose = pu_transpose - qv_transpose;
            const double h = z - z_transpose;
            H[base + row * m + col] = h;
            H[base + col * m + row] = -h;
        }
    }
}

// 仅用于连续 FP64 [batch,128,128]。四条点积独立、k 升序；不假设输入对称。
__device__ __forceinline__ void write_h(
    double* H, long long base, int i, int j, double pu, double qv, double put, double qvt)
{
    const double z = __dsub_rn(pu, qv);
    const double zt = __dsub_rn(put, qvt);
    const double h = __dsub_rn(z, zt);
    H[base + i * 128 + j] = h;
    H[base + j * 128 + i] = -h;
}

// 紧凑严格上三角块映射：0,1；0,2；1,2；0,3；...。
__device__ __forceinline__ void offdiag_pair(int packed, int& bi, int& bj)
{
    bj = 1;
    bi = packed;
    while (bi >= bj) {
        bi -= bj;
        ++bj;
    }
}

extern "C" __global__ void hvp_outer_off16(
    const double* P, const double* Q, const double* U, const double* V, double* H)
{
    const int x = threadIdx.x, y = threadIdx.y;
    int bi, bj;
    offdiag_pair(blockIdx.x, bi, bj);
    const int i = bi * 16 + y, j = bj * 16 + x;
    const long long base = (long long)blockIdx.z * 16384;
    __shared__ double pi[16][17], qi[16][17], pj[16][17], qj[16][17];
    __shared__ double ui[16][17], vi[16][17], uj[16][17], vj[16][17];
    double pu = 0.0, qv = 0.0, put = 0.0, qvt = 0.0;
    for (int start = 0; start < 128; start += 16) {
        pi[y][x] = P[base + (bi * 16 + y) * 128 + start + x];
        qi[y][x] = Q[base + (bi * 16 + y) * 128 + start + x];
        pj[y][x] = P[base + (bj * 16 + y) * 128 + start + x];
        qj[y][x] = Q[base + (bj * 16 + y) * 128 + start + x];
        ui[y][x] = U[base + (start + y) * 128 + bi * 16 + x];
        vi[y][x] = V[base + (start + y) * 128 + bi * 16 + x];
        uj[y][x] = U[base + (start + y) * 128 + bj * 16 + x];
        vj[y][x] = V[base + (start + y) * 128 + bj * 16 + x];
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            pu = fma(pi[y][t], uj[t][x], pu);
            qv = fma(qi[y][t], vj[t][x], qv);
            put = fma(pj[x][t], ui[t][y], put);
            qvt = fma(qj[x][t], vi[t][y], qvt);
        }
        __syncthreads();
    }
    write_h(H, base, i, j, pu, qv, put, qvt);
}

extern "C" __global__ void hvp_outer_diag16(
    const double* P, const double* Q, const double* U, const double* V, double* H)
{
    const int tid = threadIdx.x, tile = blockIdx.x * 16;
    const int r = tid >> 4, c = tid & 15;
    const int i = c < r ? 15 - r : r;
    const int j = c < r ? 15 - c : c;
    const long long base = (long long)blockIdx.z * 16384;
    __shared__ double p[16][17], q[16][17], u[16][17], v[16][17];
    double pu = 0.0, qv = 0.0, put = 0.0, qvt = 0.0;
    for (int start = 0; start < 128; start += 16) {
        #pragma unroll
        for (int index = tid; index < 256; index += 128) {
            const int a = index >> 4, b = index & 15;
            p[a][b] = P[base + (tile + a) * 128 + start + b];
            q[a][b] = Q[base + (tile + a) * 128 + start + b];
            u[a][b] = U[base + (start + a) * 128 + tile + b];
            v[a][b] = V[base + (start + a) * 128 + tile + b];
        }
        __syncthreads();
        if (i != j) {
            #pragma unroll
            for (int t = 0; t < 16; ++t) {
                pu = fma(p[i][t], u[t][j], pu);
                qv = fma(q[i][t], v[t][j], qv);
                put = fma(p[j][t], u[t][i], put);
                qvt = fma(q[j][t], v[t][i], qvt);
            }
        }
        __syncthreads();
    }
    if (i != j) write_h(H, base, tile + i, tile + j, pu, qv, put, qvt);
    if (tid < 16) H[base + (tile + tid) * 128 + tile + tid] = 0.0;
}

// 线搜索后半候选：逐 tile 点积、固定树归约、每题第二级汇总。
// 两项始终分别归约；矩阵点积之外的乘法/加法显式舍入，禁止隐式融合。

#define PARTIAL_PARAMETERS \
    const double* gradient, const double* delta, const double* V, \
    const double* delta_Sr, const double* trial, const double* delta_Si, \
    double* partial, int m, \
    long long gb, long long gr, long long gc, \
    long long db, long long dr, long long dc, \
    long long vb, long long vr, long long vc, \
    long long rb, long long rr, long long rc, \
    long long tb, long long tr, long long tc, \
    long long ib, long long ir, long long ic

#define PARTIAL_ARGUMENTS \
    gradient, delta, V, delta_Sr, trial, delta_Si, partial, m, \
    gb, gr, gc, db, dr, dc, vb, vr, vc, rb, rr, rc, tb, tr, tc, ib, ir, ic

template<bool BOTH>
__device__ __forceinline__ void partial_body(PARTIAL_PARAMETERS)
{
    constexpr int TILE = 16;
    constexpr int THREADS = TILE * TILE;
    constexpr int CHAINS = BOTH ? 2 : 1;
    const int x = threadIdx.x, y = threadIdx.y;
    const int tid = y * TILE + x;
    const int row = blockIdx.y * TILE + y;
    const int col = blockIdx.x * TILE + x;
    const int batch = blockIdx.z;
    const bool active = row < m && col < m;
    __shared__ double left[TILE][TILE + 1];
    __shared__ double right[TILE][TILE + 1];
    __shared__ double sums[CHAINS][THREADS];
    double product = 0.0;

    // trial @ delta_Si，每个输出元素按 k 递增累加。
    for (int start = 0; start < m; start += TILE) {
        const int kx = start + x, ky = start + y;
        left[y][x] = row < m && kx < m
            ? trial[batch * tb + row * tr + kx * tc] : 0.0;
        right[y][x] = ky < m && col < m
            ? delta_Si[batch * ib + ky * ir + col * ic] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            product = fma(left[y][k], right[k][x], product);
        __syncthreads();
    }

    double tail = 0.0, first = 0.0;
    if (active) {
        const double sum = __dadd_rn(delta_Sr[batch * rb + row * rr + col * rc], product);
        tail = __dmul_rn(V[batch * vb + row * vr + col * vc], sum);
        if constexpr (BOTH)
            first = __dmul_rn(gradient[batch * gb + row * gr + col * gc],
                             delta[batch * db + row * dr + col * dc]);
    }
    sums[0][tid] = tail;
    if constexpr (BOTH) sums[1][tid] = first;
    __syncthreads();

    // 固定二叉树；尾 tile 的无效元素严格写0，所有线程参加同步。
    #pragma unroll
    for (int offset = THREADS / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            sums[0][tid] = __dadd_rn(sums[0][tid], sums[0][tid + offset]);
            if constexpr (BOTH)
                sums[1][tid] = __dadd_rn(sums[1][tid], sums[1][tid + offset]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        const long long tiles = (long long)gridDim.x * gridDim.y;
        const long long tile = (long long)blockIdx.y * gridDim.x + blockIdx.x;
        const long long base = (long long)batch * CHAINS * tiles;
        partial[base + tile] = sums[0][0];
        if constexpr (BOTH) partial[base + tiles + tile] = sums[1][0];
    }
}



extern "C" __global__ void hvp_line_partial(PARTIAL_PARAMETERS)
{
    partial_body<true>(PARTIAL_ARGUMENTS);
}

template<bool BOTH>
__device__ __forceinline__ void finish_body(const double* partial, double* output, int tiles)
{
    constexpr int THREADS = 256;
    constexpr int CHAINS = BOTH ? 2 : 1;
    const int tid = threadIdx.x;
    const long long base = (long long)blockIdx.x * CHAINS * tiles;
    __shared__ double sums[CHAINS][THREADS];
    double tail = 0.0, first = 0.0;
    // tiles>256也固定按递增下标分配到线程，不使用atomic。
    for (int tile = tid; tile < tiles; tile += THREADS) {
        tail = __dadd_rn(tail, partial[base + tile]);
        if constexpr (BOTH)
            first = __dadd_rn(first, partial[base + tiles + tile]);
    }
    sums[0][tid] = tail;
    if constexpr (BOTH) sums[1][tid] = first;
    __syncthreads();
    #pragma unroll
    for (int offset = THREADS / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            sums[0][tid] = __dadd_rn(sums[0][tid], sums[0][tid + offset]);
            if constexpr (BOTH)
                sums[1][tid] = __dadd_rn(sums[1][tid], sums[1][tid + offset]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        if constexpr (BOTH)
            output[blockIdx.x] = __dadd_rn(sums[1][0], sums[0][0]);
        else
            output[blockIdx.x] = sums[0][0];
    }
}



extern "C" __global__ void hvp_line_finish(const double* partial, double* output, int tiles)
{
    finish_body<true>(partial, output, tiles);
}

// 预条件与 CG 逐元素更新：保持运算顺序，不合并范数/内积归约。
typedef long long i64;
#define STRIDES(name) i64 name##b, i64 name##i, i64 name##j
#define OFFSET(name, bb, ii, jj) ((bb) * name##b + (ii) * name##i + (jj) * name##j)

extern "C" __global__ void hvp_precondition(
    const double* R, const double* denominator, double* output, int m,
    STRIDES(r), STRIDES(d)) {
    const i64 linear = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    const i64 b = blockIdx.y;
    const i64 matrix_size = (i64)m * m;
    if (linear >= matrix_size) return;
    const i64 i = linear / m, j = linear % m;
    // 两个位置均真实读取；不假设 R 反对称或 denominator 精确对称。
    const double a = __ddiv_rn(R[OFFSET(r, b, i, j)], denominator[OFFSET(d, b, i, j)]);
    const double transpose = __ddiv_rn(R[OFFSET(r, b, j, i)], denominator[OFFSET(d, b, j, i)]);
    const double difference = __dsub_rn(a, transpose);
    output[b * matrix_size + linear] = __dmul_rn(difference, 0.5);
}

extern "C" __global__ void hvp_cg_update(
    const double* direction, const double* residual,
    const double* p, const double* Hp, const double* alpha,
    double* new_direction, double* new_residual, int m,
    STRIDES(d), STRIDES(r), STRIDES(p), STRIDES(h), i64 alpha_stride) {
    const i64 linear = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    const i64 b = blockIdx.y;
    const i64 matrix_size = (i64)m * m;
    if (linear >= matrix_size) return;
    const i64 i = linear / m, j = linear % m;
    const double a = alpha[b * alpha_stride];
    // 显式 rn 操作防止乘加收缩，和原 PyTorch 分开的乘、加/减顺序一致。
    const double direction_change = __dmul_rn(a, p[OFFSET(p, b, i, j)]);
    const double residual_change = __dmul_rn(a, Hp[OFFSET(h, b, i, j)]);
    new_direction[b * matrix_size + linear] = __dadd_rn(direction[OFFSET(d, b, i, j)], direction_change);
    new_residual[b * matrix_size + linear] = __dsub_rn(residual[OFFSET(r, b, i, j)], residual_change);
}

extern "C" __global__ void hvp_cg_next(
    const double* z, const double* p, const double* beta,
    const unsigned char* active, double* output, int m,
    STRIDES(z), STRIDES(p), i64 beta_stride, i64 active_stride) {
    const i64 linear = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    const i64 b = blockIdx.y;
    const i64 matrix_size = (i64)m * m;
    if (linear >= matrix_size) return;
    const i64 i = linear / m, j = linear % m;
    const double scaled = __dmul_rn(beta[b * beta_stride], p[OFFSET(p, b, i, j)]);
    const double next = __dadd_rn(z[OFFSET(z, b, i, j)], scaled);
    // 对应原先先算 beta*p、再 z+，最后 where；不改变 active 的更新位置。
    output[b * matrix_size + linear] = active[b * active_stride] ? next : 0.0;
}

#undef OFFSET
#undef STRIDES
