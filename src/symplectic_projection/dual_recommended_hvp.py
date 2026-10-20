"""推荐实现：特征基中首轮直接求解、融合 Newton-CG 更新、两级 HVP 及缓存 K 逆。
T：CPU float64 [batch, 2n, 2k]；小矩阵及 GPU 块也全部为 float64。
L 分解的是 K=I+omega@omega，不是原版的复矩阵 I+1j*omega。
要求对偶解位于正定内部；接近边界时 K 的形成和 Gram 求解会放大舍入误差。
"""
import torch
from .gram_triangle_reflect import gram_add_
from .recovery_fused import recover_block
from .hvp_fused import (inner_products, outer_skew, sandwich_front, line_change,
                       fused_precondition, cg_update, cg_next)
from .cholesky_inverse_kernel import inverse_from_cholesky

ROW_CHUNK = 1024
DEVICE = "cuda"
FEAS_TOL = 1e-8
MAX_NEWTON, MAX_CG, MAX_BACKTRACK = 100, 150, 40
# 大输出减少 CPU 回填；最终逆已缓存，所有阶数均复用。
PINNED_OUTPUT_MIN_BYTES = 64 * 2**20
FINAL_INVERSE_MIN_M = 0


def J(X):
    n = X.shape[-2] // 2
    return torch.cat((X[..., n:, :], -X[..., :n, :]), dim=-2)


def skew(A):
    return (A - A.mT) / 2


def inner(A, B):
    return torch.sum(A * B, dim=(-2, -1))


def norm(A):
    return torch.linalg.vector_norm(A, dim=(-2, -1))


def construct_inverse(I, L):
    """I 为单位矩阵；128 阶 FP64 批量因子使用专用核，其余输入沿用系统求解。"""
    if L.ndim == 3 and L.shape[-2:] == (128, 128) and L.is_cuda and L.dtype == torch.float64:
        return inverse_from_cholesky(L)
    return torch.cholesky_solve(I, L)


def inverse_parts(omega, L, I):
    """(I+i*omega)^-1 = P+i*Q，其中 P=(I+omega^2)^-1，Q=-omega@P。"""
    P = construct_inverse(I, L)
    Q = -omega @ P
    return P, Q


def to_eigenbasis(Ci, vectors):
    """只在 Newton 前转换一次；特征基中的 J 不再是标准坐标矩阵。"""
    return skew(vectors.mT @ Ci @ vectors), skew(vectors.mT @ J(vectors))


def from_eigenbasis(omega, P, Q, vectors, I):
    """返回原坐标的对偶量及一致的因子，兼容不传最终逆缓存的恢复路径。"""
    omega = skew(vectors @ omega @ vectors.mT)
    P = vectors @ P @ vectors.mT
    Q = vectors @ Q @ vectors.mT
    L = torch.linalg.cholesky(I + omega @ omega)
    return omega, L, P, Q


def sandwich(values, Ci, P, Q):
    """特征基中 C_r 为对角阵；右侧仍显式转置，不依赖浮点逆的 Hermitian 性。"""
    Yr, Yi = sandwich_front(values, Ci, P, Q)
    Sr, Si = recover_block(Yr, Yi, P, Q)
    return (Sr + Sr.mT) / 2, skew(Si)


def hvp(P, Q, Sr, Si, E):
    U, V = inner_products(E, Sr, Si)
    return outer_skew(P, Q, U, V)


@torch.no_grad()
def gram_chunks(T, blocks=None):
    """上三角 CUDA 核累加 Gram 两部；仍缓存相同的 (A,D) 块供恢复。"""
    batch, rows, m = T.shape
    n = rows // 2
    Cr = torch.zeros(batch, m, m, device=DEVICE, dtype=T.dtype)
    Ci = torch.zeros_like(Cr)
    buffer = torch.empty(batch * min(n, ROW_CHUNK) * m, dtype=T.dtype, pin_memory=True)
    for start in range(0, n, ROW_CHUNK):
        stop = min(start + ROW_CHUNK, n)
        host = buffer[:batch * (stop-start) * m].view(batch, stop-start, m)
        A = host.copy_(T[:, start:stop]).to(DEVICE)
        D = host.copy_(T[:, n+start:n+stop]).to(DEVICE)
        gram_add_(A, D, Cr, Ci)
        if blocks is not None:
            blocks.append((A, D))
        del A, D
    return Cr, Ci


@torch.no_grad()
def dual_batch(Cr, Ci, final=None):
    # 1. 独立归一化，然后将整个对偶迭代保留在 C_r 的特征基中。
    batch, m, _ = Cr.shape
    I = torch.eye(m, dtype=Cr.dtype, device=Cr.device).expand(batch, -1, -1)
    omega, L = torch.zeros_like(Cr), I.clone()
    P = I  # omega=0 时 K=I，无需求解；之后只在接受候选时更新。
    scale = (Cr.diagonal(dim1=-2, dim2=-1).sum(-1) / m).clamp_min(1)
    Cr, Ci = Cr / scale[:, None, None], Ci / scale[:, None, None]
    values, vectors = torch.linalg.eigh(Cr)
    denominator = values[:, :, None] + values[:, None, :]
    Ci, Jm = to_eigenbasis(Ci, vectors)
    Jscaled = Jm / scale[:, None, None]

    def precondition(R):
        return fused_precondition(R, denominator)

    active = torch.ones(batch, dtype=torch.bool, device=Cr.device)
    iterations = torch.zeros(batch, dtype=torch.int64, device=Cr.device)
    for newton_step in range(MAX_NEWTON):
        # 2. 复用已接受的 K 逆；本 Newton 步的 CG/HVP 共用 P、Q。
        if newton_step == 0:
            Q = torch.zeros_like(omega)
            Sr, Si = torch.diag_embed(values), Ci
        else:
            Q = -omega @ P
            Sr, Si = sandwich(values, Ci, P, Q)
        gradient = Jscaled - Si
        gradient_norm = norm(gradient)
        residual = gradient_norm * scale / m**0.5
        active &= residual >= FEAS_TOL / 2
        if not bool(active.any()):
            omega, L, P, Q = from_eigenbasis(omega, P, Q, vectors, I)
            if final is not None and m >= FINAL_INVERSE_MIN_M:
                final[:] = (P, Q)
            return omega, L, iterations
        iterations += active

        # 3. 首轮 H(E)=E*Lambda+Lambda*E 可直接求解；后续保留原批量 CG。
        if newton_step == 0:
            initial_z = precondition(torch.where(active[:, None, None], -gradient, 0))
            direction = initial_z
        else:
            direction = torch.zeros_like(gradient)
            r = torch.where(active[:, None, None], -gradient, 0)
            z = precondition(r)
            initial_z = z  # p 独立复制，后续 z 重新绑定，不会覆盖此结果。
            p, rz = z.clone(), inner(r, z)
            tolerance = residual.sqrt().clamp(1e-8, 0.1) * gradient_norm
            cg_active = active.clone()
            for _ in range(MAX_CG):
                Hp = hvp(P, Q, Sr, Si, p)
                curvature = inner(p, Hp)
                cg_active &= curvature > 0
                alpha = torch.where(cg_active, rz / torch.where(cg_active, curvature, 1), 0)
                direction, r = cg_update(direction, r, p, Hp, alpha)
                cg_active &= norm(r) > tolerance
                if not bool(cg_active.any()):
                    break
                z = precondition(r)
                next_rz = inner(r, z)
                beta = next_rz / torch.where(cg_active, rz, 1)
                p = cg_next(z, p, beta, cg_active)
                rz = next_rz
        fallback = active & (inner(gradient, direction) >= 0)
        direction = torch.where(fallback[:, None, None], initial_z, direction)

        # 4. K 正定等价于 I+i*omega 正定；保留原精确线搜索差值。
        slope = inner(gradient, direction)
        step = torch.ones_like(slope)
        pending = active.clone()
        for _ in range(MAX_BACKTRACK):
            delta = torch.where(pending, step, 0)[:, None, None] * direction
            trial = skew(omega + delta)
            new_L, info = torch.linalg.cholesky_ex(I + trial @ trial)
            valid = info == 0
            safe_L = torch.where(valid[:, None, None], new_L, L)
            P_trial = construct_inverse(I, safe_L)
            V = P_trial @ delta
            delta_Sr, delta_Si = inner_products(delta, Sr, Si)
            change = line_change(gradient, delta, V, delta_Sr, trial, delta_Si)
            accept = pending & valid & (change <= 1e-4 * step * slope)
            omega = torch.where(accept[:, None, None], trial, omega)
            L = torch.where(accept[:, None, None], new_L, L)
            P = torch.where(accept[:, None, None], P_trial, P)
            del P_trial
            pending &= ~accept
            if not bool(pending.any()):
                break
            step = torch.where(pending, step / 2, step)
        else:
            raise RuntimeError("批量线搜索未找到下降步")
    raise RuntimeError("达到批量 Newton 迭代上限")


@torch.no_grad()
def recover_matmul(T, omega, L, blocks=None, final=None):
    """用实数逆的两部恢复 X，复用 (A,D) 缓存并返回 CPU float64。"""
    batch, rows, m = T.shape
    n = rows // 2
    if final:
        P, Q = final
    else:
        I = torch.eye(m, dtype=L.dtype, device=L.device).expand_as(L)
        P, Q = inverse_parts(omega, L, I)
    direct = T.numel() * T.element_size() >= PINNED_OUTPUT_MIN_BYTES
    X = torch.empty_like(T, device="cpu", pin_memory=direct)
    buffer = None if direct and blocks is not None else torch.empty(
        batch * min(n, ROW_CHUNK) * m, dtype=T.dtype, pin_memory=True)
    for index, start in enumerate(range(0, n, ROW_CHUNK)):
        stop = min(start + ROW_CHUNK, n)
        if buffer is not None:
            host = buffer[:batch * (stop-start) * m].view(batch, stop-start, m)
        if blocks is None:
            A = host.copy_(T[:, start:stop]).to(L.device)
            D = host.copy_(T[:, n+start:n+stop]).to(L.device)
        else:
            A, D = blocks[index]
        # 用 B @ N.mH 恢复；与 S=N C N.mH 保持相同的左右求解方向。
        U_real, U_imag = recover_block(A, D, P, Q)
        if direct:
            X[:, start:stop].copy_(U_real)
            X[:, n+start:n+stop].copy_(U_imag)
        else:
            host.copy_(U_real)
            X[:, start:stop].copy_(host)
            host.copy_(U_imag)
            X[:, n+start:n+stop].copy_(host)
        del A, D, U_real, U_imag
    return X


@torch.no_grad()
def solve(T):
    """已有 CPU float64 输入 -> CPU 的 (X, omega, Newton 更新次数)。"""
    blocks, final = [], []
    Cr, Ci = gram_chunks(T, blocks)
    omega, L, iterations = dual_batch(Cr, Ci, final)
    del Cr, Ci
    X = recover_matmul(T, omega, L, blocks, final)
    return X, omega.cpu(), iterations.cpu()
