# SPDX-License-Identifier: GPL-3.0-or-later
# JensenRasmus/secondorder_spst, c121d28f2116da6a566b8eff44642a42d8971bd0.
# Manopt 7.1, 909a2f756e691a6297f737bcc45bbff829674833 (GPL-3.0-or-later).
# Manopt copyright Nicolas Boumal; underlying GenRTR is BSD-3-Clause.
# GenRTR copyright (c) 2007,2012 Christopher G. Baker, Pierre-Antoine Absil,
# Kyle A. Gallivan. All rights reserved.
# Manopt adaptation: Nicolas Boumal; cached tCG: Victor Liao and Nicolas Boumal.
# Python adaptation copyright (c) 2026 YulinXie.
"""Jensen--Zimmermann 几何上的原信赖域循环及 Manopt 缓存 tCG。

外循环对应作者仓库 trustregions.m；内循环逐式移植 Manopt 7.1 的
trs_tCG_cached.m / tCG_rejectedstep.m 默认分支（无预条件器）。保留
拒绝步缓存、模型下降检查、rho 正则化、半径更新和严格接受不等式。
共同欧氏残差作为外循环精度判断。目标始终为 ||X-T||_F^2，不含 1/2。

本适配文件随 Jensen/Manopt GPL 代码使用；GenRTR 的 BSD 署名及条款
仍须保留，许可全文见根目录 THIRD_PARTY_NOTICES.txt。这里的 memorytCG_MB 仅计缓存
张量和标量的数值载荷，不声称等于 MATLAB getsize 的对象内存。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Callable

import torch

from .common import SolveResult, inner, left_j, standard_j, synchronize, validate_target
from .geometry import orthosymplectic_initial, prepare_projection


SOURCE = {
    "repository": "https://github.com/JensenRasmus/secondorder_spst",
    "commit": "c121d28f2116da6a566b8eff44642a42d8971bd0",
    "outer_file": "trustregions.m",
    "outer_sha256": "b30b3a8efc5b1d838b6ea84f78373ff4818ae611599dfaea0067fc6d828eb28f",
    "manopt_repository": "https://github.com/NicolasBoumal/manopt",
    "manopt_version": "7.1",
    "manopt_commit": "909a2f756e691a6297f737bcc45bbff829674833",
    "inner_file": "manopt/solvers/trustregions/trs_tCG_cached.m",
    "inner_sha256": "4f7667b13a702d18525f7a9fe2a2dbf35616be6701e38e6b18be7e2539ddb090",
    "rejected_file": "manopt/solvers/trustregions/tCG_rejectedstep.m",
    "rejected_sha256": "d5a7936fc389c02e0e76b1acbbcdb843f9991028a76d19c0707c622472044414",
    "preconditioner": "identity (original default)",
    "subproblem": "trs_tCG_cached",
}


@dataclass
class JensenTrustRegionOptions:
    """专用选项；不借用旧欧氏 RTR 的内层参数或默认值。"""

    max_iter: int = 1000
    time_limit: float = 60.0
    grad_tol: float = 1e-9
    feas_tol: float = 1e-8
    miniter: int = 0
    delta_bar: float | None = None
    delta0: float | None = None
    rho_prime: float = 0.1
    rho_regularization: float = 1e3
    kappa: float = 0.1
    theta: float = 1.0
    mininner: int = 1
    maxinner: int | None = None  # 原默认：流形实维数
    trscache: bool = True
    history: bool = False

    def __post_init__(self):
        if self.max_iter < 0 or self.miniter < 0 or self.mininner < 1:
            raise ValueError("迭代预算无效")
        if self.maxinner is not None and self.maxinner < 1:
            raise ValueError("maxinner 必须为正")
        if not math.isfinite(self.time_limit) or self.time_limit <= 0:
            raise ValueError("time_limit 必须有限且为正")
        if not all(math.isfinite(v) and v > 0 for v in (self.grad_tol, self.feas_tol, self.kappa)):
            raise ValueError("共同容差与 kappa 必须有限且为正")
        if not 0 <= self.theta <= 1 or not 0 <= self.rho_prime < 0.25:
            raise ValueError("theta 或 rho_prime 超出原算法参数范围")
        if self.rho_regularization < 0 or not math.isfinite(self.rho_regularization):
            raise ValueError("rho_regularization 必须有限且非负")
        for value in (self.delta_bar, self.delta0):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("信赖域半径必须有限且为正")


@dataclass
class TCGCache:
    store_iters: list[dict] = field(default_factory=list)
    store_last: dict | None = None

    @property
    def payload_mb(self):
        total = 0
        for record in self.store_iters + ([] if self.store_last is None else [self.store_last]):
            for value in record.values():
                if isinstance(value, torch.Tensor):
                    total += value.numel() * value.element_size()
                elif isinstance(value, (float, int)):
                    total += 8
        return total / 1024**2


@dataclass
class TCGResult:
    eta: torch.Tensor
    Heta: torch.Tensor
    limitedbyTR: bool
    stats: dict
    cache: TCGCache | None


def _divide(numerator, denominator):
    """保留 MATLAB 标量除法在零分母时的 Inf/NaN 语义。"""
    if denominator == 0:
        return math.nan if numerator == 0 else math.copysign(math.inf, numerator * math.copysign(1., denominator))
    return numerator / denominator


def _sqrt(value):
    # 不裁剪负值：数值失败应显式暴露，不能悄悄改变边界步。
    return math.sqrt(value) if value >= 0 else math.nan


def _boundary(record, delta):
    tau = _divide(-record["e_Pd"] + _sqrt(record["e_Pd"]**2 + record["d_Pd"] *
                  (delta**2 - record["e_Pe"])), record["d_Pd"])
    eta = record["eta"] - tau * record["mdelta"]
    heta = record["Heta"] - tau * record["Hmdelta"]
    reason = "negative curvature" if record["d_Hd"] <= 0 else "exceeded trust region"
    return eta, heta, tau, reason


def tcg_rejectedstep(cache, delta):
    """原 tCG_rejectedstep：从同一基点的缓存重新截断，不调 Hessian。"""
    for record in cache.store_iters:
        if record["d_Hd"] <= 0 or record["normsq"] >= delta**2:
            eta, heta, _, reason = _boundary(record, delta)
            stats = dict(numinner=record["numinner"], hessvecevals=0,
                         memorytCG_MB=cache.payload_mb, numstored=len(cache.store_iters) +
                         int(cache.store_last is not None), cached=True, stopreason=reason)
            return TCGResult(eta, heta, True, stats, cache)
    if cache.store_last is None:
        raise FloatingPointError("拒绝步缓存没有覆盖缩小后的信赖域")
    last = cache.store_last
    stats = dict(numinner=last["numinner"], hessvecevals=0, memorytCG_MB=cache.payload_mb,
                 numstored=len(cache.store_iters)+1, cached=True, stopreason=last["stopreason_str"])
    return TCGResult(last["eta"], last["Heta"], False, stats, cache)


def trs_tcg_cached(geometry, grad, hessian: Callable, delta, options, *,
                   dimension=None, accepted=True, cache=None):
    """无预条件器的原默认 cached-tCG；geometry.project 对应 M.tangent。"""
    if not accepted and options.trscache and cache is not None:
        return tcg_rejectedstep(cache, delta)
    maxinner = options.maxinner if options.maxinner is not None else dimension
    if maxinner is None or maxinner < 1:
        raise ValueError("必须给出流形维数或 maxinner")
    dot = lambda a, b: float(geometry.inner(a, b))
    eta, heta = torch.zeros_like(grad), torch.zeros_like(grad)
    residual = grad
    r_r = dot(residual, residual)
    norm_r0 = _sqrt(r_r)
    z = residual  # getPrecon 在无 problem.precon 时恒等。
    z_r = dot(z, residual)
    d_pd, e_pd, e_pe = z_r, 0., 0.
    mdelta = z
    model_value, max_normsq = 0., 0.
    limited = False
    reason = "maximum inner iterations"
    stored = TCGCache() if options.trscache else None
    for j in range(1, maxinner+1):
        hmdelta = hessian(mdelta)
        d_hd = dot(mdelta, hmdelta)
        alpha = _divide(z_r, d_hd)
        e_pe_new = e_pe + 2.*alpha*e_pd + alpha*alpha*d_pd
        record = dict(normsq=e_pe_new, numinner=j, e_Pe=e_pe, d_Pd=d_pd,
                      e_Pd=e_pd, d_Hd=d_hd, eta=eta, Heta=heta,
                      mdelta=mdelta, Hmdelta=hmdelta)
        if stored is not None:
            next_smallest = ((1/16)**math.floor(-.25*(math.log2(max_normsq) -
                              math.log2(delta**2))) * delta**2 if max_normsq > 0 else 0.)
            if d_hd <= 0 or e_pe_new >= next_smallest:
                stored.store_iters.append(record)
                max_normsq = e_pe_new
        if d_hd <= 0 or e_pe_new >= delta**2:
            eta, heta, _, reason = _boundary(record, delta)
            limited = True
            break
        e_pe = e_pe_new
        new_eta = eta - alpha*mdelta
        new_heta = heta - alpha*hmdelta
        new_model = dot(new_eta, grad) + .5*dot(new_eta, new_heta)
        if new_model >= model_value:
            reason = "model increased"
            break
        eta, heta, model_value = new_eta, new_heta, new_model
        residual = residual - alpha*hmdelta
        r_r = dot(residual, residual)
        norm_r = _sqrt(r_r)
        if j >= options.mininner and norm_r <= norm_r0*min(norm_r0**options.theta, options.kappa):
            reason = ("reached target residual-kappa (linear)" if options.kappa < norm_r0**options.theta
                      else "reached target residual-theta (superlinear)")
            break
        z = residual
        zold_rold, z_r = z_r, dot(z, residual)
        beta = _divide(z_r, zold_rold)
        mdelta = geometry.project(z + beta*mdelta)
        e_pd = beta*(e_pd + alpha*d_pd)
        d_pd = z_r + beta*beta*d_pd
    if stored is not None and not limited:
        stored.store_last = dict(numinner=j, stopreason_str=reason, eta=eta, Heta=heta)
    stats = dict(numinner=j, hessvecevals=j, memorytCG_MB=0. if stored is None else stored.payload_mb,
                 numstored=0 if stored is None else len(stored.store_iters)+int(stored.store_last is not None),
                 cached=False, stopreason=reason)
    return TCGResult(eta, heta, limited, stats, stored)


def _uniform_measures(x, target, jm, scale):
    gradient, multiplier = prepare_projection(x).project_with_multiplier(2*(x-target))
    rg = float(torch.linalg.vector_norm(gradient))/(2*scale)
    rf = float(torch.linalg.vector_norm(x.mT @ left_j(x)-jm))/math.sqrt(x.shape[1])
    return rg, rf, multiplier/2


@torch.no_grad()
def solve_jensen_trustregions(target, options=None, *, initial=None):
    """论文采用的 R-TR2 搜索步骤；默认以共同欧氏残差作为停止条件。"""
    from .jensen_geometry import JensenGeometry

    validate_target(target)
    options = options or JensenTrustRegionOptions()
    if not isinstance(options, JensenTrustRegionOptions):
        raise TypeError("本方法需要 JensenTrustRegionOptions，不接受旧欧氏 RTR 选项")
    if initial is not None:
        validate_target(initial)
        if initial.shape != target.shape or initial.device != target.device:
            raise ValueError("初值形状和设备必须与目标一致")
    synchronize(target.device)
    started = time.perf_counter()
    n, k = target.shape[0]//2, target.shape[1]//2
    dimension = (4*n-2*k+1)*k
    delta_bar = options.delta_bar if options.delta_bar is not None else math.sqrt(dimension)
    delta = options.delta0 if options.delta0 is not None else delta_bar/8
    if delta > delta_bar:
        raise ValueError("delta0 不能大于 delta_bar")
    scale = max(1., float(torch.linalg.vector_norm(target)))
    jm = standard_j(k, device=target.device, dtype=target.dtype)
    x = orthosymplectic_initial(target) if initial is None else initial.detach().clone()
    history, cache = [], None
    iterations, accepted = 0, True
    status, error, omega = "max_iter", None, None
    counts = dict(accepted_steps=0, rejected_steps=0, hessvecevals=0, cached_subproblems=0,
                  retractions=0, geometry_builds=0)
    try:
        geometry = JensenGeometry(x)
        counts["geometry_builds"] += 1
        egrad = 2*(x-target)
        grad = geometry.gradient(egrad)
        cost = float(inner(x-target, x-target))
        norm_grad = float(geometry.norm(grad))
        rg, rf, omega = _uniform_measures(x, target, jm, scale)
        if rf > options.feas_tol:
            raise ValueError("初值不满足辛可行性容差")
        current = dict(iter=0, iteration=0, cost=cost, objective=cost, gradnorm=norm_grad,
                       gradient_relative=rg, feasibility_relative=rf, Delta=delta,
                       time=time.perf_counter()-started, rho=math.inf, rhonum=math.nan,
                       rhoden=math.nan, accepted=True, stepsize=math.nan, limitedbyTR=False,
                       numinner=0, hessvecevals=0, memorytCG_MB=0., cached=False)
        if options.history:
            history.append(current)
        while True:
            if not all(math.isfinite(v) for v in (cost, norm_grad, rg, rf)):
                status = "nonfinite"
                break
            if iterations >= options.miniter:
                # 对应 Manopt stoppingcriterion：时间/迭代优先于 stopfun。
                if time.perf_counter()-started >= options.time_limit:
                    status = "time_limit"
                    break
                if iterations >= options.max_iter:
                    status = "max_iter"
                    break
                if rg <= options.grad_tol and rf <= options.feas_tol:
                    status = "converged"
                    break
            old_delta = delta
            subproblem = trs_tcg_cached(geometry, grad,
                lambda v: geometry.hessian(egrad, 2*v, v), delta, options,
                dimension=dimension, accepted=accepted, cache=cache)
            cache = subproblem.cache
            counts["hessvecevals"] += subproblem.stats["hessvecevals"]
            counts["cached_subproblems"] += int(subproblem.stats["cached"])
            eta, heta = subproblem.eta, subproblem.Heta
            norm_eta = float(geometry.norm(eta))
            proposal = geometry.retract(eta)
            counts["retractions"] += 1
            proposal_cost = float(inner(proposal-target, proposal-target))
            raw_num = cost-proposal_cost
            raw_den = -float(geometry.inner(eta, grad+.5*heta))
            offset = max(1., abs(cost))*torch.finfo(target.dtype).eps*options.rho_regularization
            rhonum, rhoden = raw_num+offset, raw_den+offset
            model_decreased = rhoden >= 0
            rho = _divide(rhonum, rhoden)
            radius_action = "unchanged"
            if rho < .25 or not model_decreased or math.isnan(rho):
                delta /= 4
                radius_action = "decrease"
            elif rho > .75 and subproblem.limitedbyTR:
                delta = min(2*delta, delta_bar)
                radius_action = "increase"
            accepted = bool(model_decreased and rho > options.rho_prime)
            if accepted:
                x, cost = proposal, proposal_cost
                geometry = JensenGeometry(x)
                counts["geometry_builds"] += 1
                egrad = 2*(x-target)
                grad = geometry.gradient(egrad)
                norm_grad = float(geometry.norm(grad))
                rg, rf, omega = _uniform_measures(x, target, jm, scale)
                cache = None  # 原 StoreDB 移除旧基点；拒绝时保留全部缓存。
                counts["accepted_steps"] += 1
            else:
                counts["rejected_steps"] += 1
            iterations += 1
            current = dict(iter=iterations, iteration=iterations, cost=cost, objective=cost,
                           gradnorm=norm_grad, gradient_relative=rg, feasibility_relative=rf,
                           Delta=delta, Delta_before=old_delta, time=time.perf_counter()-started,
                           rho=rho, rho_noreg=_divide(raw_num, raw_den), rhonum=rhonum, rhoden=rhoden,
                           rho_reg_offset=offset, accepted=accepted, stepsize=norm_eta,
                           limitedbyTR=subproblem.limitedbyTR, model_decreased=model_decreased,
                           proposal_cost=proposal_cost, radius_action=radius_action, **subproblem.stats)
            if options.history:
                history.append(current)
    except (torch.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
        status, error = "numerical_failure", str(exc)
    synchronize(target.device)
    stats = dict(method="jensen_rtr2", source=SOURCE, options=asdict(options),
                 objective_convention="||X-T||_F^2", elapsed_seconds=time.perf_counter()-started,
                 counts=counts, error=error, initialization="target_polar" if initial is None else "provided",
                 stopping_mode="uniform", dimension=dimension,
                 Delta_bar=delta_bar, Delta0=options.delta0 if options.delta0 is not None else delta_bar/8,
                 memory_measure="cached numeric payload only; not MATLAB object memory")
    return SolveResult(x=x, omega=omega, status=status, iterations=iterations, history=history, stats=stats)
