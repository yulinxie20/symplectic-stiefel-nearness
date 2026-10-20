# 来源：2026-10-13_a800_symplectic_dual/symplectic_gpu/riemannian.py
"""双精度黎曼对照：稳定目标差 ABB 与真实黎曼 Hessian 的 RTR+tCG。

目标 f(X)=||X-T||²，常欧氏度量，Gao 2k Cayley 回缩。所有求解均
禁止自动微分图，不形成显式 Hessian 或 2n 阶方阵。首版控制流含
设备标量同步，重点是可验证基线，尚未实现 CUDA Graph 或融合算子。
"""
from __future__ import annotations

import math
import time
from typing import Callable

import torch

from .common import (SolverOptions, SolveResult, inner, left_j, standard_j,
                     synchronize, validate_target)
from .geometry import orthosymplectic_initial, prepare_cayley, prepare_projection


class _TimeLimit(Exception):
    pass


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value))


class EuclideanGeometry:
    """Hess f(X)[Z]=P_X(2Z-JZΛ)，Λ 为环境梯度的投影乘子。"""

    def __init__(self, x, target, counts=None, check_budget=None):
        self.counts = counts if counts is not None else {}
        self.check_budget = check_budget or (lambda: None)
        self.check_budget()
        self.counts["projection_preparations"] = self.counts.get("projection_preparations", 0) + 1
        self.cache = prepare_projection(x)
        self.gradient, self.multiplier = self.cache.project_with_multiplier(2 * (x - target))
        # 接近驻点时首次投影是两个 O(1) 量相减；重投影将法向舍入误差
        # 从 O(eps ||环境梯度||) 降到 O(eps ||黎曼梯度||)，稳定严格线搜索。
        self.gradient = self.cache.project(self.gradient)
        self.counts["projection_calls"] = self.counts.get("projection_calls", 0) + 2

    def project(self, value):
        self.check_budget()
        self.counts["projection_calls"] = self.counts.get("projection_calls", 0) + 1
        return self.cache.project(value)

    def hessian(self, tangent):
        self.check_budget()
        self.counts["hessian_vector_calls"] = self.counts.get("hessian_vector_calls", 0) + 1
        return self.project(2 * tangent - left_j(tangent) @ self.multiplier)


def objective_increment(residual: torch.Tensor, delta: torch.Tensor) -> float:
    """精确二次式的目标差，避免直接相减两个几乎相等的大目标值。"""
    return float(2 * inner(residual, delta) + inner(delta, delta))


def _boundary_parameter(eta: torch.Tensor, direction: torch.Tensor, radius: float) -> float:
    a, b = float(inner(direction, direction)), float(inner(eta, direction))
    norm_eta = _norm(eta)
    if not math.isfinite(a) or a <= 0 or norm_eta > radius * (1 + 64 * torch.finfo(eta.dtype).eps):
        raise FloatingPointError("无效的信赖域边界方程")
    c = min(0.0, (norm_eta - radius) * (norm_eta + radius))
    root = math.sqrt(max(0.0, b * b - a * c))
    return (-c / (b + root) if b + root > 0 else 0.0) if b >= 0 else (-b + root) / a


def truncated_cg(gradient: torch.Tensor, hessian: Callable, radius: float, *,
                 project: Callable | None = None, relative_tolerance: float = 0.1,
                 max_iter: int = 100, check_budget: Callable | None = None):
    """球内截断 CG；处理非正曲率，并额外复核真实模型与真实残差。"""
    if not math.isfinite(radius) or radius <= 0 or not 0 < relative_tolerance < 1 or max_iter < 1:
        raise ValueError("无效的截断 CG 参数")
    project = project or (lambda z: z)
    check_budget = check_budget or (lambda: None)
    eta = torch.zeros_like(gradient)
    residual, direction = gradient.clone(), -gradient
    norm_g = _norm(gradient)
    residual_squared = float(inner(residual, residual))
    details = dict(reason="inner_max_iter", iterations=0, boundary=False,
                   nonpositive_curvature=False)
    if norm_g == 0:
        details.update(reason="zero_gradient", predicted_decrease=0.0,
                       relative_residual=0.0, residual_satisfied=True)
        return eta, eta.clone(), details
    for iteration in range(max_iter):
        check_budget()
        direction = project(direction)
        h_direction = hessian(direction)
        curvature = float(inner(direction, h_direction))
        details["iterations"] = iteration + 1
        if not math.isfinite(curvature):
            raise FloatingPointError("截断 CG 曲率不是有限数")
        if curvature <= 0:
            eta = eta + _boundary_parameter(eta, direction, radius) * direction
            details.update(reason="nonpositive_curvature", boundary=True, nonpositive_curvature=True)
            break
        alpha = residual_squared / curvature
        proposal = eta + alpha * direction
        if _norm(proposal) >= radius:
            eta = eta + _boundary_parameter(eta, direction, radius) * direction
            details.update(reason="radius_boundary", boundary=True)
            break
        eta = proposal
        new_residual = residual + alpha * h_direction
        new_squared = float(inner(new_residual, new_residual))
        if not math.isfinite(new_squared):
            raise FloatingPointError("截断 CG 残差不是有限数")
        residual = new_residual
        if math.sqrt(new_squared) <= relative_tolerance * norm_g:
            details["reason"] = "residual"
            break
        direction = -residual + (new_squared / residual_squared) * direction
        residual_squared = new_squared
    h_eta = hessian(eta)
    true_residual = _norm(gradient + h_eta)
    predicted = -float(inner(gradient, eta)) - 0.5 * float(inner(eta, h_eta))
    satisfied = true_residual <= relative_tolerance * norm_g
    if details["reason"] == "residual" and not satisfied:
        details["reason"] = "residual_verification_failed"
    details.update(predicted_decrease=predicted, relative_residual=true_residual / norm_g,
                   residual_satisfied=satisfied)
    return eta, h_eta, details


def _abb_step(s, y, accepted_iteration):
    sy = abs(float(inner(s, y)))
    numerator, denominator = ((float(inner(s, s)), sy) if accepted_iteration % 2 == 0
                              else (sy, float(inner(y, y))))
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise FloatingPointError("ABB 步长不是有限数")
    # 舍入导致零割线时重置到环境二次项的精确步长，并记录该情况。
    if denominator <= 0 or numerator <= 0:
        return 0.5, True
    return min(1e5, max(1e-15, numerator / denominator)), False


@torch.no_grad()
def solve_riemannian(target: torch.Tensor, options: SolverOptions | None = None, *,
                     method: str = "abb", initial: torch.Tensor | None = None) -> SolveResult:
    """返回相对梯度/可行性停机的黎曼基线结果，失败不会伪装成收敛。

    grad_tol 对应 ||grad_R f||/(2 max(1,||T||))，feas_tol 对应
    ||XᵀJX-J||/sqrt(2k)。返回的 omega=Λ/2，匹配论文 K=I+iΩ。
    """
    validate_target(target)
    options = options or SolverOptions()
    if method not in {"abb", "rtr"}:
        raise ValueError("method 必须是 abb 或 rtr")
    if options.max_iter < 0 or options.max_backtracks < 1 or options.cg_max_iter < 1:
        raise ValueError("迭代和回溯次数无效")
    if not 0 < options.cg_rtol < 1:
        raise ValueError("cg_rtol 必须介于 0 和 1 之间")
    if any(not math.isfinite(v) or v <= 0 for v in (options.grad_tol, options.feas_tol, options.time_limit)):
        raise ValueError("容差与时间预算必须是正的有限数")
    if initial is not None:
        validate_target(initial)
        if initial.shape != target.shape or initial.device != target.device:
            raise ValueError("初值必须与目标具有相同形状和设备")
    target = target.detach()
    synchronize(target.device)
    started = time.perf_counter()
    counts = dict(projection_preparations=0, projection_calls=0, hessian_vector_calls=0,
                  retraction_preparations=0, retraction_trials=0, retraction_failures=0,
                  accepted_steps=0, rejected_steps=0, backtracks=0, tcg_iterations=0,
                  abb_resets=0)
    history, status, error = [], "max_iter", None
    geometry, x, iterations = None, None, 0
    last_recorded_x = None
    f = grad_norm = feasibility = None
    target_scale = max(1.0, _norm(target))
    j_small = standard_j(target.shape[1] // 2, device=target.device, dtype=target.dtype)

    def check_budget():
        if time.perf_counter() - started >= options.time_limit:
            raise _TimeLimit

    try:
        x = orthosymplectic_initial(target) if initial is None else initial.detach().clone()
        feasibility = _norm(x.mT @ left_j(x) - j_small) / math.sqrt(x.shape[1])
        if feasibility > options.feas_tol:
            raise ValueError("给定初值不满足请求的辛可行性容差")
        geometry = EuclideanGeometry(x, target, counts, check_budget)
        radius, max_radius, step = max(1.0, 0.1 * _norm(x)), max(100.0, _norm(x)), 0.5
        while True:
            residual = x - target
            f = float(inner(residual, residual))
            grad_norm = _norm(geometry.gradient)
            feasibility = _norm(x.mT @ left_j(x) - j_small) / math.sqrt(x.shape[1])
            last_recorded_x = x
            record = dict(iteration=iterations, objective=f, gradient_norm=grad_norm,
                          gradient_relative=grad_norm / (2 * target_scale),
                          feasibility_relative=feasibility,
                          elapsed_seconds=time.perf_counter() - started)
            if options.history:
                history.append(record)
            if not all(math.isfinite(v) for v in (f, grad_norm, feasibility)):
                status = "nonfinite"
                break
            if grad_norm / (2 * target_scale) <= options.grad_tol and feasibility <= options.feas_tol:
                status = "converged"
                break
            if iterations >= options.max_iter:
                break
            check_budget()
            if method == "abb":
                direction = -geometry.gradient
                derivative = -grad_norm**2
                counts["retraction_preparations"] += 1
                retraction = prepare_cayley(x, direction)
                accepted = False
                for backtrack in range(options.max_backtracks):
                    check_budget()
                    counts["retraction_trials"] += 1
                    try:
                        delta = retraction.increment(step)
                        difference = objective_increment(residual, delta)
                    except torch.linalg.LinAlgError:
                        counts["retraction_failures"] += 1
                        difference = math.inf
                    if math.isfinite(difference) and difference <= 1e-4 * step * derivative:
                        accepted = True
                        break
                    step *= 0.5
                    counts["backtracks"] += 1
                record.update(step=step, accepted=accepted,
                              objective_difference=difference if math.isfinite(difference) else None)
                if not accepted:
                    status = "line_search_failed"
                    break
                previous_gradient = geometry.gradient
                x = x + delta
                geometry = None
                geometry = EuclideanGeometry(x, target, counts, check_budget)
                counts["accepted_steps"] += 1
                iterations += 1
                # 环境割线 ABB，与历史基线相同；该向量差不是精确向量传输。
                step, reset = _abb_step(delta, geometry.gradient - previous_gradient, iterations)
                counts["abb_resets"] += int(reset)
            else:
                eta, _, inner_details = truncated_cg(
                    geometry.gradient, geometry.hessian, radius, project=geometry.project,
                    relative_tolerance=options.cg_rtol, max_iter=options.cg_max_iter,
                    check_budget=check_budget)
                counts["tcg_iterations"] += inner_details["iterations"]
                predicted = inner_details["predicted_decrease"]
                record.update(radius=radius, inner=inner_details)
                if not math.isfinite(predicted) or predicted <= 0:
                    status = "nonpositive_predicted_decrease"
                    break
                counts["retraction_preparations"] += 1
                counts["retraction_trials"] += 1
                try:
                    delta = prepare_cayley(x, eta).increment()
                    difference = objective_increment(residual, delta)
                    rho = -difference / predicted
                except torch.linalg.LinAlgError:
                    counts["retraction_failures"] += 1
                    difference, rho = math.inf, -math.inf
                accepted = math.isfinite(rho) and rho >= 0.1
                record.update(accepted=accepted, rho=rho if math.isfinite(rho) else None,
                              objective_difference=difference if math.isfinite(difference) else None)
                if not math.isfinite(rho) or rho < 0.25:
                    radius *= 0.25
                elif rho > 0.75 and inner_details["boundary"]:
                    radius = min(2 * radius, max_radius)
                iterations += 1
                if accepted:
                    counts["accepted_steps"] += 1
                    x = x + delta
                    geometry = None
                    geometry = EuclideanGeometry(x, target, counts, check_budget)
                else:
                    counts["rejected_steps"] += 1
                if radius <= torch.finfo(target.dtype).eps * max(1.0, _norm(x)):
                    status = "trust_radius_breakdown"
                    break
    except _TimeLimit:
        status = "time_limit"
    except (torch.linalg.LinAlgError, FloatingPointError) as exc:
        status, error = "numerical_failure", str(exc)
    # 超时可能发生在已经接受新点、尚未写入下一条历史之间；不返回旧点指标。
    if x is not None and last_recorded_x is not x:
        residual = x - target
        f = float(inner(residual, residual))
        grad_norm = None if geometry is None else _norm(geometry.gradient)
        feasibility = _norm(x.mT @ left_j(x) - j_small) / math.sqrt(x.shape[1])
    synchronize(target.device)
    stats = dict(method="riemannian_" + method, retraction="gao_cayley_2k",
                 metric="euclidean", objective_convention="||X-T||_F^2",
                 elapsed_seconds=time.perf_counter() - started, counts=counts,
                 error=error, objective=f, gradient_norm=grad_norm,
                 gradient_relative=None if grad_norm is None else grad_norm / (2 * target_scale),
                 feasibility_relative=feasibility,
                 model_scope="真实黎曼 Hessian；普通 Cayley 不保证欧氏二阶回缩",
                 initialization="target_polar" if initial is None else "provided",
                 preconditioner="identity")
    return SolveResult(x=x, omega=None if geometry is None else geometry.multiplier / 2,
                       status=status, iterations=iterations, history=history, stats=stats)
