# SPDX-License-Identifier: GPL-3.0-or-later
# Adapted from opt-gaobin/spopt, commit 94c808dae9da975345b3e23e0e42bffbefbf63ae.
# Original copyright (C) 2020 P.-A. Absil, Bin Gao, Nguyen Thanh Son, Tatjana Stykel.
# Python adaptation copyright (C) 2026 YulinXie.
"""作者 spopt 的欧氏度量、Cayley 回缩及非单调 ABB 分支。

保留原始的 2n/4k 阶 Cayley 分支、初始步长、ABB 奇偶顺序、五次试探
上限及非单调参考值更新。适配仅包括 PyTorch/FP64、目标相关极分解初值、
统一相对梯度与可行性停止条件，以及二次目标差的等价稳定计算。
原 MATLAB 的 xtol/ftol 不参与其实际停止条件。

本文件采用 GPL-3.0-or-later；许可全文见根目录 THIRD_PARTY_NOTICES.txt。
"""
from __future__ import annotations

import math
import time

import torch

from .common import (SolveResult, SolverOptions, inner, left_j, objective_increment,
                     standard_j, synchronize, validate_target)
from .geometry import orthosymplectic_initial, prepare_projection

SOURCE = {
    "repository": "https://github.com/opt-gaobin/spopt",
    "commit": "94c808dae9da975345b3e23e0e42bffbefbf63ae",
    "file": "spopt.m",
    "source_sha256_lf": "d8edac82463317ad369e874d3bc8e3941f0c9926f69457293034a02dd5faa243",
    "metric": "Euclidean (metric=2)",
    "retraction": "Cayley (retr=1)",
    "stepsizeRule": 1,
    "stepsize": 1e-3,
    "maxgamma": 1e5,
    "mingamma": 1e-15,
    "beta": 1e-4,
    "delta": 0.1,
    "alpha": 0.85,
    "nt": 5,
    "original_stop": "absolute projected-gradient norm < gtol (default 1e-6)",
    "original_xtol_ftol_active": False,
}


class GaoCayley:
    """与 spopt.m 的 Euclidean+Cayley 两个分支逐式一致。"""

    def __init__(self, x, ambient_gradient, gradient, multiplier):
        n, k = x.shape[0] // 2, x.shape[1] // 2
        self.x = x
        jx = left_j(x)
        xj = x @ standard_j(k, device=x.device, dtype=x.dtype)
        jxg, xx = jx.mT @ ambient_gradient, x.mT @ x
        pg = ambient_gradient - 0.5 * (xj @ jxg) - jx @ multiplier + 0.5 * (xj @ (xx @ multiplier))
        self.dense = k >= n / 2
        if self.dense:
            pgxj = -pg @ xj.mT
            h = pgxj + pgxj.mT
            self.hj = torch.cat((-h[:, n:], h[:, :n]), dim=1)
            self.rjx = h @ jx
            self.identity = torch.eye(2*n, device=x.device, dtype=x.dtype)
        else:
            self.u = torch.cat((-pg, xj), dim=1)
            v = torch.cat((xj, -pg), dim=1)
            self.vju = -v.mT @ left_j(self.u)
            self.vjx = v.mT @ jx
            self.identity = torch.eye(4*k, device=x.device, dtype=x.dtype)

    def increment(self, step):
        if self.dense:
            return torch.linalg.solve(self.identity - 0.5*step*self.hj, step*self.rjx)
        return self.u @ (step * torch.linalg.solve(self.identity + 0.5*step*self.vju, self.vjx))


def nonmonotone_update(reference_gap, q, increment):
    """用 C-f 储存非单调参考值；等价于 C+=(alpha Q C+f+)/(alpha Q+1)。"""
    next_q = SOURCE["alpha"] * q + 1.0
    return SOURCE["alpha"] * q * (reference_gap - increment) / next_q, next_q


@torch.no_grad()
def solve_gao_nonmonotone(target, options=None, *, initial=None):
    validate_target(target)
    options = options or SolverOptions()
    if initial is not None:
        validate_target(initial)
        if initial.shape != target.shape or initial.device != target.device:
            raise ValueError("初值形状和设备必须与目标一致")
    synchronize(target.device)
    started = time.perf_counter()
    target_scale = max(1.0, float(torch.linalg.vector_norm(target)))
    jm = standard_j(target.shape[1] // 2, device=target.device, dtype=target.dtype)
    x = orthosymplectic_initial(target) if initial is None else initial.detach().clone()
    gamma, reference_gap, q = SOURCE["stepsize"], 0.0, 1.0
    iterations, status, error = 0, "max_iter", None
    counts = dict(retraction_trials=0, backtracks=0, forced_fifth_trials=0,
                  objective_increases=0, max_objective_increase=0.0)
    history = []
    gradient = multiplier = None
    try:
        ambient = 2*(x-target)
        gradient, multiplier = prepare_projection(x).project_with_multiplier(ambient)
        while True:
            norm_g = float(torch.linalg.vector_norm(gradient))
            feasibility = float(torch.linalg.vector_norm(x.mT @ left_j(x)-jm))/math.sqrt(x.shape[1])
            f = float(inner(x-target, x-target))
            if not all(math.isfinite(v) for v in (norm_g, feasibility, f)):
                status = "nonfinite"
                break
            if iterations == 0 and feasibility > options.feas_tol:
                raise ValueError("初值不满足辛可行性容差")
            record = dict(iteration=iterations, objective=f, gradient_relative=norm_g/(2*target_scale),
                          feasibility_relative=feasibility, reference_gap=reference_gap,
                          reference_q=q, trial_step=gamma)
            if options.history:
                history.append(record)
            if norm_g/(2*target_scale) <= options.grad_tol and feasibility <= options.feas_tol:
                status = "converged"
                break
            if iterations >= options.max_iter:
                break
            if time.perf_counter()-started >= options.time_limit:
                status = "time_limit"
                break
            retraction = GaoCayley(x, ambient, gradient, multiplier)
            derivative = SOURCE["beta"] * abs(float(inner(ambient, gradient)))
            step = gamma
            for trial in range(SOURCE["nt"]):
                delta = retraction.increment(step)
                difference = objective_increment(x-target, delta)
                counts["retraction_trials"] += 1
                condition = math.isfinite(difference) and difference <= reference_gap - step*derivative
                if condition or trial == SOURCE["nt"]-1:
                    if not math.isfinite(difference):
                        raise FloatingPointError("第五次试探非有限，不能返回正常结果")
                    counts["forced_fifth_trials"] += int(not condition)
                    break
                step *= SOURCE["delta"]
                counts["backtracks"] += 1
            if difference > 0:
                counts["objective_increases"] += 1
                counts["max_objective_increase"] = max(counts["max_objective_increase"], difference)
            record.update(step=step, objective_increment=difference, line_search_condition=condition,
                          trials=trial+1)
            previous_gradient = gradient
            x = x + delta
            iterations += 1
            # 原作者使用环境空间割线，无额外向量传输。
            ambient = 2*(x-target)
            gradient, multiplier = prepare_projection(x).project_with_multiplier(ambient)
            y = gradient - previous_gradient
            sy = abs(float(inner(delta, y)))
            numerator, denominator = ((float(inner(delta, delta)), sy) if iterations % 2 == 0
                                      else (sy, float(inner(y, y))))
            if not math.isfinite(numerator) or not math.isfinite(denominator):
                raise FloatingPointError("ABB 割线非有限")
            if denominator == 0:
                # 梯度已满足时下一轮按统一标准停止；否则作为明确失败处理。
                if float(torch.linalg.vector_norm(gradient))/(2*target_scale) > options.grad_tol:
                    raise FloatingPointError("ABB 分母为零")
                gamma = SOURCE["mingamma"]
            else:
                gamma = min(SOURCE["maxgamma"], max(SOURCE["mingamma"], numerator/denominator))
            reference_gap, q = nonmonotone_update(reference_gap, q, difference)
    except (torch.linalg.LinAlgError, FloatingPointError) as exc:
        status, error = "numerical_failure", str(exc)
    synchronize(target.device)
    stats = dict(method="gao_nonmonotone", source=SOURCE, objective_convention="||X-T||_F^2",
                 elapsed_seconds=time.perf_counter()-started, counts=counts, error=error,
                 initialization="target_polar" if initial is None else "provided",
                 line_search_objective="exact quadratic increments and C-f recurrence")
    return SolveResult(x=x, omega=None if multiplier is None else multiplier/2,
                       status=status, iterations=iterations, history=history, stats=stats)
