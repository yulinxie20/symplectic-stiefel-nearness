"""论文算例和统一后验指标；目标一律不含 1/2。"""
from __future__ import annotations

import hashlib
import math

import torch


def left_j(x):
    n = x.shape[-2] // 2
    return torch.cat((x[..., n:, :], -x[..., :n, :]), dim=-2)


def make_target(case, seed):
    n, k, batch = case["n"], case["k"], case.get("batch", 1)
    generator = torch.Generator().manual_seed(seed)
    known = None
    if case["family"] == "gaussian":
        target = torch.randn(batch, 2*n, 2*k, dtype=torch.float64,
                             generator=generator) / math.sqrt(2*n)
    elif case["family"] == "known":
        basis = torch.linalg.qr(torch.randn(n, 2*k, dtype=torch.float64,
                                           generator=generator)).Q
        a, b = basis[:, :k], basis[:, k:]
        xr = torch.cat((math.sqrt(1.5)*a, math.sqrt(.5)*b), dim=1)
        xi = torch.cat((math.sqrt(.5)*b, math.sqrt(1.5)*a), dim=1)
        x = torch.cat((xr, xi), dim=0).unsqueeze(0)
        rho = 1-case["mu"]
        omega = rho*left_j(torch.eye(2*k, dtype=torch.float64)).unsqueeze(0)
        # 与 X-JXΩ 相同，但不用相近数相减生成 O(mu) 的列块。
        mu = case["mu"]
        tr = torch.cat(((2-mu)*math.sqrt(1.5)*a, mu*math.sqrt(.5)*b),dim=1)
        ti = torch.cat((mu*math.sqrt(.5)*b, (2-mu)*math.sqrt(1.5)*a),dim=1)
        target = torch.cat((tr,ti),dim=0).unsqueeze(0)
        known = {"x": x, "omega": omega, "objective": 4*k*rho*rho}
    elif case["family"] == "zero":
        target = torch.zeros(batch, 2*n, 2*k, dtype=torch.float64)
        known = {"objective": 2*k}
    elif case["family"] == "strict_gap":
        # 论文成对列顺序 B0=[2d,2id,a,b]，重排为标准 J4 顺序。
        b = torch.tensor([[2, 2j, 1, 1j], [4j, -4, 0, 1]], dtype=torch.complex128)
        b = b[:, [0, 2, 1, 3]]
        target = torch.cat((b.real, b.imag), dim=0).unsqueeze(0)
        known = {"dual_value": 18.0}
    else:
        raise ValueError(case["family"])
    return target, known


def target_hash(target):
    return hashlib.sha256(target.numpy().tobytes()).hexdigest()


@torch.no_grad()
def inspect_output(target, x, omega, known=None):
    """CPU FP64复算；一阶可行验收与全局精度验收分开记录。

    主乘子一律由输出点的法向投影重新计算。返回乘子仅作一致性诊断。
    数值目标差不是区间算术证书，近似可行目标差可为负。
    """
    from symplectic_projection.baselines.geometry import prepare_projection

    records = []
    m, n = target.shape[-1], target.shape[-2]//2
    jm = left_j(torch.eye(m, dtype=torch.float64))
    identity = torch.eye(m, dtype=torch.complex128)
    unknown_optimum = known is not None and "dual_value" in known and "objective" not in known
    for index in range(len(target)):
        record = {"index": index, "accepted": False, "quality_pass": False,
                  "global_accuracy_pass": None if unknown_optimum else False,
                  "multiplier_source": "output_projection"}
        stage = "output_validation"
        try:
            t, y = target[index], x[index]
            if y.shape != t.shape:
                raise ValueError(f"输出形状 {tuple(y.shape)} 与目标 {tuple(t.shape)} 不同")
            if not bool(torch.isfinite(y).all()):
                record["reason"] = "nonfinite_output"
                records.append(record)
                continue
            error, jy = y-t, left_j(y)
            f = float(torch.sum(error*error))
            scale = max(1.0, float(torch.linalg.vector_norm(t)))
            feas = float(torch.linalg.vector_norm(y.T@jy-jm)/math.sqrt(m))
            record.update(objective=f, feasibility=feas)
            if known:
                if "objective" in known:
                    fstar = known["objective"]
                    relative_error = (f-fstar)/max(1.0, abs(fstar))
                    record.update(
                        known_optimum=fstar, relative_objective_error=relative_error,
                        global_accuracy_pass=math.isfinite(relative_error)
                        and abs(relative_error) <= 1e-8)
                if "x" in known:
                    record["relative_solution_error"] = float(
                        torch.linalg.vector_norm(y-known["x"][index]) /
                        torch.linalg.vector_norm(known["x"][index]))
                if "dual_value" in known:
                    record["known_dual_value"] = known["dual_value"]
                    record["difference_from_known_dual"] = f-known["dual_value"]

            stage = "projection"
            gradient, multiplier = prepare_projection(y).project_with_multiplier(2*error)
            projected_omega = multiplier/2
            if not bool(torch.isfinite(gradient).all() and torch.isfinite(projected_omega).all()):
                raise FloatingPointError("投影梯度或投影乘子不是有限数")
            stat = float(torch.linalg.vector_norm(gradient)/(2*scale))
            record.update(stationarity=stat, accepted=feas <= 1e-8 and stat <= 1e-9,
                          multiplier_skew_residual=float(
                              torch.linalg.vector_norm(projected_omega+projected_omega.T)))

            # 无效的返回乘子不影响统一投影乘子的后验验收。
            record.update(returned_multiplier_available=False,
                          returned_multiplier_stationarity=None,
                          returned_multiplier_skew_residual=None,
                          returned_multiplier_relative_difference=None)
            try:
                returned = None if omega is None else omega[index]
                if returned is None:
                    record["returned_multiplier_reason"] = "missing"
                elif returned.shape != projected_omega.shape:
                    record["returned_multiplier_reason"] = "invalid_shape"
                elif not bool(torch.isfinite(returned).all()):
                    record["returned_multiplier_reason"] = "nonfinite"
                else:
                    record.update(
                        returned_multiplier_available=True,
                        returned_multiplier_stationarity=float(
                            torch.linalg.vector_norm(error-jy@returned)/scale),
                        returned_multiplier_skew_residual=float(
                            torch.linalg.vector_norm(returned+returned.T)),
                        returned_multiplier_relative_difference=float(
                            torch.linalg.vector_norm(returned-projected_omega) /
                            max(1.0, float(torch.linalg.vector_norm(projected_omega)))))
            except Exception as exc:
                record["returned_multiplier_reason"] = f"{type(exc).__name__}: {exc}"

            # 仅在 Ω_proj 给出的 K 数值正定时走独立复数 Cholesky 路径。
            # 边界上的因子分解失败只影响此项诊断，不丢弃已有残差或已知最优值。
            try:
                kc = identity+1j*projected_omega
                eig = torch.linalg.eigvalsh(kc)
                record["lambda_min_k"] = float(eig[0])
                record["dual_validation_status"] = "not_positive_definite"
                if float(eig[0]) > 0:
                    b = torch.complex(t[:n], t[n:])
                    factor = torch.linalg.cholesky(kc)
                    solved = torch.linalg.solve_triangular(factor,b.mH,upper=False)
                    dual = float(torch.sum(t*t)-torch.sum(torch.abs(solved)**2)
                                 -torch.sum(projected_omega*jm))
                    relative_gap = (f-dual)/max(1.0, abs(f))
                    record.update(
                        computed_dual_value=dual,
                        computed_relative_primal_dual_difference=relative_gap,
                        dual_validation_status="computed" if math.isfinite(relative_gap)
                        else "nonfinite")
                    if not unknown_optimum and not (known and "objective" in known):
                        record["global_accuracy_pass"] = (
                            math.isfinite(relative_gap) and abs(relative_gap) <= 1e-8)
            except Exception as exc:
                record["dual_validation_status"] = "failed"
                record["dual_validation_error"] = f"{type(exc).__name__}: {exc}"
            record["quality_pass"] = (
                record["accepted"] and record["global_accuracy_pass"] is True)
        except Exception as exc:
            record.update(reason=f"{stage}_failed",
                          exception=f"{type(exc).__name__}: {exc}")
        records.append(record)
    return records
