"""同步 CPU→CPU 墙钟基准；结果仅输出到 stdout，不自动保存文件。"""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
    "smoke": (2, 128, 8),
    "paper-k32": (64, 4096, 32),
    "paper-k64": (64, 4096, 64),
    "paper-n8192": (64, 8192, 64),
}
FEASIBILITY_TOL, STATIONARITY_TOL = 1e-8, 1e-9


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="smoke")
    parser.add_argument("--B", "--batch", "--batch-size", dest="B", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    for name, default in zip(("B", "n", "k"), PRESETS[args.preset]):
        if getattr(args, name) is None:
            setattr(args, name, default)
    if min(args.B, args.n, args.k, args.repeats, args.threads) < 1:
        parser.error("B、n、k、repeats 和 threads 必须为正整数")
    if args.warmup < 0 or args.k > args.n:
        parser.error("warmup 必须非负，且 k 不能大于 n")
    return args


def driver_metadata():
    """驱动名称由 nvidia-smi 查询；不可用时明确记录，不猜测版本。"""
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        rows = [line.split(",") for line in query.stdout.splitlines() if line.strip()]
        return {"gpus": [
            dict(index=int(row[0].strip()), uuid=row[1].strip(), driver=row[2].strip())
            for row in rows
        ], "query_error": None}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        return {"gpus": [], "query_error": str(error)}


def source_hashes(package):
    """只对实际加载的包源码和本入口取哈希，排除缓存与生成物。"""
    directory = Path(package.__file__).resolve().parent
    files = [p for p in directory.rglob("*") if p.suffix in (".py", ".cu", ".cuh")]
    files.append(Path(__file__).resolve())
    result = {}
    for path in sorted(files):
        label = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        result[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def residuals(torch, T, output):
    """逐题检查，避免为大规模输出再构造完整批量的临时矩阵。"""
    X, omega, iterations = output
    if any(t.device.type != "cpu" for t in output):
        raise RuntimeError("计时协议要求 solve 返回 CPU 的 X、omega 和迭代次数")
    B, rows, m = T.shape
    if X.shape != T.shape or omega.shape != (B, m, m) or iterations.shape != (B,):
        raise RuntimeError("solve 返回的张量形状与输入不一致")
    if X.dtype != torch.float64 or omega.dtype != torch.float64:
        raise RuntimeError("本基准要求 FP64 输出")
    eye = torch.eye(m, dtype=T.dtype)
    Jm = torch.cat((eye[m // 2:], -eye[:m // 2]), dim=0)
    feasibility, stationarity, objective = [], [], []
    for b in range(B):
        JX = torch.cat((X[b, rows // 2:], -X[b, :rows // 2]), dim=0)
        feasibility.append(float(torch.linalg.vector_norm(X[b].T @ JX - Jm) / math.sqrt(m)))
        error = X[b] - T[b]
        objective.append(float(0.5 * torch.sum(error * error)))
        error.addmm_(JX, omega[b], beta=1, alpha=-1)
        stationarity.append(float(torch.linalg.vector_norm(error) /
                                  torch.linalg.vector_norm(T[b]).clamp_min(1)))
        del JX, error
    finite = all(math.isfinite(x) for x in feasibility + stationarity + objective)
    # 非有限值用 null 保持标准 JSON，并将验证标记为失败。
    clean = lambda values: [x if math.isfinite(x) else None for x in values]
    return {
        "constraint_residual_per_problem": clean(feasibility),
        "stationarity_residual_per_problem": clean(stationarity),
        "objective_per_problem": clean(objective),
        "constraint_residual_max": max(feasibility) if all(map(math.isfinite, feasibility)) else None,
        "stationarity_residual_max": max(stationarity) if all(map(math.isfinite, stationarity)) else None,
        "newton_iterations": iterations.tolist(),
        "finite": finite,
        "thresholds": {"constraint_residual_max": FEASIBILITY_TOL,
                       "stationarity_residual_max": STATIONARITY_TOL},
        "passed": finite and max(feasibility) < FEASIBILITY_TOL and max(stationarity) < STATIONARITY_TOL,
        "definitions": {
            "constraint": "||Xᵀ J₂ₙ X − J₂ₖ||F / sqrt(2k)",
            "stationarity": "||X − T − J₂ₙ X omega||F / max(||T||F, 1)",
            "objective": "0.5 ||X − T||F²",
        },
        "note": "指标在最后一次计时输出上逐题计算，不计入墙钟样本。",
    }


def measure(torch, solve, T, warmup, repeats):
    # 独立首次调用负责延迟编译和模块加载，随后才开始指定次数的预热。
    output = solve(T)
    torch.cuda.synchronize()
    del output
    for _ in range(warmup):
        output = solve(T)
        torch.cuda.synchronize()
        del output
    samples = []
    for index in range(repeats):
        torch.cuda.synchronize()
        begin = time.perf_counter_ns()
        output = solve(T)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - begin) / 1e6)
        # 返回对象释放在计时区间之外；最后一份输出供数值检查复用。
        if index + 1 < repeats:
            del output
    return samples, output


def run(args):
    # 同时支持源码检出直接运行和常规安装，不依赖机器上的绝对路径。
    if (ROOT / "src" / "symplectic_projection").is_dir():
        sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    import torch
    import symplectic_projection as package
    import cupy as cp

    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available():
        raise RuntimeError("本基准需要可用的 CUDA GPU")
    torch.cuda.init()
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    environment = {
        "executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cupy": cp.__version__,
        "numpy": np.__version__,
        "pytorch_cuda": torch.version.cuda,
        "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver_api_version": cp.cuda.runtime.driverGetVersion(),
        "nvidia_driver": driver_metadata(),
        "gpu": {
            "device_index": device,
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "sm_count": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
        },
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "package_file": str(Path(package.__file__).resolve()),
    }
    hashes = source_hashes(package)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    T = torch.randn(args.B, 2 * args.n, 2 * args.k, dtype=torch.float64,
                    device="cpu", generator=generator) / math.sqrt(2 * args.n)
    samples, output = measure(torch, package.solve, T, args.warmup, args.repeats)
    metrics = residuals(torch, T, output)
    del output
    return {
        "config": {**vars(args), "dtype": "float64", "input_shape": list(T.shape)},
        "environment": environment,
        "source_sha256": hashes,
        "timing": {
            "kind": "synchronized_cpu_to_cpu_wall",
            "unit": "ms",
            "scope": "已有 CPU 输入至 CPU 输出全部可用；含分配、主机提交、传输和同步。",
            "excluded": ["输入生成", "首次调用与延迟编译", "预热", "输出释放", "数值指标"],
            "note": "不是纯 GPU 核时间；未运行 CUPTI 或推算 FLOP。",
            "samples": samples,
            "mean": statistics.mean(samples),
            "median": statistics.median(samples),
            "min": min(samples),
            "max": max(samples),
        },
        "validation": metrics,
    }


def main():
    args = parse_args()
    # 包的诊断输出也移到 stderr，stdout 始终只包含一个 JSON 对象。
    with redirect_stdout(sys.stderr):
        result = run(args)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    return 0 if result["validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
