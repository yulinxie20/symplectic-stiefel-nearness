"""运行论文实验并生成表图。"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PROTOCOL = {
    "cases": [dict(name=f"n{n}_k{k}", n=n, k=k)
              for n in (2048, 4096) for k in (16, 32, 48)],
    "round_orders": [
        ["n2048_k16", "n4096_k16", "n2048_k32", "n4096_k32", "n2048_k48", "n4096_k48"],
        ["n4096_k32", "n2048_k32", "n4096_k48", "n2048_k48", "n4096_k16", "n2048_k16"],
        ["n2048_k48", "n4096_k48", "n2048_k16", "n4096_k16", "n2048_k32", "n4096_k32"],
    ],
    "base_seed": 2026102100,
    "shuffle_seed_base": 202610210000,
    "targets_per_case": 64,
    "rounds": 3,
    "single_warmups": 2,
    "single_methods": ["dual", "gao_nonmonotone", "jensen_rtr2"],
    "batch_methods": ["dual_sequential", "dual"],
    "thresholds": {"feasibility": 1e-8, "stationarity": 1e-9, "delta": 1e-8},
    "gao_options": dict(max_iter=1000, time_limit=60.0, grad_tol=1e-9,
                        feas_tol=1e-8, history=False),
    "jensen_options": dict(max_iter=1000, time_limit=60.0, grad_tol=1e-9,
                           feas_tol=1e-8, history=False),
    "cpu_threads": 1,
}
TASKS = [(r, name) for r, names in enumerate(PROTOCOL["round_orders"], 1) for name in names]


def bind_cpu(cpu):
    """工作进程启动时绑定单核，系统调用失败即报错。"""
    if sys.platform == "win32":
        from ctypes import wintypes as w
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.argtypes = []
        kernel.GetCurrentProcess.restype = w.HANDLE
        kernel.SetProcessAffinityMask.argtypes = [w.HANDLE, ctypes.c_size_t]
        kernel.SetProcessAffinityMask.restype = w.BOOL
        if cpu >= 8 * ctypes.sizeof(ctypes.c_size_t):
            raise ValueError("CPU 编号超出 Windows 亲和性掩码范围")
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), 1 << cpu):
            raise ctypes.WinError(ctypes.get_last_error())
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
    else:
        raise RuntimeError("需要 Windows 或支持 CPU 亲和性的 Linux")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results", help="全新输出目录")
    parser.add_argument("--cpu", type=int, default=2, help="绑定的逻辑 CPU，默认 2")
    parser.add_argument("--worker", type=int, choices=range(len(TASKS)), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cpu < 0:
        parser.error("--cpu 必须为非负整数")
    sys.dont_write_bytecode = True
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    if not os.environ.get("CUDA_PATH"):
        nvcc = shutil.which("nvcc")
        candidates = [Path(sys.prefix) / "Library", Path(sys.prefix)]
        if nvcc:
            candidates.insert(0, Path(nvcc).resolve().parent.parent)
        for cuda_root in candidates:
            if (cuda_root / "include" / "cuda_runtime.h").is_file():
                os.environ["CUDA_PATH"] = str(cuda_root)
                break
    sys.path.insert(0, str(ROOT / "src"))
    output = args.output.resolve()
    if args.worker is not None:
        # 先绑核，再导入数值库。
        bind_cpu(args.cpu)
        from symplectic_projection.experiment import run_case
        round_number, name = TASKS[args.worker]
        run_case(output, PROTOCOL, name, round_number)
        return

    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(
        json.dumps(dict(PROTOCOL, worker_cpu=args.cpu), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    for task, (round_number, name) in enumerate(TASKS):
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(task),
                   "--cpu", str(args.cpu), "--output", str(output)]
        print(f"round {round_number}/3: {name}", flush=True)
        subprocess.run(command, check=True)
    os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
    from symplectic_projection.experiment import summarize_results
    summary = summarize_results(output, PROTOCOL)
    if not summary["all_calls_successful"]:
        raise RuntimeError("部分调用未通过验收；原始时间、失败记录和汇总均已保留")
    print(f"结果：{output}", flush=True)


if __name__ == "__main__":
    main()
