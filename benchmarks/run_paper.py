"""论文单问题比较、边界检查和批量吞吐；GPU运行串行，逐次保存失败。"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
# 缓存为可重建的本地文件，不写用户目录；不计入预热后的正式时间。
os.environ.setdefault("CUPY_CACHE_DIR",str(ROOT/".tmp"/"cupy"))
(ROOT/".tmp"/"compiler").mkdir(parents=True,exist_ok=True)
tempfile.tempdir=str(ROOT/".tmp"/"compiler")

import numpy as np
import torch
from symplectic_projection import solve
from symplectic_projection.baselines import SolverOptions, solve_riemannian
import cupy as cp
from paper_cases import inspect_output, make_target, target_hash

METHODS = ["dual", "rgd", "rtr"]
OPTIONS = dict(max_iter=1000, grad_tol=1e-9, feas_tol=1e-8,
               max_backtracks=40, cg_max_iter=100, cg_rtol=.1,
               time_limit=60.0, history=False)


def cases_for(suite):
    if suite == "pilot":
        return [dict(name=f"pilot_n{n}_k{k}", family="gaussian", n=n, k=k)
                for n,k in [(64,4),(256,16),(1024,32)]]
    if suite == "scaling":
        return [dict(name=f"n{n}_k{k}", family="gaussian", n=n, k=k)
                for n,k in [(256,8),(1024,8),(4096,8),(256,32),
                            (1024,32),(4096,32),(4096,64)]]
    if suite == "critical":
        return [dict(name=f"critical_n{2*k-1}_k{k}", family="gaussian", n=2*k-1, k=k)
                for k in [4,8,16]]
    if suite == "boundary":
        return [dict(name=f"known_mu{mu:g}", family="known", n=128, k=4, mu=mu)
                for mu in [.5,.01,.0001,.000001]]+[
            dict(name="zero_n16_k4",family="zero",n=16,k=4),
            dict(name="strict_gap_n2_k2",family="strict_gap",n=2,k=2)]
    if suite == "batch":
        return [dict(name=f"batch{batch}_n1024_k32",family="gaussian",
                     n=1024,k=32,batch=batch) for batch in [8,32]]
    raise ValueError(suite)


def environment():
    prop = torch.cuda.get_device_properties(0)
    query = subprocess.run(["nvidia-smi","--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader"],capture_output=True,text=True,check=False)
    git = subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,check=False)
    files = sorted((ROOT/"src").rglob("*.py"))+sorted((ROOT/"src").rglob("*.cu"))
    files += sorted((ROOT/"benchmarks").glob("*.py"))
    return dict(executable=sys.executable,python=sys.version,platform=platform.platform(),
                processor=platform.processor(),torch=torch.__version__,numpy=np.__version__,
                cupy=cp.__version__,pytorch_cuda=torch.version.cuda,
                cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                gpu=prop.name,sm_count=prop.multi_processor_count,memory_bytes=prop.total_memory,
                nvidia_smi=query.stdout.strip(),cpu_threads=torch.get_num_threads(),
                commit=git.stdout.strip(),source_sha256={
                    p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in files})


def call_method(method, target, history=False):
    if method == "dual":
        x, omega, iterations = solve(target)
        return x, omega, iterations.tolist(), ["returned"]*len(target), None
    xs, omegas, iterations, statuses, histories = [], [], [], [], []
    for index,t in enumerate(target):
        options = SolverOptions(**{**OPTIONS,"history":history})
        result = solve_riemannian(t.to("cuda"),options,method="abb" if method=="rgd" else "rtr")
        xs.append(result.x.cpu() if result.x is not None else torch.full_like(t,float("nan")))
        omegas.append(result.omega.cpu() if result.omega is not None else
                      torch.full((t.shape[1],t.shape[1]),float("nan"),dtype=t.dtype))
        iterations.append(result.iterations)
        statuses.append(result.status)
        histories.append(result.history)
    return torch.stack(xs),torch.stack(omegas),iterations,statuses,histories if history else None


def safe_call(method,target,known,*,measure=True,history=False):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter_ns()
    output=None
    elapsed=None
    try:
        output=call_method(method,target,history)
        torch.cuda.synchronize()
        elapsed=(time.perf_counter_ns()-started)/1e6
        allocated=torch.cuda.max_memory_allocated()
        x,omega,iterations,statuses,histories=output
        validation=inspect_output(target,x,omega,known)
        normal_return=all(s in ("returned","converged") for s in statuses)
        return dict(milliseconds=elapsed,exception=None,iterations=iterations,status=statuses,
                    validation=validation,accepted=all(v["accepted"] for v in validation)
                    and normal_return,
                    quality_pass=all(v.get("quality_pass",False) for v in validation) and normal_return,
                    peak_torch_allocated_bytes=allocated,history=histories)
    except Exception as exc:
        torch.cuda.synchronize()
        return dict(milliseconds=elapsed if elapsed is not None else (time.perf_counter_ns()-started)/1e6,
                    exception=f"{type(exc).__name__}: {exc}",accepted=False,
                    quality_pass=False,batch_aborted=len(target)>1,
                    failed_stage="validation" if output is not None else "solver",
                    peak_torch_allocated_bytes=torch.cuda.max_memory_allocated())
    finally:
        del output


def clean(value):
    if isinstance(value,float) and not math.isfinite(value):
        return None
    if isinstance(value,dict):
        return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):
        return [clean(v) for v in value]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",choices=["pilot","scaling","critical","boundary","batch"],required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=[20261021,20261022,20261023])
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--warmup",type=int,default=2)
    parser.add_argument("--only",nargs="+")
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=METHODS)
    parser.add_argument("--history",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("需要CUDA")
    cases=cases_for(args.suite)
    if args.only:
        cases=[c for c in cases if c["name"] in args.only]
    if not cases:
        raise ValueError("没有选中算例")
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"raw.jsonl").exists():
        raise FileExistsError("输出目录已有raw.jsonl；请另选目录以保留原始数据")
    metadata={"suite":args.suite,"cases":cases,"seeds":args.seeds,"repeats":args.repeats,
              "warmup":args.warmup,"methods":args.methods,"options":OPTIONS,
              "timing":"CPU input to CPU output; synchronized wall; initialization included",
              "excluded":["input generation","compilation","warmup","validation","output release"],
              "batch_baselines":"sequential per-target GPU calls; dual uses native batch",
              "thresholds":{"feasibility":1e-8,"stationarity":1e-9,"global_relative_accuracy":1e-8},
              "global_accuracy":"known optimum absolute relative objective error; otherwise independently recomputed positive-definite complex dual difference; strict-gap case unknown",
              "environment":environment()}
    (args.output/"config.json").write_text(json.dumps(clean(metadata),indent=2),encoding="utf-8")
    # 单独触发动态编译；不以首个正式样本充当编译计时。
    cold,_=make_target(dict(family="gaussian",n=32,k=4),20261020)
    compilation={}
    for method in args.methods:
        with redirect_stdout(sys.stderr):
            compilation[method]=safe_call(method,cold,None)
    (args.output/"first_call.json").write_text(json.dumps(clean(compilation),indent=2),encoding="utf-8")
    if any(row.get("exception") for row in compilation.values()):
        raise RuntimeError("首次调用失败；先排除环境或编译问题，再开始正式比较。")
    with (args.output/"raw.jsonl").open("w",encoding="utf-8") as stream:
        for case_index,case in enumerate(cases):
            for seed_index,seed in enumerate(args.seeds):
                target,known=make_target(case,seed)
                digest=target_hash(target)
                for warmup in range(args.warmup):
                    for method in args.methods:
                        with redirect_stdout(sys.stderr):
                            result=safe_call(method,target,known)
                        row=dict(case=case,seed=seed,target_sha256=digest,method=method,
                                 case_ids=[f'{case["name"]}:seed{seed}:item{i}' for i in range(len(target))],
                                 phase="warmup",repeat=warmup,**result)
                        stream.write(json.dumps(clean(row),ensure_ascii=False)+"\n");stream.flush()
                for repeat in range(args.repeats):
                    shift=(case_index+seed_index+repeat)%len(args.methods)
                    order=args.methods[shift:]+args.methods[:shift]
                    for method in order:
                        with redirect_stdout(sys.stderr):
                            result=safe_call(method,target,known,history=args.history)
                        row=dict(case=case,seed=seed,target_sha256=digest,method=method,
                                 case_ids=[f'{case["name"]}:seed{seed}:item{i}' for i in range(len(target))],
                                 phase="measured",repeat=repeat,**result)
                        stream.write(json.dumps(clean(row),ensure_ascii=False)+"\n");stream.flush()
                        print(json.dumps({"case":case["name"],"seed":seed,"repeat":repeat,
                                          "method":method,"ms":result["milliseconds"],
                                          "accepted":result["accepted"],
                                          "quality_pass":result["quality_pass"],
                                          "exception":result.get("exception")}),flush=True)


if __name__=="__main__":
    main()
