"""论文实验、计时、验收及表图输出。"""
from contextlib import redirect_stdout
import csv
from importlib.metadata import version
import json
import math
import platform
import random
from statistics import fmean
import subprocess
import sys
import time

import cupy as cp
import numpy as np
import torch

from . import initialize_kernels, solve
from .baselines.common import SolverOptions, left_j
from .baselines.geometry import prepare_projection
from .baselines.gao_nonmonotone import solve_gao_nonmonotone
from .baselines.jensen_trustregions import JensenTrustRegionOptions, solve_jensen_trustregions


def clean(value):
    """将非有限数转换为 JSON null。"""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def make_inputs(case, case_index, protocol):
    seeds = [protocol["base_seed"] + 1000 * case_index + j
             for j in range(protocol["targets_per_case"])]
    targets = torch.empty(len(seeds), 2 * case["n"], 2 * case["k"], dtype=torch.float64)
    for item, seed in enumerate(seeds):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        targets[item] = torch.randn(2 * case["n"], 2 * case["k"], dtype=torch.float64,
                                   generator=generator) / math.sqrt(2 * case["n"])
    return targets, seeds


def environment():
    prop = torch.cuda.get_device_properties(0)
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                 "--format=csv,noheader"], capture_output=True, text=True, check=False)
        driver_text = driver.stdout.strip()
    except OSError:
        driver_text = None
    return dict(executable=sys.executable, python=sys.version, platform=platform.platform(),
                processor_model=platform.processor(), numpy=np.__version__, torch=torch.__version__,
                cupy=cp.__version__, matplotlib=version("matplotlib"), pytorch_cuda=torch.version.cuda,
                cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), nvidia_smi=driver_text,
                gpu=prop.name, memory_bytes=prop.total_memory, cpu_threads=torch.get_num_threads())


def call_method(method, target, protocol):
    if method == "dual":
        x, omega, iterations = solve(target)
        return x, omega, iterations.tolist(), ["returned"] * len(target), None
    xs, omegas, iterations, statuses, errors = [], [], [], [], []
    for t in target:
        if method == "dual_sequential":
            x, omega, iteration = solve(t.unsqueeze(0))
            xs.append(x[0])
            omegas.append(omega[0])
            iterations.append(int(iteration[0]))
            statuses.append("returned")
        else:
            if method == "gao_nonmonotone":
                options = SolverOptions(**protocol["gao_options"])
                result = solve_gao_nonmonotone(t.to("cuda"), options)
            elif method == "jensen_rtr2":
                options = JensenTrustRegionOptions(**protocol["jensen_options"])
                result = solve_jensen_trustregions(t.to("cuda"), options)
            else:
                raise ValueError(method)
            xs.append(result.x.cpu())
            omegas.append(result.omega.cpu() if result.omega is not None else
                          torch.full((t.shape[1], t.shape[1]), float("nan"), dtype=t.dtype))
            iterations.append(result.iterations)
            statuses.append(result.status)
            if result.stats.get("error"):
                errors.append(str(result.stats["error"]))
    return torch.stack(xs), torch.stack(omegas), iterations, statuses, "; ".join(errors) or None


def numeric_pass(record, thresholds):
    return all(record.get(key) is not None and math.isfinite(record[key])
               and abs(record[key]) <= threshold for key, threshold in thresholds.items())


@torch.no_grad()
def inspect_output(target, x):
    """在 CPU FP64 中独立复算三个验收指标。"""
    records = []
    m, n = target.shape[-1], target.shape[-2] // 2
    jm = left_j(torch.eye(m, dtype=torch.float64))
    identity = torch.eye(m, dtype=torch.complex128)
    for index in range(len(target)):
        record = dict(index=index, feasibility=None, stationarity=None, delta=None)
        try:
            t, y = target[index], x[index]
            if y.shape != t.shape or not bool(torch.isfinite(y).all()):
                raise ValueError("输出形状错误或包含非有限值")
            error, jy = y - t, left_j(y)
            objective = float(torch.sum(error * error))
            scale = max(1.0, float(torch.linalg.vector_norm(t)))
            record["feasibility"] = float(torch.linalg.vector_norm(y.T @ jy - jm) / math.sqrt(m))
            gradient, multiplier = prepare_projection(y).project_with_multiplier(2 * error)
            omega = multiplier / 2
            if not bool(torch.isfinite(gradient).all() and torch.isfinite(omega).all()):
                raise FloatingPointError("投影梯度或乘子包含非有限值")
            record["stationarity"] = float(torch.linalg.vector_norm(gradient) / (2 * scale))
            kc = identity + 1j * omega
            b = torch.complex(t[:n], t[n:])
            factor = torch.linalg.cholesky(kc)
            solved = torch.linalg.solve_triangular(factor, b.mH, upper=False)
            dual = float(torch.sum(t * t) - torch.sum(torch.abs(solved) ** 2)
                         - torch.sum(omega * jm))
            record["delta"] = abs((objective - dual) / max(1.0, abs(objective)))
        except Exception as exc:
            record["exception"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    return records


def timed_call(method, target, protocol):
    """同步计时 CPU 输入至 CPU 输出，随后验收。"""
    result = dict(milliseconds=None, exception=None, validation=[], status=[], iterations=[])
    output = None
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    try:
        output = call_method(method, target, protocol)
    except Exception as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
    torch.cuda.synchronize()
    result["milliseconds"] = (time.perf_counter_ns() - started) / 1e6
    if output is not None:
        x, _, iterations, statuses, error = output
        result.update(iterations=iterations, status=statuses, exception=error)
        result["validation"] = inspect_output(target, x)
    normal = (result["exception"] is None and len(result["status"]) == len(target)
              and all(status in ("returned", "converged") for status in result["status"]))
    result["success"] = (normal and len(result["validation"]) == len(target)
                         and all(numeric_pass(record, protocol["thresholds"])
                                 for record in result["validation"]))
    return result


def schedule(protocol, case_index, round_number):
    single, batch = protocol["single_methods"], protocol["batch_methods"]
    for repeat in range(protocol["single_warmups"]):
        for method in single:
            yield "warmup", "single", method, 0, repeat
    for position in range(protocol["targets_per_case"]):
        shift = (position + round_number - 1 + case_index) % len(single)
        for method in single[shift:] + single[:shift]:
            yield "measured", "single", method, position, None
    for method in batch:
        yield "warmup", "batch", method, None, 0
    for method in (batch[::-1] if (round_number - 1 + case_index) % 2 else batch):
        yield "measured", "batch", method, None, None


def run_case(output, protocol, name, round_number):
    case_index = next(i for i, case in enumerate(protocol["cases"]) if case["name"] == name)
    case = protocol["cases"][case_index]
    folder = output / f"round{round_number}" / name
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(protocol["cpu_threads"])
    if round_number == 1 and name == protocol["round_orders"][0][0]:
        write_json(output / "environment.json", environment())
    initialize_kernels()
    targets, seeds = make_inputs(case, case_index, protocol)
    shuffle_seed = protocol["shuffle_seed_base"] + 100 * round_number + case_index
    item_order = list(range(protocol["targets_per_case"]))
    random.Random(shuffle_seed).shuffle(item_order)
    ordered = targets[item_order].contiguous()
    del targets
    config = dict(case=case, round=round_number, seeds=seeds,
                  shuffle_seed=shuffle_seed, item_order=item_order)
    write_json(folder / "config.json", config)
    with (folder / "raw.jsonl").open("w", encoding="utf-8") as stream:
        for phase, experiment, method, position, warmup_index in schedule(protocol, case_index, round_number):
            item = None if position is None else item_order[position]
            target = ordered if position is None else ordered[position:position + 1]
            with redirect_stdout(sys.stderr):
                result = timed_call(method, target, protocol)
            row = dict(case=name, round=round_number, phase=phase, experiment=experiment,
                       method=method, item=item, warmup_index=warmup_index, **result)
            stream.write(json.dumps(clean(row), ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            if experiment == "batch" or position % 8 == 0 or not row["success"]:
                print(json.dumps(dict(case=name, round=round_number, phase=phase, experiment=experiment,
                                      item=item, method=method, milliseconds=row["milliseconds"],
                                      success=row["success"]), ensure_ascii=False), flush=True)


def complete_mean(values):
    if not values or any(value is None or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("正式计时缺失或无效")
    return fmean(values)


def summarize_results(output, protocol):
    """检查调用数量，按全部正式时间生成统计及表图。"""
    cases, formal, warmups = [], [], []
    for case in protocol["cases"]:
        rows = []
        for round_number in range(1, protocol["rounds"] + 1):
            folder = output / f"round{round_number}" / case["name"]
            records = [json.loads(line) for line in (folder / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
            rows.extend(records)
        measured = [row for row in rows if row["phase"] == "measured"]
        formal.extend(measured)
        warmups.extend(row for row in rows if row["phase"] == "warmup")
        item = dict(case=case)
        for group, methods, per_round in (("single", protocol["single_methods"], protocol["targets_per_case"]),
                                           ("batch", protocol["batch_methods"], 1)):
            item[group] = {}
            for method in methods:
                selected = [row for row in measured if row["experiment"] == group and row["method"] == method]
                by_round = [[row for row in selected if row["round"] == r]
                            for r in range(1, protocol["rounds"] + 1)]
                expected = per_round * protocol["rounds"]
                if len(selected) != expected or any(len(part) != per_round for part in by_round):
                    raise ValueError(f"正式记录不完整：{case['name']}/{group}/{method} 应有 {expected} 次调用")
                item[group][method] = dict(calls=len(selected), successful_calls=sum(row["success"] for row in selected),
                    all_calls_successful=all(row["success"] for row in selected),
                    milliseconds=dict(mean=complete_mean([row["milliseconds"] for row in selected])),
                    round_mean_ms=[complete_mean([row["milliseconds"] for row in part]) for part in by_round])
            item[group]["speedups"] = {}
            for numerator in (methods[1:] if group == "single" else methods[:1]):
                upper, lower = item[group][numerator], item[group]["dual"]
                mean_upper, mean_lower = upper["milliseconds"]["mean"], lower["milliseconds"]["mean"]
                item[group]["speedups"][numerator] = (mean_upper / mean_lower if upper["all_calls_successful"]
                    and lower["all_calls_successful"] and mean_lower > 0 else None)
        cases.append(item)
    validations = [value for row in formal for value in row["validation"]]
    accuracy = dict(outputs=len(validations))
    for key in protocol["thresholds"]:
        accuracy["max_" + key] = max((value[key] for value in validations
            if value.get(key) is not None and math.isfinite(value[key])), default=None)
    all_rows = formal + warmups
    summary = dict(schema="paper-summary-v1", complete=True, cases=cases, accuracy=accuracy,
        counts=dict(formal_calls=len(formal), formal_successes=sum(row["success"] for row in formal),
                    warmup_calls=len(warmups), warmup_successes=sum(row["success"] for row in warmups)),
        all_calls_successful=all(row["success"] for row in all_rows),
        failure_records=[row for row in all_rows if not row["success"]])
    try:
        summary["tables"] = write_tables(cases, output)
        summary["figures"] = write_comparison_figure(cases, output)
    finally:
        write_json(output / "summary.json", summary)
    return summary


def write_tables(cases, output):
    tables = output / "tables"
    tables.mkdir(exist_ok=True)
    groups = (
        ("single", ("dual", "gao_nonmonotone", "jensen_rtr2"), ("Dual", "RSD", "R--TR2"), 1, 1),
        ("batch", ("dual_sequential", "dual"), ("Sequential", "Batched"), 1000, 3),
    )

    def formatted(value, places):
        return "--" if value is None else f"{value:.{places}f}"

    files = []
    for group, methods, labels, divisor, places in groups:
        ratios = methods[1:] if group == "single" else methods[:1]
        ratio_labels = [label + "/Dual" for label in labels[1:]] if group == "single" else ["Speedup"]
        header, rows = ["n", "k", *labels, *ratio_labels, "Successes"], []
        for item in cases:
            times = [item[group][method]["milliseconds"]["mean"] for method in methods]
            rows.append([str(item["case"]["n"]), str(item["case"]["k"]),
                         *(formatted(value / divisor if value is not None else None, places) for value in times),
                         *(formatted(item[group]["speedups"][method], 2) for method in ratios),
                         "/".join(str(item[group][method]["successful_calls"]) for method in methods)])
        csv_path, tex_path = tables / f"{group}.csv", tables / f"{group}.tex"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows([header, *rows])
        tex = [r"\begin{tabular}{" + "r" * len(header) + "}", r"\toprule",
               " & ".join(header) + r" \\", r"\midrule"]
        tex.extend(" & ".join(row) + r" \\" for row in rows)
        tex.extend([r"\bottomrule", r"\end{tabular}"])
        tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
        files.extend(path.relative_to(output).as_posix() for path in (csv_path, tex_path))
    return files


def write_comparison_figure(cases, output):
    groups = (
        ("single", ("dual", "gao_nonmonotone", "jensen_rtr2"), ("Dual", "RSD", "R-TR2"),
         1.0, (18, 2000), (20, 50, 100, 200, 500, 1000), "Single-target time (ms)"),
        ("batch", ("dual_sequential", "dual"), ("Sequential", "Batched"),
         0.001, (0.045, 3.5), (0.05, 0.1, 0.2, 0.5, 1, 2), "Total time for 64 targets (s)"),
    )
    ranges = []
    for group, methods, labels, factor, limits, ticks, ylabel in groups:
        values = [value for item in cases for method in methods
                  for value in [item[group][method]["milliseconds"]["mean"],
                                *item[group][method]["round_mean_ms"]]]
        if not values or any(value is None or not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"{group} 比较图需要完整且为正的耗时")
        low, high = min(values) * factor, max(values) * factor
        ranges.append(((min(limits[0], low * 0.8), max(limits[1], high * 1.2)),
                       ticks if low >= limits[0] and high <= limits[1] else None))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    styles = {
        "dual": dict(color="#1f4e79", marker="o", linestyle="-"),
        "gao_nonmonotone": dict(color="#686868", marker="s", linestyle="--"),
        "jensen_rtr2": dict(color="#9c4a20", marker="^", linestyle="-."),
    }
    styles["dual_sequential"] = styles["gao_nonmonotone"]
    rc = {
        "font.family": "DejaVu Serif", "font.size": 8.3,
        "axes.labelsize": 8.3, "axes.titlesize": 8.6,
        "legend.fontsize": 8.3, "xtick.labelsize": 8.1, "ytick.labelsize": 8.1,
        "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42,
        "axes.linewidth": 0.65, "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
    }
    destination = output / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    pdf, png = destination / "comparison_en.pdf", destination / "comparison_en.png"
    with plt.rc_context(rc):
        figure, axes = plt.subplots(2, 2, figsize=(5.125, 4.2), sharex=True, sharey="row")
        try:
            figure.subplots_adjust(left=0.135, right=0.985, bottom=0.11,
                                   top=0.855, hspace=0.58, wspace=0.16)
            for row_index, (group, methods, labels, factor, _, _, ylabel) in enumerate(groups):
                limits, ticks = ranges[row_index]
                for col, n in enumerate((2048, 4096)):
                    selected = sorted((item for item in cases if item["case"]["n"] == n),
                                      key=lambda item: item["case"]["k"])
                    xs, axis = [item["case"]["k"] for item in selected], axes[row_index, col]
                    for method in methods:
                        style = styles[method]
                        for run, offset in enumerate((-1.0, 0.0, 1.0)):
                            axis.plot([x + offset for x in xs],
                                      [item[group][method]["round_mean_ms"][run] * factor for item in selected],
                                      color=style["color"], marker=style["marker"], linestyle="None",
                                      markersize=2.8, markerfacecolor="white", markeredgewidth=0.65, zorder=3)
                        axis.plot(xs, [item[group][method]["milliseconds"]["mean"] * factor for item in selected],
                                  **style, linewidth=1.15, markersize=4.3, markeredgewidth=0.65, zorder=4)
                    axis.set(yscale="log", xlim=(12, 52), xticks=(16, 32, 48), ylim=limits)
                    axis.grid(axis="y", which="major", color="#d8d8d8", linewidth=0.55)
                    axis.minorticks_off()
                    axis.tick_params(axis="both", length=3, pad=2.5)
                    axis.set_title(rf"({'abcd'[row_index * 2 + col]}) $n={n}$", pad=6.5)
                    if ticks is not None:
                        axis.yaxis.set_major_locator(FixedLocator(ticks))
                    axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
                    axis.yaxis.set_minor_formatter(NullFormatter())
                    if row_index == 1:
                        axis.set_xlabel(r"$k$", labelpad=2)
                axes[row_index, 0].set_ylabel(ylabel, labelpad=5)
                handles = [Line2D([], [], **styles[method], linewidth=1.2, markersize=4.3,
                                  markeredgewidth=0.7, label=label) for method, label in zip(methods, labels)]
                location, y, spacing = (("upper center", 0.998, 1.4), ("center", 0.485, 2.0))[row_index]
                figure.legend(handles=handles, loc=location, bbox_to_anchor=(0.56, y), ncol=len(methods),
                              frameon=False, columnspacing=spacing, handlelength=2.0, handletextpad=0.55)
            figure.savefig(pdf, format="pdf", metadata={"CreationDate": None, "ModDate": None})
            figure.savefig(png, format="png", dpi=240)
        finally:
            plt.close(figure)
    return [path.relative_to(output).as_posix() for path in (pdf, png)]
