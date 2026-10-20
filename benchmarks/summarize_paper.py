"""核验并汇总论文正式实验；仅使用 CPU，四组完整后才生成交付数据。

默认读取 results/paper_20261020/{scaling,critical,boundary,batch}。
先对每个种子取重复时间的中位数，再对种子中位数取中位数；失败调用
仍计入时间和计数。只有全部配对重复均通过质量验收，才报告配对时间比。
边界组仅提供诊断，不作速度比较。原始 JSON 文件逐字节归档。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SUITES = ("scaling", "critical", "boundary", "batch")
METHODS = ("dual", "rgd", "rtr")
LABELS = {"dual": "Dual", "rgd": "RGD", "rtr": "RTR"}
NORMAL_STATUSES = {"returned", "converged"}
REQUIRED_FILES = ("config.json", "raw.jsonl", "first_call.json")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SUMMARY_FIELDS = (
    "suite", "case", "family", "n", "k", "batch", "method", "seed_count",
    "repeats", "calls", "time_median_ms", "seed_median_min_ms",
    "seed_median_max_ms", "quality_pass_calls", "first_order_pass_calls",
    "exception_calls", "abnormal_status_calls", "items_expected",
    "items_observed", "items_unavailable", "quality_pass_items",
    "first_order_pass_items", "global_accuracy_unknown_items",
    "abnormal_status_items", "validation_failure_items", "iteration_min",
    "iteration_max", "feasibility_max", "stationarity_max",
    "absolute_relative_objective_error_max",
    "absolute_relative_primal_dual_difference_max", "relative_solution_error_max",
    "lambda_min_k_min", "peak_torch_allocated_bytes_max", "paired_ratio",
    "paired_ratio_min", "paired_ratio_max", "paired_ratio_eligible_seeds",
    "paired_ratio_reason", "amortized_ms_per_item",
    "quality_verified_items_per_second",
)
BOUNDARY_FIELDS = (
    "case", "family", "n", "k", "mu", "seed", "repeat", "method", "item_index",
    "target_sha256", "status", "exception", "reason", "accepted", "quality_pass",
    "global_accuracy_pass", "objective", "known_optimum", "relative_objective_error",
    "relative_solution_error", "lambda_min_k", "feasibility", "stationarity",
    "computed_dual_value", "computed_relative_primal_dual_difference",
    "known_dual_value", "difference_from_known_dual", "iterations",
)


class IncompleteResults(ValueError):
    """结果不完整或追溯校验失败。"""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(data, location):
    try:
        return json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IncompleteResults(f"{location}: JSON 无法完整读取：{exc}") from exc


def require(condition, message):
    if not condition:
        raise IncompleteResults(message)


def load_suite(source_root, suite):
    directory = source_root / suite
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    require(not missing, f"{suite}: 缺少 {', '.join(missing)}")
    snapshots = {
        p.name: p.read_bytes() for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix in {".json", ".jsonl"}
    }
    config = read_json(snapshots["config.json"], f"{suite}/config.json")
    require(config.get("suite") == suite, f"{suite}: config 中 suite 不匹配")
    cases, seeds, methods = config.get("cases", []), config.get("seeds", []), config.get("methods", [])
    repeats, warmup = config.get("repeats"), config.get("warmup")
    require(cases and all(isinstance(c, dict) and c.get("name") for c in cases),
            f"{suite}: cases 为空或缺少名称")
    names = [case["name"] for case in cases]
    require(len(set(names)) == len(names), f"{suite}: case 名称重复")
    require(seeds and len(set(seeds)) == len(seeds), f"{suite}: seeds 为空或重复")
    require(len(methods) == 3 and set(methods) == set(METHODS),
            f"{suite}: 正式比较应完整包含 dual、rgd、rtr")
    require(isinstance(repeats, int) and not isinstance(repeats, bool) and repeats > 0,
            f"{suite}: repeats 必须为正整数")
    require(isinstance(warmup, int) and not isinstance(warmup, bool) and warmup >= 0,
            f"{suite}: warmup 必须为非负整数")
    case_map = {case["name"]: case for case in cases}
    expected = {
        (phase, name, seed, repeat, method)
        for phase, count in (("measured", repeats), ("warmup", warmup))
        for name, seed, repeat, method in itertools.product(names, seeds, range(count), methods)
    }
    first_call = read_json(snapshots["first_call.json"], f"{suite}/first_call.json")
    require(set(first_call) == set(methods), f"{suite}: first_call 方法不完整")
    require(not any(row.get("exception") for row in first_call.values()),
            f"{suite}: first_call 存在异常，不能视为完成正式预热")
    rows, seen, target_hashes = [], set(), {}
    for line_number, line in enumerate(snapshots["raw.jsonl"].splitlines(), 1):
        require(bool(line.strip()), f"{suite}/raw.jsonl:{line_number}: 空行")
        row = read_json(line, f"{suite}/raw.jsonl:{line_number}")
        name = row.get("case", {}).get("name")
        key = (row.get("phase"), name, row.get("seed"), row.get("repeat"), row.get("method"))
        require(key in expected, f"{suite}/raw.jsonl:{line_number}: 非预期样本 {key}")
        require(key not in seen, f"{suite}/raw.jsonl:{line_number}: 重复样本 {key}")
        require(row["case"] == case_map[name], f"{suite}: {key} 的 case 参数不同于 config")
        seen.add(key)
        digest = row.get("target_sha256", "")
        require(isinstance(digest, str) and SHA256.fullmatch(digest), f"{suite}: {key} 无有效目标哈希")
        target_key = (name, row["seed"])
        require(target_hashes.setdefault(target_key, digest) == digest,
                f"{suite}: {target_key} 在方法、重复或预热之间使用了不同目标")
        batch = row["case"].get("batch", 1)
        expected_ids = [f"{name}:seed{row['seed']}:item{i}" for i in range(batch)]
        require(row.get("case_ids") == expected_ids, f"{suite}: {key} 的逐题标识不完整")
        require(finite_number(row.get("milliseconds")) and row["milliseconds"] >= 0,
                f"{suite}: {key} 缺少有限的非负实际耗时")
        require(isinstance(row.get("accepted"), bool) and isinstance(row.get("quality_pass"), bool),
                f"{suite}: {key} 缺少分开的验收布尔值")
        if row.get("exception"):
            require(not row["accepted"] and not row["quality_pass"],
                    f"{suite}: {key} 异常样本却标为通过")
        else:
            for field in ("status", "iterations", "validation"):
                require(isinstance(row.get(field), list) and len(row[field]) == batch,
                        f"{suite}: {key} 的 {field} 未逐题保存")
            require([v.get("index") for v in row["validation"]] == list(range(batch)),
                    f"{suite}: {key} 的后验题号不完整")
            for item in row["validation"]:
                require(isinstance(item.get("accepted"), bool)
                        and isinstance(item.get("quality_pass"), bool),
                        f"{suite}: {key} 缺少逐题验收")
                require(item.get("global_accuracy_pass") in (True, False, None),
                        f"{suite}: {key} 的全局验收字段无效")
                require(item["quality_pass"] == (
                    item["accepted"] and item.get("global_accuracy_pass") is True),
                    f"{suite}: {key} 的逐题综合验收不一致")
                if row["case"]["family"] == "strict_gap":
                    require(item.get("global_accuracy_pass") is None,
                            f"{suite}: 严格间隙算例的全局精度必须保留为未知")
            normal = all(status in NORMAL_STATUSES for status in row["status"])
            require(row["accepted"] == (normal and all(v["accepted"] for v in row["validation"])),
                    f"{suite}: {key} 的一阶验收汇总不一致")
            require(row["quality_pass"] == (normal and all(v["quality_pass"] for v in row["validation"])),
                    f"{suite}: {key} 的质量验收汇总不一致")
        rows.append(row)
    missing_keys = expected - seen
    require(not missing_keys,
            f"{suite}: 正式/预热记录尚不完整，缺少 {len(missing_keys)} 条；"
            f"示例 {sorted(missing_keys, key=str)[:5]}")
    measured = [row for row in rows if row["phase"] == "measured"]
    return dict(suite=suite, config=config, snapshots=snapshots, rows=measured,
                target_hashes=target_hashes, warmup_rows=len(rows)-len(measured),
                source_directory=str(directory.resolve()))


def verify_targets_and_environment(data, source_root):
    """在 CPU 单线程重建目标；不导入任何 GPU 求解器模块。"""
    # 先核验源码，再执行目标生成；公开归档可证明仅 CRLF/LF 的字节差异。
    source_manifest(data, source_root)
    import torch
    from paper_cases import make_target, target_hash

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for suite_data in data:
            config = suite_data["config"]
            for case, seed in itertools.product(config["cases"], config["seeds"]):
                target, _ = make_target(case, seed)
                actual = target_hash(target)
                require(actual == suite_data["target_hashes"][(case["name"], seed)],
                        f"{suite_data['suite']}: CPU 重建目标 {case['name']}, seed={seed} 哈希不同")
                del target
    finally:
        torch.set_num_threads(previous_threads)
    first_env = data[0]["config"]["environment"]
    fields = ("executable", "python", "torch", "numpy", "cupy", "gpu",
              "nvidia_smi", "cpu_threads", "pytorch_cuda", "cuda_runtime")
    core_hashes = {
        name: digest for name, digest in first_env["source_sha256"].items()
        if name.startswith("src/") or name in {
            "benchmarks/run_paper.py", "benchmarks/paper_cases.py", "benchmarks/run_benchmark.py"}
    }
    for suite_data in data[1:]:
        env = suite_data["config"]["environment"]
        for field in fields:
            require(env.get(field) == first_env.get(field),
                    f"{suite_data['suite']}: 实验环境字段 {field} 与其他正式组不同")
        for name, digest in core_hashes.items():
            require(env["source_sha256"].get(name) == digest,
                    f"{suite_data['suite']}: 求解/验收源码 {name} 与其他正式组不同")
    return torch.__version__


def metric(values, key, *, absolute=False, minimum=False):
    numbers = [abs(v[key]) if absolute else v[key] for v in values if finite_number(v.get(key))]
    return (min(numbers) if minimum else max(numbers)) if numbers else None


def aggregate_method(suite, case, method, rows, config):
    by_seed = defaultdict(list)
    validations, iterations, statuses = [], [], []
    normal_quality_items = normal_first_order_items = 0
    for row in rows:
        by_seed[row["seed"]].append(row)
        items = row.get("validation", [])
        validations.extend(items)
        iterations.extend(v for v in row.get("iterations", []) if finite_number(v))
        statuses.extend(row.get("status", []))
        for status, item in zip(row.get("status", []), items):
            normal_quality_items += status in NORMAL_STATUSES and item["quality_pass"]
            normal_first_order_items += status in NORMAL_STATUSES and item["accepted"]
    seed_medians = []
    for seed in config["seeds"]:
        samples = by_seed[seed]
        times = [row["milliseconds"] for row in samples]
        seed_medians.append(dict(
            suite=suite, case=case["name"], method=method, seed=seed,
            milliseconds=statistics.median(times), repeat_min_ms=min(times),
            repeat_max_ms=max(times), repeats=len(times),
            all_quality_pass=all(row["quality_pass"] for row in samples),
            quality_pass_calls=sum(row["quality_pass"] for row in samples),
            first_order_pass_calls=sum(row["accepted"] for row in samples),
            exception_calls=sum(bool(row.get("exception")) for row in samples),
        ))
    times = [item["milliseconds"] for item in seed_medians]
    batch = case.get("batch", 1)
    time_median = statistics.median(times)
    all_quality = all(row["quality_pass"] for row in rows)
    result = dict(
        suite=suite, case=case["name"], family=case["family"],
        n=case["n"], k=case["k"], batch=batch, method=method,
        seed_count=len(config["seeds"]), repeats=config["repeats"], calls=len(rows),
        time_median_ms=time_median, seed_median_min_ms=min(times),
        seed_median_max_ms=max(times),
        quality_pass_calls=sum(row["quality_pass"] for row in rows),
        first_order_pass_calls=sum(row["accepted"] for row in rows),
        exception_calls=sum(bool(row.get("exception")) for row in rows),
        abnormal_status_calls=sum(any(s not in NORMAL_STATUSES for s in row.get("status", []))
                                  for row in rows),
        items_expected=len(rows)*batch, items_observed=len(validations),
        items_unavailable=len(rows)*batch-len(validations),
        quality_pass_items=normal_quality_items, first_order_pass_items=normal_first_order_items,
        global_accuracy_unknown_items=sum(v.get("global_accuracy_pass") is None for v in validations),
        abnormal_status_items=sum(s not in NORMAL_STATUSES for s in statuses),
        validation_failure_items=sum(bool(v.get("reason") or v.get("exception")) for v in validations),
        iteration_min=min(iterations) if iterations else None,
        iteration_max=max(iterations) if iterations else None,
        feasibility_max=metric(validations, "feasibility"),
        stationarity_max=metric(validations, "stationarity"),
        absolute_relative_objective_error_max=metric(validations, "relative_objective_error", absolute=True),
        absolute_relative_primal_dual_difference_max=metric(
            validations, "computed_relative_primal_dual_difference", absolute=True),
        relative_solution_error_max=metric(validations, "relative_solution_error"),
        lambda_min_k_min=metric(validations, "lambda_min_k", minimum=True),
        peak_torch_allocated_bytes_max=metric(rows, "peak_torch_allocated_bytes"),
        status_counts=dict(sorted(Counter(statuses).items())),
        exception_counts=dict(sorted(Counter(
            row["exception"] for row in rows if row.get("exception")).items())),
        metric_observations={key: sum(finite_number(v.get(key)) for v in validations)
                             for key in ("feasibility", "stationarity", "relative_objective_error",
                                         "computed_relative_primal_dual_difference",
                                         "relative_solution_error", "lambda_min_k")},
        seed_medians=seed_medians, paired_ratio=None, paired_ratio_min=None,
        paired_ratio_max=None, paired_ratio_eligible_seeds=0, paired_ratio_reason=None,
        amortized_ms_per_item=time_median/batch,
        quality_verified_items_per_second=1000*batch/time_median
        if all_quality and time_median > 0 and suite != "boundary" else None,
    )
    return result


def paired_ratios(methods, suite):
    records = []
    dual_seeds = {row["seed"]: row for row in methods["dual"]["seed_medians"]}
    for method in ("rgd", "rtr"):
        ratios = []
        for baseline in methods[method]["seed_medians"]:
            dual = dual_seeds[baseline["seed"]]
            eligible = (suite != "boundary" and baseline["all_quality_pass"]
                        and dual["all_quality_pass"] and dual["milliseconds"] > 0)
            reason = None if eligible else (
                "boundary_diagnostics_only" if suite == "boundary" else
                "paired_quality_not_passed" if not (
                    baseline["all_quality_pass"] and dual["all_quality_pass"])
                else "zero_dual_time")
            ratio = baseline["milliseconds"]/dual["milliseconds"] if eligible else None
            records.append(dict(
                suite=suite, case=baseline["case"], method=method, seed=baseline["seed"],
                baseline_median_ms=baseline["milliseconds"], dual_median_ms=dual["milliseconds"],
                paired_ratio=ratio, eligible=eligible, reason=reason))
            if eligible:
                ratios.append(ratio)
        result = methods[method]
        result["paired_ratio_eligible_seeds"] = len(ratios)
        if len(ratios) == len(dual_seeds):
            result.update(paired_ratio=statistics.median(ratios),
                          paired_ratio_min=min(ratios), paired_ratio_max=max(ratios))
        else:
            result["paired_ratio_reason"] = (
                "boundary_diagnostics_only" if suite == "boundary" else
                "at_least_one_seed_has_unverified_or_failed_calls")
    return records


def boundary_items(rows):
    result = []
    for row in rows:
        case = row["case"]
        items = row.get("validation", [])
        for item in items or [None]:
            index = None if item is None else item["index"]
            record = {
                "case": case["name"], "family": case["family"], "n": case["n"], "k": case["k"],
                "mu": case.get("mu"), "seed": row["seed"], "repeat": row["repeat"],
                "method": row["method"], "item_index": index,
                "target_sha256": row["target_sha256"],
                "status": None if index is None else row["status"][index],
                "exception": row.get("exception"),
                "iterations": None if index is None else row["iterations"][index],
            }
            if item is not None:
                record.update({key: item[key] for key in BOUNDARY_FIELDS if key in item})
            else:
                record.update(accepted=False, quality_pass=False,
                              global_accuracy_pass=None if case["family"] == "strict_gap" else False)
            result.append(record)
    return result


def build_summary(data):
    suites, flat, seeds, ratios, boundaries = {}, [], [], [], []
    for suite_data in data:
        suite, config = suite_data["suite"], suite_data["config"]
        grouped = defaultdict(list)
        for row in suite_data["rows"]:
            grouped[(row["case"]["name"], row["method"])].append(row)
        cases = []
        for case in config["cases"]:
            methods = {
                method: aggregate_method(suite, case, method,
                                         grouped[(case["name"], method)], config)
                for method in METHODS
            }
            ratios.extend(paired_ratios(methods, suite))
            for method in METHODS:
                flat.append(methods[method])
                seeds.extend(methods[method]["seed_medians"])
            cases.append(dict(parameters=case, methods=methods))
        suites[suite] = dict(
            measured_calls=len(suite_data["rows"]), warmup_calls_excluded=suite_data["warmup_rows"],
            seeds=config["seeds"], repeats=config["repeats"], warmup=config["warmup"], cases=cases,
            config_file=f"{suite}/config.json", raw_file=f"{suite}/raw.jsonl",
            first_call_file=f"{suite}/first_call.json",
            target_sha256=[dict(case=name, seed=seed, sha256=digest)
                           for (name, seed), digest in suite_data["target_hashes"].items()],
        )
        if suite == "boundary":
            boundaries = boundary_items(suite_data["rows"])
    return suites, flat, seeds, ratios, boundaries


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validated_source_record(name, runtime, raw, archived=None):
    """保留运行时哈希；只允许可由归档清单核对的 CRLF/LF 差异。"""
    require(bool(runtime), f"{name}: 缺少原始运行哈希")
    recorded = next(iter(runtime.values()))
    require(isinstance(recorded, str) and SHA256.fullmatch(recorded)
            and all(digest == recorded for digest in runtime.values()),
            f"{name}: 各正式组记录的原始运行哈希不同或无效")
    current, normalized = sha256(raw), sha256(raw.replace(b"\r\n", b"\n"))
    entry = archived if archived and archived.get("sha256") == recorded else None
    if current != recorded:
        require(entry is not None,
                f"{name}: 当前字节哈希与运行哈希不同，且 source-root 下没有匹配原始哈希的源码清单条目")
    if entry is not None:
        require(entry.get("path") == name and entry.get("normalized_lf_sha256") == normalized,
                f"{name}: 当前 LF 规范化哈希与归档源码清单不同")
        archived_runtime = entry.get("runtime_sha256", {})
        require(isinstance(archived_runtime, dict)
                and all(digest == recorded for digest in archived_runtime.values()),
                f"{name}: 归档源码清单中的运行哈希与 config 不一致")
        # sha256、normalized_lf_sha256 和原始行尾统计均来自原始运行记录；
        # LF checkout 的当前字节信息单列，不改写成所谓“原始”哈希。
        result = dict(entry)
    else:
        crlf = raw.count(b"\r\n")
        result = dict(
            path=name, sha256=recorded, normalized_lf_sha256=normalized,
            crlf_newlines=crlf, lf_only_newlines=raw.count(b"\n")-crlf,
            line_count=len(raw.splitlines()))
    current_crlf = raw.count(b"\r\n")
    result.update(
        runtime_sha256=runtime, runtime_hash_matches_current=current == recorded,
        validation="raw_sha256" if current == recorded else "archived_normalized_lf_sha256",
        current_working_copy=dict(
            sha256=current, normalized_lf_sha256=normalized, crlf_newlines=current_crlf,
            lf_only_newlines=raw.count(b"\n")-current_crlf, line_count=len(raw.splitlines())))
    return result


def source_manifest(data, source_root):
    """记录原始运行与当前工作副本；必要时读取指定来源内的公开源码清单。"""
    first = data[0]["config"]["environment"]["source_sha256"]
    names = sorted(
        name for name in first
        if name.startswith("src/") or name in {
            "benchmarks/run_paper.py", "benchmarks/paper_cases.py", "benchmarks/run_benchmark.py"}
    )
    archived_entries = {}
    archived_path = Path(source_root)/"summary"/"source_manifest.json"
    if archived_path.is_file():
        archived = read_json(archived_path.read_bytes(), str(archived_path))
        require(isinstance(archived.get("files"), list), f"{archived_path}: 缺少 files 列表")
        for entry in archived["files"]:
            require(isinstance(entry, dict) and isinstance(entry.get("path"), str),
                    f"{archived_path}: 源码清单条目无效")
            require(entry["path"] not in archived_entries,
                    f"{archived_path}: 重复源码条目 {entry['path']}")
            archived_entries[entry["path"]] = entry
    files = []
    for name in names:
        raw = (ROOT/name).read_bytes()
        runtime = {
            item["suite"]: item["config"]["environment"]["source_sha256"].get(name)
            for item in data
        }
        files.append(validated_source_record(name, runtime, raw, archived_entries.get(name)))
    return dict(
        normalization="CRLF bytes replaced by LF bytes; all other bytes retained",
        git_attributes_file=".gitattributes",
        note="Runtime hashes identify actual working-copy bytes. Git's LF normalization can "
             "give a different hash for a CRLF working copy without changing source text. "
             "A raw-hash mismatch is accepted only when the specified source-root manifest "
             "matches the config's original raw hash and the current LF-normalized hash. "
             "Original hashes and newline counts are retained; current bytes are recorded separately. "
             "summarize_paper.py was not executed by the GPU benchmark and is excluded.",
        files=files)


def tex_number(value, *, scientific=False, digits=3):
    if value is None:
        return r"\text{--}"
    if scientific or (value != 0 and (abs(value) < .01 or abs(value) >= 100000)):
        mantissa, exponent = f"{value:.2e}".split("e")
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"
    return f"{value:.{digits}g}"


def tex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(char, char) for char in str(value))


def write_tables(directory, suites, boundaries):
    directory.mkdir(parents=True, exist_ok=True)
    for suite in ("scaling", "critical", "batch"):
        timing = [
            "% Columns: n, k, B, Dual ms [seed-median min,max], RGD ms [min,max],",
            "% RTR ms [min,max], paired RGD/Dual, paired RTR/Dual.",
            "% All measured calls enter times; unavailable quality-verified ratios are --.",
        ]
        quality = [
            "% Columns: n, k, B, method, Q/N calls, A/N calls, exceptions, abnormal-status calls,",
            "% iteration range, max feasibility, max stationarity, max absolute relative dual difference.",
            "% Maxima include all available measured outputs; missing diagnostics are --.",
        ]
        for case_data in suites[suite]["cases"]:
            case, methods = case_data["parameters"], case_data["methods"]
            cells = [str(case["n"]), str(case["k"]), str(case.get("batch", 1))]
            for method in METHODS:
                row = methods[method]
                time = tex_number(row["time_median_ms"])
                low, high = tex_number(row["seed_median_min_ms"]), tex_number(row["seed_median_max_ms"])
                cells.append(rf"${time}\,[{low},{high}]$")
            cells.extend(f"${tex_number(methods[method]['paired_ratio'])}$" for method in ("rgd", "rtr"))
            timing.append(" & ".join(cells) + r" \\")
            for method in METHODS:
                row = methods[method]
                iteration = (r"\text{--}" if row["iteration_min"] is None else
                             f"{row['iteration_min']}-{row['iteration_max']}")
                cells = [
                    str(case["n"]), str(case["k"]), str(case.get("batch", 1)), LABELS[method],
                    f"{row['quality_pass_calls']}/{row['calls']}",
                    f"{row['first_order_pass_calls']}/{row['calls']}",
                    str(row["exception_calls"]), str(row["abnormal_status_calls"]), f"${iteration}$",
                    f"${tex_number(row['feasibility_max'], scientific=True)}$",
                    f"${tex_number(row['stationarity_max'], scientific=True)}$",
                    f"${tex_number(row['absolute_relative_primal_dual_difference_max'], scientific=True)}$",
                ]
                quality.append(" & ".join(cells) + r" \\")
        (directory/f"{suite}_timing_rows.tex").write_text("\n".join(timing)+"\n", encoding="utf-8")
        (directory/f"{suite}_quality_rows.tex").write_text("\n".join(quality)+"\n", encoding="utf-8")
    lines = [
        "% Columns: case, seed, repeat, method, f, signed relative known-optimum difference,",
        "% relative solution error, lambda_min(K_proj), status, global-accuracy pass.",
        "% Each row is an actual measured output; unknown global accuracy is ?.",
    ]
    case_order = {name: index for index, name in enumerate(dict.fromkeys(row["case"] for row in boundaries))}
    for row in sorted(boundaries, key=lambda row: (
            case_order[row["case"]], row["seed"], row["repeat"], METHODS.index(row["method"]))):
        case_label = (rf"$\mu={tex_number(row['mu'])}$" if row["family"] == "known"
                      else r"$T=0$" if row["family"] == "zero" else "Strict gap")
        global_pass = row.get("global_accuracy_pass")
        status = row.get("status") or ("exception" if row.get("exception") else "unavailable")
        cells = [
            case_label, str(row["seed"]), str(row["repeat"]), LABELS[row["method"]],
            f"${tex_number(row.get('objective'), digits=10)}$",
            f"${tex_number(row.get('relative_objective_error'), scientific=True)}$",
            f"${tex_number(row.get('relative_solution_error'), scientific=True)}$",
            f"${tex_number(row.get('lambda_min_k'), scientific=True)}$",
            tex_escape(status), "?" if global_pass is None else ("yes" if global_pass else "no"),
        ]
        lines.append(" & ".join(cells) + r" \\")
    (directory/"boundary_rows.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")


def plot_scaling(path, suite):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"dual": "#0072B2", "rgd": "#D55E00", "rtr": "#009E73"}
    markers, styles = {"dual": "o", "rgd": "s", "rtr": "^"}, {"dual": "-", "rgd": "--", "rtr": ":"}
    settings = {
        "font.family": "DejaVu Serif", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 10, "legend.fontsize": 8, "pdf.fonttype": 42,
        "mathtext.fontset": "dejavuserif", "axes.spines.top": False, "axes.spines.right": False,
    }
    with plt.rc_context(settings):
        figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=False)
        failure_present = False
        for axis, k in zip(axes, (8, 32)):
            cases = sorted((c for c in suite["cases"] if c["parameters"]["k"] == k),
                           key=lambda c: c["parameters"]["n"])
            require(bool(cases), f"scaling: 没有 k={k} 的实际数据，无法生成指定图")
            ns = [case["parameters"]["n"] for case in cases]
            for method in METHODS:
                rows = [case["methods"][method] for case in cases]
                center = [row["time_median_ms"] for row in rows]
                require(all(value > 0 for value in center), "对数时间图不能包含零耗时")
                lower = [max(0, row["time_median_ms"]-row["seed_median_min_ms"]) for row in rows]
                upper = [max(0, row["seed_median_max_ms"]-row["time_median_ms"]) for row in rows]
                axis.errorbar(ns, center, yerr=[lower, upper], color=colors[method],
                              linestyle=styles[method], linewidth=1.3, capsize=3,
                              marker=markers[method], markersize=4, label=LABELS[method])
                for n, value, row in zip(ns, center, rows):
                    if row["quality_pass_calls"] != row["calls"]:
                        failure_present = True
                        axis.plot(n, value, color=colors[method], marker=markers[method],
                                  markerfacecolor="white", markeredgewidth=1.3, markersize=6)
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xticks(ns, [str(n) for n in ns])
            axis.set_xlabel(r"$n$")
            axis.set_ylabel("CPU-to-CPU wall time (ms)")
            axis.set_title(rf"$k={k}$")
            axis.grid(axis="y", which="major", color="#dddddd", linewidth=.5)
        handles = [Line2D([], [], color=colors[m], marker=markers[m], linestyle=styles[m],
                          label=LABELS[m], markersize=4) for m in METHODS]
        figure.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
        if failure_present:
            figure.text(.5, .015, "Open markers: at least one call did not pass quality validation.",
                        ha="center", fontsize=7)
        figure.subplots_adjust(left=.105, right=.985, bottom=.20, top=.79, wspace=.34)
        figure.savefig(path, format="pdf", bbox_inches="tight",
                       metadata={"Title": "Single-instance FP64 scaling",
                                 "Creator": "benchmarks/summarize_paper.py",
                                 "CreationDate": None, "ModDate": None})
        plt.close(figure)
    return matplotlib.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT/"results"/"paper_20261020")
    parser.add_argument("--output-root", type=Path, default=ROOT/"paper_results"/"2026-10-20")
    parser.add_argument("--check-only", action="store_true", help="只核验四组完整性和目标，不写文件")
    parser.add_argument("--_verify-targets", dest="verify_targets_child", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    data, errors = [], []
    for suite in SUITES:
        try:
            data.append(load_suite(args.source_root, suite))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(str(exc))
    if errors:
        print("正式结果未完整通过校验，未生成任何汇总：\n- "+"\n- ".join(errors), file=sys.stderr)
        return 2
    try:
        if args.verify_targets_child:
            print(json.dumps({"torch_version": verify_targets_and_environment(data, args.source_root)}))
            return 0
        # 此 Windows 环境的 PyTorch 与绘图库携带不同 OpenMP 运行库。CPU 目标
        # 核验放在同一 Python 环境的独立进程，不使用允许重复 OpenMP 的不安全选项。
        verification = subprocess.run(
            [sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
             "--source-root", str(args.source_root.resolve()), "--_verify-targets"],
            capture_output=True, text=True, encoding="utf-8", check=False)
        require(verification.returncode == 0,
                "CPU 目标核验失败："+verification.stderr.strip())
        torch_version = json.loads(verification.stdout)["torch_version"]
        import numpy
        import matplotlib
        for suite_data in data:
            for name, snapshot in suite_data["snapshots"].items():
                require((args.source_root/suite_data["suite"]/name).read_bytes() == snapshot,
                        f"{suite_data['suite']}/{name}: 校验期间文件变化，请等待实验结束")
        suites, flat, seeds, ratios, boundaries = build_summary(data)
        if args.check_only:
            print(json.dumps({suite: suites[suite]["measured_calls"] for suite in SUITES}))
            print("四组记录、环境、源码和 CPU 重建目标哈希均通过校验；未写入文件。")
            return 0
        # 先检查全部归档目标，已有不同原始数据时不覆盖。
        for suite_data in data:
            for name, snapshot in suite_data["snapshots"].items():
                destination = args.output_root/suite_data["suite"]/name
                require(not destination.exists() or destination.read_bytes() == snapshot,
                        f"{destination}: 已有不同原始数据，拒绝覆盖")
        for suite_data in data:
            directory = args.output_root/suite_data["suite"]
            directory.mkdir(parents=True, exist_ok=True)
            for name, snapshot in suite_data["snapshots"].items():
                if not (directory/name).exists():
                    (directory/name).write_bytes(snapshot)
        (args.output_root/"figures").mkdir(exist_ok=True)
        plot_scaling(args.output_root/"figures"/"scaling.pdf", suites["scaling"])
        write_tables(args.output_root/"tables", suites, boundaries)
        write_csv(args.output_root/"summary.csv", flat, SUMMARY_FIELDS)
        write_csv(args.output_root/"seed_medians.csv", seeds, (
            "suite", "case", "method", "seed", "milliseconds", "repeat_min_ms", "repeat_max_ms",
            "repeats", "all_quality_pass", "quality_pass_calls", "first_order_pass_calls", "exception_calls"))
        write_csv(args.output_root/"paired_seed_ratios.csv", ratios, (
            "suite", "case", "method", "seed", "baseline_median_ms", "dual_median_ms",
            "paired_ratio", "eligible", "reason"))
        write_csv(args.output_root/"boundary.csv", boundaries, BOUNDARY_FIELDS)
        source_directory = args.output_root/"summary"
        source_directory.mkdir(exist_ok=True)
        source_manifest_bytes = (
            json.dumps(source_manifest(data, args.source_root), indent=2, allow_nan=False)+"\n").encode("utf-8")
        (source_directory/"source_manifest.json").write_bytes(source_manifest_bytes)
        summary = dict(
            schema_version=1, generated_at_utc=datetime.now(timezone.utc).isoformat(),
            source_root=str(args.source_root.resolve()),
            runtime=dict(executable=sys.executable, python=sys.version, torch=torch_version,
                         numpy=numpy.__version__, matplotlib=matplotlib.__version__, device="cpu"),
            summarizer_sha256=sha256(Path(__file__).read_bytes()),
            source_manifest_file="summary/source_manifest.json",
            source_manifest_sha256=sha256(source_manifest_bytes),
            definitions={
                "time": "median across seeds of per-seed median across all measured repetitions; failures retained",
                "time_range": "minimum and maximum of the per-seed medians, not a confidence interval",
                "paired_ratio": "per-seed baseline median / dual median, then median across seeds; "
                                "reported only when every paired repetition passes quality validation",
                "quality": "normal termination and first-order/feasibility gates and global-accuracy gate",
                "first_order": "normal termination and first-order/feasibility gates",
                "counts": "calls and per-item observations include every measured row; no successful-subset filtering",
                "metric_maxima": "all finite available diagnostics; observation counts retained; "
                                 "relative objective and primal-dual differences use absolute values",
                "boundary": "no timing ratio; strict-gap global accuracy is unknown",
                "batch": "native dual batch versus sequential per-item GPU baselines; "
                         "quality-verified throughput only when every call passes",
                "excluded": ["warmup", "first_call"],
                "source_provenance": "config files are copied unchanged; current and cross-suite "
                                     "source checks cover src/*, run_paper.py, paper_cases.py, "
                                     "and run_benchmark.py. summarize_paper.py entries captured "
                                     "during the experiment describe analysis code that was not "
                                     "executed by the GPU benchmark; the actual analysis hash is "
                                     "recorded separately as summarizer_sha256. Raw-byte mismatches "
                                     "are accepted only through a source-root manifest entry matching "
                                     "the config's original raw hash and the current LF-normalized hash; "
                                     "original source hashes are never replaced by checkout hashes.",
            },
            source_files={suite_data["suite"]: {
                name: sha256(snapshot) for name, snapshot in suite_data["snapshots"].items()
            } for suite_data in data},
            suites=suites, boundary_items=boundaries, paired_seed_ratios=ratios,
        )
        (args.output_root/"summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        artifacts = {
            path.relative_to(args.output_root).as_posix(): sha256(path.read_bytes())
            for path in sorted(args.output_root.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        (args.output_root/"manifest.json").write_text(
            json.dumps({"sha256": artifacts}, indent=2)+"\n", encoding="utf-8")
        print(f"已汇总 {sum(suites[s]['measured_calls'] for s in SUITES)} 条正式调用："
              f"{args.output_root.resolve()}")
        return 0
    except ImportError as exc:
        print(f"默认环境缺少汇总依赖，未切换环境：{exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"汇总失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
