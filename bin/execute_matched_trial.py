#!/usr/bin/env python3
"""Execute one matched compiler-search trial for a scaled-down model-serving kernel."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels.model_serving_programs import program_cards


def utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head(folder: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(folder), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "N/A"


def import_runtime_modules() -> tuple[Any, Any]:
    import tvm
    from tvm import meta_schedule as meta_schedule

    return tvm, meta_schedule


def observed_records(run_folder: Path) -> int:
    total = 0
    for record_file in run_folder.rglob("*tuning*record*.json"):
        try:
            total += sum(1 for line in record_file.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
        except OSError:
            pass
    return total


def measure_runtime(module: Any, device: Any, buffers: Sequence[Any], number: int, repeat: int, min_repeat_ms: int) -> Dict[str, Any]:
    evaluator = module.time_evaluator(module.entry_name, device, number=number, repeat=repeat, min_repeat_ms=min_repeat_ms)
    sample_seconds = [float(value) for value in evaluator(*buffers).results]
    sample_ms = [value * 1000.0 for value in sample_seconds]
    return {
        "best_latency_ms": min(sample_ms),
        "median_latency_ms": statistics.median(sample_ms),
        "latency_samples_ms": sample_ms,
    }


def assemble_search_policy(args: argparse.Namespace) -> Any:
    from tvm.meta_schedule.search_strategy import MCTSSearchPyFull

    shared_options = dict(
        population_size=3,
        init_measured_ratio=0,
        init_min_unmeasured=3,
        max_fail_count=20,
        genetic_num_iters=3,
        genetic_mutate_prob=0.85,
        genetic_max_fail_count=2,
        trace_commit=True,
        mcts_num_threads=1,
        mcts_num_rollouts_per_expansion=1,
        use_llm=True,
        llm_budget=args.llm_budget,
        verbose=args.verbose,
    )
    if args.engine == "baseline":
        return MCTSSearchPyFull(**shared_options, llm_model_name=args.strong_model)
    return MCTSSearchPyFull(
        **shared_options,
        default_llm_model_name=args.strong_model,
        llm_bucket=[args.strong_model, args.light_model],
    )


def execute(args: argparse.Namespace) -> Dict[str, Any]:
    tvm, meta_schedule = import_runtime_modules()
    catalog = program_cards()
    card = catalog[args.kernel]
    target = "llvm --num-cores=1"
    device = tvm.cpu(0)
    module = card.builder(tvm)
    buffers = card.input_factory(tvm, device, args.seed)
    run_token = f"{args.engine}_{args.kernel}_seed{args.seed}_trials{args.trials}_llm{args.llm_budget}_{args.cost_model}"
    run_folder = PROJECT_ROOT / "measurements" / "work_dirs" / run_token
    run_folder.mkdir(parents=True, exist_ok=True)

    search_policy = assemble_search_policy(args)
    start = time.perf_counter()
    database = meta_schedule.tune_tir(
        mod=module,
        target=target,
        max_trials_global=args.trials,
        num_trials_per_iter=min(args.num_trials_per_iter, args.trials),
        work_dir=str(run_folder),
        strategy=search_policy,
        num_tuning_cores=args.num_tuning_cores,
        seed=args.seed,
        cost_model=args.cost_model,
    )
    tuning_time_sec = time.perf_counter() - start
    schedule = meta_schedule.tir_integration.compile_tir(database, module, target)
    built_module = tvm.build(schedule.mod, target=target)
    latency = measure_runtime(built_module, device, buffers, args.number, args.repeat, args.min_repeat_ms)
    source_root = Path(os.environ["BASELINE_TVM_HOME"] if args.engine == "baseline" else os.environ["COORDINATED_TVM_HOME"])

    return {
        "status": "ok",
        "run_id": run_token,
        "engine": args.engine,
        "method_name": "reasoning_compiler_single_model_mcts" if args.engine == "baseline" else "shared_tree_multi_model_search",
        "workload_name": args.kernel,
        "workload_type": card.family,
        "shape": card.shape,
        "target": target,
        "device": "cpu",
        "seed": args.seed,
        "trials": args.trials,
        "llm_budget": args.llm_budget,
        "num_trials_per_iter": args.num_trials_per_iter,
        "cost_model": args.cost_model,
        "strong_model": args.strong_model,
        "light_model": args.light_model if args.engine == "coordinated" else "N/A",
        "total_tuning_time_sec": tuning_time_sec,
        "num_measured_candidates": observed_records(run_folder),
        "number": args.number,
        "repeat": args.repeat,
        "min_repeat_ms": args.min_repeat_ms,
        "tvm_version": getattr(tvm, "__version__", "N/A"),
        "tvm_file": getattr(tvm, "__file__", "N/A"),
        "repo_path": str(source_root),
        "repo_commit": git_head(source_root),
        "api_key_presence_only": "yes" if os.environ.get("OPENAI_API_KEY") else "no",
        "timestamp_utc": utc_stamp(),
        "work_dir": str(run_folder),
        **latency,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["baseline", "coordinated"], required=True)
    parser.add_argument("--kernel", choices=sorted(program_cards()), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--llm-budget", type=int, default=4)
    parser.add_argument("--cost-model", default="xgb")
    parser.add_argument("--strong-model", default=os.environ.get("STRONG_MODEL_NAME", "gpt-5.4"))
    parser.add_argument("--light-model", default=os.environ.get("LIGHT_MODEL_NAME", "gpt-5.4-mini"))
    parser.add_argument("--num-trials-per-iter", type=int, default=8)
    parser.add_argument("--num-tuning-cores", type=int, default=1)
    parser.add_argument("--number", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--min-repeat-ms", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = execute(args)
    except Exception as exc:  # keep failed runs explicit and machine-readable
        payload = {
            "status": "failed",
            "run_id": f"{args.engine}_{args.kernel}_seed{args.seed}_trials{args.trials}_llm{args.llm_budget}_{args.cost_model}",
            "engine": args.engine,
            "workload_name": args.kernel,
            "seed": args.seed,
            "trials": args.trials,
            "llm_budget": args.llm_budget,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "timestamp_utc": utc_stamp(),
        }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
