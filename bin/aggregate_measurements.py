#!/usr/bin/env python3
"""Aggregate raw matched-run JSON files into CSV/JSON tables."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "measurements" / "raw"
MEASURE_DIR = ROOT_DIR / "measurements"


def as_float(value: Any) -> Optional[float]:
    if value in (None, "", "N/A"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def rounded(value: Optional[float], digits: int = 6) -> Any:
    return "N/A" if value is None else round(float(value), digits)


def read_payloads() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in sorted(RAW_DIR.glob("*.json")):
        try:
            row = json.loads(entry.read_text(encoding="utf-8"))
        except Exception:
            continue
        if row.get("engine") in {"baseline", "coordinated"}:
            row["_raw_json"] = str(entry)
            rows.append(row)
    return rows


def write_tables(rows: List[Dict[str, Any]], json_file: Path, csv_file: Path) -> None:
    json_file.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def pair_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (row.get("workload_name"), row.get("seed"), row.get("trials"), row.get("llm_budget"), row.get("cost_model"))


def paired_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        buckets.setdefault(pair_key(row), {})[row["engine"]] = row

    pairs: List[Dict[str, Any]] = []
    for key, item in sorted(buckets.items()):
        baseline = item.get("baseline")
        coordinated = item.get("coordinated")
        if not baseline or not coordinated:
            continue
        base_latency = as_float(baseline.get("median_latency_ms"))
        coord_latency = as_float(coordinated.get("median_latency_ms"))
        base_time = as_float(baseline.get("total_tuning_time_sec"))
        coord_time = as_float(coordinated.get("total_tuning_time_sec"))
        base_llm = as_float(baseline.get("llm_calls"))
        coord_llm = as_float(coordinated.get("llm_calls"))
        base_strong = as_float(baseline.get("strong_model_call_count"))
        coord_strong = as_float(coordinated.get("strong_model_call_count"))
        improvement = (base_latency - coord_latency) / base_latency * 100 if base_latency and coord_latency else None
        pairs.append(
            {
                "workload_name": key[0],
                "workload_type": baseline.get("workload_type"),
                "shape": baseline.get("shape"),
                "seed": key[1],
                "trials": key[2],
                "llm_budget": key[3],
                "cost_model": key[4],
                "baseline_median_latency_ms": rounded(base_latency),
                "coordinated_median_latency_ms": rounded(coord_latency),
                "latency_improvement_percent": rounded(improvement, 4),
                "speedup_ratio": rounded(base_latency / coord_latency if base_latency and coord_latency else None, 4),
                "baseline_tuning_time_sec": rounded(base_time, 4),
                "coordinated_tuning_time_sec": rounded(coord_time, 4),
                "tuning_time_reduction_percent": rounded((base_time - coord_time) / base_time * 100 if base_time and coord_time else None, 4),
                "baseline_llm_calls": rounded(base_llm, 0),
                "coordinated_llm_calls": rounded(coord_llm, 0),
                "llm_call_reduction_percent": rounded((base_llm - coord_llm) / base_llm * 100 if base_llm else None, 4),
                "baseline_strong_model_calls": rounded(base_strong, 0),
                "coordinated_strong_model_calls": rounded(coord_strong, 0),
                "strong_model_call_reduction_percent": rounded((base_strong - coord_strong) / base_strong * 100 if base_strong else None, 4),
                "baseline_run_id": baseline.get("run_id"),
                "coordinated_run_id": coordinated.get("run_id"),
                "baseline_tvm_file": baseline.get("tvm_file"),
                "coordinated_tvm_file": coordinated.get("tvm_file"),
            }
        )
    return pairs


def describe(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"median": None, "mean": None, "std": None, "min": None, "max": None}
    return {
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def validated_rows(pairs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in pairs:
        group_key = (row["workload_name"], row["workload_type"], row["shape"], row["trials"], row["llm_budget"], row["cost_model"])
        groups.setdefault(group_key, []).append(row)

    summaries: List[Dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        seeds = sorted(int(row["seed"]) for row in group)
        base_latencies = [as_float(row["baseline_median_latency_ms"]) for row in group]
        coord_latencies = [as_float(row["coordinated_median_latency_ms"]) for row in group]
        base_times = [as_float(row["baseline_tuning_time_sec"]) for row in group]
        coord_times = [as_float(row["coordinated_tuning_time_sec"]) for row in group]
        base_strong = [as_float(row["baseline_strong_model_calls"]) for row in group]
        coord_strong = [as_float(row["coordinated_strong_model_calls"]) for row in group]
        base_stats = describe([value for value in base_latencies if value is not None])
        coord_stats = describe([value for value in coord_latencies if value is not None])
        base_median = base_stats["median"]
        coord_median = coord_stats["median"]
        base_strong_median = statistics.median([value for value in base_strong if value is not None]) if any(value is not None for value in base_strong) else None
        coord_strong_median = statistics.median([value for value in coord_strong if value is not None]) if any(value is not None for value in coord_strong) else None
        improvement = (base_median - coord_median) / base_median * 100 if base_median and coord_median else None
        strong_reduction = (base_strong_median - coord_strong_median) / base_strong_median * 100 if base_strong_median else None
        summaries.append(
            {
                "workload_name": key[0],
                "workload_type": key[1],
                "shape": key[2],
                "trials": key[3],
                "llm_budget": key[4],
                "cost_model": key[5],
                "num_repetitions": len(group),
                "seeds": ",".join(str(seed) for seed in seeds),
                "baseline_median_latency_ms": rounded(base_median),
                "coordinated_median_latency_ms": rounded(coord_median),
                "baseline_mean_latency_ms": rounded(base_stats["mean"]),
                "coordinated_mean_latency_ms": rounded(coord_stats["mean"]),
                "baseline_std_latency_ms": rounded(base_stats["std"]),
                "coordinated_std_latency_ms": rounded(coord_stats["std"]),
                "baseline_min_latency_ms": rounded(base_stats["min"]),
                "coordinated_min_latency_ms": rounded(coord_stats["min"]),
                "baseline_max_latency_ms": rounded(base_stats["max"]),
                "coordinated_max_latency_ms": rounded(coord_stats["max"]),
                "latency_improvement_percent": rounded(improvement, 4),
                "speedup_ratio": rounded(base_median / coord_median if base_median and coord_median else None, 4),
                "baseline_median_tuning_time_sec": rounded(statistics.median([value for value in base_times if value is not None]) if any(value is not None for value in base_times) else None, 4),
                "coordinated_median_tuning_time_sec": rounded(statistics.median([value for value in coord_times if value is not None]) if any(value is not None for value in coord_times) else None, 4),
                "baseline_median_strong_model_calls": rounded(base_strong_median, 0),
                "coordinated_median_strong_model_calls": rounded(coord_strong_median, 0),
                "strong_model_call_reduction_percent": rounded(strong_reduction, 4),
                "strict_latency_win": bool(improvement is not None and improvement >= 2),
                "cost_quality_win": bool(coord_median is not None and base_median is not None and coord_median <= base_median * 1.01 and strong_reduction is not None and strong_reduction >= 50),
            }
        )
    return summaries


def main() -> int:
    MEASURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_payloads()
    baseline_rows = [row for row in rows if row.get("engine") == "baseline"]
    coordinated_rows = [row for row in rows if row.get("engine") == "coordinated"]
    pairs = paired_rows(rows)
    summaries = validated_rows(pairs)
    write_tables(baseline_rows, MEASURE_DIR / "baseline_runs.json", MEASURE_DIR / "baseline_runs.csv")
    write_tables(coordinated_rows, MEASURE_DIR / "coordinated_runs.json", MEASURE_DIR / "coordinated_runs.csv")
    write_tables(pairs, MEASURE_DIR / "paired_eval_table.json", MEASURE_DIR / "paired_eval_table.csv")
    write_tables(summaries, MEASURE_DIR / "validated_summary.json", MEASURE_DIR / "validated_summary.csv")
    print(f"wrote {len(rows)} raw rows, {len(pairs)} paired rows, {len(summaries)} summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
