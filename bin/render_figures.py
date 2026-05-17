#!/usr/bin/env python3
"""Render publication-friendly figures from validated summary CSV files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
MEASURE_DIR = ROOT_DIR / "measurements"
FIGURE_DIR = ROOT_DIR / "figures"


def read_summary() -> List[Dict[str, Any]]:
    with (MEASURE_DIR / "validated_summary.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_number(value: Any) -> float:
    return float(value)


def workload_labels(rows: List[Dict[str, Any]]) -> List[str]:
    mapping = {
        "flux_conv_large": "FLUX Conv",
        "llama_attention_qk_av": "Llama Attention",
        "llama_mlp_swiglu": "Llama MLP",
        "simplified_moe_ffn": "MoE FFN",
    }
    return [mapping.get(row["workload_name"], row["workload_name"]) for row in rows]


def latency_bars(rows: List[Dict[str, Any]]) -> None:
    labels = workload_labels(rows)
    baseline = [as_number(row["baseline_median_latency_ms"]) for row in rows]
    coordinated = [as_number(row["coordinated_median_latency_ms"]) for row in rows]
    indices = range(len(rows))
    width = 0.36
    plt.figure(figsize=(8, 4.5))
    plt.bar([i - width / 2 for i in indices], baseline, width, label="Reasoning Compiler")
    plt.bar([i + width / 2 for i in indices], coordinated, width, label="Shared-MCTS")
    plt.xticks(list(indices), labels, rotation=15, ha="right")
    plt.ylabel("Median latency (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "latency_by_operator.png", dpi=180)
    plt.close()


def speedup_bars(rows: List[Dict[str, Any]]) -> None:
    labels = workload_labels(rows)
    speedups = [as_number(row["speedup_ratio"]) for row in rows]
    colors = ["#2f7d5f" if value >= 1 else "#9a4b4b" for value in speedups]
    plt.figure(figsize=(8, 4.2))
    plt.bar(labels, speedups, color=colors)
    plt.axhline(1.0, color="black", linewidth=1)
    plt.ylabel("Speedup over baseline")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "speedup_by_operator.png", dpi=180)
    plt.close()


def call_count_bars(rows: List[Dict[str, Any]]) -> None:
    labels = workload_labels(rows)
    baseline = [as_number(row["baseline_median_strong_model_calls"]) for row in rows]
    coordinated = [as_number(row["coordinated_median_strong_model_calls"]) for row in rows]
    indices = range(len(rows))
    width = 0.36
    plt.figure(figsize=(8, 4.2))
    plt.bar([i - width / 2 for i in indices], baseline, width, label="Reasoning Compiler")
    plt.bar([i + width / 2 for i in indices], coordinated, width, label="Shared-MCTS")
    plt.xticks(list(indices), labels, rotation=15, ha="right")
    plt.ylabel("Median strong-model calls")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "llm_usage_by_operator.png", dpi=180)
    plt.close()


def pareto_scatter(rows: List[Dict[str, Any]]) -> None:
    plt.figure(figsize=(6, 4.5))
    for row, label in zip(rows, workload_labels(rows)):
        plt.scatter(as_number(row["baseline_median_latency_ms"]), as_number(row["baseline_median_strong_model_calls"]), marker="o", label=f"{label} baseline")
        plt.scatter(as_number(row["coordinated_median_latency_ms"]), as_number(row["coordinated_median_strong_model_calls"]), marker="x", label=f"{label} shared")
    plt.xlabel("Median latency (ms)")
    plt.ylabel("Strong-model calls")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "latency_vs_strong_calls.png", dpi=180)
    plt.close()


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_summary()
    latency_bars(rows)
    speedup_bars(rows)
    call_count_bars(rows)
    pareto_scatter(rows)
    print(f"wrote figures to {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
