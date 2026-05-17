# Shared MCTS Compiler Optimization

This repository contains a course-scale project on shared-MCTS multi-model compiler optimization for TVM tensor programs. It extends the Reasoning Compiler direction with coordinated model selection during search.

This repository is intended for a graduate course project, not for claiming a full paper-scale reproduction.

## Key Takeaway

Shared-MCTS multi-model search improved latency on 2 out of 3 scaled-down Reasoning-Compiler-family TVM workloads, but did not reduce strong-model calls. The result supports selected-workload latency improvement, not cost saving.

## Overview

TVM tensor program optimization has a large schedule transformation search space. Transformations such as tiling, fusion, vectorization, layout changes, and unrolling are highly interdependent, so finding a good schedule can require many compile-and-measure samples.

Reasoning Compiler improves this problem by formulating compiler optimization as a sequential, context-aware decision process guided by an LLM and structured Monte Carlo Tree Search (MCTS). However, a single-model LLM-guided search still has a model-selection limitation: a stronger model may produce better proposals but can increase cost, while a smaller model may be cheaper but less capable.

This project explores a shared-MCTS search direction where multiple models use the same search tree. At each step, the search can choose both a compiler transformation and the next model to query. The goal is to obtain a better search trajectory and improve latency on TVM tensor workloads.

## Reference

- REASONING COMPILER: LLM-Guided Optimizations for Efficient Model Serving
- Venue: NeurIPS 2025 poster
- Link: https://openreview.net/forum?id=2D4TuZyNnr

## What This Repository Is

- A course-scale implementation and evaluation of shared-MCTS multi-model compiler search.
- A matched comparison against the official Reasoning Compiler implementation on scaled-down TVM workloads.
- A system optimization project focused on compiler search behavior and latency.

## What This Repository Is Not

- It is not a full reproduction of the original Reasoning Compiler benchmark suite.
- It is not a claim that shared-MCTS search always outperforms Reasoning Compiler.
- It is not a GPU/CUDA performance claim.
- It is not a cost-saving claim.

## Why Scaled-down Workloads?

The public Reasoning Compiler repository did not include exact runnable configurations for the full benchmark workloads. To make a matched five-seed evaluation feasible for a course project, we preserved the operator structure of the workload families while reducing tensor dimensions.

These workloads are designed to represent the structure of attention, convolution, and MLP tensor programs, but they are not exact full-scale reproductions.

## Workloads

| Workload | Scaled-down definition |
|---|---|
| Llama-style Attention | `B=1, H=8, S=128, D=64`; QK matmul followed by AV matmul; softmax omitted |
| FLUX-style Convolution | `N=1, IC=128, H=64, W=64, OC=128, K=3, S=1, P=1` |
| Llama-style MLP | `T=128, H=512, I=2048`; gate projection, up projection, elementwise gate*up, down projection |

## Experimental Setup

The main experiment compares the official Reasoning Compiler implementation against the shared-MCTS multi-model search implementation under matched conditions.

| Item | Setting |
|---|---|
| Target | LLVM CPU |
| Seeds | `0, 1, 2, 3, 4` |
| Trials | `32` |
| LLM budget | `4` total LLM calls per tuning run, not per MetaSchedule trial |
| Comparison | Same workload, shape, target, trials, LLM budget, seeds, and measurement harness |
| Baseline | Official Reasoning Compiler implementation |
| Proposed method | Shared-MCTS / multi-model search |
| TVM import | Verified separately for both implementations |

## Results

All results are based on matched LLVM CPU experiments with five seeds, 32 trials, and a total LLM budget of 4 calls per tuning run.

| Workload | Reasoning Compiler median latency | Shared-MCTS median latency | Improvement |
|---|---:|---:|---:|
| scaled-down FLUX Conv | `2.508199 ms` | `2.338621 ms` | `+6.76%` |
| scaled-down Llama Attention | `0.157998 ms` | `0.151024 ms` | `+4.41%` |
| scaled-down Llama MLP | `2.932757 ms` | `3.039299 ms` | `-3.63%` |

Aggregate result:

| Metric | Result |
|---|---:|
| Faster workloads | `2 / 3` |
| Geomean speedup | `1.0268x` |
| Median latency improvement | `4.414%` |
| Strong-model call reduction | `0%` |
| Cost-saving claim | Not supported |

### Figures

![Median latency comparison](figures/latency_by_operator.png)

![Speedup over Reasoning Compiler baseline](figures/speedup_by_operator.png)

![Strong-model call comparison](figures/llm_usage_by_operator.png)

![Latency versus strong-model calls](figures/latency_vs_strong_calls.png)

## Interpretation

The shared-MCTS multi-model search improved median latency on 2 out of 3 scaled-down Reasoning-Compiler-family workloads. It achieved 6.76% improvement on the FLUX-style convolution workload and 4.41% improvement on the Llama-style attention workload.

However, it was 3.63% slower on the Llama-style MLP workload. Therefore, the result should be interpreted as selected-workload latency improvement, not universal superiority.

The experiment did not reduce strong-model calls. Therefore, this repository does not claim cost saving or strong-model call reduction.

## Safe Claims

This repository supports the following claim:

> On three scaled-down Reasoning-Compiler-family TVM workloads under matched LLVM CPU conditions, shared-MCTS multi-model search improved median latency on 2 out of 3 workloads, achieving 1.0268x geomean speedup over the official Reasoning Compiler baseline.

This repository does not claim:

- full benchmark reproduction,
- universal improvement,
- GPU improvement,
- strong-model call reduction,
- cost saving.

## Repository Structure

```text
.
+-- bin/
|   +-- execute_matched_trial.py
|   +-- run_smoke_matrix.sh
|   +-- run_validation_matrix.sh
|   +-- aggregate_measurements.py
|   +-- render_figures.py
+-- kernels/
|   +-- model_serving_programs.py
+-- measurements/
|   +-- baseline_runs.csv
|   +-- coordinated_runs.csv
|   +-- paired_eval_table.csv
|   +-- validated_summary.csv
+-- figures/
+-- docs/
|   +-- operator_family_map.md
|   +-- matched_study_report.md
|   +-- korean_project_note.md
+-- requirements.txt
+-- LICENSE
+-- README.md
```

## Reproduction

The scripts expect two separate TVM environments:

- `BASELINE_TVM_HOME`: official Reasoning Compiler TVM source tree
- `COORDINATED_TVM_HOME`: shared-MCTS multi-model TVM source tree

`requirements.txt` only contains local plotting and result-processing dependencies. TVM and the two compiler-search implementations must be built separately.

Example:

```bash
pip install -r requirements.txt

export CUDA_VISIBLE_DEVICES=0
export BASELINE_TVM_HOME=/path/to/reasoning-compiler
export COORDINATED_TVM_HOME=/path/to/shared-mcts-tvm

TVM_HOME=$BASELINE_TVM_HOME \
PYTHONPATH=$BASELINE_TVM_HOME/python:$PYTHONPATH \
LD_LIBRARY_PATH=$BASELINE_TVM_HOME/build:$LD_LIBRARY_PATH \
TVM_LIBRARY_PATH=$BASELINE_TVM_HOME/build/libtvm.so \
python3 -c "import tvm; print(tvm.__version__, tvm.__file__)"

TVM_HOME=$COORDINATED_TVM_HOME \
PYTHONPATH=$COORDINATED_TVM_HOME/python:$PYTHONPATH \
LD_LIBRARY_PATH=$COORDINATED_TVM_HOME/build:$LD_LIBRARY_PATH \
TVM_LIBRARY_PATH=$COORDINATED_TVM_HOME/build/libtvm.so \
python3 -c "import tvm; print(tvm.__version__, tvm.__file__)"

bash bin/run_validation_matrix.sh
python3 bin/aggregate_measurements.py
python3 bin/render_figures.py
```

For more details, see:

- `docs/operator_family_map.md`
- `docs/matched_study_report.md`
- `docs/korean_project_note.md`

API keys are not stored in this repository. All latency numbers should be interpreted as course-scale experimental results, not full paper-scale reproduction results.

## License

This project is released under the MIT License.
