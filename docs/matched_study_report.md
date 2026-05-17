# Matched Study Report

## 1. Purpose

This study evaluates whether shared-MCTS multi-model compiler search improves over the official Reasoning Compiler implementation on TVM tensor workloads derived from the Reasoning Compiler experiment families.

The main comparison is not based on arbitrary small MatMul or simple Conv2D kernels. Those kernels are treated only as supplementary sanity checks. The main study focuses on model-serving-style operator families: attention, convolution, and MLP.

## 2. Why Simple Kernels Are Insufficient

MatMul 256/512 and a small CPU Conv2D are useful for checking whether the build, MetaSchedule integration, and MCTS search path work. However, they do not represent the larger, more interdependent transformation spaces emphasized by Reasoning Compiler.

For the final project claim, the more relevant question is whether shared search helps on Reasoning-Compiler-family workloads with attention, convolution, and feed-forward structure.

## 3. Workload Extraction

The Reasoning Compiler paper describes layer-wise workloads including:

- Llama-3-8B Attention
- DeepSeek-R1 MoE
- FLUX Attention
- FLUX Convolution
- Llama-4-Scout MLP

The public repository did not include exact runnable benchmark configurations, tensor shapes, or launch scripts for the full benchmark suite. To keep the comparison matched and feasible for a course project, this repository uses scaled-down workloads that preserve the operator structure while reducing tensor dimensions.

## 4. Matched Comparison Setup

Both methods used:

- same workload semantics,
- same tensor shape,
- same LLVM CPU target,
- same trial budget,
- same LLM budget,
- same seeds,
- same latency measurement harness,
- separate verified TVM imports.

The validation setting was:

- target: `llvm --num-cores=1`
- seeds: `0, 1, 2, 3, 4`
- trials: `32`
- LLM budget: `4`
- cost model: `xgb`

## 5. Results

| Workload | Baseline median latency | Shared-MCTS median latency | Improvement | Speedup |
|---|---:|---:|---:|---:|
| scaled-down FLUX Conv | 2.508199 ms | 2.338621 ms | +6.7609% | 1.0725x |
| scaled-down Llama Attention | 0.157998 ms | 0.151024 ms | +4.4140% | 1.0462x |
| scaled-down Llama MLP | 2.932757 ms | 3.039299 ms | -3.6328% | 0.9649x |

Aggregate:

- faster workloads: 2 / 3
- geomean speedup: 1.0268x
- median latency improvement: 4.414%
- strong-model call reduction: 0%

## 6. Main Conclusion

On three scaled-down Reasoning-Compiler-family TVM workloads under matched LLVM CPU conditions, shared-MCTS multi-model search improved median latency on 2 out of 3 workloads and achieved 1.0268x geomean speedup over the official Reasoning Compiler baseline.

The result is workload-dependent. The method improved the scaled-down FLUX convolution and Llama attention workloads, but regressed on the scaled-down Llama MLP workload.

## 7. What Can Be Claimed

The safe claim is:

> On three scaled-down Reasoning-Compiler-family TVM workloads under matched LLVM CPU conditions, shared-MCTS multi-model search improved median latency on 2 out of 3 workloads, achieving 1.0268x geomean speedup over the official Reasoning Compiler baseline.

## 8. What Cannot Be Claimed

- This is not a full paper-scale benchmark reproduction.
- This is not an exact reproduction of the full original benchmark suite.
- This is not a GPU/CUDA performance claim.
- This is not a universal superiority claim.
- This is not a cost-saving claim, because strong-model calls did not decrease.
