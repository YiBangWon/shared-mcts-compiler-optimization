# 팀원 공유용 요약

## 현재 결과

공식 Reasoning Compiler 구현과 shared-MCTS multi-model search 구현을 같은 TVM tensor workload, 같은 target, 같은 seed, 같은 trial budget, 같은 LLM budget 조건에서 비교했습니다.

이번 메인 결과는 단순 MatMul/Conv2D sanity check가 아니라, Reasoning Compiler 실험 방향에 맞춘 scaled-down model-serving workload 3개를 기준으로 합니다.

## 선택한 workload

| Workload | 설명 | 결과 |
|---|---|---|
| `flux_conv_large` | FLUX convolution 계열 scaled-down workload | shared-MCTS가 6.76% 빠름 |
| `llama_attention_qk_av` | Llama attention 계열 scaled-down workload | shared-MCTS가 4.41% 빠름 |
| `llama_mlp_swiglu` | Llama MLP/FFN 계열 scaled-down workload | shared-MCTS가 3.63% 느림 |

전체적으로는 3개 중 2개 workload에서 latency가 개선되었고, geomean speedup은 1.0268x입니다.

## 발표에서 안전하게 말할 수 있는 claim

“세 개의 scaled-down Reasoning-Compiler-family TVM workload에서 공식 Reasoning Compiler baseline과 matched comparison을 수행했고, shared-MCTS multi-model search가 2/3 workload에서 median latency를 개선했으며 geomean 1.0268x speedup을 보였습니다.”

## 발표에서 하면 안 되는 claim

- 전체 논문 benchmark suite를 완전히 재현했다고 말하면 안 됩니다.
- GPU/CUDA 성능 개선을 주장하면 안 됩니다.
- 모든 workload에서 항상 더 좋다고 말하면 안 됩니다.
- 비용 절감이나 strong-model call 감소를 주장하면 안 됩니다. 이번 결과에서는 strong-model call reduction이 0%입니다.

## Scaled-down의 의미

원 논문 workload family의 operator 구조는 유지하되 tensor dimension을 줄였습니다. 예를 들어 attention은 QK matmul과 AV matmul 구조를 유지하지만 sequence length와 head 수를 course-scale 실험이 가능한 크기로 줄였고, softmax는 생략했습니다. 따라서 이 결과는 full paper-scale reproduction이 아니라 course-scale matched reproduction으로 해석해야 합니다.
