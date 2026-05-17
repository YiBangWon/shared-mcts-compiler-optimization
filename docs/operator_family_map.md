# Operator Family Map

## Reasoning-Compiler Experiment Families

The Reasoning Compiler paper reports model-serving layer workloads. The relevant families for this course-scale artifact are:

| Paper family | Public exact runnable config | Course-scale workload used here | Status |
|---|---|---|---|
| Llama-3-8B Attention | Not found in runnable form | `llama_attention_qk_av` | selected |
| DeepSeek-R1 MoE | Not found in runnable form | `simplified_moe_ffn` | mapped, not in final 3-workload validation |
| FLUX Attention | Not found in runnable form | attention-family mapping overlaps with `llama_attention_qk_av` | not separately validated |
| FLUX Convolution | Not found in runnable form | `flux_conv_large` | selected |
| Llama-4-Scout MLP | Not found in runnable form | `llama_mlp_swiglu` | selected |

## Selected Workloads

### `llama_attention_qk_av`

- Family: Llama-style attention
- Shape: `B=1,H=8,S=128,D=64`
- Operator structure: QK reduction followed by AV reduction
- Simplification: softmax omitted for course-scale TIR tuning

### `flux_conv_large`

- Family: FLUX-style convolution
- Shape: `N=1,IC=128,H=64,W=64,OC=128,K=3,S=1,P=1`
- Operator structure: NCHW convolution with adjacent elementwise operation
- Role: larger convolutional workload than the supplementary sanity Conv2D

### `llama_mlp_swiglu`

- Family: Llama-style MLP / FFN
- Shape: `T=128,H=512,I=2048`
- Operator structure: gate projection, up projection, elementwise gate*up, down projection

## Supplementary Sanity Kernels

MatMul 256/512 and simple CPU Conv2D were used only to check whether the compiler search pipelines ran under matched conditions. They are not main Reasoning-Compiler-family benchmark workloads in this repository.
