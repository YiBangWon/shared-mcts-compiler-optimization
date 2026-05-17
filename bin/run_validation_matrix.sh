#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT_DIR/measurements/raw"
mkdir -p "$RAW_DIR"

: "${BASELINE_TVM_HOME:?Set BASELINE_TVM_HOME to the official Reasoning Compiler TVM tree}"
: "${COORDINATED_TVM_HOME:?Set COORDINATED_TVM_HOME to the shared-MCTS TVM tree}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

run_one() {
  local engine="$1"
  local kernel="$2"
  local seed="$3"
  local tvm_home
  if [[ "$engine" == "baseline" ]]; then
    tvm_home="$BASELINE_TVM_HOME"
  else
    tvm_home="$COORDINATED_TVM_HOME"
  fi
  export TVM_HOME="$tvm_home"
  export PYTHONPATH="$TVM_HOME/python:$ROOT_DIR:${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="$TVM_HOME/build:${LD_LIBRARY_PATH:-}"
  export TVM_LIBRARY_PATH="$TVM_HOME/build/libtvm.so"
  python3 "$ROOT_DIR/bin/execute_matched_trial.py" \
    --engine "$engine" \
    --kernel "$kernel" \
    --seed "$seed" \
    --trials 32 \
    --llm-budget 4 \
    --cost-model xgb \
    --out "$RAW_DIR/${engine}_${kernel}_seed${seed}_trials32_llm4_xgb.json"
}

for kernel in flux_conv_large llama_attention_qk_av llama_mlp_swiglu; do
  for seed in 0 1 2 3 4; do
    run_one baseline "$kernel" "$seed"
    run_one coordinated "$kernel" "$seed"
  done
done
