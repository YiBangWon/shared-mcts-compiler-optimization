#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT_DIR/measurements/raw"
mkdir -p "$RAW_DIR"

: "${BASELINE_TVM_HOME:?Set BASELINE_TVM_HOME to the official Reasoning Compiler TVM tree}"
: "${COORDINATED_TVM_HOME:?Set COORDINATED_TVM_HOME to the shared-MCTS TVM tree}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for engine in baseline coordinated; do
  if [[ "$engine" == "baseline" ]]; then
    export TVM_HOME="$BASELINE_TVM_HOME"
  else
    export TVM_HOME="$COORDINATED_TVM_HOME"
  fi
  export PYTHONPATH="$TVM_HOME/python:$ROOT_DIR:${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="$TVM_HOME/build:${LD_LIBRARY_PATH:-}"
  export TVM_LIBRARY_PATH="$TVM_HOME/build/libtvm.so"
  python3 "$ROOT_DIR/bin/execute_matched_trial.py" \
    --engine "$engine" \
    --kernel llama_attention_qk_av \
    --seed 0 \
    --trials 16 \
    --llm-budget 2 \
    --cost-model xgb \
    --out "$RAW_DIR/${engine}_llama_attention_qk_av_seed0_trials16_llm2_xgb.json"
done
