#!/bin/bash
# generated for run P3-cold-normal-warn
source "/workspace/qwen-gdn-rca/evidence/runs/P3-cold-normal-warn/run-env.sh"
source /workspace/venv-qwen-gdn-rca/bin/activate
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-7200}
exec vllm serve "/workspace/models/Qwen3.8-27B-FP8" \
  --port 8139 \
  --jit-monitor-mode warn --jit-monitor-verbose \
  --max-model-len 4096 --max-num-seqs 32 --gpu-memory-utilization 0.90
