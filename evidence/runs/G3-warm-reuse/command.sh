#!/bin/bash
source "/workspace/qwen-gdn-rca/evidence/runs/G3-warm-reuse/run-env.sh"
source /workspace/venv-qwen-gdn-rca/bin/activate
export VLLM_ENGINE_READY_TIMEOUT_S=7200
exec vllm serve /workspace/models/Qwen3.5-0.8B --port 8138 --jit-monitor-mode warn --jit-monitor-verbose --max-model-len 2048 --max-num-seqs 16
