#!/bin/bash
# Build vLLM (pinned SHA 31e2443) from source with the COHERENT venv ROCm 7.14
# toolchain. Produces evidence/environment/vllm-build-rocm714.txt (via caller).
set -euo pipefail
VENV=/workspace/venv-qwen-gdn-rca
DEV=$VENV/lib/python3.12/site-packages/_rocm_sdk_devel
SRC=/workspace/vllm-zero-jit-rca

source $VENV/bin/activate
export PATH="$DEV/bin:$VENV/bin:/usr/local/cargo/bin:/usr/bin:/bin"
export ROCM_PATH="$DEV"
export HIP_PATH="$DEV"
export LD_LIBRARY_PATH="$DEV/lib"
export CMAKE_PREFIX_PATH="$DEV"
export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH=gfx1100
export MAX_JOBS=48
export VLLM_TARGET_MESSAGE=OFF
export PYTHONUNBUFFERED=1

echo "=== build env ==="
echo "PATH=$PATH"
echo "ROCM_PATH=$ROCM_PATH  HIP_PATH=$HIP_PATH"
which hipcc cmake ninja python pip cargo || true
hipcc --version | head -3
python -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip)"
python -c "from torch.utils.cpp_extension import ROCM_HOME; print('ROCM_HOME', ROCM_HOME)"
echo "=== source ==="
cd $SRC
git rev-parse HEAD
git status --short | head -5 || true

echo "=== pip install -e . --no-deps --no-build-isolation ==="
pip install -e . --no-deps --no-build-isolation
echo "BUILD_SCRIPT_DONE rc=$?"
