#!/bin/bash
# Environment setup for the Phase-1 RCA. Recreates the venv and installs the
# mission-specified PyTorch ROCm 7.14 stack from the official wheel index.
set -euo pipefail
VENV=/workspace/venv-qwen-gdn-rca
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools
# Mission spec: official PyTorch ROCm 7.14 index, exactly this URL.
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.14
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.hip:", torch.version.hip)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.device_count():
    print("device_name:", torch.cuda.get_device_name(0))
    print("device_props:", torch.cuda.get_device_properties(0))
    print("gcnArchName:", getattr(torch.cuda.get_device_properties(0), "gcnArchName", None))
PY
