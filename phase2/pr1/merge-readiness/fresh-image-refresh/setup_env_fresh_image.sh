#!/bin/bash
# Fresh-image environment setup (2026-10-07). Same official PyTorch ROCm 7.14
# index and same resolved stack as scripts/setup_env.sh, with one documented
# deviation: the rocm metapackage is pinned to 7.14.0.post1 with explicit
# [devel,libraries,device-gfx1100] extras BEFORE torch is installed.
#
# Reason: on this fresh image an unpinned resolution picked rocm==7.14.1 whose
# device-all expansion (~25 GB of gfx* code-object wheels, driven by torch's
# rocm[device-all,libraries] requirement) exhausted the 4 GB tmpfs at /tmp
# (pip download staging) - OSError Errno 28. The 7.14.0.post1 + single-gfx1100
# set matches the audited historical stack exactly (rocm 7.14.0.post1,
# rocm-sdk-core/libraries/devel 7.14.0, rocm-sdk-device-gfx1100 7.14.0,
# torch 2.14.1+rocm7.14, triton-rocm 3.8.0) and avoids the 7.14.1 upgrade the
# mission forbids. TMPDIR is moved off the small tmpfs.
set -euo pipefail
export TMPDIR=/root/pip-tmp
mkdir -p "$TMPDIR"
VENV=/workspace/venv-qwen-gdn-rca
rm -rf "$VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# ROCm 7.14.0 SDK family, gfx1100-only device code, devel toolchain included.
# rocm-sdk-devel is not mirrored on the PyTorch index; AMD's official ROCm
# wheel index (repo.amd.com/rocm/whl-multi-arch) provides 7.14.0.
pip3 install "rocm[devel,libraries,device-gfx1100]==7.14.0.post1" \
  --index-url https://download.pytorch.org/whl/rocm7.14 \
  --extra-index-url https://repo.amd.com/rocm/whl-multi-arch/

# torch's small pure-python deps + triton-rocm (exact pins from torch METADATA)
pip3 install filelock "typing-extensions>=4.10.0" "setuptools>=77.0.3" \
  "sympy>=1.13.3" "networkx>=2.5.1" jinja2 "fsspec>=0.8.5" numpy
pip3 install "triton-rocm==3.8.0" \
  --index-url https://download.pytorch.org/whl/rocm7.14

# torch/torchvision with --no-deps so pip cannot silently swap the SDK family
# (documented intentional deviation: rocm[device-all] extra left unsatisfied;
# single-arch device code is a wheel-size optimization, not a semantic need).
pip3 install --no-deps torch==2.14.1+rocm7.14 torchvision==0.29.1+rocm7.14 \
  --index-url https://download.pytorch.org/whl/rocm7.14

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.hip:", torch.version.hip)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.device_count():
    print("device_name:", torch.cuda.get_device_name(0))
    print("gcnArchName:", getattr(torch.cuda.get_device_properties(0), "gcnArchName", None))
PY
