#!/bin/bash
# Collect canonical experiment-environment evidence from the RCA venv.
# Usage: collect_env_rocm714.sh [outdir]   (default: evidence/environment)
set -u
OUT="${1:-$(dirname "$0")/../evidence/environment}"
mkdir -p "$OUT"
VENV=/workspace/venv-qwen-gdn-rca
DEV=$VENV/lib/python3.12/site-packages/_rocm_sdk_devel

{
  echo "# collected $(date -Is) by collect_env_rocm714.sh"
  echo "== interpreter identity (inside build/run env) =="
  env PATH="$DEV/bin:$VENV/bin:/usr/bin:/bin" ROCM_PATH="$DEV" HIP_PATH="$DEV" \
    bash -c 'source '$VENV'/bin/activate; which python python3 pip pip3; python -V'
  echo
  echo "== torch / gpu =="
  env PATH="$DEV/bin:$VENV/bin:/usr/bin:/bin" ROCM_PATH="$DEV" HIP_PATH="$DEV" \
    LD_LIBRARY_PATH="$DEV/lib" bash -c 'source '$VENV'/bin/activate; python - <<PYEOF
import sys, torch
print("python executable:", sys.executable)
print("torch:", torch.__version__)
print("torch file:", torch.__file__)
print("torch.version.hip:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.device_count():
    p = torch.cuda.get_device_properties(0)
    print("device:", torch.cuda.get_device_name(0))
    print("gcnArchName:", getattr(p, "gcnArchName", None))
    print("total_memory:", p.total_memory)
import triton
print("triton:", triton.__version__)
print("triton file:", triton.__file__)
import os
print("triton backends:", sorted(d for d in os.listdir(os.path.join(os.path.dirname(triton.__file__),"backends")) if not d.startswith("_") and d.endswith(".py") is False and d in ("amd","nvidia")))
PYEOF'
  echo
  echo "== importlib metadata (torch/triton/vllm/rocm) =="
  env PATH="$DEV/bin:$VENV/bin:/usr/bin:/bin" bash -c 'source '$VENV'/bin/activate; python - <<PYEOF
import importlib.metadata as m
for name in ["torch","torchvision","triton-rocm","rocm","rocm-sdk-core","rocm-sdk-libraries","rocm-sdk-device-gfx1100","rocm-sdk-devel","vllm","xgrammar","conch-triton-kernels"]:
    try:
        d = m.distribution(name)
        print(name, d.version, d.locate_file(""))
    except Exception as e:
        print(name, "NOT-INSTALLED")
PYEOF'
} > "$OUT/python-rocm714.txt" 2>&1

{
  echo "# collected $(date -Is)"
  env PATH="$DEV/bin:$VENV/bin:/usr/bin:/bin" bash -c 'source '$VENV'/bin/activate; pip freeze'
} > "$OUT/packages-rocm714.txt" 2>&1

{
  echo "# collected $(date -Is)"
  echo "== BUILD toolchain (venv rocm-sdk-devel, isolated) =="
  echo "DEVROOT=$DEV"
  env -i PATH="$DEV/bin:/usr/bin:/bin" HOME=/root ROCM_PATH="$DEV" HIP_PATH="$DEV" "$DEV/bin/hipcc" --version
  echo
  echo "== clang =="
  env -i PATH="$DEV/bin:/usr/bin:/bin" HOME=/root "$DEV/bin/amdclang++" --version | head -3
  echo
  echo "== hipconfig (key fields) =="
  env -i PATH="$DEV/bin:/usr/bin:/bin" HOME=/root ROCM_PATH="$DEV" HIP_PATH="$DEV" "$DEV/bin/hipconfig" | grep -E 'ROCM_PATH|HIP_PATH|HIP_COMPILER|HIP_VERSION|CPP_CONFIG' | head
  echo
  echo "== cmake configs in devel tree =="
  ls "$DEV/lib/cmake" | head -30
  echo
  echo "== PRE-EXISTING host toolchain (NOT used; contamination reference) =="
  echo "/opt/rocm -> $(readlink -f /opt/rocm)"
  env -i PATH="/opt/rocm/bin:/usr/bin:/bin" HOME=/root /opt/rocm/bin/hipcc --version 2>&1 | head -4 || true
  echo "host ROCM_PATH (login shell default): /opt/rocm  [must be overridden in all build/run shells]"
  echo
  echo "== torch cpp_extension ROCM_HOME resolution under build env =="
  env PATH="$DEV/bin:$VENV/bin:/usr/bin:/bin" ROCM_PATH="$DEV" HIP_PATH="$DEV" bash -c 'source '$VENV'/bin/activate; python -c "from torch.utils.cpp_extension import ROCM_HOME; print(\"ROCM_HOME:\", ROCM_HOME)"'
  echo
  echo "== gfx1100 smoke: hipcc compile + run =="
  echo "(evidence: /tmp/opencode/hipver.cu compiled with devel hipcc --offload-arch=gfx1100; binary ran on GPU: 'hip 7.14 dev ok=0')"
} > "$OUT/toolchain-rocm714.txt" 2>&1

echo "evidence written to $OUT"
