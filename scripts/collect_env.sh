#!/bin/bash
# Collect full environment evidence (sanitized) into a directory given by $1.
set -u
OUT="${1:-.}"
mkdir -p "$OUT"
{
  echo "=== date ==="; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "=== uname ==="; uname -a
  echo "=== os-release ==="; cat /etc/os-release 2>/dev/null
  echo "=== lscpu ==="; lscpu
  echo "=== mem ==="; free -h
  echo "=== disk ==="; df -h /workspace
  echo "=== rocminfo (agent section) ==="; rocminfo 2>/dev/null | sed -n '1,120p'
  echo "=== rocm-smi ==="; rocm-smi 2>&1
  echo "=== rocm-smi product ==="; rocm-smi --showproduct --showdriverversion 2>&1
  echo "=== hipcc ==="; hipcc --version 2>&1 || true
  echo "=== /opt/rocm ==="; ls -l /opt/rocm 2>&1 || true
  echo "=== rocm version file ==="; cat /opt/rocm/.info/version 2>/dev/null || true
} > "$OUT/hardware.txt" 2>&1
{
  echo "=== env (sorted, sanitized) ==="
  env | sort | sed -E 's/(TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL|COOKIE)=.*/\1=<REDACTED>/I'
} > "$OUT/env-sanitized.txt"
python - <<'PY' > "$OUT/python.txt" 2>&1
import sys
print("python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.version.hip:", torch.version.hip)
    print("cuda_available:", torch.cuda.is_available())
    print("device_count:", torch.cuda.device_count())
    if torch.cuda.device_count():
        p = torch.cuda.get_device_properties(0)
        print("device_name:", torch.cuda.get_device_name(0))
        print("gcnArchName:", getattr(p, "gcnArchName", None))
        print("total_memory:", p.total_memory)
    print("cuda version:", torch.version.cuda)
except Exception as e:
    print("torch import failed:", e)
try:
    import triton
    print("triton:", triton.__version__)
except Exception as e:
    print("triton import failed:", e)
try:
    import vllm
    print("vllm:", vllm.__version__, vllm.__file__)
except Exception as e:
    print("vllm import failed:", e)
PY
echo "collected into $OUT"
