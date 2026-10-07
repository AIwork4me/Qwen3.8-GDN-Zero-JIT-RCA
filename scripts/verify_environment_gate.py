#!/usr/bin/env python3
"""Phase-1 environment gate.

Fails unless the running interpreter proves the coherent ROCm 7.14 / gfx1100 /
pinned-vLLM experiment contract:

  * python executable is inside /workspace/venv-qwen-gdn-rca
  * torch is 2.14.1+rocm7.14 with torch.version.hip == 7.14.*
  * GPU 0 is gfx1100
  * triton is triton-rocm (AMD backend present, NVIDIA-only driver absent)
  * (optional, after vLLM install) vllm imports from /workspace/vllm-zero-jit-rca
  * (optional, with --build-env) the build toolchain is the venv devel tree,
    not /opt/rocm 7.2
"""
import importlib.metadata as md
import os
import sys

VENV = "/workspace/venv-qwen-gdn-rca"
VLLM_SRC = "/workspace/vllm-zero-jit-rca"
DEV = f"{VENV}/lib/python3.12/site-packages/_rocm_sdk_devel"

fails: list[str] = []
notes: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


py = sys.executable
check(py.startswith(VENV + "/"), f"python inside RCA venv (got {py})")

import torch  # noqa: E402

check(torch.__version__ == "2.14.1+rocm7.14", f"torch version (got {torch.__version__})")
hip = torch.version.hip
check(hip is not None and hip.startswith("7.14."), f"torch HIP 7.14.x (got {hip})")
check(torch.cuda.is_available(), "torch.cuda.is_available()")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    arch = getattr(p, "gcnArchName", "")
    check(arch.startswith("gfx1100"), f"device 0 gcnArchName gfx1100 (got {arch})")

import triton  # noqa: E402

check(triton.__version__.startswith("3.8"), f"triton 3.8.x (got {triton.__version__})")
backends = os.listdir(os.path.join(os.path.dirname(triton.__file__), "backends"))
check("amd" in backends, f"triton amd backend present (backends={backends})")
try:
    md.distribution("triton")
    check(False, "CUDA 'triton' distribution must NOT be installed alongside triton-rocm")
except md.PackageNotFoundError:
    check(True, "no CUDA 'triton' distribution (only triton-rocm)")
try:
    v = md.version("triton-rocm")
    check(v == "3.8.0", f"triton-rocm 3.8.0 (got {v})")
except md.PackageNotFoundError:
    check(False, "triton-rocm installed")

nv = [d.metadata["Name"] for d in md.distributions() if (d.metadata["Name"] or "").startswith("nvidia-")]
check(not nv, f"no nvidia-* contamination (found {nv})")

rocm_compiler_ok = os.path.isfile(f"{DEV}/bin/hipcc")
check(rocm_compiler_ok, "venv devel hipcc present (rocm-sdk-devel 7.14.0)")
if rocm_compiler_ok and "--build-env" in sys.argv:
    from torch.utils.cpp_extension import ROCM_HOME

    check(str(ROCM_HOME) == DEV, f"cpp_extension ROCM_HOME is devel tree (got {ROCM_HOME})")
    cc = os.environ.get("CC", "")
    if cc:
        check("/opt/rocm" not in cc, f"CC not pointing into /opt/rocm (got {cc})")

try:
    import vllm  # noqa: E402

    origin = os.path.dirname(vllm.__file__)
    check(
        origin.startswith(VLLM_SRC),
        f"vllm imports from pinned source (got {origin})",
    )
    check(not origin.startswith("/opt/venv"), "vllm NOT from old /opt/venv")
except ImportError:
    notes.append("vllm not installed yet (pre-build state)")

print()
for n in notes:
    print("NOTE", n)
print("ENVIRONMENT GATE:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
