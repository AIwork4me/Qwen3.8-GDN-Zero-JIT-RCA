# 02 — Experiment Environment

Updated: 2026-10-07 (session continuing from checkpoint `3d331dc`).
Raw evidence: `evidence/environment/*-rocm714.txt`.

## Host

```text
GPU        : AMD Radeon Pro W7900D (card model 0x744b), gfx1100, 48 GiB
Driver     : 6.16.13
CPU        : AMD EPYC 9334 (128 threads)
RAM        : 503 GiB
OS         : Ubuntu 24.04.4 LTS, kernel 6.8.0-79-generic
Host ROCm  : /opt/rocm -> 7.2.1 (PRE-EXISTING; used ONLY for rocm-smi/rocminfo host tools)
```

## Experiment venv (canonical)

```text
venv             : /workspace/venv-qwen-gdn-rca
python           : 3.12.3 (/workspace/venv-qwen-gdn-rca/bin/python)
torch            : 2.14.1+rocm7.14  (official index https://download.pytorch.org/whl/rocm7.14)
torch.version.hip: 7.14.60850
torchvision      : 0.29.1+rocm7.14
triton           : triton-rocm 3.8.0 (AMD backend present)
rocm SDK (pip)   : rocm==7.14.0.post1, rocm-sdk-core/libraries/device-gfx1100/devel 7.14.0
GPU smoke        : bf16 matmul OK on cuda:0 (gfx1100)
triton smoke     : @triton.jit kernel compiled + executed on gfx1100 (cache-isolated)
hipcc smoke      : HIP C++ kernel compiled --offload-arch=gfx1100 + executed (devel hipcc)
```

## Source-build ROCm toolchain resolution

### Discovery

1. Host `/opt/rocm` = ROCm **7.2.1** (`hipcc 7.2.53211`) — pre-existing, FIRST on the
   default login `PATH`, and the login shell exports `ROCM_PATH=/opt/rocm`,
   `LD_LIBRARY_PATH=.../opt/rocm/lib`. Using it to build vLLM against a
   ROCm-7.14 PyTorch would produce exactly the mixed toolchain forbidden by
   Blocker B.
2. `rocm-sdk-core/libraries/device-gfx1100` 7.14.0 (pip) provide the *runtime*
   (libamdhip64, hip runtime, gfx1100 ISA code, headers) inside site-packages
   but ship **no compiler**.
3. The compiler lives in the separate **`rocm-sdk-devel` 7.14.0** wheel
   (529,800,802 bytes, fetched from `https://repo.amd.com/rocm/whl-multi-arch/`,
   zip-integrity verified). Installed and expanded via `python -m rocm_sdk init`
   into `.../site-packages/_rocm_sdk_devel` providing:

```text
bin/hipcc, bin/amdclang++, bin/hipconfig
lib/llvm/bin/clang-23 (AMD clang 23.0.0git, roc-7.14 toolchain)
include/ (hip + rocm headers 7.14)
lib/ (libamdhip64 7.14, etc.)
lib/cmake/* (hip, hipblas, rocprim, miopen, rccl, ... CMake configs)
```

4. **Contamination vector proven**: with the default login environment, the
   devel-tree `hipcc --version` reports `InstalledDir: /opt/rocm-7.2.1/lib/llvm/bin`
   (7.2 clang). With an isolated environment
   (`PATH=$DEV/bin:...`, `ROCM_PATH=$DEV`, `HIP_PATH=$DEV`,
   no `/opt/rocm` on `LD_LIBRARY_PATH`), `hipcc --version` reports
   `HIP version 7.14.60850` / `AMD clang 23.0.0git` with
   `InstalledDir: .../_rocm_sdk_devel/lib/llvm/bin` — coherent.
5. End-to-end proof: a HIP source kernel compiled with the isolated devel
   `hipcc --offload-arch=gfx1100`, linked against devel `lib/`, executed on the
   W7900 (`/tmp/opencode/hipver.cu` → `hip 7.14 dev ok=0`).
6. `torch.utils.cpp_extension.ROCM_HOME` resolves to `/opt/rocm` under the
   default shell (Blocker B trap confirmed live) and to the devel tree when
   `ROCM_PATH=$DEV` is exported — the vLLM build env must always set it.

### Classification

```text
COHERENT ROCm 7.14
```

with the mandatory build/run environment contract:

```bash
DEV=/workspace/venv-qwen-gdn-rca/lib/python3.12/site-packages/_rocm_sdk_devel
export PATH="$DEV/bin:/workspace/venv-qwen-gdn-rca/bin:/usr/bin:/bin"
export ROCM_PATH="$DEV"
export HIP_PATH="$DEV"
export LD_LIBRARY_PATH="$DEV/lib"          # NEVER /opt/rocm/lib
export CMAKE_PREFIX_PATH="$DEV"
export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH=gfx1100
```

## Environment incidents and repairs (2026-10-07, evidence preserved)

### Incident 1 — torch was actually a CUDA wheel (manifest lied)

Fresh collection revealed the previously-committed manifest (torch 2.14.1+rocm7.14)
did NOT match the venv: `torch 2.14.1+cu130`, `torch.version.hip: None`,
`device_count: 0`. The 1.39 GB official rocm7.14 wheel was already present in
`/workspace/downloads` (integrity OK). Repair: forced reinstall
(`--no-deps`) → torch 2.14.1+rocm7.14, HIP 7.14.60850, gfx1100 visible,
bf16 matmul verified.

### Incident 2 — `pip install -r requirements/rocm.txt` replaced torch again

pip resolved `xgrammar==0.2.8 → torch → rocm[device-all,libraries]==7.14.*`
(the official rocm7.14 torch wheel metadata requires the `device-all` extra,
i.e. all 25 ISA device packages). Our intentionally minimal install has only
`rocm-sdk-device-gfx1100`, so pip backtracked and **silently replaced torch
with the vanilla PyPI 2.14.1 (CUDA cu130) build**, dragging in 15 `nvidia-*`
packages and a CUDA `triton 3.8.0` that shadowed `triton-rocm`.

Repairs (all verified in `evidence/environment/packages-rocm714.txt`):

1. reinstalled `torch 2.14.1+rocm7.14` (`--no-deps`, local wheel);
2. uninstalled all 15 `nvidia-*` packages + the CUDA `triton`;
3. reinstalled `triton-rocm 3.8.0` from the local wheel (the uninstall had
   removed shared `triton/` files); AMD backend present; gfx1100 kernel
   compile+run verified;
4. **documented metadata workaround**: patched the *installed*
   `torch-2.14.1+rocm7.14.dist-info/METADATA` line
   `Requires-Dist: rocm[device-all,libraries]==7.14.*`
   →
   `Requires-Dist: rocm[device-gfx1100,libraries]==7.14.*`
   to stop pip from re-triggering this on every resolution. The wheel binary
   is untouched; the change makes installed metadata describe the actual
   single-gfx1100 environment. Known residue: `pip check` reports
   `xgrammar 0.2.8 requires triton, which is not installed` because
   triton-rocm does not `Provides-Dist: triton` — accepted and documented;
   importing `triton` works.
5. All later pip operations use `--no-deps` plus explicit verification.

### wheels.vllm.ai precompiled path — unavailable on this host

Current main supports `VLLM_USE_PRECOMPILED=1` installs from
`https://wheels.vllm.ai/rocm/<commit>/<variant>/` (setup.py,
`precompiled_wheel_utils`). The host egress proxy answers
`CONNECT wheels.vllm.ai:443` with **403 Forbidden** (proxy policy, origin never
reached) → the prebuilt-wheel path is not usable here; recorded as a deviation.
Consequence: vLLM must be built from source with the coherent devel toolchain
above (which also removes any Blocker-B ambiguity from the build).

## Gate check

`scripts/verify_environment_gate.py` asserts the whole contract
(venv interpreter, torch HIP 7.14.x, gfx1100, triton-rocm, toolchain
isolation, vLLM import origin). Raw output:
`evidence/environment/gate-check.txt`.
