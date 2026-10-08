# Segfault root cause — `import vllm` on fresh image (2026-10-08)

## Symptom

`import vllm` (and therefore pytest collection of any test importing vllm)
segfaults deterministically (8/8) on the freshly rebuilt ROCm 7.14 venv,
in `triton/knobs.py:15` (`from triton._C.libtriton import getenv`), i.e.
during the C-level initialization of triton-rocm's `libtriton.so`.

## Mechanism (isolated experimentally)

- `vllm/env_override.py:148` calls `_maybe_promote_torch_symbols_for_rocm()`
  before `import torch` (upstream #56190, landed 2026-09-10, present since
  before the previously validated base 3ca00a82; NOT part of the 33-commit
  drift). On ROCm it does `ctypes.CDLL(torch/lib/libtorch_cpu.so,
  mode=RTLD_GLOBAL)`.
- Direct repro A: `ctypes.CDLL(libtorch_cpu, RTLD_GLOBAL); import triton.knobs`
  -> SEGFAULT (no output).
- Control B: `import triton.knobs` alone -> OK.
- Control C: `ctypes.CDLL(libtorch_cpu, RTLD_LOCAL); import triton.knobs` -> OK.
- Fix D: `import triton.knobs; import vllm` -> OK 3/3.

Conclusion: promoting libtorch_cpu's symbols into the global scope before
libtriton.so's static initializers run corrupts the later load (classic
mixed-static-init clash; both objects carry large statically-linked C++
runtime/LLVM-ish symbol sets). Importing triton first loads libtriton in a
clean RTLD_LOCAL scope and the subsequent promote is harmless.

The same code existed on the previous (deleted) image where validation at
3ca00a82 passed; the clash is a fresh-image layout/library-resolution
interaction (exact trigger difference not identifiable post-hoc — the
previous image no longer exists). No upstream file in the drift range
changed this path (env_override.py: 0 commits since 3ca00a82).

## Mitigation (environment-level, no upstream changes)

`/workspace/venv-qwen-gdn-rca/lib/python3.12/site-packages/zz_triton_knobs_first.pth`
contains a single executed line `import triton.knobs`, which runs during
site initialization for every process in this venv (a plain
`sitecustomize.py` in the venv is shadowed by Ubuntu's
/usr/lib/python3.12/sitecustomize.py, which precedes venv site-packages on
sys.path — the .pth mechanism is the canonical venv-level startup hook and
runs before sitecustomize is imported). This replicates the safe import
order for pytest and all spawned engine-core children without touching the
PR diff, the vLLM source tree, or test semantics. vLLM imports triton in
its normal chain regardless, so no behavior changes beyond load order.

## Impact on evidence validity

- The jit-monitor semantics under test (activate/is_active/error-raise) are
  Python-level and unaffected by shared-object load order.
- Negative-control sensitivity (armed monitor raises on runtime JIT) is
  unaffected.
- This is recorded as an environment workaround, not an upstream fix; a
  separate upstream investigation may be warranted but is out of scope for
  PR #60395 (tests+CI only).
