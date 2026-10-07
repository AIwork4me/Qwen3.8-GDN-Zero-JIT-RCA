## Overview

Adds e2e zero-runtime-JIT regression coverage for the Qwen3.5 GDN (Gated DeltaNet
linear-attention) path on ROCm: a second test in
`tests/jit_monitor/test_no_runtime_jit_rocm.py` plus the matching AMD MI355
mirror timeout bump in `.buildkite/test_areas/jit_monitor.yaml`.

## Claims

- The only existing ROCm zero-JIT e2e test boots a dense model (Qwen3-0.6B),
  which never executes Qwen GDN kernels — a Qwen GDN warmup regression would
  pass today's ROCm jit-monitor CI and only surface as first-inference JIT on
  user GDN deployments (the GDN warmup area is regression-prone, e.g. the RDNA
  warmup fix #45000).
- The new test `test_qwen_gdn_no_runtime_jit_rocm` boots Qwen3.5-0.8B (dummy
  weights, two-layer hybrid truncation: one GDN + one full-attention layer) in
  the default graph configuration with `jit_monitor_mode="error"`, fresh
  per-test caches, and the shared prefill/decode/sampler battery; it asserts
  the monitor is armed before inference and that the booted model actually
  contains GDN layers, so a GDN-less boot cannot pass green.
- It runs the generic in-tree Triton/FLA GDN path (`VLLM_ROCM_USE_AITER=0`),
  so the regression does not depend on optional AITER availability.

## Validation

On current main `3ca00a82`, ROCm 7.14, gfx1100 (W7900), dummy weights:

- `pytest tests/jit_monitor/test_no_runtime_jit_rocm.py::test_v2_sampler_warmup_rocm` — PASS (131 s)
- `pytest tests/jit_monitor/test_no_runtime_jit_rocm.py::test_qwen_gdn_no_runtime_jit_rocm` — PASS (279 s; monitor `(True, "error")` and GDN layer count > 0 asserted in-test)
- full file `tests/jit_monitor/test_no_runtime_jit_rocm.py` — 2/2 PASS (394 s)
- `ruff check` / `ruff format --check` clean; yaml parses; `pytest --collect-only` 2 tests

Sensitivity (disposable local mutation, reverted): disabling the Qwen GDN warmup
makes the test **fail** with
`RuntimeError: Triton kernel JIT compilation during inference: _fused_post_conv_kernel`;
reverting restores PASS. In-process proof on the same main: architecture
`Qwen3_5ForConditionalGeneration`, `layer_types=[linear_attention, full_attention]`,
all six GDN Triton kernel families present in the startup cache, 0 new cache
artifacts after the battery.

First MI355 run in CI is the authoritative hardware signal for the new timeout
budget (per-test 300→600 s, job 10→25 min for the second engine boot).

## Details

- Scope: Qwen3.5-0.8B on the generic Triton/FLA path only. Non-goals: other Qwen
  GDN variants (Qwen3-Next, Qwen3.8, Qwen4-Exp), AITER coverage, eager mode,
  speculative decoding, any runtime/kernel/warmup change.
- On ROCm the monitor raises on inference-time Triton JIT compiles; autotune
  misses are print-only and inductor/cudagraph-capture compiles are unmonitored.
- Shared boot/cache-isolation setup is factored into
  `_run_rocm_shape_battery(model, hf_overrides)` / `_isolate_rocm_jit_caches(...)`;
  the existing dense test's behavior is unchanged (same model, kwargs, env, order).
- The GDN-presence assertion reuses the warmup's own `_iter_qwen_gdn_layers`
  predicate over `compilation_config.static_forward_context`, so config/registry
  drift that stops constructing GDN layers fails the test instead of passing
  vacuously.
- Duplicate-work check (2026-10-07): no open/draft/recently-merged PR adds ROCm
  Qwen3.5 GDN zero-JIT regression coverage; `tests/jit_monitor/` on main contains
  only the dense ROCm test. Nearby GDN PRs are kernel/perf/numerics work without
  `jit_monitor_mode="error"` e2e coverage.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>
