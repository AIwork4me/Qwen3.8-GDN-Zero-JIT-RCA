# Final PR description (as submitted)

- Upstream PR: https://github.com/vllm-project/vllm/pull/60395 (DRAFT)
- Submitted body: pr-body-upstream.md (repo PR template format)
- Fork head: 9e370fac11bd6fe3a5a92d762318b674e80af827 (tree 48f1d43b)
- The narrative text below is retained for the RCA record; the template-form
  pr-body-upstream.md is the verbatim submission.

## Title

[ROCm][Test] Add Qwen3.5 GDN zero-runtime-JIT regression coverage

## Body

### Problem

ROCm currently has one e2e zero-runtime-JIT regression test
(`tests/jit_monitor/test_no_runtime_jit_rocm.py`), and it boots a **dense** model
(Qwen3-0.6B). Dense Qwen never executes the Qwen GDN (Gated DeltaNet
linear-attention) kernels, so a regression in the Qwen GDN Triton warmup would
pass today's ROCm jit-monitor CI and only surface as first-inference JIT
compilation on user GDN deployments (the GDN warmup area is regression-prone —
e.g. the RDNA warmup fix #45000).

### What this PR adds

A second test in the same file, `test_qwen_gdn_no_runtime_jit_rocm`:

- boots **Qwen/Qwen3.5-0.8B** with `load_format="dummy"` and a two-layer hybrid
  truncation (`layer_types=["linear_attention", "full_attention"]` — one GDN +
  one full-attention layer, the minimal layout that still exercises the hybrid
  KV-cache manager Qwen3.5 is served with);
- default graph configuration (`enforce_eager=False`) with
  `jit_monitor_mode="error"`;
- asserts the JIT monitor is armed in the worker before inference
  (`(is_active(), mode) == (True, "error")`) and that the booted model actually
  contains GDN layers (reusing the warmup's own `_iter_qwen_gdn_layers`
  predicate), so a GDN-less boot cannot pass green;
- fresh per-test `TRITON_CACHE_DIR` / `VLLM_CACHE_ROOT` /
  `TORCHINDUCTOR_CACHE_DIR` and a spawned engine core, so a previous run's
  caches cannot hide a missing warmup specialization;
- runs the shared prefill/decode/sampler shape battery from
  `test_no_runtime_jit.py`;
- runs the generic in-tree Triton GDN path (`VLLM_ROCM_USE_AITER=0`), so the
  regression does not depend on optional AITER availability — AITER GDN coverage
  is left to future work.

The shared runner/cache-isolation setup used by both tests is factored into
`_run_rocm_shape_battery(model, hf_overrides)` / `_isolate_rocm_jit_caches(...)`;
the existing dense test's behavior is unchanged.

CI: the AMD MI355 mirror job in `.buildkite/test_areas/jit_monitor.yaml` now
boots two models, so its per-test timeout goes 300 s → 600 s and the job
timeout 10 → 25 min.

### Why this test is meaningful

A local disposable mutation that disables the Qwen GDN warmup makes the new
test **fail** on inference-time Triton JIT (`jit_monitor_mode="error"` raise);
reverting the mutation makes it pass again. I.e. the test is sensitive to
exactly the regression class it targets.

### Validation

(all on current main `3ca00a8261c7790ba7487e2001ab9004ad8065f2`, ROCm 7.14, gfx1100/W7900, dummy weights)

- `pytest tests/jit_monitor/test_no_runtime_jit_rocm.py::test_v2_sampler_warmup_rocm` — PASS (131 s)
- `pytest tests/jit_monitor/test_no_runtime_jit_rocm.py::test_qwen_gdn_no_runtime_jit_rocm` — PASS (279 s;
  monitor `(True, "error")` and GDN layer count > 0 asserted in-test)
- full file `tests/jit_monitor/test_no_runtime_jit_rocm.py` — 2/2 PASS (394 s total)
- negative control (Qwen GDN warmup disabled in a disposable local mutation) — test FAILS
  with `RuntimeError: Triton kernel JIT compilation during inference: _fused_post_conv_kernel`;
  revert → PASS (267 s)
- `ruff check` / `ruff format --check` (ruff 0.14.0) clean; yaml parses; `pytest --collect-only` 2 tests
- in-process proof on the same main: architecture `Qwen3_5ForConditionalGeneration`,
  `layer_types=[linear_attention, full_attention]`, 1 GDN layer, all six GDN Triton kernel
  families present in the startup cache, 0 new cache artifacts after the battery

### Scope / non-goals

- tests + CI config only; no runtime, kernel, or warmup changes
- covers Qwen3.5-0.8B only; other Qwen GDN variants (Qwen3-Next, Qwen3.8,
  Qwen4-Exp) are not covered
- no AITER coverage, no eager-mode coverage, no speculative decoding
- on ROCm the monitor raises on inference-time Triton JIT compiles; autotune
  misses are print-only and inductor/cudagraph-capture compiles are unmonitored
- not a fix for any open issue; additive regression protection

### Duplicate-work check

Searched open/draft/recently-merged PRs and current `tests/jit_monitor/`
(2026-10-07): no existing ROCm Qwen3.5 GDN zero-JIT regression coverage; the ROCm
jit-monitor file contains only the dense Qwen3-0.6B test. Nearby GDN PRs are
kernel/perf/numerics work without `jit_monitor_mode="error"` e2e coverage.

### AI disclosure

Implementation and validation were AI-assisted under human direction
(AIwork4me); every changed line and all validation results were reviewed by the
submitter.
