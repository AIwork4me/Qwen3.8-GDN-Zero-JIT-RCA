# Phase-2 PR-1 — PR Draft (NOT SUBMITTED)

## Title

[ROCm][Test] Add Qwen GDN zero-runtime-JIT regression coverage

## Problem

ROCm currently has a single e2e zero-runtime-JIT regression test
(`tests/jit_monitor/test_no_runtime_jit_rocm.py::test_v2_sampler_warmup_rocm`),
which boots a dense model (Qwen3-0.6B) and asserts the MRV2 sampler warmup
registry covers the standard shape battery under `jit_monitor_mode="error"`.
It does not execute any Gated DeltaNet (GDN) code: the GDN linear-attention
kernels (causal-conv update, gated-delta recurrent decode, FLA chunk prefill,
fused post-conv) are only reachable through `QwenGatedDeltaNetAttention`
layers, which dense models never construct. A warmup regression limited to
the GDN path (e.g. losing the `qwen_triton_warmup` registrations) would
therefore pass today's CI and only surface as first-inference JIT on user
GDN deployments.

## Why this coverage matters

The zero-JIT-during-inference property is the acceptance gate tracked in
#49349, checked with `--jit-monitor-mode error`. GDN models are a
JIT-heavy path with a documented history of runtime JIT reports (e.g. the
Qwen3.5 GDN gated-RMSNorm `layer_norm_fwd_kernel` report in #49349,
addressed by #54251). The generic `test_no_runtime_jit.py` is module-level
skipped pending warmup migrations, and its model list contains no GDN
architecture — so nothing in CI today protects GDN zero-JIT on any backend.
This PR closes that gap for ROCm, the backend with the live jit-monitor CI
lane.

## Existing coverage gap

- `test_no_runtime_jit_rocm.py::test_v2_sampler_warmup_rocm` — dense
  Qwen3-0.6B only; no GDN kernels compiled.
- `test_no_runtime_jit.py` — `pytest.mark.skip` (whole module), no GDN model
  in `JIT_MONITOR_MODELS`.
- Merged GDN warmup PRs (#54251, #54797) added warmup-unit tests, not e2e
  jit-monitor regression coverage.

## What this PR adds

A second test in `tests/jit_monitor/test_no_runtime_jit_rocm.py`:

- `test_qwen_gdn_no_runtime_jit_rocm` boots `Qwen/Qwen3.5-0.8B`
  (18 linear-attention + 6 full-attention layers) with dummy weights,
  truncated to a two-layer hybrid (`["linear_attention",
  "full_attention"]`) for CI cost;
- `enforce_eager=False` (default graph configuration), fresh
  `TRITON_CACHE_DIR`/`VLLM_CACHE_ROOT`/`TORCHINDUCTOR_CACHE_DIR` per test;
- asserts the JIT monitor is armed `(True, "error")` in the worker before
  inference, then runs the shared `_run_shape_battery`;
- any inference-time Triton compilation raises and fails the test.

Shared plumbing with the existing test is factored into
`_run_rocm_shape_battery(model, hf_overrides)` and
`_isolate_rocm_jit_caches(...)`; the existing dense test's behavior is
unchanged. The MI355 CI step label/timeout is updated (per-test 600 s,
job 25 min) to fit the second boot.

This PR is tests-only: no runtime code, kernels, warmup logic, or defaults
change.

## Why Qwen3.5-0.8B

Smallest public Qwen GDN checkpoint; `layer_types` is
`linear_attention`-dominant, so even the truncated two-layer dummy model
constructs `QwenGatedDeltaNetAttention` and compiles/executes the GDN kernel
families (conv1d update, packed recurrent decode, FLA chunk prefill,
fused post-conv, gated RMSNorm). GDN kernels specialize on shapes/dtypes,
not weights, so dummy loading does not weaken the property.

## ROCm execution path

`VLLM_ROCM_USE_AITER=0` selects vLLM's in-tree generic Triton GDN path —
the same deterministic configuration as the existing ROCm jit-monitor test —
so the regression does not depend on optional AITER availability. The AITER
GDN path is explicitly out of scope for this test.

## Validation

Upstream base: main @ `68088ed3` (build from source, ROCm 7.14, gfx1100
W7900D, torch 2.14.1+rocm7.14, triton-rocm 3.8.0):

- `test_v2_sampler_warmup_rocm` (baseline, unchanged behavior): PASS, 137 s
- `test_qwen_gdn_no_runtime_jit_rocm`: PASS, 263–278 s (cold cache, monitor
  armed, 0 raises)
- full file: 2 passed, 392 s
- ruff check + format: clean
- GDN execution verified in-process: model contains
  `QwenGatedDeltaNetAttention`; startup Triton cache contains the GDN
  families; zero cache files created after the battery starts (disk-level
  zero-runtime-JIT confirmation)

## Negative-control sensitivity

A disposable local mutation (early-return in
`vllm/model_executor/warmup/qwen_triton_warmup.py`, i.e. GDN warmup lost)
makes the new test FAIL with exactly the protected error:

```text
RuntimeError: Triton kernel JIT compilation during inference:
_fused_post_conv_kernel. This causes a latency spike; consider extending
warmup to cover this shape/config.
```

The mutation was reverted; the clean tree passes again. This demonstrates
the test detects missing GDN warmup coverage rather than merely asserting
model startup.

Known scope limit (honesty): decode-side kernels whose only needed
specializations fall inside FULL-graph capture sizes can be covered by
capture rather than the warmup registry; this test pins the combined
startup coverage (warmup + capture) of the battery's compile keys, which is
the user-visible property.

## Scope

- ROCm only (mirrors the existing ROCm jit-monitor CI lane on MI355).
- Generic non-AITER GDN path.
- Language-only battery prompts (dummy weights).

## Non-goals

- Not a fix for any runtime-JIT bug: current main passes this test.
- No AITER GDN coverage; no CUDA lane (the generic test file is skipped
  pending warmup migrations; enabling it is a separate effort).
- No speculative-decoding, structured-output, or multimodal GDN coverage.
- No changes to kernels, warmup registration, dispatch, or defaults.

## AI assistance disclosure

Test authored with AI assistance (AMD ROCm infra investigation workflow);
validated on gfx1100 with sensitivity/negative-control evidence as
described above.
