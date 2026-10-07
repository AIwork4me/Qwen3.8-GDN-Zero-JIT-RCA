# Phase-2 PR-1 — Duplicate-Work Audit (2026-10-07)

Independent subagent search record (verbatim verdict + evidence; full
prose record also reflected in ../duplicate-search.md).

## Verdict

CLEAR TO IMPLEMENT

## Evidence

1. `tests/jit_monitor/test_no_runtime_jit_rocm.py` (main @ 68088ed3) —
   contains exactly one test, `test_v2_sampler_warmup_rocm`, booting dense
   `Qwen/Qwen3-0.6B` with `jit_monitor_mode="error"`,
   `VLLM_ROCM_USE_AITER=0`, fresh caches, asserting the monitor is armed and
   running `_run_shape_battery`. No GDN model. The GitHub commits API shows
   the only commit ever touching this file is 96e203b (PR #58092, merged
   2026-09-23, "[ROCm][Bugfix] Register MRV2 sampler JIT warmups"), so no
   open PR modifies it.
2. `tests/jit_monitor/test_no_runtime_jit.py` (main) — `JIT_MONITOR_MODELS`
   contains no GDN architecture (Qwen3-0.6B dense, DeepSeek-V2-Lite/V3,
   granite-4.0-tiny, DS-MTP drafts); the entire module is skipped
   (`pytest.mark.skip`) pending warmup migrations tracked in #49349. Grep
   of `tests/jit_monitor/` for gdn|GDN|QwenGatedDeltaNet|qwen3_5|Qwen3_5|
   qwen3_next → zero matches.
3. Grep of all jit_monitor usages in tests/ (main) — appears only in
   `tests/jit_monitor/*`, `tests/engine/test_arg_utils.py`,
   `tests/v1/worker/test_gpu_worker.py`, and tilelang check scripts. None
   involve a GDN model. Qwen/Qwen3.5-0.8B appears elsewhere (registry,
   mamba/GDN kernel tests, attention tests, incl. ROCm-specific
   `test_gdn_rocm_layout_dispatch.py`) but never in a jit-monitor context.
4. Open PR #56693 (2026-09-13, "[Kernel][CI] Enable --jit-monitor-mode
   error on CI for DSv4, Gemma4 and gpt-oss") rewrites
   `test_no_runtime_jit.py`, removes the skip, replaces the model list —
   no GDN model, does not touch the ROCm file.
5. Merged GDN-warmup PRs #54251 (Qwen GDN gated RMSNorm warmup, 2026-09-03)
   and #54797 (extend Qwen Triton warmup, 2026-09-07) add unit tests of
   warmup modules; neither adds an e2e zero-JIT/jit-monitor regression test
   nor anything ROCm-specific. Other inspected open PRs (#59260, #59976,
   #48363, #51134, #50848) touch no `tests/jit_monitor/` files.
6. Issue #49349 (OPEN since 2026-07-21, updated 2026-10-05): a 2026-08-28
   comment (@zupengwang) reports a remaining inference-time Triton JIT in
   the Qwen3.5 GDN output-projection path on CUDA main — GDN zero-JIT is a
   live, uncovered concern; no ROCm GDN test or CI request in the thread.
   Issue #52663 (OPEN, rocm): unrelated RDNA3 FP8 tuned-config timeout; no
   jit-monitor/GDN content.
7. CI: `.buildkite/test_areas/jit_monitor.yaml` runs the ROCm file on the
   MI355 DPX pool with directory-wide `source_file_dependencies`; a new
   test in the same file is scheduled automatically.

## Reasoning

No equivalent ROCm Qwen GDN zero-runtime-JIT test exists in upstream main;
the generic file is skipped and lists no GDN architecture; no open PR adds
one. The nearest work is warmup-unit tests, not e2e jit-monitor coverage.
The new test extends the existing ROCm harness with GDN-specific
incremental coverage and runs in the existing CI lane.
