# Phase-2 PR-1 — Duplicate-Work Search (HARD GATE)

Date: 2026-10-07
Base: upstream main @ 68088ed3927e91bce8918db78f2efd9e644a4134
Tip at preparation: 43b4aaea3e40e20ef53ce2bfecfa1b72060bf2b5 (target
jit_monitor test files byte-identical between both)

## Verdict

**CLEAR TO IMPLEMENT**

## Search performed (independent subagent + local grep)

1. `tests/jit_monitor/test_no_runtime_jit_rocm.py` (main @ 68088ed3): exactly
   one test, `test_v2_sampler_warmup_rocm`, dense `Qwen/Qwen3-0.6B`,
   `jit_monitor_mode="error"`, `VLLM_ROCM_USE_AITER=0`, fresh tmp caches,
   `(True, "error")` monitor-state assertion, shared `_run_shape_battery`.
   No GDN model. Only commit ever touching the file: 96e203b (PR #58092,
   merged 2026-09-23, "[ROCm][Bugfix] Register MRV2 sampler JIT warmups").
   No open PR modifies it.
2. `tests/jit_monitor/test_no_runtime_jit.py`: `JIT_MONITOR_MODELS` has no
   GDN architecture (Qwen3-0.6B dense, DeepSeek-V2-Lite/V3, granite-4.0-tiny,
   DS-MTP drafts); whole module `pytest.mark.skip` pending #49349 warmup
   migrations. `grep -r 'gdn|QwenGatedDeltaNet|qwen3_5|qwen3_next'
   tests/jit_monitor/` → zero matches.
3. jit_monitor usage across tests/: only jit_monitor/*, engine/test_arg_utils,
   v1/worker/test_gpu_worker, kernels tilelang checks — none GDN.
4. Open PR #56693 (2026-09-13, "[Kernel][CI] Enable --jit-monitor-mode error
   on CI for DSv4, Gemma4 and gpt-oss"): rewrites the generic model list
   (DeepSeek-V4, Gemma-4, gpt-oss) — no GDN, does not touch the ROCm file.
5. Merged GDN warmup PRs #54251 (Qwen GDN gated RMSNorm warmup) and #54797
   (extend Qwen Triton warmup) add warmup-unit tests, not e2e jit-monitor
   regression coverage; not ROCm-specific. Other inspected open PRs
   (#59260, #59976, #48363, #51134, #50848) touch no tests/jit_monitor files.
6. Issue #49349 (OPEN, tracking): 2026-08-28 comment (@zupengwang) reports a
   remaining inference-time Triton JIT in the Qwen3.5 GDN output-projection
   path on CUDA main — GDN zero-JIT is a live uncovered concern; no ROCm GDN
   test/CI request anywhere in the thread. Issue #52663 (OPEN, rocm): FP8
   tuned-config timeout; no jit-monitor/GDN test content.
7. CI: `.buildkite/test_areas/jit_monitor.yaml` runs
   `pytest tests/jit_monitor/test_no_runtime_jit_rocm.py` on the MI355 DPX
   pool with directory-wide `source_file_dependencies`; a new test in the
   same file is scheduled automatically — no pipeline changes needed.

## Conclusion

The only ROCm zero-JIT e2e regression test covers the dense-model sampler
warmup path. No GDN (QwenGatedDeltaNet / qwen3_5 / qwen3_next) coverage
exists or is being contributed in any jit-monitor context. Adding a
Qwen3.5-0.8B GDN test to `tests/jit_monitor/test_no_runtime_jit_rocm.py`
is incremental, non-duplicate coverage that would run in the existing CI
lane.
