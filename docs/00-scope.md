# 00 — Scope

## Mission

Phase 1 only: rigorous, reproducible, 360-degree investigation of Qwen3.8 /
Gated DeltaNet runtime JIT compilation on AMD Radeon PRO W7900 (gfx1100),
ROCm 7.14 (official PyTorch wheel index), current vLLM upstream `main`.

## Primary questions

1. Does runtime JIT still exist on current vLLM `main`?
2. Exactly which kernels compile during inference rather than startup warmup?
3. Which execution paths trigger them?
4. Is the behavior RDNA3/ROCm-specific, Qwen3.8-specific, generic to Qwen GDN,
   FP8-caused, missing-warmup-coverage, compile-key mismatch, backend
   selection, or something else?
5. Is the historical root-cause claim of vLLM issue #52663 still true?
6. What is the real current root cause?

## Hard phase boundary — ALLOWED

- environment setup; build upstream vLLM; run workloads
- cache isolation / clearing; experiment scripts; logs
- source and git-history inspection; upstream issue/PR research
- diagnostic instrumentation for tracing only (separate worktree)
- profilers; minimized reproducers; execution-path comparisons
- commit evidence + analysis to GitHub

## Hard phase boundary — FORBIDDEN (Phase 2 material)

- adding `register_warmup()` as a fix
- converting kernels to `VllmJitKernel`
- modifying compile keys to solve the problem
- precompiling missing kernels as a solution
- changing Triton launch parameters to avoid JIT
- upstream-ready behavioral patches; vLLM/Triton PRs; upstream comments

Candidate fixes are documented in `docs/phase2_candidates.md` only.

## Identity of the experiment

- vLLM SHA: see `manifests/upstream.json` (never "latest main" without SHA)
- GPU must resolve to gfx1100 or the run is not a valid W7900 result
- PyTorch must remain the official `rocm7.14` wheel build throughout;
  replacement by pip during vLLM install is a failed environment gate
- Every run gets explicit cache dirs; "cold" is proven by pre-run manifest
