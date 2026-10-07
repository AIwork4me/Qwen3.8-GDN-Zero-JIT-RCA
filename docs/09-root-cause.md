# 09 — Root Cause (Phase-1 verdict)

## Scope

Qwen3.8 / Gated DeltaNet runtime JIT on AMD Radeon PRO W7900D (gfx1100),
ROCm 7.14 stack, pinned vLLM main `31e2443c90542a33a4a4a293ea7186fba2796c67`,
model Qwen/Qwen3.8-27B-FP8 @ 017b9c7a, cold/warm controlled caches.

## Verdict

```text
Phase-1 COMPLETE — HISTORICAL ROOT CAUSE DISPROVED (for the default
configuration at the pinned SHA); the residual runtime-JIT mechanism is
fully identified and is upstream-expected diagnostic-mode behavior.
```

## Root cause statement (one sentence, evidence-backed)

At pinned current main on gfx1100, **zero runtime JIT occurs in the default
(graph) configuration** — warmup (104 compile keys) + cudagraph capture
(sizes [1..64]) cover every compile key the first requests need
(P5/P3: 428 startup-side entries each, 0 runtime compiles; P2 error-mode
acceptance passes) — while the "kernels compile during first
inference" behavior reproduces **only** under `--enforce-eager`, where
upstream deliberately disables cudagraph capture AND the JIT warmup
(`vllm/config/vllm.py:1758`, `gpu_worker.py:1056`), so the V1 profile run
(which never executes a decode step) leaves the entire decode leg
(`_causal_conv1d_update_kernel`, `fused_recurrent_gated_delta_rule_packed_decode_kernel`,
decode paged-attention, sampler batch kernels) plus 4 additional
`_w8a8_triton_block_scaled_mm` Triton-M-specialization variants uncompiled until the first
real decode request (+36-42 s on the first two requests; GPU idle during
compile; JIT monitor intentionally silent).

## Compile-key mechanism (one sentence)

Runtime compile keys in eager = decode-leg specializations and w8a8
Triton-M-specialization variants that in normal mode are generated at
startup by cudagraph capture-at-N and the 104-key warmup registry
(normal-mode coverage proof: docs/07; kernel-level equality across independent cold
runs: docs/04).

## Consequences for the historical reports

- #52663's "first inference triggers ~43 min Triton JIT" (vllm
  0.27.1+rocm723, Aug 2026): not reproducible at this SHA in any tested
  configuration; the largest first-request compile cost measured is +42 s
  (eager). The FP8 startup cost remains real but is a STARTUP timeline
  phenomenon (8.5 min cold vs 3.0 min warm; no 23-min W8A8 autotune cliff
  observed in this configuration).
- #49349's Qwen-GDN `layer_norm_fwd_kernel` runtime JIT: fixed at this SHA
  (#54251 warmup lineage) — layer_norm never runtime-compiled in any run,
  including eager.
- The 600 s engine-ready timeout class of #52663: not observed (cold
  startup ≤ 8.5 min ≪ 600 s default here), noting config differences
  (max-model-len 4096, single GPU, current main).

## Generic-GDN and FP8 attribution

- Generic GDN issue: the eager-mode runtime-JIT mechanism is GDN-path
  generic (G1/G2 prove it for Qwen3.5-0.8B), and the normal-mode coverage
  is equally generic (G2 zero events).
- FP8 causal to runtime JIT: NO (FP8 w8a8 compiles are startup-timeline in
  normal mode; in eager the extra w8a8 runtime variants ride along with the
  same startup-coverage mechanism, they are not GDN-specific).
