# 09 — Root Cause (Phase-1 verdict)

## Scope

Qwen3.8 / Gated DeltaNet runtime JIT on AMD Radeon PRO W7900D (gfx1100),
ROCm 7.14 stack, pinned vLLM main `31e2443c90542a33a4a4a293ea7186fba2796c67`,
model Qwen/Qwen3.8-27B-FP8 @ 017b9c7a, cold/warm controlled caches.

## Verdict

```text
Default graph mode: CONFIRMED — no runtime JIT was observed in the tested
default/graph execution envelope.

Enforce-eager: CONFIRMED — runtime JIT still occurs under --enforce-eager
because graph capture and JIT warmup are intentionally disabled (and the
JIT monitor is intentionally inactive) by current upstream design.

Historical #52663: the historical ~43-minute first-inference runtime-JIT
impact is not reproducible on the current tested stack; the eager-mode
runtime-JIT mechanism itself remains reproducible (P1: 216 post-battery
Triton artifacts, ~36-42 s first-request penalty).
```

Precise statement: at the pinned current vLLM SHA, the historical
first-inference runtime-JIT production impact is no longer reproducible in
the default graph configuration: startup JIT warmup and graph capture cover
all compile keys required by the tested Qwen3.8/GDN inference envelope in
normal-mode execution (one eager-only specialization is not covered, by
design — docs/07).
However, the underlying eager-mode behavior remains reproducible:
`--enforce-eager` intentionally disables graph capture and JIT warmup, so
first-request Triton compilation still occurs, with substantially lower
observed impact than the historical report.

## Root cause statement (one sentence, evidence-backed)

At pinned current main on gfx1100, **zero runtime JIT occurs in the default
(graph) configuration within the tested envelope** — warmup (104 compile
keys) + cudagraph capture (sizes [1..64]) cover all compile keys required by
the tested Qwen3.8/GDN inference envelope in normal-mode execution. Primary
proof is disk-level: in P5/P3 no file in ANY cache
section (triton/vllm/xdg/torchinductor) has an mtime after the battery t0
(max triton mtime 03:06:50 < monitor arming 03:07:22 < battery 03:14:40),
which also bounds JIT invisible to the monitor (inductor-generated Triton);
secondary: armed-monitor zero events (P5/P3), P2 error-mode acceptance
(0 raises, 33/33 OK), flat first-request latency. Adversarial audit:
ROOT CAUSE CONFIRMED — while the "kernels compile during first
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

## Scope envelope (adversarial-review caveat)

Untested execution paths in ALL runs (normal and eager alike): prefix-cache-hit
scheduling (hit rate 0.0% throughout), preemption/recompute (KV peaked 9%),
non-greedy sampling (top-p/top-k kernels warmup-compiled but never executed),
structured outputs, chat path, speculative decode. The zero-runtime-JIT
statement is proven for the battery envelope (prompts <=128 tokens x decode
batches <=4, max_tokens 16, greedy, max_num_seqs 32, capture <=64,
max_num_batched_tokens 2048 chunking); the battery demonstrably exercises
every decode-leg and w8a8 M-specialization bucket it can reach (the identical
battery under eager compiled 27 entries).

## Consequences for the historical reports

- #52663's "first inference triggers ~43 min Triton JIT" (vllm
  0.27.1+rocm723, Aug 2026): the historical ~43-minute first-inference
  runtime-JIT **impact** is NOT REPRODUCIBLE at this SHA/stack in the
  default graph configuration (different version, defaults, and possibly
  aiter/wrapped W8A8 configs); the eager-mode runtime-JIT **mechanism**
  itself remains reproducible (P1), with the largest first-request compile
  cost measured at +42 s (eager), far below ~43 minutes. The FP8 startup
  cost remains real but is a STARTUP timeline
  phenomenon (8.5 min cold vs 3.0 min warm; no 23-min W8A8 autotune cliff
  observed in this configuration).
- #49349's Qwen-GDN `layer_norm_fwd_kernel` runtime JIT: the ROWS_PER_BLOCK
  gap is closed at this SHA for the tested geometry (#54251 warmup lineage)
  — layer_norm never runtime-compiled in any committed run, including
  eager.
- The 600 s engine-ready timeout class of #52663: not observed (cold
  startup ≤ 8.5 min = 510 s < 600 s default here — a modest margin),
  noting config differences (max-model-len 4096, single GPU, current main).

## Generic-GDN and FP8 attribution

- Eager-mode runtime JIT is not GDN-exclusive: N1 (dense non-GDN
  Qwen2.5-0.5B) also produced post-battery Triton artifacts in eager
  (80 files / ~11 compile entries — generic decode/sampler/paged-attention
  infra), while G1 adds the GDN-specific families (conv1d_update, packed
  decode) on top of the same generic infra set. The mechanism is a property
  of the eager configuration (capture + warmup disabled), not of GDN.
- Normal-mode zero-runtime-JIT coverage holds for both GDN models tested
  (27B P5/P3, 0.8B G2: zero post-battery artifacts, monitor armed, 0
  events). N1 was not run in normal mode, so the dense-model normal-mode
  case is untested here.
- FP8 causal to the runtime-JIT mechanism: NO, within the tested envelope —
  non-FP8 models runtime-JIT in eager too (G1, N1), and in normal mode FP8
  w8a8 compiles are startup-timeline (0 runtime FP8 compiles). In eager, 4
  extra w8a8 M-specialization variants do compile at runtime, riding the
  same missing-startup-coverage mechanism (not GDN-specific).
