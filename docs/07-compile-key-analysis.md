# 07 — Compile-Key Analysis (core RCA)

Question: why did startup fail (P1/eager) or succeed (P5/normal) in producing
the exact compile artifacts the first real requests needed?

## The runtime keys (eager, cold — P1)

`_causal_conv1d_update_kernel` (2 specializations), 
`fused_recurrent_gated_delta_rule_packed_decode_kernel` (2),
`_w8a8_triton_block_scaled_mm` (+4 m-bucket variants), 
`fused_sigmoid_gating_delta_rule_update_kernel` (+1),
`kernel_paged_attention_2d` (1), and 14 sampler/KV-infra kernels (1-3 each).

## Mechanism per kernel family

### `_causal_conv1d_update_kernel` / packed decode / paged attention / sampler — class H + F

In **normal mode** these compile during STARTUP because:
1. cudagraph capture REPLAYS decode steps at sizes {1,2,4,...,512}
   (`cudagraph_capture_sizes` in P5 config log) → executes the decode leg
   for those batch sizes → Triton compiles each specialization (conv1d_update
   6 variants = capture-size family);
2. the 104-key JIT warmup (`qwen_triton_warmup` etc.) launches the
   compile-only variants (registry from #50174).

In **eager mode** `--enforce-eager` sets `kernel_config.enable_jit_warmup=False`
and disables cudagraph capture (log: "Enforce eager set, disabling
torch.compile, CUDAGraphs, and JIT kernel warmup", `vllm/config/vllm.py:1758`)
and the V1 profile run only exercises a PREFILL-shaped dummy forward
(`_forward_core` with `attn_metadata=None` → `_warmup_prefill_kernels`;
num_decodes==0 → decode leg never runs at startup). The startup phase
therefore produces ZERO decode-leg artifacts; the first real decode request
compiles them → runtime JIT.
Classification: **H (decode path never exercised during startup)** — by
upstream design in eager; plus **F** in the sense that "warmup vs runtime
configuration differs" (eager vs capture).

### `_w8a8_triton_block_scaled_mm` — class B (registered? no: startup-shape-dependent)

FP8 W8A8 GEMM kernel selection keys on the dynamic-M bucket
(configs map m ∈ {1,4,16,32,...,512} → BLOCK_SIZE_M etc.). The eager startup
profile run uses ONE m shape (2 variants compiled); runtime requests across
the battery m-range needed 4 more → class **B (startup missed runtime
shapes)**. In normal mode, capture at 19 batch sizes + warmup runs compile
all 6 variants at startup.

### `layer_norm_fwd_kernel` — class I (already fixed at this SHA)

The historical #49349 mismatch (startup ROWS_PER_BLOCK=1 vs runtime 4) is
covered at this SHA: `warmup_layer_norm_fwd` (VllmJitKernel warmup, PR
#54251) + profile run produce 3 variants; the runtime needed 0 more — even
in EAGER mode. The startup profile-run forward alone covered it here
(27B/48-layer geometry: rows_per_token = hv//tp = 48).

### `fused_sigmoid_gating_delta_rule_update_kernel` — class H (eager)

+1 runtime variant in eager (decode-side); 2 at startup in normal (warmup +
capture).

## Superset proof (normal mode covers everything)

Per-kernel variant counts: every P1-runtime kernel has ≥ count in
P5-startup (table in docs/05 §2; 20/20 YES). P3 independent cold repeat
reproduced the identical 67-kernel startup set → deterministic coverage.

## Conclusion

For the DEFAULT (normal) configuration at pinned SHA 31e2443 on gfx1100:

```text
compile-key cause of runtime JIT: NONE — no gap observed
```

For DIAGNOSTIC (eager) configuration:

```text
runtime JIT root cause = startup produces no decode-leg artifacts and
only one w8a8 m-bucket, because enforce_eager disables cudagraph capture
AND the JIT warmup registry (vllm/config/vllm.py:1758,
vllm/v1/worker/gpu_worker.py:1056), while the V1 profile run never
executes a decode step; all decode-leg Triton kernels therefore compile
at the first real decode request.
```

This is upstream-expected behavior ("runtime JIT compilation is expected"
per gpu_worker comment), not a defect — but it is the mechanism behind the
"first inference compiles kernels" observations, and it is invisible to the
JIT monitor (which is intentionally disabled in eager).
