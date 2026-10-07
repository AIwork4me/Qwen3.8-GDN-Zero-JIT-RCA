# G2 — Qwen3.5-0.8B (GDN control), cold, warn, NORMAL mode (primary GDN zero-JIT test)

- model: Qwen/Qwen3.5-0.8B @ 2fc06364 (18 GDN + 6 full-attn layers, generic
  FLA/Triton path — no aiter)
- flags: (NO --enforce-eager) --max-model-len 2048 --max-num-seqs 16
  --jit-monitor-mode warn --jit-monitor-verbose
- startup: JIT kernel warmup ran (88 compile keys), cudagraph capture sizes
  [1,2,4,8,16,24,32] (max 32, per server.log engine config), "Kernel JIT
  monitor activated; ... mode=warn" BEFORE first request
- cold caches proven (cache-before empty, PREFLIGHT PASS)
- battery: tokens {1,4,8,15,16,17,31,32,33,64,128} × batches {1,2,4}; all OK

## Result

```text
runtime JIT events (monitor-armed, post-startup): 0
```

Every GDN kernel family that runtime-compiled in EAGER G1 —
`_causal_conv1d_update_kernel` (6 variants here vs 2 in eager),
`fused_recurrent_gated_delta_rule_packed_decode_kernel` (2),
`_causal_conv1d_fwd_kernel` (2), `layer_norm_fwd_kernel`, FLA chunk
autotune set, `batch_memcpy_kernel` — was compiled during STARTUP
(warmup + cudagraph capture) in normal mode, with per-kernel variant counts
≥ eager's (family-level coverage; entry-level eager superset not claimed).
No latency spikes in
requests.jsonl; no jit_monitor lines in server.log.

## Interpretation

For the small GDN model at pinned SHA 31e2443, upstream's warmup+cudagraph
machinery **already achieves zero runtime JIT** for this battery. The
historical #49349 Qwen-GDN report (layer_norm_fwd ROWS_PER_BLOCK mismatch,
main d9dabfa, 2026-08-28) is **no longer reproducible for this model/shape
battery at this SHA** (gap closed at this SHA for the tested geometry; warmup
coverage traces to the #54251/#54797 lineage).

=> The open question is now strictly about Qwen3.8-27B-FP8: FP8 W8A8 GEMM
autotune (startup, #52663) and whether any runtime compile key escapes at
27B geometry/dtype.
