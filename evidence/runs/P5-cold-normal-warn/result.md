# P5 — Qwen3.8-27B-FP8 primary, cold, warn, NORMAL mode (primary zero-JIT test)

- model: Qwen3.8-27B-FP8 (rev 017b9c7a), 48 GDN + 16 full-attn, FP8 W8A8
  block [128,128], generic GDN path on gfx1100 (W7900D, ROCm 7.14 venv)
- flags: (NO enforce-eager) --max-model-len 4096 --max-num-seqs 32
  --gpu-memory-utilization 0.90 --jit-monitor-mode warn --jit-monitor-verbose
- startup ≈ 8.5 min: torch.compile, JIT kernel warmup (104 compile keys),
  cudagraph capture to size 512, then "Kernel JIT monitor activated;
  ... mode=warn"
- cold caches proven (empty dirs, PREFLIGHT PASS); battery: tokens
  {1,4,8,15,16,17,31,32,33,64,128} × batches {1,2,4}, max_tokens 16; all OK

## Result

```text
runtime JIT events: 0
first-request latency spike: NONE (26.13 s request #1 == steady state 26.1-26.2 s)
triton cache: 428 entries / 71 kernels, ALL mtime-attributed STARTUP; RUNTIME entries: 0
```

Every kernel family that runtime-compiled in eager P1 —
`_causal_conv1d_update_kernel`, `fused_recurrent_gated_delta_rule_packed_decode_kernel`,
`_w8a8_triton_block_scaled_mm` (all 6 variants), `kernel_paged_attention_2d`,
sampler/KV infra — was fully compiled during startup in normal mode
(cudagraph capture sizes 1..512 + 104-key JIT warmup superset).

## Reading

- The historical "first inference triggers ~43 min Triton JIT" (#52663
  comment, vllm 0.27.1+rocm723, Aug 2026) is **not reproducible** on pinned
  current main in the default (graph) configuration: the decode-path kernels
  compile during cudagraph capture at startup, and the 104-key warmup covers
  the rest.
- In `--enforce-eager` mode (P1) the same model DOES runtime-JIT (first two
  requests +36 s/+42 s; conv1d_update ×2, packed decode ×2, +4 w8a8, decode
  infra) because eager disables both cudagraph capture and the JIT warmup
  (and the monitor itself).
