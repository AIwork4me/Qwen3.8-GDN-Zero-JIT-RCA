# P1 — Qwen3.8-27B-FP8 primary, cold, warn, EAGER (runtime-JIT inventory)

- model: local snapshot of Qwen/Qwen3.8-27B-FP8 rev 017b9c7a
  (config sha256 74227dd6…; 48 GDN + 16 full-attn layers; FP8 W8A8 block
  [128,128], dynamic activation)
- vLLM: pinned main 31e2443c, ROCm 7.14 venv, gfx1100 (W7900D), generic GDN
  path (no aiter), packed-recurrent decode on, GDN decode kernel auto
  fallback cuda→triton
- flags: --enforce-eager --max-model-len 4096 --max-num-seqs 32
  --gpu-memory-utilization 0.90 --jit-monitor-mode warn --jit-monitor-verbose
- cold caches proven; startup → "Application startup complete" ≈ 6.5 min
  (no 23-min autotune cliff in eager: startup profile run compiled the
  prefill-side kernels; see below)
- battery: tokens {1,4,8,15,16,17,31,32,33,64,128} × batches {1,2,4},
  max_tokens 16, greedy; all requests OK

## Result — runtime compile inventory (mtime-attributed vs battery t0)

STARTUP (before first request) — 61 entries, incl. FLA chunk autotune
family (chunk_fwd_kernel_o 36, chunk_scaled_dot_kkt_fwd 27,
recompute_w_u_fwd 9, …), layer_norm_fwd_kernel ×3, causal_conv1d_fwd ×1,
fused_post_conv ×3, w8a8_triton_block_scaled_mm ×2, batch_memcpy ×1,
vision/mrope kernels.

RUNTIME (first requests) — 27 entries (audit-verified count):

| kernel | variants | note |
|---|---|---|
| `_causal_conv1d_update_kernel` | 2 | GDN decode conv (historical #52663 family) |
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 2 | GDN decode recurrent (historical) |
| `_w8a8_triton_block_scaled_mm` | +4 | FP8 W8A8 GEMM variants beyond the 2 startup ones |
| `fused_sigmoid_gating_delta_rule_update_kernel` | +1 | extra decode-update variant |
| `kernel_paged_attention_2d` | 1 | full-attn decode |
| sampler/KV infra (`_apply_write_kernel`, `_zero_kv_blocks_kernel`, `_gather_block_tables_kernel`, `_prepare_prefill_inputs_kernel`, mamba align pre/post, …) | 1-3 each | decode/scheduling infra |

Latency evidence: request wall times 62.7 s and 68.3 s for the first two
requests vs 26.4 s steady state (16 decode tokens each) → ~36-42 s of
first/second-request compile latency (timeline B), GPU idle during compile
(eager, single stream).

## Notes

- JIT monitor was NOT active (upstream disables it with enforce_eager —
  gpu_worker.py:1056); inventory above is from TRITON_CACHE_DIR artifacts +
  mtime attribution + latency, not from monitor warnings.
- `layer_norm_fwd_kernel` did NOT runtime-compile: the 3 startup variants
  (from the V1 profile-run forward) covered the battery (contrast with the historical #49349 ROWS_PER_BLOCK gap, fixed at this SHA by #54251). runtime JIT events: 0 (monitor inactive).
- Historical
  historical #49349 ROWS_PER_BLOCK gap, fixed at this SHA by #54251).
- The runtime w8a8 variants are Triton dynamic-arg specializations on M
  (startup compiled the M%16-divisible variant; runtime M values produced
  4 more). gfx1100 has no tuned W8A8 configs in-tree, so the default
  config is used for all M (m-bucket lookup inert) — corrected in docs/07.
- Full artifact lists: `triton-kernels.txt`, `cache-after.txt`,
  `requests.jsonl`.
