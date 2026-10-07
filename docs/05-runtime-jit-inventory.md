# 05 — Runtime-JIT Inventory

Sources: P1 (27B eager, mtime-attributed triton cache + latency), G1 (0.8B
eager), P5/P3 (27B normal, startup-side), G2 (0.8B normal). Monitor events:
zero everywhere the monitor was armed (normal mode) — see docs/04.

## 1. RUNTIME compiles observed (eager mode, cold cache, first requests)

### Qwen3.8-27B-FP8 (P1) — 20 kernels, 24 variants, after first request

| kernel | variants | family | startup (eager) variants |
|---|---|---|---|
| `_causal_conv1d_update_kernel` | 2 | GDN decode conv | 0 |
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 2 | GDN decode SSM | 0 |
| `_w8a8_triton_block_scaled_mm` | +4 | FP8 W8A8 GEMM (m-bucket variants) | 2 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | +1 | GDN decode update | 1 |
| `kernel_paged_attention_2d` | 1 | full-attn decode | 0 |
| `_apply_write_kernel` | 3 | sampler | 0 |
| `_fwd_kernel` | 1 | sampler/topk | 0 |
| `_gather_block_tables_kernel`, `_compute_slot_mappings_kernel`, `_prepare_prefill_inputs_kernel`, `_prepare_pos_seq_lens_kernel`, `_post_update_kernel`, `_scatter_num_accepted_kernel`, `_combine_sampled_and_draft_tokens_kernel`, `_get_num_sampled_and_rejected_kernel`*, `_zero_kv_blocks_kernel`, `get_aligned_state_indices_multi_group_kernel`, `reshape_and_cache_kernel_flash`, `pre/process_mamba_fused_kernel`, `precopy_mamba_align_fused_kernel` | 1 each | KV/scheduler/mamba-align infra | mostly 0 |

(*compiled at startup in eager)

Latency: first request 62.7 s, second 68.3 s, steady 26.4 s (16 decode
tokens) → compile cost concentrated in first two requests (~36-42 s), GPU
idle during compile.

### Qwen3.5-0.8B (G1) — 35 kernels; GDN-relevant runtime set matches P1's
(conv1d_update ×2, packed decode ×2, layer_norm ×3*, conv fwd ×1*,
fused_post_conv ×3*, sigmoid_gating ×2*, batch_memcpy ×1*, FLA chunk
autotune family). (*some hit at startup profile; full list in run dir.)

## 2. STARTUP compiles (normal mode) that cover the runtime keys (P5)

428 cache entries / 67 kernels, ALL mtime-attributed before first request:
warmup (104 compile keys) + cudagraph capture sizes {1..512} + V1 profile
run. Coverage proof (P1-runtime ⊆ P5-startup, per-kernel variant counts):

```text
_causal_conv1d_update_kernel                                   2 → 6  YES
fused_recurrent_gated_delta_rule_packed_decode_kernel          2 → 2  YES
_w8a8_triton_block_scaled_mm                                   4 → 6  YES
fused_sigmoid_gating_delta_rule_update_kernel                  1 → 2  YES
kernel_paged_attention_2d                                      1 → 2  YES
(all 20 P1-runtime kernels covered; 51 extra startup-only kernels)
```

## 3. Timeline separation (27B FP8)

| phase | cold normal (P5) | warm (P4) |
|---|---|---|
| engine init → ready | 8.5 min (compile+warmup+capture) | 3.0 min (cache hit) |
| first request | 26.1 s, **0 compiles** | 26.1 s, 0 compiles |

FP8 W8A8 compilation/autotune is a STARTUP-timeline cost at this SHA; no
runtime (post-ready) FP8 compile in normal mode.
