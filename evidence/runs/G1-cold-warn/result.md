# G1 — Qwen3.5-0.8B (GDN control), cold, warn, EAGER

- model: Qwen/Qwen3.5-0.8B @ 2fc06364715b967f1860aea9cf38778875588b17
  (Qwen3_5ForConditionalGeneration; 18 GDN + 6 full-attn layers →
  QwenGatedDeltaNetAttention generic path, GDN_AITER_TRITON_AVAILABLE=False)
- flags: --enforce-eager --max-model-len 2048 --max-num-seqs 16
  --jit-monitor-mode warn --jit-monitor-verbose
- cold caches proven; battery {1,4,8,15,16,17,31,32,33,64,128} × batches {1,2,4}, max_tokens 8

## Findings

1. **`--enforce-eager` disables the JIT monitor by upstream design**
   (`vllm/v1/worker/gpu_worker.py:1056` `_maybe_activate_jit_monitor` returns
   early when `enable_jit_warmup` is False; log line: "Enforce eager set,
   disabling torch.compile, CUDAGraphs, and JIT kernel warmup").
   => 0 `jit_monitor` warnings here is EXPECTED and NOT evidence of zero JIT.

2. **Runtime compilation happened and is measurable via latency + cache**:
   first battery request (prompt=1, batch=1) wall = **4.782 s**; request #2
   (prompt=4) still 10.335 s (further compiles), then ~0.18-0.2 s steady — the first real decode compiled kernels.

3. Triton cache inventory — JIT monitor was NOT active during inference (eager); inventory built from cache artifacts. **Triton cache inventory** (`triton-kernels.txt`, 35 distinct kernels)
   includes exactly the historical GDN families:
   `_causal_conv1d_update_kernel` (2 variants),
   `fused_recurrent_gated_delta_rule_packed_decode_kernel` (2 variants),
   `_causal_conv1d_fwd_kernel` (1), `layer_norm_fwd_kernel` (3 variants),
   FLA chunk prefill autotune (`chunk_fwd_kernel_o` 36, `chunk_scaled_dot_kkt_fwd_kernel` 27,
   `recompute_w_u_fwd_kernel` 9, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 8,
   `chunk_local_cumsum_scalar_kernel` 4), `fused_post_conv_kernel` (3),
   `fused_sigmoid_gating_delta_rule_update_kernel` (2), `batch_memcpy_kernel`,
   mamba align pre/post copy kernels, sampler kernels, full-attn
   `kernel_paged_attention_2d`.
   Closure mtime attribution (see `cache-boundary-verification.txt`): of the
   total inventory, only the **decode-leg set** (23 entries / 19 kernels —
   conv1d_update ×2, packed decode ×2, apply_write ×3, sigmoid_gating +1,
   paged_attention ×1, mamba-align/sampler infra) compiled AFTER battery
   start; the prefill families above (FLA chunk set, conv1d_fwd,
   fused_post_conv, layer_norm, batch_memcpy) compiled during the eager
   startup profile run.

Interpretation: in EAGER mode (no warmup, no cudagraphs) the GDN decode path
compiles on the first real decode request. The zero-JIT acceptance question
is answered by the NORMAL-mode runs (G2/P5...): does startup (warmup +
cudagraph capture) produce the same compile keys the runtime needs?
