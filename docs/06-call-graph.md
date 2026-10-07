# 06 — Dynamic Call Graph (runtime-proven; corrected after independent audit)

Built from: P1 runtime cache artifacts (which kernels actually ran/compiled
at first decode), P5 startup artifacts, source reading at pinned SHA
(file:line references), and platform detection values captured live
(`RocmPlatform`, `GDN_AITER_TRITON_AVAILABLE=False`, `gdn_decode_kernel`
auto-fallback cuda→triton, packed-recurrent decode on). Line references
independently re-derived by the call-graph audit subagent.

## Qwen3.8-27B-FP8 → GDN layer → generic ROCm path (live path at runtime)

```mermaid
flowchart TD
    REQ["HTTP /v1/completions (battery: token ids)"] --> SCHED["V1 scheduler → GPUModelRunner.execute_model"]
    SCHED --> FWD["CompiledModel / eager forward: Qwen3_5ForConditionalGeneration.language_model (qwen3_5.py:520)"]
    FWD --> DEC["Qwen3_5Model (qwen3_5.py:220) → Qwen3_5DecoderLayer (qwen3_5.py:123)"]
    DEC -->|layer_type=linear_attention ×48| GDN["QwenGatedDeltaNetAttention (qwen3_5.py:147; class qwen_gdn_linear_attn.py:372)"]
    DEC -->|layer_type=full_attention ×16| FA["Qwen3NextAttention (qwen3_5.py:155; qwen3_next.py:273)"]

    GDN -->|forward (qwen_gdn_linear_attn.py:844); _forward_method = forward_hip :414| AITER{GDN_AITER_TRITON_AVAILABLE? :74}
    AITER -->|"False (no aiter pkg; rocm_aiter_ops.py:2472)"| FC["forward_cuda fallback :898"]
    AITER -->|True (not taken)| AITERPATH["_forward_core_decode_aiter :1570"]

    FC -->|"in_proj_qkvz / in_proj_ba (FP8 linears :913-914); torch.ops.vllm.qwen_gdn_attention_core (:964, use_aiter=False) → op impl :1900 → _forward_core :1250"| DECIDE{"enable_packed_recurrent_decode (envs.py:132, default on) ∧ num_prefills==0 ∧ num_decodes>0 :1278"}
    DECIDE -->|pure decode step| NONSPEC["_forward_core_decode_non_spec :1637"]
    DECIDE -->|prefill step| PREF["causal_conv1d_fn :1361 → _causal_conv1d_fwd_kernel (causal_conv1d.py:712; kernel def :17)<br/>fused_post_conv_prep :1424 (fused_gdn_prefill_post_conv.py:152)<br/>chunk_gated_delta_rule :1514 → FLA chunk family (chunk_o.py:42, chunk_delta_h.py:45, chunk_scaled_dot_kkt.py:47, wy_fast.py:29, cumsum.py:27)"]
    DECIDE -->|mixed prefill+decode batch| PEELED["peeled decode → fused_sigmoid_gating_delta_rule_update :1480 (fused_sigmoid_gating.py:24)"]

    NONSPEC --> CU["causal_conv1d_update :1665 → _causal_conv1d_update_kernel (causal_conv1d.py:1231; def :763)"]
    NONSPEC --> PD["fused_recurrent_gated_delta_rule_packed_decode :1675 → kernel (fused_recurrent.py:456; def :256)"]
    GDN --> OUTPROJ["_output_projection :850-862 = RMSNormGated (forward_cuda layernorm.py:404 → rmsnorm_fn layernorm_guard.py:562 → layer_norm_fwd :441 → layer_norm_fwd_kernel :388) + out_proj (FP8)"]

    FA --> PA["RocmAttentionImpl.forward (rocm_attn.py:380) → chunked_prefill_paged_decode :468 → kernel_paged_attention_2d (chunked_prefill_paged_decode.py:461); KV write → reshape_and_cache_kernel_flash (rocm_attn.py:532)"]
```

Runtime evidence binding the graph:

- `_causal_conv1d_update_kernel` + `fused_recurrent_gated_delta_rule_packed_decode_kernel`
  compiled exactly at the FIRST decode request in P1 (mtime > battery t0;
  first-request 62.7 s) — proving the decode leg executes `_forward_core_decode_non_spec`.
- `fused_sigmoid_gating_delta_rule_update_kernel` (+1 runtime variant in
  P1): invoked from the recurrent-attention step of `_forward_core`
  (:1453/:1480/:1538) — i.e. the peeled-decode path of mixed batches —
  NOT from `_output_projection` (which contains only RMSNormGated + out_proj).
- `layer_norm_fwd_kernel` (3 variants) compiled only during startup.
  Attribution differs by mode: in EAGER (P1/G1) the JIT warmup registry is
  disabled, so the startup variants come from the V1 profile-run forward
  (P1/G1 result.md); in NORMAL mode the same 3 M-specializations are
  produced by `qwen_triton_warmup`'s `warmup_layer_norm_fwd` enumeration
  (qwen_triton_warmup.py:148-168 → layernorm_guard.py:270-331).
- `_w8a8_triton_block_scaled_mm`: launched from every FP8 block linear
  (`init_fp8_linear_kernel` kernels/linear/__init__.py:701 → ROCm kernel
  list :468-471 → aiter absent → `TritonFp8BlockScaledMMKernel`
  scaled_mm/triton.py:166 → `w8a8_triton_block_scaled_mm` fp8_utils.py:880
  → launch :985). Multiple compiled variants arise from Triton
  dynamic-arg specialization on M (see docs/07).
- Full-attention leg: `kernel_paged_attention_2d` runtime-compiled at first
  decode in P1 (1 variant), 2 startup variants in P5.

## Startup (timeline A) call sites producing the covering keys (normal mode)

1. `GPUModelRunner` V1 profile run: `attn_metadata=None` → GDN
   `_forward_core` bails into `_warmup_prefill_kernels`
   (qwen_gdn_linear_attn.py:1272-1274 → :1105/:1143) — compiles the FLA
   chunk-autotune family while memory is plentiful (its stated purpose).
2. `kernel_warmup` (kernel_warmup.py:158) against the jit_warmup_registry:
   `qwen_triton_warmup` (104 compile keys at 27B): layer_norm
   M-specializations, causal_conv1d_fn dummy, fused_post_conv lengths
   {1,2,16}, fused_sigmoid_gating update; `mamba_triton_warmup`:
   batch_memcpy.
3. cudagraph capture: `_warmup_and_capture` with `force_attention=FULL`
   builds real decode metadata (gpu_model_runner.py:6881, :6031-6046) and
   captures sizes [1,2,4,8,16,24,32,40,48,56,64] (11 sizes; P5 config log)
   → executes + compiles the decode-leg kernels per capture size.
4. jit monitor armed AFTER all of the above (`gpu_worker.py:1040`).
