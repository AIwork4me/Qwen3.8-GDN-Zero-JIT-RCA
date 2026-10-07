# 03 — Model Architecture: Qwen/Qwen3.8-27B-FP8 on pinned vLLM SHA

Verified 2026-10-07 from `/workspace/models/Qwen3.8-27B-FP8/config.json`
(sha256 `74227dd6...`, see `manifests/model-qwen38.json`).

## Config → registry → class trace (all references at SHA 31e2443)

```text
config.json
  architectures: ["Qwen3_5ForConditionalGeneration"]
  model_type: "qwen3_5"            (multimodal wrapper: vision_config + text_config)
  text_config.model_type: "qwen3_5_text"
        │
        ▼
vllm/model_executor/models/registry.py:587
  "Qwen3_5ForConditionalGeneration": ("qwen3_5", "Qwen3_5ForConditionalGeneration")
        │
        ▼
vllm/model_executor/models/qwen3_5.py:470
  class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid)
  (vision tower + language model; language-only requests never touch vision)
        │  text decoder layers
        ▼
vllm/model_executor/models/qwen3_5.py:120 Qwen3_5DecoderLayer(Qwen3NextDecoderLayer)
  layer_types from config (64 layers, full_attention_interval=4):
    * "linear_attention" (48) → qwen3_5.py:148
        QwenGatedDeltaNetAttention(prefix=f"{prefix}.linear_attn", ...)
    * "full_attention"   (16) → qwen3_5.py:156 Qwen3NextAttention
        │
        ▼
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:372
  class QwenGatedDeltaNetAttention(GatedDeltaNetAttention)
```

The trace DOES reach `QwenGatedDeltaNetAttention` for all 48 GDN layers.

## GDN geometry (text_config)

```text
linear_num_key_heads   = 16   linear_key_head_dim  = 128
linear_num_value_heads = 48   linear_value_head_dim= 128
linear_conv_kernel_dim = 4    (→ conv_dim = 2*16*128 + 48*128 = 9728)
hidden_size = 5120, head_dim (full attn) = 256, num_attention_heads = 24
num_key_value_heads = 4, full-attention interval = 4 (48 GDN + 16 full)
mamba_ssm_dtype = float32 (SSM state dtype)
mtp_num_hidden_layers = 1 (MTP module present; only used with speculative
                          decoding — not part of the eager baseline)
max_position_embeddings = 262144
```

## Quantization (STARTUP-JIT relevant)

```text
quant_method      = fp8
fmt               = e4m3
activation_scheme = dynamic
weight_block_size = [128, 128]   → W8A8 BLOCK FP8 path
modules_to_not_convert = vision-tower attention/MLP projections (+ MTP?)
```

→ the layer goes through
`vllm/model_executor/layers/quantization/utils/fp8_utils.py:963` →
`get_w8a8_block_fp8_configs()` (fp8_utils.py:844), which loads device-keyed
tuned configs from
`vllm/model_executor/layers/quantization/utils/configs/`.

**In-tree state at our SHA: 245 config files, 28× MI300X, 0 for any
Radeon/gfx1100 device name** → on `AMD Radeon Pro W7900D`
(`device_name=AMD_Radeon_Graphics`) the lookup misses and cold startup falls
into Triton W8A8 autotune per GEMM shape (#52663 still unfixed at this SHA).
This is **STARTUP JIT**, tracked separately from runtime JIT (docs/04+).

## Backend selection at runtime (gfx1100, natural environment)

`QwenGatedDeltaNetAttention.__init__` (qwen_gdn_linear_attn.py:386):

```text
current_platform.is_rocm() → _forward_method = forward_hip  (line 415)
forward_hip (line 864):
    GDN_AITER_TRITON_AVAILABLE (line 74:
        rocm_aiter_ops.are_gdn_triton_kernels_available()
        or rocm_aiter_ops.is_rdna_gdn_triton_kernels_available())
    → requires the optional `aiter` package (vllm/_aiter_ops.py:2472-2484);
      `aiter` is NOT in requirements/rocm.txt and NOT installed in the
      experiment venv
    → False on this host ⇒ fallback: forward_cuda (generic path)  (line 898)
```

Decode-kernel env defaults (vllm/envs.py:132-133):

```text
VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE = 1 (default)
VLLM_GDN_DECODE_KERNEL = "cuda" (default) → on this host unsupported
    (qwen_gdn_linear_attn.py:516-528: is_cuda_alike() true on ROCm but fused
    decode unsupported → silent fallback to "triton")
```

Net effect for a plain (non-spec) decode step:

```text
_forward_core (line ~1281)
  enable_packed_recurrent_decode && no spec && num_prefills==0 && num_decodes>0
  → _forward_core_decode_non_spec (line 1637)
      → causal_conv1d_update  → _causal_conv1d_update_kernel
        (vllm/model_executor/layers/mamba/ops/causal_conv1d.py:763/1231)
      → fused_recurrent_gated_delta_rule_packed_decode
        → fused_recurrent_gated_delta_rule_packed_decode_kernel
        (vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:256/456)
```

Prefill (first request chunked prefill):

```text
forward_cuda → causal_conv1d_fn (_causal_conv1d_fwd_kernel)
            → fused_post_conv_prep (FLA)
            → chunk_gated_delta_rule (FLA chunked kernels, autotuned)
            → _output_projection → RMSNormGated → layer_norm_fwd_kernel
```

## What this establishes (static)

1. Qwen3.8-27B-FP8 executes `QwenGatedDeltaNetAttention` 48× per forward.
2. Natural gfx1100 backend = generic FLA/Triton path, NOT AITER.
3. The FP8 startup autotune gap (#52663) exists at our SHA → expected
   STARTUP JIT; the engine-ready timeout must be raised for experiments.
4. The decode-path kernels `_causal_conv1d_update_kernel` and
   `fused_recurrent_gated_delta_rule_packed_decode_kernel` have **no warmup
   module coverage** at our SHA (docs/01 §4) — runtime JIT candidates; to be
   proven by experiment, not assumed.
