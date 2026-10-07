# Phase-2 PR-1 — Test Design

## Objective

Convert the Phase-1 RCA result (Qwen GDN models on ROCm complete the standard
shape battery under the default graph configuration with zero runtime JIT)
into durable upstream regression coverage.

## Chosen shape (Option A: add one dedicated GDN test)

Add one test to the existing `tests/jit_monitor/test_no_runtime_jit_rocm.py`
(mirror of the current `test_v2_sampler_warmup_rocm` contract):

- model: `Qwen/Qwen3.5-0.8B` (HF id; config verified: `Qwen3_5ForConditionalGeneration`,
  18 linear_attention + 6 full_attention layers; registry.py:587 →
  `qwen3_5` → `QwenGatedDeltaNetAttention` for layer_type `linear_attention`)
- `enforce_eager=False` (default graph configuration — the Zero-JIT contract)
- `jit_monitor_mode="error"` (inference-time JIT fails the test)
- `load_format="dummy"` + `hf_overrides=dummy_hf_overrides` (1-layer
  truncation; layer index 0 is `linear_attention`, so the truncated model
  still executes exactly the GDN path — index-based selection proven at
  qwen3_5.py:256)
- assert worker monitor state `== (True, "error")` via `collective_rpc`
  (reuses `_worker_monitor_state`)
- run the shared `_run_shape_battery(llm)` (prefill/decode/sampler shapes)
- env: `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_ROCM_USE_AITER=0` (in-tree
  generic GDN/Triton path, not optional AITER), `VLLM_WORKER_MULTIPROC_METHOD=spawn`,
  `VLLM_ALLOW_INSECURE_SERIALIZATION=1`, per-test `TRITON_CACHE_DIR` /
  `VLLM_CACHE_ROOT` / `TORCHINDUCTOR_CACHE_DIR` under `tmp_path`
  (fresh caches: a previous run must not hide missing warmup coverage)
- resources: `max_model_len=2048`, `max_num_seqs=8`,
  `gpu_memory_utilization=0.03`, `kv_cache_memory_bytes=256MiB` (identical
  to the existing test — 1-layer truncated 0.8B ≈ 0.5GB weights incl.
  embeddings)
- test name: `test_qwen_gdn_no_runtime_jit_rocm`

## Why not Option B (parameterize the existing test)

Parameterizing `test_v2_sampler_warmup_rocm` over [dense, GDN] would couple
the dense sampler-warmup regression (PR #58092's subject) to the GDN
coverage; separate tests give independent failure attribution with ~15
shared lines duplicated, matching how the upstream file already factors
shared helpers (`_worker_monitor_state`, `_run_shape_battery`).

## Non-goals

- No runtime code changes; no warmup infrastructure changes; no AITER
  coverage; no eager-mode assertions (eager intentionally disables the
  monitor — testing eager would test the wrong property); no 27B/FP8 in CI.

## GDN-execution proof plan (local evidence, not committed introspection)

- module inspection: loaded model contains `QwenGatedDeltaNetAttention`
- config proof: 18 linear_attention layers (full model) / layer 0 GDN
  (truncated)
- cache inventory: post-battery Triton artifacts == 0 with monitor armed;
  startup side contains GDN families (`_causal_conv1d_update_kernel`,
  `fused_recurrent_gated_delta_rule_packed_decode_kernel`) proving the GDN
  kernels were compiled (at startup) — i.e. GDN executed

## Risks / open items for local validation

1. `gpu_memory_utilization=0.03` on 48 GB (W7900) for the truncated 0.8B
   model — raise slightly only if local validation OOMs (must stay valid on
   MI355 CI pool too).
2. Dummy-weight truncation must not disable the GDN path — verified by
   design (layer index 0), re-verified at runtime in validation evidence.
3. `Qwen/Qwen3.5-0.8B` must be fetchable (config+tokenizer only) from CI.
