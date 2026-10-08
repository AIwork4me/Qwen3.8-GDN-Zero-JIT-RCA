# Source-Level Assumption Gate — fresh image, tree b61df6031e (main d6fe5dca + PR test)

| assumption | evidence | verdict |
|---|---|---|
| jit_monitor_mode="error" arms monitor; monitored inference-time Triton JIT raises | `vllm/utils/jit_monitor.py` unchanged vs 3ca00a82 (0 upstream commits); `activate(mode)` / `is_active()` / error-raise semantics intact | HOLDS |
| enforce_eager=False → normal graph/warmup path | test passes `enforce_eager=False`; `kernel_warmup.py` calls `qwen_triton_warmup(worker.model_runner, worker.vllm_config.model_config)` at line 191 | HOLDS |
| Qwen3.5 layer_types → constructs QwenGatedDeltaNetAttention | `vllm/model_executor/models/qwen3_5.py:147` `if self.layer_type == "linear_attention": self.linear_attn = QwenGatedDeltaNetAttention(...)` | HOLDS |
| `_iter_qwen_gdn_layers(...)` still the warmup GDN predicate | `qwen_triton_warmup.py:78` defines it; `_qwen_gdn_warmup_config` consumes it; `_QWEN_MODEL_TYPES` contains `qwen3_5` | HOLDS |
| VLLM_ROCM_USE_AITER=0 selects generic in-tree ROCm path | `vllm/envs.py:138` default False; `_resolve_gdn_prefill_backend` on ROCm returns "triton" unless explicit `gdn_prefill_backend=aiter_flydsl` opt-in (new #57560 backend is opt-in only, requires CDNA3+ and AITER=1) | HOLDS |

Verdict: SOURCE-LEVEL ASSUMPTIONS: ALL HOLD on d6fe5dca.
