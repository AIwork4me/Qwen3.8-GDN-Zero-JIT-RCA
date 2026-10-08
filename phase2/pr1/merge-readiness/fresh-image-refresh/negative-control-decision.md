# Negative-Control Decision — fresh-image refresh vs upstream main d6fe5dca

Prior successful negative control (2026-10-07, base 3ca00a82): temporarily
disabled the GDN warmup → new GDN test FAILED (armed monitor raised on
runtime JIT) → reverted → PASS.

## Mechanism-change check since that control (3ca00a82 → d6fe5dca, 33 commits)

| load-bearing piece | commits | status |
|---|---|---|
| vllm/model_executor/warmup/qwen_triton_warmup.py | 0 | UNCHANGED |
| vllm/utils/jit_monitor.py | 0 | UNCHANGED |
| tests/jit_monitor/test_no_runtime_jit.py (shared battery) | 0 | UNCHANGED |
| Qwen3.5 GDN call path (models/qwen3_5.py) | 0 | UNCHANGED |
| qwen_gdn_linear_attn.py backend resolution | 1 (#57560) | CHANGED — new `aiter_flydsl` backend is explicit opt-in only (`gdn_prefill_backend` additional-config); auto/default still resolves to "triton" on ROCm. Fresh run log confirms: `Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=128)` |
| kernel_warmup.py | 1 (#58165) | FlashInfer autotune rounding only; not the Triton warmup path |

## Verdict

NEGATIVE CONTROL:
NOT RE-RUN
Reason: load-bearing warmup/monitor mechanism unchanged;
historical sensitivity proof remains applicable.
(The only adjacent change is opt-in-only and proven inactive on this path
by the fresh run's own backend-resolution log line.)
