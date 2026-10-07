# Audit 02 — Test Design (merge-readiness, current main @ 3ca00a82)

Reviewer: independent general-purpose subagent, fresh context, read-only access to the
fresh worktree. Verbatim report preserved below.

## Verbatim subagent report

> ## 1. Smallest valuable test?
>
> Yes. The only new machinery is `_gdn_hf_overrides` (test_no_runtime_jit_rocm.py:30-42), a parameter on the shared `_run_rocm_shape_battery` (:46), the extracted `_isolate_rocm_jit_caches` (:66-77), and the test entry (:90-105). Everything else is reuse (`_run_shape_battery` from test_no_runtime_jit.py:52-91, `dummy_hf_overrides`, `create_new_process_for_each_test`). Nothing is cuttable without losing the contract: `VLLM_ROCM_USE_AITER=0` pins the Triton GDN backend (scope per the docstring), the monitor-state assert is inherited and load-bearing, cache isolation is what makes a cold-cache JIT check meaningful. Only trim candidate: the 7-line `_gdn_hf_overrides` docstring is verbose by repo standards (AGENTS.md "keep docstrings brief") — cosmetic.
>
> ## 2. Could it PASS without exercising GDN?
>
> No, with one residual hole.
> - Layer retention is explicit: :41 sets `layer_types=["linear_attention","full_attention"]`, consumed at qwen3_5.py:256 (`get_layer`) → `QwenGatedDeltaNetAttention` at qwen3_5.py:147-154. This does not depend on the hub layout. Verified against the real config (`/workspace/models/Qwen3.5-0.8B/config.json:19-44`): `layer_types[0]=="linear_attention"`, first full attention at index 3 — so the docstring claim at :33-34 is correct, and pinning beats relying on layer 0 being GDN.
> - Battery coverage: prefill (varlen 4/32/128), 16-step decode, greedy + top_k/top_p/min_p + heterogeneous sampling (test_no_runtime_jit.py:59-91). Yes.
> - Dummy weights are valid: Triton specialization keys on dtypes/shapes/constexprs, not values; the negative control (warmup no-op → `_fused_post_conv_kernel` failure) proves e2e sensitivity.
> - Correctly acknowledges that cudagraph-captured kernels don't re-trigger hooks (test_no_runtime_jit.py:6-8; docstring at :96-98 says "JIT warmup and cudagraph capture" jointly — accurate).
> - Residual hole: nothing asserts a GDN module actually exists in the booted model. If a refactor made `Qwen3_5Model` ignore `config.layer_types`, `qwen_triton_warmup` would just log "no Qwen GDN layer found" (qwen_triton_warmup.py:144) and the test would pass green with zero GDN coverage. See recommended change.
>
> ## 3. Monitor inactive without the test noticing?
>
> No. The `collective_rpc` assert at :59-60 reads the worker-process globals (`jit_monitor.is_active()`, jit_monitor.py:46-48; `_active` set at :70) plus the worker's `observability_config.jit_monitor_mode`. Activation is gated on `enable_jit_warmup` (gpu_worker.py:1056-1061, default True at config/kernel.py:290), which also gates the qwen/MRV2 warmups (kernel_warmup.py:172, gpu_worker.py:914-918) and is force-disabled by `enforce_eager` (config/vllm.py:1746-1764) — any of these flipping makes `is_active()` False → assert fails red. `assert states and ...` also rejects an empty RPC result. Hook-install failure raises during activation (no try/except in `_setup_triton_jit_hook`, jit_monitor.py:238-275) → boot fails red. Solid.
>
> ## 4. Dense-path refactor behavior-preserving?
>
> Yes. The 7 env vars moved verbatim to `_isolate_rocm_jit_caches` in identical order (old :51-57 → new :71-77); LLM kwargs identical; `@create_new_process_for_each_test("spawn")` retained (:45); the try/finally + collective_rpc assert unchanged. The only delta is args now flow through the spawn payload (tests/utils.py:1910-1916) — `"Qwen/Qwen3-0.6B"` and `dummy_hf_overrides` (module-level, importable) pickle by reference; the child resolves the target by qualname import. No behavior change.
>
> ## 5. Cache isolation sufficient?
>
> Yes. `TRITON_CACHE_DIR` (Triton disk cache incl. autotune results), `TORCHINDUCTOR_CACHE_DIR`, and `VLLM_CACHE_ROOT` (vLLM caches, FlashInfer autotune file, and the ROCm-segmented-attn Triton cache at rocm_segmented_attn.py:207) are all redirected; `tmp_path` is unique per test, and each test runs in a fresh spawned interpreter (fresh in-process JITFunction caches). Inheritance is sound: spawn child copies `os.environ` (tests/utils.py:1944), and the engine core/workers spawn from it. AITER is off and UltraQuant isn't selected, so no fixed-path kernel cache on this path. A previous run's caches cannot produce a green here.
>
> ## 6. Two-layer hybrid necessary?
>
> Yes, keep it. One clarification to the premise: `layer_types=None` default generates interval-4 layout — **every 4th** layer full attention (qwen3_5.py:88-96; hub `full_attention_interval: 4`), not every 3rd. With plain `dummy_hf_overrides` (1 layer) you'd get a pure-GDN model: zero full-attention layers. That (a) is a layout Qwen3.5 never ships with (real config is 3:1 linear:full), (b) bypasses the hybrid two-spec KV-cache manager (PagedSpec + MambaSpec groups; counted at config/model.py:1748-1758) that production serves, and (c) leaves the full-attention backend selection (GQA, head_dim 256) and boundary-layer logic (arg_utils.py:2274) unexercised — and pure-GDN-through-a-hybrid-model-class (`is_hybrid` stays True at config/model.py:2012-2020 → MambaHybridModelState via model_states/__init__.py:52-55, but with zero attention groups) is an unsupported, unserved configuration. Sampling/logits themselves don't need full attention; the hybrid cache path does. Two layers is minimal for both.
>
> ## 7. Timeout change defensible?
>
> Yes. Isolation guarantees every test recompiles Triton + inductor from cold and captures cudagraphs; two full boots on MI355 within 2×600 s + overhead fits 25 min, and mirrors the NVIDIA sibling's headroom (900 s/test, 45 min job, jit_monitor.yaml:8,22). Note the file-level pytest-timeout uses the default signal method (no `--timeout-method=thread` on the AMD command) — pre-existing, works here because the watchdog only needs to interrupt the pytest process joining the spawned child.
>
> ## 8. Docstring wording?
>
> Accurate. :33-34 verified against the hub config; :96-98 correctly credits cudagraph capture jointly with JIT warmup (avoiding the overclaim that the monitor alone sees decode misses); :100-102 correctly scopes AITER out. Minor nit: `full_attention_interval` semantics (every 4th, per qwen3_5.py:90) are only implied; a half-line "3:1 linear:full" would preempt the exact misreading in your question 6. Optional.
>
> ## 9. Current-main drift interactions?
>
> - **FlashInfer autotune (kernel_warmup.py:264-265, 342-430)**: gated on `has_flashinfer()`; capability check is `>= 90` (interface.py:467) so MI355 (9.5) passes — hinges on flashinfer being absent in the AMD image. If present, it runs entirely pre-monitor (activation at gpu_worker.py:1040), so it can only add boot time or fail red, never false-green; the new hybrid token-bucket dummy runs warm more shapes, which helps.
> - **rocm.py UltraQuant (rocm.py:522-527)**: engaged only for `cache_dtype=="ultraquant_4bit"` — not set; no interaction. Full-attention layer falls to ROCM_ATTN/TRITON_ATTN with AITER off, same as the historical passing run.
> - **MRV2 mamba_hybrid (model_states/mamba_hybrid.py)**: this is exactly the path under test. Watch: prefix caching default-on flips `mamba_cache_mode` to `"align"` (models/config.py:757-759), whose per-step Triton kernels (`preprocess_mamba_align_fused_kernel`, mamba_hybrid.py:205; `_scatter_num_accepted_kernel`, :351) are skipped by dummy runs (mamba_hybrid.py:184-186) — they're warmed by MRV2's scheduler-realistic `run_mixed_prefill_decode_warmup` (gpu/warmup.py:87-217, invoked at gpu_worker.py:914-918) pre-monitor. That warmup can be *silently skipped* if `num_blocks <= required_blocks` (gpu/warmup.py:125-132); with `kv_cache_memory_bytes=256MB` this is the one spot where the fresh re-run could turn red. If it does, bump `kv_cache_memory_bytes` — don't touch the warmup. Any failure there is red, not green, so it can't invalidate the contract.
>
> **FINAL VERDICT: APPROVED WITH SMALL CHANGES**
>
> 1. RECOMMENDED — close the last false-green hole: assert GDN presence, e.g. extend the existing RPC in test_no_runtime_jit_rocm.py:59 to also assert the worker model contains a `QwenGatedDeltaNet` layer (or any module matching `qwen_triton_warmup._is_qwen_gdn_layer`, qwen_triton_warmup.py:60), so silent layer_types/config drift can't pass a GDN-less boot. ~5 lines in `_worker_monitor_state`.
> 2. RECOMMENDED — in `_gdn_hf_overrides`'s docstring, state the concrete interval ("every 4th layer full attention, 3:1") to preempt the "every 3rd" misreading; trim the rest per repo brevity.
> 3. RECOMMENDED — before merge, confirm the fresh current-main run passes and check the log for "Skipping V2 mixed prefill+decode warmup" (gpu/warmup.py:126); if present with a runtime-JIT failure, raise `kv_cache_memory_bytes` at test_no_runtime_jit_rocm.py:52.

## Disposition of reviewer changes

| # | change | disposition |
|---|---|---|
| 1 | GDN-presence assert | **ADOPTED** — implemented as `_worker_gdn_layer_count` worker RPC reusing the warmup's own in-tree predicate `_iter_qwen_gdn_layers` over `compilation_config.static_forward_context` (the exact source the warmup itself reads), gated by `expect_gdn=True` only in the GDN test; dense test behavior unchanged. ~11 added lines, no runtime change, no new dependency. Satisfies the mission's ≤10–15-line in-test GDN-execution assertion rule. |
| 2 | docstring interval wording | **ADOPTED** — `_gdn_hf_overrides` docstring now states "alternates GDN with a full-attention layer every 4th layer (3:1 linear:full)". Verified against the local hub config (full_attention_interval=4, full-attn idx 3,7,11,...). |
| 3 | watch for silently-skipped mixed warmup | **NOTED** — fresh current-main validation will grep the run log for the skip message; escalation path documented (bump `kv_cache_memory_bytes`, never touch the warmup). |

## Verdict

```text
APPROVED WITH SMALL CHANGES → all changes adopted → design final
```
