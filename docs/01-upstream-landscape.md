# 01 — Upstream Landscape (as of pinned SHA `31e2443c`, 2026-10-06/07)

Sources: `evidence/upstream/issue-52663.json`, `evidence/upstream/issue-49349.json`
(fetched 2026-10-06), GitHub API queries (2026-10-07), local git archaeology of
the pinned source tree. All line references are to the pinned SHA unless noted.

## 1. Issues

### #52663 — FP8 on RDNA3/gfx1100 exceeds 600 s engine-ready timeout (OPEN)

- Reported on `vllm==0.27.1+rocm723` (August 2026), W7900D gfx1100.
- Root finding: **zero tuned W8A8 block-FP8 Triton configs shipped for
  `AMD_Radeon_Graphics` / `AMD_Radeon_RX_7900_XTX`** (only MI300X/MI325X and
  NVIDIA devices ship configs) → `get_w8a8_block_fp8_configs()` returns None →
  **startup autotune** of the W8A8 GEMM (5 shapes, ~23.5 min) → first engine
  init 1474 s > 600 s default `VLLM_ENGINE_READY_TIMEOUT_S`.
- Commenters also observed (historical): after startup, **first inference
  triggers ~43 min of Triton JIT**; warm cache → 49 s init.
- Independent reproduction on RX 7900 XTX; tuned gfx1100 config JSONs posted
  in-issue (num_stages=2; num_stages>2 on RDNA3 → 51-64 s compiles or hangs).
- Classification: **historical observation + partially addressed by community
  config proposals; no RDNA3 device-keyed configs merged into the tree at our
  SHA** (see §4 for tree state). The 43-min "first inference JIT" claim is a
  *hypothesis to re-test*, not a current fact.

### #49349 — Zero JIT compilation during runtime (OPEN, tracking issue)

- Tracks adoption of the shared JIT warmup infrastructure (RFC issue #47456,
  still open).
- Foundation PRs merged: #47451 (1/N, 2026-07-20), #49315 (2/N, 2026-08-11),
  #50174 (3/N provider registry + orchestration, 2026-08-17).
- Runtime JIT reports in-thread (all pre-our-SHA):
  - DSPARK/Kimi-K3 decode on B200: `_prepare_dflash_inputs_kernel`,
    `_causal_conv1d_fwd_kernel`, `_gather_initial_states_kernel`, KDA kernels.
  - `batch_memcpy_kernel` JIT exactly at prompt length 17 (16-token block
    boundary) on hybrid-mamba align path; probe showed **registration timing
    matters**: registering from lazily-created buffers is *after* the warmup
    registry closes → runtime JIT persists; registering while
    `GPUModelRunner.jit_warmup_registry` is active → eliminated.
  - CuTeDSL flashinfer JITs + `BuildPrefillChunkMetadataKernel` on GLM-5.2.
  - **Qwen3.5 GDN output projection**: `layer_norm_fwd_kernel` runtime JIT via
    `QwenGatedDeltaNet._output_projection → RMSNormGated.forward_cuda`; naive
    dummy launch compiled `ROWS_PER_BLOCK=1` while runtime needed `4`
    (compile-key mismatch lesson). Contributor took the focused subtask.
- Acceptance signal used in-thread: **`--jit-monitor-mode error`** with cold
  caches as the pass/fail gate (also CI goal via #50109).

## 2. Runtime monitor infrastructure (current, in our SHA)

- `vllm/utils/jit_monitor.py` — monitors Triton (+ CuTeDSL) compiles after
  engine warmup completes:
  - CLI `--jit-monitor-mode warn|error` (`vllm/engine/arg_utils.py:1670`,
    `vllm/config/observability.py:105`), `--jit-monitor-verbose` logs full
    specialization (constexprs/signature) per compile.
  - Origin: #46167 (CuTeDSL monitor, merged 2026-06-23); Triton + error mode
    added later (#49349 workstream).
- Warmup orchestration: `vllm/model_executor/warmup/jit_warmup.py`
  (`VllmJitKernel`-style compile-only warmups, predicate/dispatch tracing),
  `kernel_warmup.py` (provider registry; runs against
  `worker.model_runner.jit_warmup_registry`), per-family warmup modules in
  `vllm/model_executor/warmup/`.

## 3. Qwen GDN path at our SHA (static map)

Model classes: `Qwen3_5ForConditionalGeneration` /
`Qwen3_5ForCausalLM` etc. in `vllm/model_executor/models/registry.py:196-197,587`
→ `vllm/model_executor/models/qwen3_5.py:46,148` uses
`QwenGatedDeltaNetAttention` (defined in
`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:372`;
`Qwen3.8-27B-FP8` = model_type `qwen3_5` multimodal wrapper → text model with
48 GDN + 16 full-attention layers; see docs/03).

- `forward_hip` (qwen_gdn_linear_attn.py:864): if `GDN_AITER_TRITON_AVAILABLE`
  (line 74, requires `aiter` package importable + kernels present) → AITER
  fused path; **else falls back to `forward_cuda`** (generic path).
- `aiter` is NOT a dependency of the vLLM ROCm requirements
  (`requirements/rocm.txt` has no aiter) and is NOT installed in the
  experiment venv → **natural gfx1100 baseline = generic fallback path**.
- Generic prefill: `chunk_gated_delta_rule` (FLA chunked kernels, vendored at
  `vllm/third_party/flash_linear_attention/`, moved there by #48500) +
  `causal_conv1d_fn` (`_causal_conv1d_fwd_kernel`) +
  `fused_post_conv_prep` + gated RMSNorm (`layer_norm_fwd_kernel`).
- Generic non-spec decode (default flags: `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1`,
  `VLLM_GDN_DECODE_KERNEL=cuda` → on ROCm `is_cuda_alike()` true but fused
  decode unsupported → auto-fallback to "triton", qwen_gdn_linear_attn.py:516-528):
  `_forward_core_decode_non_spec` (line 1637) →
  `causal_conv1d_update` (`_causal_conv1d_update_kernel`,
  `vllm/model_executor/layers/mamba/ops/causal_conv1d.py:763,1231`) +
  `fused_recurrent_gated_delta_rule_packed_decode`
  (`fused_recurrent_gated_delta_rule_packed_decode_kernel`,
  `vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:256,456`).
- Startup prefill autotune warmup: `_warmup_prefill_kernels`
  (qwen_gdn_linear_attn.py:1060) runs during V1 profiling (startup) to
  pre-populate the FLA chunk autotuner before KV allocation.

## 4. Warmup coverage for the GDN kernel families at our SHA

`vllm/model_executor/warmup/qwen_triton_warmup.py` (introduced #49903
2026-07-28, extended by #54251 2026-09-03 "Warm up Qwen GDN gated RMSNorm" and
#54797 2026-09-07 "Extend Qwen Triton warmup to avoid first-request latency
spikes"; #56742 2026-10-02 added Qwen4Exp) warms, for model_types
{qwen3_next, qwen3_5*, qwen4_exp*}:

| Runtime kernel | Warmed at startup? | Warmup site |
|---|---|---|
| `layer_norm_fwd_kernel` (gated RMSNorm) | YES (via `warmup_layer_norm_fwd`, `VllmJitKernel.warmup`) | qwen_triton_warmup.py:148-168,305 |
| `_causal_conv1d_fwd_kernel` (prefill conv) | YES (`causal_conv1d_fn` dummy launch) | qwen_triton_warmup.py:171-205,306 |
| FLA `fused_post_conv_prep` (lengths 1,2,16) | YES | qwen_triton_warmup.py:208-234,307 |
| `fused_sigmoid_gating_delta_rule_update` | YES (non-pooling) | qwen_triton_warmup.py:237-275,310 |
| `batch_memcpy_kernel` (mamba prefix copy) | YES (mamba_triton_warmup.py, from #54797/#49903 work) | mamba_triton_warmup.py:29-53 |
| **`_causal_conv1d_update_kernel` (decode)** | **NOT FOUND in any warmup module at our SHA** | — |
| **`fused_recurrent_gated_delta_rule_packed_decode_kernel` (decode)** | **NOT FOUND in any warmup module at our SHA** (kimi_k3 warmup covers the KDA variant only) | — |
| FLA chunk prefill autotune kernels | YES (autotune cache; `_warmup_prefill_kernels`) | qwen_gdn_linear_attn.py:1060 |

Caveat: "warmed" means a warmup launch exists; **compile-key equivalence is a
separate question** (cf. the ROWS_PER_BLOCK=1-vs-4 lesson in #49349, and open
PR #52611 "Fix for Issue 52413: causal_conv1d Alignment Specialization Race
Condition" indicating alignment specialization can produce unseen variants).

## 5. Classification summary

| Item | Class |
|---|---|
| #52663 FP8 startup autotune / engine-ready timeout on gfx1100 | historical observation; still open; **startup JIT**, separable from runtime JIT |
| #52663 "43 min first-inference Triton JIT" (Aug 2026, v0.27.1+rocm723) | historical observation; hypothesis H8 territory — re-test on pinned SHA |
| #49349 zero-runtime-JIT workstream | active upstream work |
| Shared warmup infra (#47451/#49315/#50174) | current implementation (in our SHA) |
| Qwen GDN gated-RMSNorm runtime JIT (zupengwang report, main d9dabfa) | **already fixed** at our SHA by #54251 (verify at runtime) |
| `batch_memcpy` 17-token runtime JIT | **already fixed** at our SHA by #49903/#54797 (verify) |
| `_causal_conv1d_fwd_kernel` runtime JIT (Kimi/DSPARK report) | warmup exists for Qwen paths at our SHA; compile-key equivalence unverified |
| `_causal_conv1d_update_kernel` + `fused_recurrent_gated_delta_rule_packed_decode_kernel` decode runtime JIT | **no warmup found at our SHA — candidate gap** |
| PR #49930 fp8 einsum warmup | open (not in SHA) |
| PR #50109 CI `--jit-monitor-mode error` | open (not in SHA) |
| PR #52611 causal_conv1d alignment specialization race | open; hypothesis H9 relevance |
| AITER RDNA GDN work (`fused_reshape_causal_conv1d_update_single_token`, RDNA path) | current implementation behind optional `aiter` dependency; not part of natural baseline (aiter not in rocm requirements) |

## 6. What must be decided by experiment (Phase-1 runtime)

1. Does the pinned SHA reproduce ANY runtime (post-engine-ready) Triton JIT on
   gfx1100 with cold caches for Qwen3.8-27B-FP8 — and for which kernels?
2. Are the decode-path kernels (`_causal_conv1d_update_kernel`,
   `fused_recurrent_gated_delta_rule_packed_decode_kernel`) compiled at
   startup (e.g. via profile/dummy runs) or only at first real decode?
3. Do the newly-added warmups (#54251/#54797) cover the exact runtime compile
   keys (specialization equality), or do alignment/constexpr variants escape?
4. Is FP8 W8A8 autotune confined to startup (STARTUP JIT) on this host?
