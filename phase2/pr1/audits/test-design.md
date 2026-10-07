# Phase-2 PR-1 — Test-Design Review (2026-10-07)

Independent maintainer-style review of the proposed test BEFORE
implementation (verbatim subagent output).

## Answers

**1. Smallest test?** Almost. Right level (e2e is the only level where
"warmup covers runtime shapes" is observable), but the first draft
duplicated the runner body; parameterize the shared runner instead (see 8).

**2. Does it exercise Qwen GDN?** Yes on kernel families.
`qwen3_5.py:256` picks `layer_types[extract_layer_index(prefix)]`;
`layer_types[0] == "linear_attention"`, and `dummy_hf_overrides` sets
`num_hidden_layers=1` while leaving `layer_types` 24-long. Plain
truncation gives 1 GDN layer and **0 full-attention layers** — not the
config anyone serves and an untested hybrid-corner (registry precedent
keeps a full-attention layer: Ling 3.0 registry.py:785-790). Recommended:
wrapper keeping `["linear_attention", "full_attention"]` with
`num_hidden_layers=2`. GDN kernels are per-layer-type and shared across
layers, so one layer exercises conv1d prefill/update, packed recurrent
decode (env default on), FLA chunk prefill; the mixed prefill+decode
sigmoid-gating path is covered by MRV2's mixed warmup, not this battery.
"mamba align" is out of scope (mamba_cache_mode default "none").

**3. Redundant?** No. The dense test never runs a GDN kernel; the generic
module is skipped and lists no GDN model. Only overlap is the sampler half
of the battery (harmless reuse).

**4. Dummy loading sufficient?** Yes. Every monitored kernel specializes on
shapes/dtypes/strides/constexprs, not weight values — the same assumption
underpins the existing jit_monitor tests and `_warmup_prefill_kernels`.

**5. Cache isolation?** Correct. monkeypatch.setenv mutates the parent env
before spawn; EngineCore is a spawned descendant; identical to the proven
dense pattern.

**6. ROCm scope / AITER=0 — the big one.** With `VLLM_ROCM_USE_AITER=0`,
`forward_hip` falls back to `forward_cuda` (generic FLA path) — the
ROCm-specific AITER GDN path is never executed. Either cover AITER too or
explicitly document AITER-off-only scope in the docstring. (Resolution
adopted: AITER=0 documented in the docstring; AITER out of scope — this
matches the existing ROCm jit-monitor test's contract and avoids depending
on optional AITER availability.)

**7. Resources.** Dummy bf16 ≈ 0.51 GB tied embedding + 1 GDN layer + tiny
vision tower ≈ 0.65 GB. On MI355 (288 GB): 0.03 → ~8.6 GB, fine. On 48 GB:
0.03 → 1.44 GB — tight but validated. KV: mamba state ~1 MB fp32 + conv
~36 KB per slot; 256 MiB ≈ 250 slots ≫ max_num_seqs=8. Fine.

**8. Conventions.** Deduplicate the runner
(`_run_rocm_shape_battery(model, hf_overrides)`); fill Google-style
docstring (model, truncation, AITER, runner, battery scope); update the
module docstring.

**9. VLLM_USE_V2_MODEL_RUNNER=1** — V2 is the default at this SHA;
Qwen3.5 supports V2 (mamba_hybrid model states, MRV2-aware gdn_attn).
Pinning is correct, default-matching explicitness.

**10. Vision tower.** Config has vision_config; tower loads (truncated)
but the battery is text-only TokensPrompt; the monitor activates only
after compile/warmup completes, so the tower cannot false-positive.
Hub access for config/tokenizer required in CI (repo id valid,
registry.py:1332).

## Verdict: APPROVED WITH CHANGES

1. (Blocker) Resolve the AITER question — adopted: AITER=0, documented as
   generic-path-only, AITER out of scope.
2. (Blocker) Keep a full-attention layer via the wrapper override —
   adopted: `_gdn_hf_overrides` with `num_hidden_layers=2`,
   `layer_types=["linear_attention", "full_attention"]`.
3. Deduplicate the runner — adopted.
4. Docstrings — adopted.
5. (Nit) memory fraction — kept 0.03 (validated locally; same as the
   existing test).
