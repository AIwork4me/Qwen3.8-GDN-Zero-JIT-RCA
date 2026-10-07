# Phase-2 PR-1 — Implementation Audit (2026-10-07)

Patch under review: `tests/jit_monitor/test_no_runtime_jit_rocm.py` only
(diff 98 lines, applied on upstream main `68088ed3`, worktree
`/workspace/vllm-rocm-gdn-zerojit-pr1`, branch `rocm-gdn-zero-runtime-jit-test`).

## Method

Reviewed as a vLLM maintainer, verifying every claim against the tree at this
SHA rather than trusting the patch description:

1. **Diff & scope**: read `diff.patch` in full; `git status`/`git log` in the
   worktree to confirm exactly one file changed on top of `68088ed3`.
2. **Test file**: read the patched file line-by-line against the pre-patch
   version (recovered from the diff context) to confirm the existing dense
   test's behavior is preserved exactly (env set, order, LLM args,
   assertions) and that the shared env/battery helpers are pure extractions.
3. **Shared machinery**: read `tests/jit_monitor/test_no_runtime_jit.py`
   (`_run_shape_battery`, module-level `skip` mark — checked it does not
   propagate through import), `tests/utils.py::spawn_new_process_for_each_test`
   (env propagation via `os.environ.copy()`, tests/utils.py:1944-1952),
   `tests/jit_monitor/conftest.py` (autouse monitor-reset fixture).
4. **Runtime sources at this SHA**:
   - `vllm/model_executor/models/qwen3_5.py:147-165,253-262` — layer-type
     routing and `config.layer_types` indexing;
   - `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
     (`QwenGatedDeltaNetAttention` :372, `forward_hip` :864-898, GDN prefill
     backend resolver :96-149 — returns `"triton"` on all non-CUDA platforms);
   - `vllm/_aiter_ops.py` — `_AITER_ENABLED = envs.VLLM_ROCM_USE_AITER`
     (:1999), `are_gdn_triton_kernels_available` (:2472-2479),
     `is_rdna_aiter_enabled` (:2123-2133), env-independence docstring
     (:205-217);
   - `vllm/transformers_utils/config.py:363-374, 968-973` — both
     `hf_overrides` call sites (dummy-config `.model_type` probe and
     `config = hf_overrides_fn(config)` reassignment);
   - `vllm/utils/jit_monitor.py` — `is_active`, `activate(mode="error")`,
     error message text ("... during inference: ...");
   - `vllm/model_executor/warmup/qwen_triton_warmup.py` — `_QWEN_MODEL_TYPES`
     contains `qwen3_5_text` (:22), `_warm_fused_post_conv_kernel` (:208,
     invoked :307);
   - `.buildkite/test_areas/jit_monitor.yaml` — AMD MI355 mirror:
     `timeout_in_minutes: 10`, `pytest --timeout=300`, `source_file_dependencies`
     already includes `tests/jit_monitor/`.
   - `tests/models/utils.py::dummy_hf_overrides` (:448-611) — mutation order,
     `num_hidden_layers=1` default, `layer_types` (not truncated for text
     config; only `vision_config.layer_types` and `indexer_types` are).
5. **Evidence review**: baseline (137.3s PASS), positive (GDN alone 277.5s /
   263.1s post-revert; full file 391.8s; `gdn-execution-proof.txt` — arch
   `Qwen3_5ForConditionalGeneration`, `layer_types [linear_attention,
   full_attention]`, gdn_layers=1/full=1, monitor `(True,'error')`, 2777→2777
   cache files (0 created across battery), six in-tree FLA GDN kernel families
   compiled+executed), negative control (no-op of `qwen_triton_warmup` → FAIL
   with `RuntimeError: Triton kernel JIT compilation during inference:
   _fused_post_conv_kernel`, then reverted and re-confirmed green). Evidence
   chronology is internally consistent (08:16 → 09:51).
6. **Tooling**: re-ran `ruff check` + `ruff format --check` (clean) and
   `pytest --collect-only` (2 tests collected, no import errors).

## Findings (numbered, severity-tagged)

1. **[PASS] Scope: tests-only, minimal, no runtime changes.** `git status`
   shows only `tests/jit_monitor/test_no_runtime_jit_rocm.py` modified. The
   98-line diff is three mechanical pieces: module docstring update, env-block
   extraction into `_isolate_rocm_jit_caches`, parameterization of
   `_run_rocm_shape_battery`, plus the new test and the `_gdn_hf_overrides`
   wrapper. The existing `test_v2_sampler_warmup_rocm` keeps an identical env
   set (same seven variables, same order), identical LLM kwargs, and identical
   assertions; only the call shape changed. Follows AGENTS.md "reuse before
   create" (extends an existing file, reuses `_run_shape_battery`,
   `dummy_hf_overrides`, `create_new_process_for_each_test`).

2. **[PASS] Model routing is correct at this SHA.** `Qwen/Qwen3.5-0.8B`
   resolves to `Qwen3_5ForConditionalGeneration` (evidence log, model.py:730).
   `Qwen3_5DecoderLayer` routes `"linear_attention"` →
   `QwenGatedDeltaNetAttention` (qwen3_5.py:147-154; class defined at
   `layers/mamba/gdn/qwen_gdn_linear_attn.py:372`) and `"full_attention"` →
   `Qwen3NextAttention` (qwen3_5.py:155-163), with `make_layers` indexing
   `config.layer_types` (qwen3_5.py:256). The wrapper's
   `["linear_attention", "full_attention"]` therefore yields exactly one GDN +
   one full-attention layer, confirmed by the evidence worker_state
   (`gdn_layers: 1, full_attn_layers: 1`). The wrapper docstring correctly
   explains why plain `dummy_hf_overrides` (num_hidden_layers=1) would drop
   all full-attention layers for this hybrid.

3. **[PASS] ROCm assumptions verified.** The module-level
   `pytestmark = skipif(not current_platform.is_rocm())` covers both tests.
   `VLLM_ROCM_USE_AITER=0` semantics checked at this SHA: both the CDNA gate
   (`are_gdn_triton_kernels_available` → `_AITER_ENABLED`,
   _aiter_ops.py:2479) and the RDNA4 gate (`is_rdna_aiter_enabled` →
   `_AITER_ENABLED`, _aiter_ops.py:2133) honor the env var, so
   `GDN_AITER_TRITON_AVAILABLE` is False and `forward_hip` falls back to the
   generic `forward_cuda` FLA path (qwen_gdn_linear_attn.py:870-898). GDN
   prefill is Triton/FLA on all non-CUDA platforms regardless
   (qwen_gdn_linear_attn.py:118-119). The test docstring's "in-tree generic
   Triton GDN path" claim is accurate, and the kernel-family evidence
   (`_causal_conv1d_*`, `fused_recurrent_gated_delta_rule_*`,
   `fused_sigmoid_gating_delta_rule_*`, `chunk_fwd_kernel_o` — all in
   `vllm/third_party/flash_linear_attention/`) confirms the generic path
   actually executed.

4. **[PASS] Cache isolation is sound.** `monkeypatch.setenv` in the test
   process → `spawn_new_process_for_each_test` copies `os.environ` into the
   child (tests/utils.py:1944-1952) → EngineCore is itself spawned
   (`VLLM_WORKER_MULTIPROC_METHOD=spawn`) and inherits again. Each test gets a
   pytest-unique `tmp_path`, so `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT` (which
   carries the AOT/torch.compile cache), and `TORCHINDUCTOR_CACHE_DIR` are
   fresh per test — a previous run's or the sibling test's cache cannot hide a
   missing warmup specialization. The evidence (0 cache files created across
   the battery) confirms the battery ran purely on startup-compiled kernels.

5. **[PASS] Monitor-activation assertion is adequate for the known false-green
   mode.** `collective_rpc(_worker_monitor_state)` runs in the engine-core
   worker (where the process-global hooks live) before the battery, and
   `assert states and all(state == (True, "error"))` fails loudly if the
   monitor never activated or the mode string changed/renamed — the classic
   "green because the monitor was off" failure. Activation ordering is safe
   (monitor arms at the end of engine warmup; the RPC happens after LLM init).
   The autouse reset fixture in `tests/jit_monitor/conftest.py` keeps parent
   state clean between tests. This assertion was already present for the dense
   test and is preserved verbatim.

6. **[PASS] The test protects GDN zero-runtime-JIT, not just startup.** The
   negative control is decisive: no-op'ing `qwen_triton_warmup` alone makes
   the new test FAIL with `RuntimeError: Triton kernel JIT compilation during
   inference: _fused_post_conv_kernel`, and that kernel is warmed exactly at
   `qwen_triton_warmup.py:307` for `model_type=qwen3_5_text`. Combined with
   the positive-side proof (six in-tree FLA GDN families compiled+executed
   from the startup cache, zero compiles during the battery), the test
   genuinely pins the GDN warmup contract. The mutation was fully reverted and
   the test re-confirmed green (post-revert-confirm.txt).

7. **[MAJOR — required companion change in the same PR] CI budget is too
   tight for the measured runtimes.** The AMD mirror in
   `.buildkite/test_areas/jit_monitor.yaml` runs the whole file under
   `timeout_in_minutes: 10` and `pytest --timeout=300` (per test). Measured:
   GDN test alone 263–278s → only ~8–12% headroom under the 300s per-test
   watchdog; full file 392s (~35% headroom on the 10-min job). Two
   aggravators: (a) evidence was collected on a W7900D (RDNA3 workstation),
   while CI runs MI355 DPX (CDNA4) — timing parity unproven; (b) CI adds cold
   HF download of the Qwen3.5-0.8B config/tokenizer. A per-test timeout flake
   would be indistinguishable from a real regression in the dashboard.
   Keeping both tests in one file is fine organizationally (they share the
   helpers by design); the fix is step config: bump `--timeout` to ~600s and
   `timeout_in_minutes` to ~15–20, or split the GDN test into its own mirror
   step. `source_file_dependencies` already covers `tests/jit_monitor/`, so no
   dependency edits are needed. Shipping the test without touching the yaml
   risks immediate CI flakes.

8. **[PASS] `hf_overrides` wrapper is correct for both call sites.** The
   wrapper returns `config` (required: config.py:973 does
   `config = hf_overrides_fn(config)` and config.py:373 reads
   `.model_type` off the return value). Mutation order is correct:
   `dummy_hf_overrides` first sets `num_hidden_layers=1` (via `update_dict`),
   then the wrapper raises it to 2 and replaces `layer_types`; the wrapper's
   values win. At the dummy-config call site the extra attributes land on a
   throwaway `PreTrainedConfig` and `model_type` is untouched, so architecture
   resolution is identical to plain `dummy_hf_overrides`. `text_config =
   hf_config.get_text_config()` returns the same object the helper already
   mutated. Residual sharp edge checked: `dummy_hf_overrides` truncates
   `indexer_types` to its own `num_hidden_layers=1`; Qwen3.5 text config has
   no `indexer_types` (and the run passed), so no length mismatch arises.

9. **[MINOR] Style/typing nits, non-blocking.** (a) New helper params are
   partially unannotated: `hf_overrides` in `_run_rocm_shape_battery`,
   `tmp_path` in `_isolate_rocm_jit_caches`, `hf_config` in
   `_gdn_hf_overrides` (sibling code annotates more; ruff/mypy configured
   here do not enforce). (b) Docstrings omit Google `Args:`/`Returns:`
   sections — consistent with the neighboring helpers in both jit_monitor
   files, so acceptable, but a maintainer may ask. (c) The battery helper
   import from a module whose collection is module-level skipped
   (`test_no_runtime_jit.py`) is safe (the mark applies only to tests
   collected in that module) — verified, no issue, noting for reviewers.

10. **[LOW] Residual false-green surface (pre-existing, not introduced
    here).** (a) No CI test exercises the real Triton
    `jit_post_compile_hook` contract against an installed Triton
    (`test_hooks.py` mocks `knobs`); a Triton upgrade silently breaking the
    hook would turn both ROCm tests green while detecting nothing. The manual
    negative control covers this SHA only — consider a periodic canary (a
    tiny never-warmed Triton kernel asserted to raise under `mode="error"`).
    (b) The battery bounds the contract: prefill 4/32/128 tokens, decode ≤16
    steps, batch ≤4 (decode shapes beyond capture sizes rely on cudagraph
    padding/replay, where misses surface at capture time, pre-activation, not
    to the monitor); spec-decode GDN (`qwen_gdn_attention_core_fused_norm_packed`)
    and the AITER GDN decode path are out of scope — the docstring says so
    for AITER; same philosophy as the dense test. (c) New external dependency
    for this CI step: Qwen/Qwen3.5-0.8B config/tokenizer download — fails red
    (not green) if unavailable, but it is a new availability dependency.
    (d) The 0.8B checkpoint is multimodal (`Qwen3_5ForConditionalGeneration`),
    so the test also pays for ViT construction + multimodal warmup (~15-90s
    of its runtime); unavoidable if this family is the target, but it is the
    main reason finding 7's margin is thin.

## Verdict: PASS WITH NOTES

The test change itself is correct, minimal, convention-following, and
evidence-backed (positive + negative control + revert confirm). Merge after
addressing:

- **Required (same PR): finding 7** — adjust `.buildkite/test_areas/jit_monitor.yaml`
  (raise `--timeout` / job timeout, or split the GDN test into its own step).
  Tests-only is fine for the code; the CI step config is in-repo and currently
  sized for one test.
- **Optional: finding 9** typing nits; **finding 10a** hook canary idea for a
  follow-up.
