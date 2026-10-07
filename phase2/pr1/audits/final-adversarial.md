# Phase-2 PR-1 — Final Adversarial Review (2026-10-07)

Branch `rocm-gdn-zero-runtime-jit-test` at `/workspace/vllm-rocm-gdn-zerojit-pr1`
(base upstream main `68088ed3927e91bce8918db78f2efd9e644a4134`). Materials audited:
`diff.patch` (verified byte-identical to the worktree `git diff`), `design.md`,
`duplicate-search.md`, `upstream-base.json`, evidence
{baseline, positive/{test_qwen_gdn_no_runtime_jit_rocm.txt, full_file.txt,
post-revert-confirm.txt, gdn-execution-proof.txt}, negative-control/mutation1-gdn-warmup-noop.txt},
audits {implementation.md, test-evidence.md}, and the vLLM sources at the base SHA.

**Materials discrepancy up front**: `phase2/pr1/pr-draft.md` — listed as a review
input and the artifact the overclaim sanity-check targets — **does not exist**
(not at the stated path, nor anywhere under `/workspace`). The overclaim audit
of the PR description therefore **could not be performed**; see Blocking issues.
(`audits/duplicate-work.md` / `audits/test-design.md` are also absent, but
equivalents exist as `duplicate-search.md` / `design.md`; only the draft has no
equivalent anywhere.)

## Method

Adversarial pass with independent re-verification, not a read-through of the
prior audits:

1. Confirmed the worktree is exactly base `68088ed3` + the two-file diff
   (`git status`: only `.buildkite/test_areas/jit_monitor.yaml` and
   `tests/jit_monitor/test_no_runtime_jit_rocm.py` modified; `git diff` matches
   `diff.patch` after stripping `index` lines).
2. Read the patched test file and the pre-patch contract line-by-line; confirmed
   the dense test's env set (order included), LLM kwargs, and assertions are
   preserved verbatim through the extraction/parameterization
   (test_no_runtime_jit_rocm.py:66-87 vs. the old inline block).
3. Re-derived the load-bearing runtime facts from source at this SHA:
   `qwen_triton_warmup.py:22-31,296-318` (model-type gate; `_fused_post_conv_kernel`
   warmup), `qwen3_5.py:147-163,248-262` (layer routing by `config.layer_types`),
   `qwen_gdn_linear_attn.py:94-149,864-898` (prefill resolver; `forward_hip` AITER
   gate), `_aiter_ops.py:1999,205-217,2123-2133,2472-2479` (AITER env gates),
   `envs.py:138,1264-1266` (`VLLM_ROCM_USE_AITER` default False),
   `config.py:363-374,968-973` (both `hf_overrides` call sites),
   `gpu_worker.py:912-924,1035-1059` (warmup → capture → monitor-activation
   ordering; eager off-switch), `jit_monitor.py:15-21,46-48,84,120-140`
   (hook scope, `is_active`, error path), `tests/models/utils.py:448-611`
   (`dummy_hf_overrides` mutation order; `layer_types` only truncated for
   vision/indexer configs), `tests/utils.py:1900-1960` (spawn wrapper:
   cloudpickled args, `env = os.environ.copy()`),
   `tests/jit_monitor/conftest.py` (autouse monitor reset).
4. Re-checked tooling: `ruff check` + `ruff format --check` clean;
   `pytest --collect-only` collects exactly 2 tests, no import errors; import of
   `_run_shape_battery` from the module-skipped generic file is safe (skip mark
   applies at collection of that module only).
5. Re-read all four positive evidence artifacts, the baseline, and the full
   negative-control log (mutation diff, hook-chain traceback, battery-item
   chronology, exit codes); cross-checked timing arithmetic against the CI yaml.
6. Searched the tree for duplicate jit-monitor usage (`grep -rl jit_monitor
   tests/`) and for GDN-kernel reachability outside GDN layers.
7. Attempted each of the 10 attack angles as specified; outcomes below.

## Attack angles and outcomes (10 numbered)

### 1. Duplicate coverage — REFUTED (no duplicate; one process caveat)

- The sibling `test_v2_sampler_warmup_rocm` runs dense `Qwen/Qwen3-0.6B`
  (model_type `qwen3`). `qwen_triton_warmup` early-returns for any model_type
  outside `_QWEN_MODEL_TYPES = {qwen3_next, qwen3_5, qwen3_5_text, qwen3_5_moe,
  qwen3_5_moe_text, qwen4_exp, qwen4_exp_text}` (qwen_triton_warmup.py:22-31,
  gate at :301-303), and a dense model constructs no
  `QwenGatedDeltaNetAttention` (only `linear_attention` layers do;
  qwen3_5.py:147-154). Zero kernel overlap; the new test is the only one in the
  file that compiles or executes any GDN kernel.
- The generic `tests/jit_monitor/test_no_runtime_jit.py` is module-skipped
  (:21-24, pending #49349) and its `JIT_MONITOR_MODELS` (:34-49) has no GDN
  architecture; grep for `gdn|QwenGatedDeltaNet|qwen3_5|qwen3_next` under
  `tests/jit_monitor/` (pre-patch) → zero matches. Even if Qwen3.5 were added
  there it would collect nothing — the module skip suppresses it.
- `duplicate-search.md` (2026-10-07, base-verified) found no open PR touching
  the ROCm file or adding GDN jit-monitor coverage: #58092 (merged) authored the
  file's only prior commit; open #56693 rewrites the generic model list
  (DeepSeek-V4/Gemma-4/gpt-oss, no GDN, doesn't touch this file); merged
  #54251/#54797 are warmup unit tests, not e2e jit-monitor regressions.
- **Caveat (process, non-blocking)**: the search is a snapshot against main tip
  `43b4aaea` (2026-10-07 04:55 UTC). AGENTS.md requires `gh pr list` checks at
  PR-open time; re-run before submission.

### 2. Too slow for CI / is the bump offensive — NOT BLOCKING (acceptable; one arithmetic nit)

- Measured: GDN test alone 277.51s and 263.10s (positive, post-revert), full
  file 391.79s, baseline single dense test 137.30s — all on W7900D (gfx1100).
  New budget: per-test `--timeout=600` (~2.2x headroom on the slowest measured
  GDN run) and job `timeout_in_minutes: 20` (~3x headroom on the full file).
- The bump is confined to the single AMD mirror step in
  `.buildkite/test_areas/jit_monitor.yaml` (no other lane touched) and stays
  within repo norms: `kernels.yaml` steps run 20-26 min, `lm_eval.yaml` up to
  40+; the NVIDIA jit-monitor lane itself is 45 min / `--timeout=900` per test.
  A dedicated regression lane paying ~6.5 min wall for two e2e engine boots is
  proportionate; no `source_file_dependencies` changes were needed
  (`tests/jit_monitor/` already listed, yaml:35).
- **Nit**: per-test 600s x 2 tests = 1200s equals the 20-min job limit, so in a
  both-tests-hang scenario Buildkite kills the job before either pytest
  watchdog reports. Cosmetic (that scenario is red regardless); 25 min or
  per-test 480s would be tidier if a maintainer cares.
- Honest weaknesses already on record (test-evidence audit §6): timings are
  gfx1100-only; MI355 (CDNA4) adds cold HF config/tokenizer/processor fetch and
  may select different backends. Failure mode in either case is a loud red
  timeout, not a false green — acceptable for a regression lane, and the first
  CI run on the PR will settle MI355 parity.

### 3. Does 1 GDN layer really exercise "Qwen GDN" — YES (kernel coverage is depth-invariant); docstring first line slightly broad

- GDN Triton compile keys depend on head dims/dtypes/shape template params, not
  on layer count; the truncation preserves the real Qwen3.5-0.8B text-config
  dims (proof log: `Using Triton/FLA GDN prefill kernel (requested=auto,
  head_k_dim=128)`, qwen_gdn_linear_attn.py:177). The wrapper *forces*
  `layer_types=["linear_attention","full_attention"]` and `num_hidden_layers=2`
  (test:38-41), so layer 0 is GDN by construction regardless of upstream config
  drift — `make_layers`' `get_layer` indexes `config.layer_types`
  (qwen3_5.py:248-256). `dummy_hf_overrides` sets `num_hidden_layers=1` first
  (tests/models/utils.py, `update_dict`) and never truncates *text*
  `layer_types` (only vision/indexer), so the wrapper's assignments compose and
  win. Runtime confirmation: worker_state `gdn_layers: 1, full_attn_layers: 1`
  (gdn-execution-proof.txt:85); six in-tree FLA GDN kernel families compiled and
  executed with zero post-battery cache writes (proof :137-144). The executed
  implementation is the production `qwen_gdn_attention_core` splitting-op path —
  the negative-control failure traceback runs through
  qwen_gdn_linear_attn.py:1932/:1424 into
  `fused_gdn_prefill_post_conv.py:215`.
- **Wording nit (non-blocking)**: the docstring's first line — "Qwen GDN models
  must not JIT-compile during inference on ROCm" — is broader than the artifact
  (one checkpoint family, `qwen3_5`, one graph config, one battery). The body
  scopes it correctly ("every compile key the battery needs", AITER out of
  scope); recommend tightening the first line to "The Qwen3.5 GDN
  (linear-attention) path..." when the draft is authored. Same for the module
  docstring's "the Qwen GDN (Gated DeltaNet) linear-attention path" — accurate
  as a path statement, fine as is.

### 4. Monitor inactive while test passes — assertion is sufficient for activation; hook *efficacy* remains an unasserted, pre-existing gap

- The pre-battery `collective_rpc(_worker_monitor_state)` executes in the
  EngineCore worker process (uniproc executor) and asserts
  `state == (True, "error")` (test:59-60): any config drift that disarms the
  monitor (e.g., the `enable_jit_warmup` off-switch at gpu_worker.py:1056-1059,
  a mode-string rename) turns the test red. Activation ordering is warmup →
  capture → activate (gpu_worker.py:912-924, :1040), so the state read is
  post-startup by construction; runtime logs confirm
  (`Kernel JIT monitor activated ... mode=error` then worker_state
  `[true, "error"]`).
- **Residual (pre-existing, non-blocking)**: the assertion proves activation,
  not that the installed Triton `jit_post_compile_hook` actually fires — a
  future Triton API change could leave both ROCm tests green while detecting
  nothing (`test_hooks.py` mocks `knobs`; no CI test exercises the real hook
  contract). The negative control proves efficacy at this SHA only. This gap is
  inherited from the existing dense test and the monitor itself, not introduced
  here; the suggested canary (tiny never-warmed kernel asserted to raise under
  `mode="error"`) is the right follow-up (implementation.md finding 10a).

### 5. Cache contamination between tests/runs — REFUTED (mechanism + empirics)

- All three compile caches are redirected under the per-test pytest `tmp_path`
  (`TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT` incl. torch.compile/AOT,
  `TORCHINDUCTOR_CACHE_DIR`, test:75-77); the double spawn (test child via
  `create_new_process_for_each_test` with `env = os.environ.copy()`,
  EngineCore via `VLLM_WORKER_MULTIPROC_METHOD=spawn`) carries them to where
  compilation happens — proven empirically by the negative-control log showing
  the pytest `tmp_path` inside the EngineCore process
  (`.../test_qwen_gdn_no_runtime_jit_r0/vllm/torch_compile_cache/...`).
- The decisive control: the negative-control run (09:38) executed *after* both
  positive runs (08:34, 08:42) on the same machine/venv and still FAILED on a
  runtime Triton compile — a shared or stale cache that could mask a missing
  warmup key would have produced a pass exactly there. Positive-side
  corroboration: `startup=2777 final=2777 created_after_battery=0`
  (gdn-execution-proof.txt:137).
- Sequential GPU sharing between the two tests in one file is safe: each test
  runs in its own spawned child with its own EngineCore, `shutdown(timeout=30)`
  in a `finally` (test:62-63), and the autouse conftest fixture resets the
  parent's monitor globals. No contamination vector found.

### 6. ROCm-only scoping unjustified — DEFENSIBLE (it is the only live lane; CUDA belongs to #49349)

- The generic jit-monitor battery is module-skipped pending #49349
  (test_no_runtime_jit.py:21-24) — the H200 lane currently runs a file that
  collects nothing; the AMD mirror of this yaml is the only jit_monitor e2e
  lane that actually executes in CI. Adding GDN coverage anywhere else would
  add zero executed tests.
- On CUDA, Qwen3.5 GDN zero-JIT is a *live open problem* (#49349 comment
  reports an inference-time Triton JIT in the Qwen3.5 GDN output-projection
  path on CUDA main). A CUDA twin of this test would be red on main today —
  that would be a bug-fix PR, not regression coverage, and is exactly what the
  PR must not claim to be. ROCm-only is the honest, green-able increment; the
  AITER-disabled generic path is also the deterministic one (see #8).
- Consequence for the (missing) PR draft: it must scope the contribution to
  ROCm regression coverage and must not claim progress on #49349's CUDA item.

### 7. Does the dense battery already compile GDN kernels — NO (verified, not assumed)

- Kernel reachability: the Qwen-family GDN kernels are reachable only through
  `QwenGatedDeltaNetAttention` (constructed solely for `linear_attention`
  layers of qwen3_5/qwen3_next/interns2_mobius/qwen4_exp; grep confirms) or the
  analogous kimi/olmo GDN modules — none of which exist in dense Qwen3-0.6B.
  Independently, the GDN warmup itself is model-type-gated
  (qwen_triton_warmup.py:301-303) and never runs for `qwen3`.
- Nuance for the record (does not change the verdict): some *shared* kernels
  (`causal_conv1d_update`, `fused_recurrent_gated_delta_rule`) are also used by
  the Mamba2 family (`mamba_mixer2.py`) — but the dense sibling test uses a
  dense Qwen3 model with no SSM layers either, so the docstring's "kernels ...
  that dense models never touch" is accurate for what this file runs.

### 8. Hidden AITER dependence — REFUTED (deterministic generic path)

- `VLLM_ROCM_USE_AITER` defaults to False (envs.py:138, parser at :1264-1266)
  and the test explicitly sets `"0"` (test:72) in the pytest parent *before*
  the spawn chain, so `_AITER_ENABLED` (read at class-definition/import time of
  `_aiter_ops` in the fresh child) is False regardless of whether aiter is
  installed on the MI355 image.
- Both GDN AITER gates honor it: CDNA `are_gdn_triton_kernels_available`
  returns `cls._AITER_ENABLED and importable` (_aiter_ops.py:2472-2479) and
  RDNA4 `is_rdna_gdn_triton_kernels_available` routes through
  `is_rdna_aiter_enabled` = `on_rdna4() and cls._AITER_ENABLED`
  (:2123-2133). Hence `GDN_AITER_TRITON_AVAILABLE` is False → `forward_hip`
  falls back to the generic `forward_cuda` FLA path (qwen_gdn_linear_attn.py:
  864-898), and the prefill backend resolves to `"triton"` for all non-CUDA
  platforms (:118-119). AITER presence/absence in CI cannot change which GDN
  kernels run; the executed generic path is confirmed by the cache-inventory
  families (all under `vllm/third_party/flash_linear_attention/`).

### 9. Overengineered — NO (minimal, reuse-conformant; buildkite delta is 3 lines)

- The test-file refactor is three mechanical moves (env block →
  `_isolate_rocm_jit_caches`, battery → parameterized
  `_run_rocm_shape_battery(model, hf_overrides)`, plus `_gdn_hf_overrides`).
  The alternative — copy-pasting the 7-env block and 11 LLM kwargs — duplicates
  ~20 lines and violates the repo's "reuse before create" guidance; the
  parameterization preserves the dense test byte-for-byte in behavior (same env
  order, same kwargs, same assertions). Pushing `num_layers`/`layer_types`
  knobs into shared `tests/models/utils.py` would widen blast radius for one
  consumer; a file-local wrapper is the right scope. The spawn decorator
  forwards the new args correctly (cloudpickled args/kwargs,
  tests/utils.py:1906-1920; `_gdn_hf_overrides` is module-level, picklable).
- The buildkite change is exactly a label rename, `10→20` minutes, and
  `--timeout=300→600` — the minimum that matches the measured runtimes, with no
  dependency edits. Not offensive (see #2).
- Minor: new helper params are partially unannotated (`hf_overrides`,
  `tmp_path`, `hf_config`) — consistent enough with sibling helpers; ruff/mypy
  pass. Non-blocking.

### 10. Evidence does not prove the test protects runtime JIT — REFUTED (control is valid); scope limits are honestly documented

- The mutation no-ops exactly the GDN warmup (early `return` after the
  `_QWEN_MODEL_TYPES` gate; mutation log lines 4-17), leaving every other
  warmup (generic 72-key, VL, M-RoPE, Mamba) intact. The run then fails through
  the monitor's own error path on a *GDN* kernel — full Triton hook chain
  `jit.py:904 → jit_post_compile_hook → jit_monitor.py:270/230/134 →
  RuntimeError: Triton kernel JIT compilation during inference:
  _fused_post_conv_kernel` — during battery item 2 (varlen 4/32/128 prefill),
  after item 1 had passed. Test FAILED, exit 1; mutation reverted; test re-green
  (263.10s). That is precisely the regression class the test exists to catch
  (warmup/key-coverage loss on the GDN path), demonstrated end-to-end at the
  public-API level, and it simultaneously proves the GDN kernels execute under
  dummy weights.
- Honesty of scope: the docstring credits "JIT warmup **and cudagraph
  capture**" jointly (test:95-97) — correct, and the decode-side corollary
  (FULL-graph capture compiles captured decode shapes at startup, so a
  decode-only warmup regression within captured sizes is indistinguishable from
  coverage) is the upstream-documented NOTE (test_no_runtime_jit.py:6-9) and is
  explicitly recorded in test-evidence.md §3/note 1. Within the asserted
  property — no runtime JIT for what the battery runs under this config — that
  masking is sound: production serving under the same config gets the same
  capture-time compilation. Sensitivity is bounded by battery shapes (e.g., a
  single dropped warmup length outside the battery's varlen totals is unproven)
  — acknowledged in test-evidence.md §5 caveat. I consider the PR honest about
  all of this at the code/doc level.
- Two durable-coverage residuals worth stating (non-blocking, from
  test-evidence.md notes 1-2): (a) no in-test assert that a GDN layer actually
  executed — a future routing refactor that silently drops `linear_attention`
  modules would leave the test green while guarding nothing (a one-line
  `collective_rpc` module-introspection assert would close it; consider it in
  this PR or a follow-up); (b) the monitor-hook-efficacy gap from #4.

## PR-draft overclaim sanity check — COULD NOT BE PERFORMED

`phase2/pr1/pr-draft.md` does not exist. What *can* be checked is clean: the
diff's docstrings, `design.md`, and both audits frame the change strictly as
regression coverage; nothing in any material claims to fix an AMD bug or
#52663/#49349. But the actual PR description — the text maintainers read — is
unwritten and unauditable, and AGENTS.md additionally requires it to carry the
duplicate-work statement, test commands/results, and the AI-assistance
disclosure.

## Blocking issues (or none)

Exactly one, and it is procedural, not a code defect:

1. **PR draft missing** (`phase2/pr1/pr-draft.md` absent) — the mandated
   overclaim check ("must not claim to fix any AMD bug or #52663/#49349 — only
   add regression coverage") cannot be completed, and AGENTS.md-required PR
   description content (non-duplication statement, test commands + results,
   AI-assistance statement) has no artifact. Fix: author the draft with those
   constraints and contents, then this audit's overclaim section can be cleared
   verbatim.

No blocking issues in the code diff, the CI change, or the evidence. Recommended
non-blocking fixes to fold in while authoring the draft: tighten the new test's
first docstring line to "The Qwen3.5 GDN path" (attack 3); optionally add the
cheap GDN-execution introspection assert (attack 10a); optionally resolve the
600s x 2 == 20 min timeout arithmetic (attack 2 nit); re-run the duplicate
search against current main at PR-open time (attack 1 caveat). For the record:
the multimodal 0.8B checkpoint pays ViT + 14.7s multimodal warmup; a text-only
registry example (`codecho/Qwen3.5-0.8B-text-only`, registry.py:495) would trim
runtime, but the canonical `Qwen/` id is the defensible choice — noting the
alternative, not requesting it.

## Verdict: READY FOR MAINTAINER REVIEW / NOT READY

**NOT READY** — solely on Blocking issue 1 (PR draft absent; overclaim audit
incomplete). The code diff itself has zero blocking findings: it is correct at
this SHA, minimal, duplicate-free, deterministic on the AITER axis,
cache-isolated, monitor-asserted, and supported by a valid positive/negative/
revert evidence triplet; the CI budget bump is proportionate and confined to
one step. Clearing the draft item (a document-writing task, no code change)
converts this verdict to READY FOR MAINTAINER REVIEW without re-review of the
diff.

## Closure re-check (post-fixes)

Re-audit 2026-10-07 (post 10:13 UTC). Inputs: `phase2/pr1/pr-draft.md` (now
existing, written 10:13 UTC), the worktree at `/workspace/vllm-rocm-gdn-zerojit-pr1`
(HEAD `7c0ce7b0`, clean status), `diff.patch` (regenerated 10:13), the fork
commit via GitHub API, and the unchanged evidence set.

### (a) Overclaim check on pr-draft.md — ALL REQUIRED NON-CLAIMS HOLD

- **No claim to fix any AMD bug**: PASS. The only "fix" language is the
  explicit negation in Non-goals ("Not a fix for any runtime-JIT bug: current
  main passes this test"); the Problem/What-this-adds sections claim coverage
  only ("This PR closes that gap" — antecedent is the CI coverage gap, no
  issue number, no GitHub autoclose keyword).
- **No claim to fix RDNA3/GDN runtime JIT**: PASS. GDN runtime-JIT history is
  cited as motivation only, with the historical fix credited to #54251, not
  this PR.
- **No W7900 issues claimed/fixed**: PASS. `W7900D` appears once, as the
  honest validation-environment line.
- **No #52663 claim**: PASS. Zero occurrences of #52663 in the draft.
- **No #49349 fix/progress claim**: PASS. Two mentions, both contextual
  ("the acceptance gate tracked in #49349"; the gated-RMSNorm report
  "addressed by #54251"); Non-goals explicitly defers the CUDA lane
  ("enabling it is a separate effort"). No closes/fixes/progresses wording.
- **Factual claims vs evidence**: all verified — baseline 137.30 s PASS
  (baseline log:16); new test 277.51 s / post-revert 263.10 s ("263–278 s"
  ✓); full file 2 passed 391.79 s ("392 s" ✓); monitor armed
  (`worker_state monitor [true, "error"]`, proof :85) + "ZERO raises"
  (proof :136); disk-level confirmation `startup=2777 final=2777
  created_after_battery=0` (proof :137); six GDN families PRESENT
  (proof :139-144); negative-control error text matches the draft verbatim
  (mutation log :289, FAILED exit 1, reverted, post-revert PASS). The
  "model contains QwenGatedDeltaNetAttention" claim rests on the forced
  `layer_types` hybrid + `gdn_layers: 1` + the FLA GDN prefill init line
  (proof :31) + the negative-control execution traceback — the
  `attention_module_classes` omission was already adjudicated a
  non-blocking inspection-script artifact (test-evidence.md:114-120).
  Config claim "18 linear-attention + 6 full-attention" re-verified against
  `/workspace/models/Qwen3.5-0.8B/config.json` (24 layers: 18+6). AGENTS.md
  PR-description contents (non-duplication statement, test results,
  AI-assistance disclosure) all present in the draft.

### (b) CI arithmetic fix — CONFIRMED

Worktree HEAD yaml: `timeout_in_minutes: 25`, `--timeout=600`. 2 × 600 s =
1200 s = 20 min < 25 min job limit — the both-tests-hang scenario from attack
2's nit is resolved. No other lane touched.

### (c) Diff/fork consistency — CONFIRMED

- `git diff 68088ed3 7c0ce7b0` (the two files) vs `phase2/pr1/diff.patch`:
  byte-identical (stronger than the index-line-stripped equality used in the
  original audit).
- Local commit `7c0ce7b0` tree: `66911b2809a324ad7b832a2a3485d6f92c8c1290`;
  worktree clean (status 0 entries).
- Fork commit `80fb44931a94409eb3ac10744f1fe21698f452c7` (GitHub commits
  API): tree `66911b2809a324ad7b832a3485d6f92c8c1290` (identical), parent
  `68088ed3927e91bce8918db78f2efd9e644a4134` (correct upstream base),
  message matches the local commit including the 25-min bump. Local commit,
  fork branch, and diff.patch carry identical content.

### (d) Final verdict

**READY FOR MAINTAINER REVIEW.** Blocking issue 1 is cleared: the draft
exists, contains no prohibited claim, satisfies the AGENTS.md PR-description
requirements, and its factual statements match the evidence. Non-blocking
recommendations remain open and unchanged (docstring first-line tightening;
optional in-test GDN-execution assert; re-run the duplicate search at
PR-open time — the search snapshot predates submission by hours, per attack
1's caveat). Per the manual-review gate, the upstream PR remains unopened
pending owner approval.
