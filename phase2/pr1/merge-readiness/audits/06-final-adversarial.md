# Audit 06 — Final Adversarial Review (PR #60395)

Reviewer: independent hostile-review subagent, fresh context, read-only + live
GitHub API access. Verbatim report below.

## Verbatim subagent report

> # Hostile review — PR #60395 (head 9e370fac, parent 3ca00a82)
>
> **1. Duplicate coverage — REJECTED (no duplicate).** Live `gh` search today (`GDN jit monitor`, `no_runtime_jit`, exact test name, `is:pr is:open jit_monitor`, `Qwen3.5 GDN ROCm`): the only hit for this coverage class is #60395 itself. Nearby PRs (#57560 FlyDSL prefill, #48363 Mamba2 warmup, #56693 CUDA jit-monitor CI, #59356 monitor semantics, #58091 numerics test) are perf/kernel/CUDA work with no `jit_monitor_mode="error"` ROCm GDN e2e test. `tests/jit_monitor/` at main contains the same 6 files, no GDN test.
>
> **2. Stale base — REJECTED.** PR `mergeable: MERGEABLE`; main is only 2 commits ahead of the parent, touching 5 files (`glm5next/*`, `qwen4_exp/qsa.py`, `test_fused_shared_expert.py`) — none are PR files. Both PR files are blob-identical base↔main (`.buildkite/.../jit_monitor.yaml` `9e40c9d1`, `test_no_runtime_jit_rocm.py` `a99b8e1e` via contents API). No rebase needed.
>
> **3. False-green — REJECTED.** Three independent guards: (a) `assert states and all(state == (True, "error"))` (test_no_runtime_jit_rocm.py:69) — monitor inactive ⇒ red; (b) `expect_gdn` count assert via the warmup's own `_iter_qwen_gdn_layers` predicate (lines 70-72) — GDN-less boot ⇒ red; (c) per-test fresh `TRITON_CACHE_DIR`/`VLLM_CACHE_ROOT`/`TORCHINDUCTOR_CACHE_DIR` under `tmp_path` + `spawn` (fresh in-process JITFunction caches) — stale cache can't mask a miss. Kill chain *demonstrated*, not argued: negative control raises `RuntimeError: Triton kernel JIT compilation during inference: _fused_post_conv_kernel` (mutation1 log line 184). Residual limits (cudagraph-captured kernels don't re-hook; battery shapes only) are inherent to the suite and disclosed in the docstring + PR body.
>
> **4. GDN not exercised — REJECTED.** `gdn-execution-proof.txt`: `arch=['Qwen3_5ForConditionalGeneration']`, `layer_types=[linear_attention, full_attention]`, `gdn_layers=1 monitor=(True,'error')`, six FLA/Triton GDN families present at startup, `created_after_battery=0` (2777→2777 cache files). Battery output tokens flowed (processed prompts logs).
>
> **5. Monitor inactive — REJECTED.** Activation is at end of `compile_or_warm_up_model` (jit_monitor.py docstring/log line "Kernel JIT monitor activated... mode=error" post-graph-capture); the in-test RPC reads worker-process globals (`jit_monitor.is_active()`), so any gate flipping (enforce_eager, `enable_jit_warmup` off) ⇒ `(False, ...)` ⇒ assert red. Recorded `(True,'error')` in both proof and test runs.
>
> **6. Cache contamination — REJECTED.** Per-test `tmp_path` caches + spawned child (env inherited) + spawned EngineCore; negative control *with fresh caches* still failed, proving within-run warmup-miss detection is not cache-maskable.
>
> **7. Negative control invalid — REJECTED.** Mutation (early-return in `qwen_triton_warmup`) → monitor raise on `_fused_post_conv_kernel`, 1 FAILED; revert → 1 passed (post-revert-confirm.txt: "1 passed ... 267.02s"). Genuine sensitivity proof.
>
> **8. CI cost — REJECTED as attack.** 2 engine boots from cold caches justify 10→25 min / 300→600s; local full-file 394s (W7900) gives ~2× per-test and ~3.8× job headroom; the CUDA sibling lane runs 900s/test, 45 min — the ROCm lane stays cheaper. First MI355 run is declared authoritative; over-tight timeout would cause flaky red, not green.
>
> **9. Overbroad claims — REJECTED.** Body scopes: Qwen3.5-0.8B only with explicit non-goals (Qwen3-Next/3.8/4-Exp, AITER, eager, spec-decode); monitor semantics disclosed ("autotune misses print-only; inductor/cudagraph-capture unmonitored" — matches jit_monitor.py); #45000 cited only as "warmup area is regression-prone"; title says "Qwen3.5". Audit 04 found 3 wording defects pre-submission; all fixed in the live PR body (verified byte-identical to pr-body-upstream.md).
>
> **10. AITER leakage — REJECTED.** `_AITER_ENABLED = envs.VLLM_ROCM_USE_AITER` (_aiter_ops.py:1999); both `are_gdn_triton_kernels_available()` (:2479) and `is_rdna_gdn_triton_kernels_available()` (:2484) gate on it, so `GDN_AITER_TRITON_AVAILABLE` is False at import in the spawned child; `_resolve_gdn_prefill_backend` returns `"triton"` on non-CUDA. Corroborated by the cached kernel names (FLA family, not aiter's `fused_rearrange_*`).
>
> **11. Unrelated diff — REJECTED.** `gh pr view --json files`: exactly 2 files, +70/−13, tests+CI only; `git status` clean of strays.
>
> **12. Upstream already changed the files — REJECTED.** Blob SHAs at base `3ca00a82` and at `main` identical for both files (checked via contents API). Behavior under test unchanged on current main.
>
> **13. MI355 not validated — REJECTED as a defect.** pre-run-check failure is a documented policy gate (author has 1 merged PR < 4; label must come from a human; check text forbids AI agents from requesting it — check-states.txt). PR is DRAFT, body explicitly states the first MI355 run is the authoritative signal and never presents gfx1100 evidence as MI355 evidence. This is the honest state; no author-fixable action exists (requesting the label is prohibited).
>
> **14. HF dependency — REJECTED.** `Qwen/Qwen3.5-0.8B` is a public repo (downloaded during validation; full config/tokenizer locally); `load_format="dummy"` fetches config/tokenizer only, and prompts are raw token-ids (`TokensPrompt`). Identical dependency class to the pre-existing dense test (`Qwen/Qwen3-0.6B`, merged #58092). Hub outage ⇒ red network error, never green.
>
> **15. Problem no longer exists — REJECTED.** At current main, `tests/jit_monitor/test_no_runtime_jit_rocm.py` (blob `a99b8e1e`) still contains only `test_v2_sampler_warmup_rocm` (dense Qwen3-0.6B). A GDN warmup regression would still pass today's ROCm lane. The gap is real.
>
> ## FINAL VERDICT
>
> **HIGH-QUALITY MERGE-READY**
>
> As a DRAFT awaiting the policy-gated human label and first MI355 CI run, no remaining defect is author-fixable: duplicates cleared (re-verified live), base non-stale (files blob-identical to main, MERGEABLE), false-green closed by three in-test asserts plus a demonstrated mutation kill chain, diff minimal (2 files), claims accurately scoped, AITER provably unreachable, and the unvalidated-MI355 state is honestly documented rather than hidden. All attacks I could mount failed on evidence.

## Acceptance

Verdict accepted without override. Note the verdict's own scoping: "merge-ready"
means no author-fixable defect remains while CI is policy-blocked; Ready-for-Review
still requires the §41 load-bearing gates (upstream CI green incl. MI355), which are
documented in audits/05-ci-readiness.md.
