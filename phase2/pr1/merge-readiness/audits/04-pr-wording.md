# Audit 04 — PR Wording / Overclaim (merge-readiness)

Reviewer: independent subagent, read the final PR draft + final diff + test file.
Initial verdict: **PR WORDING FAIL** with 3 required edits. All 3 applied; re-verified
below. Verbatim report preserved.

## Verbatim subagent report

> ## Overclaims / mis-scoped claims found
>
> **1. #45000 citation misattributes the regression class (line 16-17) — REQUIRED EDIT**
> Verified via GitHub API: #45000 is "[Perf][ROCm] Fix GDN KKT warmup regression on RDNA by avoiding fp32 tl.dot" — a warmup-*duration* perf regression (150 s → 2400 s), fixed as a perf PR. It is **not** an instance of "passes jit-monitor CI and only surfaces as first-inference JIT compilation on user deployments." The actually documented first-inference-JIT-on-RDNA event is #52663's secondary observation, which the draft deliberately doesn't cite.
> Smallest fix: "(Qwen GDN Triton warmup regressions on RDNA have happened before, e.g. the warmup-perf fix #45000)" — i.e., cite it as evidence the GDN warmup area is regression-prone, not as an instance of the pass-CI/surface-at-inference class.
>
> **2. Title "Qwen GDN" is family-wide; test covers Qwen3.5-0.8B only — REQUIRED EDIT**
> `qwen_triton_warmup.py` serves `qwen3_next`, `qwen3_5*`, `qwen4_exp` model types; the test boots only Qwen3.5-0.8B with a 2-layer truncation. Qwen3-Next / Qwen3.8 / Qwen4-Exp GDN variants (different head dims, quantization — e.g. #52663 was Qwen3.8-FP8) are not exercised, and no non-goal says so.
> Smallest fix: retitle to "[ROCm][Test] Add **Qwen3.5** GDN zero-runtime-JIT regression coverage" and add a non-goal: "covers Qwen3.5-0.8B only; other Qwen GDN variants (Qwen3-Next, Qwen3.8, Qwen4-Exp) not covered."
>
> **3. Stale footer contradicts the validation section (lines 92-94) — REQUIRED EDIT**
> The footer says numbers "go here after current-main runs complete," but the runs demonstrably DID complete on `3ca00a826` (evidence: 130.75 s / 278.56 s / 394.14 s, six kernel families, `created_after_battery=0`, negative-control raise for `_fused_post_conv_kernel`, `post-revert-confirm.txt`). Shipping this text "verbatim" with the placeholder note would leave maintainers unable to tell whether validation is real.
> Smallest fix: delete the footer before submission.
>
> ## Properly scoped (no edit required)
>
> - **"all of ROCm"** — no such claim; `[ROCm]` is a platform tag, validation is scoped to gfx1100 local + MI355 CI, test gated on `is_rocm()`. PASS.
> - **"all inference shapes"** — no such claim. Battery is concrete: prefill 4/32/128 tokens, greedy ≤16-step decode, mixed-length batched prefill, top_k/top_p/min_p(+combo) sampling ≤8 tokens, one heterogeneous batch of 4; docstring scopes to "every compile key the battery needs." PASS.
> - **"all runtime JIT"** — no universal claim; "zero-runtime-JIT" matches the existing upstream suite/file naming, and the negative control demonstrates the Triton error hook. Recommended (not required) one-liner in Scope: on ROCm the monitor raises on Triton JIT; autotune misses are print-only, TileLang hook is disabled on ROCm (#51159, jit_monitor.py:81), inductor/cudagraph-capture compiles are unmonitored.
> - **"all GDN backends" / AITER** — explicitly generic Triton/FLA path, `VLLM_ROCM_USE_AITER=0`, AITER named as future work *and* non-goal. PASS.
> - **#52663** — zero mentions; "not a fix for any open issue" non-goal present. PASS.
> - **Validation scoping** — SHA matches `upstream/main` = HEAD exactly; hardware, ROCm version, dummy weights all stated; all claimed numbers verified against evidence logs. PASS except the stale footer (item 3).
> - CI timeout claims, "existing dense test's behavior unchanged," monitor-armed and GDN-layer assertions — all match the diff/test. PASS.
>
> **FINAL VERDICT: PR WORDING FAIL**
>
> Required edits before submission:
> 1. Rescope the #45000 parenthetical to a warmup-area regression example, not a pass-CI→first-inference-JIT precedent.
> 2. Retitle to "Qwen3.5 GDN" and/or add the variant non-goal (Qwen3-Next/Qwen3.8/Qwen4-Exp not covered).
> 3. Delete the stale placeholder footer.

## Disposition

| # | edit | disposition |
|---|---|---|
| 1 | #45000 rescoped to "the GDN warmup area is regression-prone — e.g. the RDNA warmup fix #45000" | ADOPTED |
| 2 | title → "[ROCm][Test] Add Qwen3.5 GDN zero-runtime-JIT regression coverage"; non-goal "covers Qwen3.5-0.8B only; other Qwen GDN variants (Qwen3-Next, Qwen3.8, Qwen4-Exp) are not covered" | ADOPTED |
| 3 | placeholder footer deleted | ADOPTED |
| rec | monitor-semantics one-liner in Scope ("on ROCm the monitor raises on inference-time Triton JIT compiles; autotune misses are print-only and inductor/cudagraph-capture compiles are unmonitored") | ADOPTED |

## Verdict after edits

```text
PR WORDING PASS
```
