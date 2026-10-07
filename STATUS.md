# STATUS — Phase 1 RCA (CLOSED)

| Item | State |
|---|---|
| Environment (W7900/gfx1100, ROCm 7.14 wheels, vLLM main SHA) | **DONE** — gate PASS, independent audit PASS WITH NOTES (all notes closed) |
| Upstream landscape (docs/01) | **DONE** |
| Model architecture trace (docs/03) | **DONE** |
| Baseline reproduction (warn mode) | **DONE** — P5/P3 zero runtime JIT; P1 eager inventory |
| `--jit-monitor-mode error` cold run | **DONE** — P2 PASS (0 raises, 33/33 rows OK) |
| Shape battery | **DONE** — {1..128}×batches{1,2,4} |
| Controls (warm cache / small GDN / non-GDN / eager-vs-normal) | **DONE** — P4, G1-G3, N1 |
| Call graph + compile keys (docs/06, 07) | **DONE** |
| Falsification matrix (docs/10) | **DONE** |
| Root cause report (docs/09) | **DONE** |
| Independent audits (subagents) | environment PASS(W/N); reproduction/call-graph/compile-key/adversarial — see evidence/subagent-reviews/; closure audits in evidence/subagent-reviews/closure/ |
| Phase-1 closure (matrix/timeline/boundary verifier/wording) | **DONE** — see docs/08, per-run timeline.csv, evidence/closure/cache-boundary-verification.txt |

## Phase-1 verdict (final, closure-reviewed)

```text
Default graph mode:
CONFIRMED — no runtime JIT was observed in the tested default/graph
execution envelope (prompts <=128 tokens, batches <=4, max_tokens 16,
greedy, max_num_seqs 32, capture sizes through 64, single gfx1100,
generic non-AITER GDN path).

Enforce-eager:
CONFIRMED — runtime JIT still occurs under --enforce-eager because graph
capture and JIT warmup are intentionally disabled (and the JIT monitor is
intentionally inactive) by current upstream design (216 post-battery
Triton artifacts in P1; first-request penalty ~36-42 s).

Historical #52663 interpretation:
The historical ~43-minute first-inference runtime-JIT impact is not
reproducible on the current tested stack (pinned SHA 31e2443c, ROCm 7.14,
default graph configuration). The eager-mode runtime-JIT mechanism itself
remains reproducible, with substantially lower observed impact
(~36-42 s on the first two requests, not ~43 minutes).

Evidence packaging: PASS (closure audits A-D in
evidence/subagent-reviews/closure/; cache-boundary verification PASS;
experiment matrix regenerated from committed commands with 0 mismatches;
derived timelines with provenance for all runs).
```

Phase 1: **CLOSED / PASS**.

Phase 2: **AUTHORIZED** (see docs/phase2_candidates.md; PR-1 progress in
phase2/pr1/).

---

# Phase 2 PR-1 Status (appended 2026-10-07)

```text
Phase 1:
CLOSED / PASS

Phase 2 PR-1:
IMPLEMENTATION COMPLETE
LOCAL VALIDATION PASS
UPSTREAM PR NOT OPENED

Branch:
AIwork4me/vllm:rocm-gdn-zero-runtime-jit-test
(fork commit 37363be53af1528426e1e3cb1dc4249ac4b4b564,
 tree identical to local commit 6ae52736; base upstream main 68088ed3)

Status:
READY FOR MANUAL REVIEW

Next step:
MANUAL REVIEW BEFORE ANY UPSTREAM PR
(re-run duplicate search immediately before opening; see
phase2/pr1/audits/final-adversarial.md closure notes)
```

- New test: `tests/jit_monitor/test_no_runtime_jit_rocm.py::test_qwen_gdn_no_runtime_jit_rocm`
  (Qwen3.5-0.8B GDN, default graph config, `jit_monitor_mode="error"`,
  dummy weights, fresh caches, monitor-armed assertion, shared battery)
- Validation: baseline PASS; new test PASS (263–278 s); full file 2-pass;
  GDN execution proven (module inspection + GDN kernel families in startup
  cache + 0 post-battery cache files); negative control FAILS the test via
  the monitor when GDN warmup is removed, then reverts clean
- Audits: duplicate-work CLEAR TO IMPLEMENT; test-design APPROVED WITH
  CHANGES (adopted); implementation PASS WITH NOTES (CI budget adopted);
  test-evidence PASS WITH NOTES (no false green); final adversarial
  READY FOR MAINTAINER REVIEW
- Hard stop honored: no upstream PR, no upstream issue comments, no
  eager-mode/warmup code changes, no FP8 config JSONs, no kernel changes

---

## Phase 2 PR-1 — Merge-Readiness Closure (2026-10-07)

Evidence lineage: `phase2/pr1/merge-readiness/` (historical evidence untouched).

- Upstream refresh: previous base `68088ed3` → current main `3ca00a82`
  (retrieved 2026-10-07T10:55Z; 60 commits / 282 files drift; classification
  LOW-MEDIUM; both patch files blob-identical base↔main)
- Final duplicate gate: CLEAR (primary + independent subagent; re-verified
  immediately before PR creation and again during final adversarial review)
- Fresh branch: `rocm-gdn-zero-runtime-jit-test-main` on AIwork4me/vllm
  (worktree /workspace/vllm-rocm-gdn-zerojit-final; local commit tree `48f1d43b`;
  fork head `9e370fac`; historical branch `rocm-gdn-zero-runtime-jit-test`
  untouched)
- Final diff: 2 files, tests+CI only (+70/−13); minimality subagent verdict
  MINIMAL / MAINTAINER-FRIENDLY; no production code changes
- Current-main validation (gfx1100, ROCm 7.14): baseline dense PASS 131 s;
  new GDN test PASS 279 s (monitor (True,"error") + GDN layer count asserted
  in-test); full file 2/2 PASS 394 s; GDN proof: arch Qwen3_5, layer_types
  [linear_attention, full_attention], 1 GDN layer, six GDN kernel families in
  startup cache, 0 post-battery cache artifacts; negative control: warmup
  no-op → RuntimeError _fused_post_conv_kernel → revert → PASS 267 s;
  ruff check/format clean
- Test-design audit: APPROVED WITH SMALL CHANGES — all adopted (in-test GDN
  presence assertion added via the warmup's own `_iter_qwen_gdn_layers`,
  docstring interval fix, mixed-warmup skip watch-item checked: not present)
- PR wording audit: PASS after 3 required edits (#45000 rescoped, title scoped
  to Qwen3.5, variant non-goals added)
- Upstream PR: **https://github.com/vllm-project/vllm/pull/60395** (DRAFT)
  - DCO fixed twice (missing sign-off; sign-off email mismatch) → SUCCESS
  - pre-run-check FAILURE = repository policy gate (author <4 merged PRs; the
    'verified'/'ready' label must be added by a human; AI agents are
    explicitly forbidden from requesting it) → pre-commit + Buildkite lanes
    (incl. AMD MI355 mirror) NOT STARTED
  - all runnable checks green (DCO, CodeRabbit, readthedocs, Summary, Meta)
- Final adversarial audit: HIGH-QUALITY MERGE-READY (all 15 attack vectors
  rejected on evidence; "no remaining author-fixable defect")

Status:
PHASE 2 PR-1 — CI BLOCKED

Next step:
HUMAN ACTION REQUIRED — add the 'verified' or 'ready' label to PR #60395
(or merge-gate account to 4 merged PRs) so upstream CI (incl. the AMD MI355
mirror lane) starts; then re-assess per phase2/pr1/merge-readiness/
audits/05-ci-readiness.md and reviewer-plan.md
