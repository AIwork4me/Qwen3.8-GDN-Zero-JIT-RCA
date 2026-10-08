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

---

## 2026-10-08 — Fresh-image bootstrap + latest-main refresh + Ready for Review

Environment deleted with the previous Linux image; rebuilt from zero and the
PR refreshed onto execution-time current main. Full evidence lineage under
`phase2/pr1/merge-readiness/fresh-image-refresh/`.

- Fresh image: Ubuntu 24.04.4, EPYC 9334, AMD Radeon Pro W7900D (gfx1100,
  Chip ID 0x744b), 503 GiB RAM; host /opt/rocm-7.2.1 present and explicitly
  NOT used for the build (compiler identity = venv `_rocm_sdk_devel`,
  hipcc HIP 7.14.60850 driving clang 23 ROCm/llvm-project)
- Environment: venv `/workspace/venv-qwen-gdn-rca`; torch 2.14.1+rocm7.14
  (hip 7.14.60850) + rocm 7.14.0.post1 SDK family (core/libraries/devel/
  device-gfx1100 all 7.14.0) from the official PyTorch rocm7.14 index +
  AMD repo.amd.com wheel index for rocm-sdk-devel; triton-rocm 3.8.0; no
  nvidia-*/cuda-* contamination. Documented deviations on the fresh image:
  (1) rocm metapackage pinned to [devel,libraries,device-gfx1100]==7.14.0.post1
  before torch to avoid the 7.14.1 device-all (~25 GB) resolution that
  exhausted the 4 GB /tmp tmpfs; (2) torch installed --no-deps; (3) amdsmi
  7.0.2 installed per requirements/build/rocm.txt (ROCm platform detection);
  (4) venv-level zz_triton_knobs_first.pth preload works around an
  RTLD_GLOBAL libtorch_cpu / libtriton segfault (root cause + proof in
  segfault-root-cause.md)
- Network: github.com git transport intermittently blocked (CONNECT 503);
  bounded retries succeeded for clone/fetch/push. CMake FetchContent
  triton_kernels clone blocked → NETWORK WORKAROUND — SOURCE IDENTITY
  PRESERVED: codeload tarball of the exact pinned commit 669b31ac via
  TRITON_KERNELS_SRC_DIR; installed tree byte-identical to pinned source
  (triton-kernels-consumption-proof.txt)
- Upstream main discovered at execution time: cycle 1 d6fe5dca (33 commits
  past the previously validated 3ca00a82; drift LOW); during validation the
  tip advanced 9 more commits (only a58bdd0d touches a watched file —
  jit_monitor TileLang-hook internals, Triton path untouched) → cycle 2
  rebase onto fdfc171a; drift analysis refreshed, classification LOW
- New PR head: `90b1a661407a43b2d500c1cdae2188c8d83abf64` (single commit on
  fdfc171a; author + Signed-off-by AIwork4me <AIwork4me@qq.com> preserved;
  DCO SUCCESS on GitHub after push). Duplicate search (pre-refresh and
  final): CLEAR
- Build: vllm 0.1.dev22635+g90b1a6614.rocm714 editable from
  /workspace/vllm-zero-jit-rca; environment gate PASS pre- and post-build
- Fresh validation on 90b1a661 (= fdfc171a + test): full ROCm jit-monitor
  file 2/2 PASS 425 s (per-test ~143 s dense / ~268 s GDN on the same-day
  cycle-1 head); GDN proof in-log: jit_monitor_mode='error', Using
  Triton/FLA GDN prefill kernel (requested=auto), Warming up Qwen GDN
  Triton kernels; negative control NOT RE-RUN (warmup/monitor mechanism
  unchanged; historical sensitivity proof applicable — see
  negative-control-decision.md); ruff check/format, git diff --check, YAML
  parse, pytest --collect-only all PASS
- Subagent audits: environment — all toolchain/runtime checks PASS after
  adding the triton_kernels consumption proof (initial FAIL was an
  evidence-completeness gap only); freshness — PASS (tip movement inspected,
  validation sufficient); diff — MINIMAL / MAINTAINER-FRIENDLY (notes: CI
  mirror source_file_dependencies omits warmup dir — pre-existing, recorded;
  no blocking findings)
- Upstream PR #60395 updated: remote head verified 90b1a661; 1 commit;
  MERGEABLE; files exactly the 2 intended; PR body refreshed to fdfc171a
  validation; marked READY FOR REVIEW
- Reviewers: intended single request AndreasKaratzas could not be submitted
  (fork author lacks requestReviewsByLogin permission on upstream). On the
  ready transition GitHub auto-requested the CODEOWNERS set {tjtanaa,
  AndreasKaratzas, Harry-Chen, khluu} (/.buildkite @Harry-Chen @khluu rule
  among them); removal also requires unavailable permissions. Preferred
  reviewer AndreasKaratzas is included; no promotional comments posted
- CI authorization: NOT available — pre-run-check FAILURE is the repository
  policy gate (author has 1 merged PR < 4; 'verified'/'ready' label must
  come from a human; AI agents must not request it). DCO, CodeRabbit,
  readthedocs, Summary, Meta all SUCCESS; pre-commit + Buildkite lanes
  (incl. AMD MI355 mirror) not started

Status:
PHASE 2 PR-1 — WAITING FOR MAINTAINER CI AUTHORIZATION

Next step:
maintainer adds 'verified' or 'ready'/'ready-run-all-tests' label (or
approves) so upstream CI including the AMD MI355 mirror lane starts; then
re-assess per phase2/pr1/merge-readiness/audits/05-ci-readiness.md (first
command preference: /amd-ci run; failure policy in the mission contract)
