# Phase-2 PR-1 — Merge-Readiness Closure (current-main refresh)

Fresh evidence lineage for taking the ROCm Qwen GDN zero-runtime-JIT regression
test from the historical Phase-2 base to a DRAFT upstream PR against **current**
`vllm-project/vllm:main`. Historical evidence under `phase2/pr1/evidence/` is
untouched; this directory is a new lineage.

## Contents

| path | what |
|---|---|
| upstream-refresh.json | CURRENT_UPSTREAM_MAIN SHA + drift measurement method |
| duplicate-search-final.md | final duplicate gate (primary + independent subagent) |
| audits/01-upstream-drift.md | 60-commit drift review, classification LOW-MEDIUM |
| audits/02-test-design.md | skeptical-maintainer subagent review + dispositions |
| audits/03-diff-minimality.md | final diff audit (subagent verdict) |
| audits/04-pr-wording.md | overclaim audit of final PR text |
| audits/05-ci-readiness.md | CI-green readiness review |
| audits/06-final-adversarial.md | final adversarial review (verbatim) |
| diff-review.md | line-by-line review of the final diff |
| validation.md | current-main local validation record |
| pr-final.md | final PR description |
| ci-results.md | upstream CI statuses |
| reviewer-plan.md | reviewer strategy |
| evidence/local/ | environment gate, build log, test runs |
| evidence/negative-control/ | current-main negative control |
| evidence/upstream-ci/ | persisted PR check states |

## Base / branch

- previous Phase-2 base: `68088ed3927e91bce8918db78f2efd9e644a4134` (historical, frozen)
- current upstream main (retrieved 2026-10-07T10:55Z): `3ca00a8261c7790ba7487e2001ab9004ad8065f2`
- fresh fork branch: `rocm-gdn-zero-runtime-jit-test-main` (worktree
  `/workspace/vllm-rocm-gdn-zerojit-final`), no merge commits, historical branch untouched
- intended diff: `tests/jit_monitor/test_no_runtime_jit_rocm.py` +
  `.buildkite/test_areas/jit_monitor.yaml` (tests/CI only)
