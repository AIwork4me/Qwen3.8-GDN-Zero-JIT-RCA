# Phase-2 PR-1 — ROCm Qwen GDN Zero Runtime JIT Regression Coverage

Outcome: upstream-ready tests-only patch, locally validated on gfx1100,
pushed to the AIwork4me fork. **Upstream PR intentionally NOT opened**
(manual review gate first).

## Contents

| path | what |
|---|---|
| upstream-base.json | upstream base SHA, tree verification, worktree provenance |
| duplicate-search.md | HARD-GATE search record — CLEAR TO IMPLEMENT |
| design.md | test design + reviewer-mandated changes |
| diff.patch | the exact patch (matches the pushed branch commit) |
| validation.md | full local validation record (positive/negative/lint) |
| pr-draft.md | PR description draft (not submitted) |
| evidence/baseline/ | existing ROCm test alone (PASS) |
| evidence/positive/ | new test, full file, post-revert confirm, GDN-execution proof |
| evidence/negative-control/ | GDN-warmup no-op mutation → monitor raise → revert |
| audits/ | duplicate-work, test-design, implementation, test-evidence, final-adversarial |

## Patch summary

- `tests/jit_monitor/test_no_runtime_jit_rocm.py`: add
  `test_qwen_gdn_no_runtime_jit_rocm` (Qwen3.5-0.8B GDN, default graph
  config, `jit_monitor_mode="error"`, dummy weights, fresh caches,
  monitor-armed assertion, shared shape battery); factor shared runner +
  cache isolation; existing dense test behavior unchanged.
- `.buildkite/test_areas/jit_monitor.yaml`: AMD mirror label/timeout
  update for the second boot (per-test 600 s, job 25 min).

## Verdict chain

1. duplicate-work: CLEAR TO IMPLEMENT
2. test-design: APPROVED WITH CHANGES (all adopted)
3. implementation audit: PASS WITH NOTES (CI-budget note adopted)
4. test-evidence audit: PASS WITH NOTES (no false green found)
5. final adversarial: see audits/final-adversarial.md
