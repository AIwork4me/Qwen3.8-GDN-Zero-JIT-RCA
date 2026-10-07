# Reviewer plan — PR #60395

Current posture: DRAFT, upstream CI blocked on the 'verified'/'ready' label policy
gate; no reviewer requests made (per mission §33 and the pre-run-check's explicit
instruction that AI agents must not request labels/attention).

## Preferred strategy once CI runs / a human marks the PR ready

1. Respect automatically assigned reviewers / CODEOWNERS first.
2. If no reviewer is auto-assigned, request the smallest set:
   - primary ROCm/test reviewer: **@AndreasKaratzas**
   - second ROCm owner if needed or if the primary is unavailable: **@tjtanaa**
   - CI owner **@khluu** only if the Buildkite timeout/label change needs explicit review.
3. One request each; no group @-mentions; no pressure; no self-approval; no merge.

## Review-response policy (pre-committed)

- Optimize for merge, not authorship pride: if a maintainer asks for simplification
  (drop the two-layer hybrid, keep old timeout, parameterize differently, rename
  helpers, trim docstrings), adopt their design unless it breaks the test's
  scientific validity (monitor-armed assert, cache isolation, GDN-execution oracle,
  negative-control sensitivity must survive).
- No scope expansion during review: newly exposed issues go to follow-up
  issues/PRs, not into this tests-only change.
- If MI355 later fails the new test: classify per mission §35-§37 (PATCH FAILURE /
  UPSTREAM REGRESSION / CI INFRA / TIMEOUT / UNRELATED / UNKNOWN) before touching
  anything; a genuine Zero-JIT regression discovery converts this into a
  separate fix + regression-test effort rather than a weakened test.
