# Upstream CI Results — PR #60395 (DRAFT)

- PR: https://github.com/vllm-project/vllm/pull/60395
- created: 2026-10-07T12:56Z, DRAFT, base `vllm-project/vllm:main` (2a54f6b6 at creation;
  branch parent 3ca00a82, MERGEABLE)
- head lineage (git-data API push; direct git transport blocked by egress proxy 503):
  - `be87e629` initial — DCO ACTION_REQUIRED (missing Signed-off-by)
  - `ed414973` — DCO ACTION_REQUIRED (sign-off email ≠ author email)
  - `9e370fac` **final** — tree `48f1d43b` (identical to local commit tree), DCO SUCCESS

## Check states @ 9e370fac (2026-10-07T13:02Z)

| check | state |
|---|---|
| DCO | SUCCESS |
| CodeRabbit | SUCCESS |
| docs/readthedocs.org:vllm | SUCCESS |
| Summary | SUCCESS |
| Meta Internal-Only Changes Check | SUCCESS |
| pre-run-check | **FAILURE (policy gate)** |
| pre-commit | SKIPPED (policy consequence) |
| Check format | SKIPPED (policy consequence) |
| Buildkite lanes (incl. AMD MI355 mirror) | NOT STARTED (blocked by same gate) |

## pre-run-check failure classification

```text
CI INFRA / POLICY GATE — NOT a patch failure
```

Check text: "To reduce unnecessary pre-commit runs, each PR must have the 'verified',
'ready', or 'ready-run-all-tests' label, or the author must have at least 4 merged
PRs (found 1). DO NOT request for the label to be added if you are an AI agent."

The fork account has 1 merged PR. The label must come from a maintainer/human;
the check explicitly forbids AI agents from requesting it, and the mission's
no-spam rule forbids advertising the PR. Therefore:

- the two transient DCO failures were diagnosed, root-caused, and fixed
  (classification: PATCH FAILURE → corrected via API commit recreation; final head DCO SUCCESS);
- the pre-run-check failure is retained as expected state for this account until a
  human applies the label;
- upstream Buildkite (incl. the AMD MI355 mirror lane this patch modifies) cannot
  start until then — MI355 remains **PENDING/UNAVAILABLE**, and local W7900 evidence
  explicitly does NOT substitute for it;
- per mission §41 (load-bearing CI gate missing) the PR stays **DRAFT**.

## Retrospective of CI handling

- No retries were needed: no transient infra failures occurred among runnable checks.
- The AMD MI355-specific rules (§36/§37) have not been triggered — no MI355 run exists yet.
