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
