# STATUS — Phase 1 RCA

| Item | State |
|---|---|
| Environment (W7900/gfx1100, ROCm 7.14 wheels, vLLM main SHA) | **DONE** — gate PASS, independent audit PASS WITH NOTES (all notes closed) |
| Upstream landscape (docs/01) | **DONE** |
| Model architecture trace (docs/03) | **DONE** |
| Baseline reproduction (warn mode) | **DONE** — P5/P3 zero runtime JIT; P1 eager inventory |
| `--jit-monitor-mode error` cold run | **DONE** — P2 PASS (0 raises, 33/33 OK) |
| Shape battery | **DONE** — {1..128}×batches{1,2,4} |
| Controls (warm cache / small GDN / non-GDN / eager-vs-normal) | **DONE** — P4, G1-G3, N1 |
| Call graph + compile keys (docs/06, 07) | **DONE** |
| Falsification matrix (docs/10) | **DONE** |
| Root cause report (docs/09) | **DONE** |
| Independent audits (subagents) | environment PASS(W/N); reproduction/call-graph/compile-key/adversarial — see evidence/subagent-reviews/ |

Verdict: **Phase-1 COMPLETE — HISTORICAL ROOT CAUSE DISPROVED**
(default-configuration runtime JIT at pinned SHA: none; eager-mode runtime
JIT mechanism fully identified, upstream-expected). See docs/09-root-cause.md.

Phase 2: NOT STARTED.
