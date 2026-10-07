# Qwen3.8 / GDN Zero Runtime JIT — Phase 1 Root Cause Investigation (RCA)

Evidence repository for a Phase-1-only root cause investigation of **runtime JIT
compilation during inference** for Qwen3.8 / Gated DeltaNet (GDN) models on:

- GPU: AMD Radeon PRO W7900 — `gfx1100` (RDNA3, 48 GB)
- ROCm stack: PyTorch official `rocm7.14` wheels (torch 2.14.1+rocm7.14)
- vLLM: current upstream `main` (SHA recorded in `manifests/upstream.json`)

**This is Phase 1 (RCA) only.** No fixes are implemented here. See
`docs/00-scope.md` for the full scope and hard phase boundary.

## Verdict (Phase-1 closure)

```text
Default graph mode: CONFIRMED — no runtime JIT was observed in the tested
default/graph execution envelope.

Enforce-eager: CONFIRMED — runtime JIT still occurs under --enforce-eager
because graph capture and JIT warmup are intentionally disabled (and the
JIT monitor is intentionally inactive) by current upstream design.

Historical #52663: the historical ~43-minute first-inference runtime-JIT
impact is not reproducible on the current tested stack; the eager-mode
runtime-JIT mechanism itself remains reproducible (P1: 216 post-battery
Triton artifacts, ~36-42 s first-request penalty).
```

See `docs/09-root-cause.md`, `STATUS.md`, and the machine-verified boundary
report `evidence/closure/cache-boundary-verification.txt`.

## Repository layout

- `docs/` — numbered investigation reports (scope, upstream landscape,
  environment, model architecture, runtime-JIT inventory, call graph,
  compile-key analysis, experiment matrix, root cause, falsification,
  phase-2 candidates)
- `scripts/` — setup / collection / reproduction harness scripts
- `evidence/` — raw committed evidence (upstream research, environment,
  per-run artifacts under `evidence/runs/<RUN_ID>/`)
- `instrumentation/` — diagnostic-only patch applied to a separate vLLM
  worktree (observability only; no behavioral changes)
- `manifests/` — machine-readable records (upstream SHA, environment,
  experiment matrix)

## Rules

- Every conclusion must be traceable to committed evidence.
- Untouched-upstream reproduction comes before instrumented diagnostics.
- Every run uses explicitly named, isolated cache directories.
- Raw logs are never edited; sanitized copies note the original hash.
