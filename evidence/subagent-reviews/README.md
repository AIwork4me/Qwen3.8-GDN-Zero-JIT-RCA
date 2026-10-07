# Subagent Reviews

## Historical reviews (pre-closure)

**Honesty note (2026-10-07 closure):** historical audit *summaries* were
retained in commit messages and the list below, but the original verbatim
subagent outputs were **not persisted as standalone files** at the time
they ran. The summaries below are therefore second-hand records, not raw
transcripts. New closure audits (2026-10-07) ARE persisted verbatim under
`closure/` (see next section).

Historical review trail (2026-10-07, pre-closure checkpoint 82f7c9e):

- environment gate: PASS WITH NOTES (notes closed before experiments in
  commits 7c142c5..15ff117)
- reproduction gate: FAIL initially (P5 cache-before clobbered by warm-mode
  harness bug + N1 wording) -> corrected in 4494ba9; underlying runs
  independently corroborated as genuine (distinct processes, cold-proof
  cache roots, correct envs, correct graph modes)
- call-graph gate: FAIL initially (5 doc discrepancies) -> all corrected;
  independent derivation otherwise agrees on every load-bearing edge
- compile-key gate: SUPPORTED -> claims tightened to what evidence proves
  (normal-mode runtime needs ⊆ normal-mode startup artifacts; w8a8
  mechanism corrected to Triton M-specialization; capture sizes [1..64];
  scope caveats)
- adversarial final review (after corrections): ROOT CAUSE CONFIRMED
  (disk-mtime proof cited as primary; envelope caveats folded into
  docs/09) — historical wording; the closure-qualified verdict is in
  docs/09 ("historical impact not reproducible; eager mechanism remains
  reproducible")

## Closure audits (2026-10-07) — verbatim subagent outputs

- `closure/01-evidence-integrity-audit.md` — Subagent A
- `closure/02-technical-wording-audit.md` — Subagent B
- `closure/03-experiment-matrix-audit.md` — Subagent C
- `closure/04-final-phase1-adversarial-audit.md` — Subagent D

## Closure audit outcomes and resolution

| Audit | Verdict | Resolution |
|---|---|---|
| A — evidence integrity | PASS WITH NOTES | non-blocking notes resolved (G2 capture-size prose fixed to [1..32]; N1 "all requests OK" bullet qualified; N1 artifact count unit clarified as files) |
| B — technical wording | FAIL (blocking B-1..B-5) | ALL blocking + recommended B-6..B-16 + notes B-17..B-19 applied to docs/04/05/06/07/09/10, README, STATUS, and run result.md files |
| C — experiment matrix | PASS WITH NOTES (0 unexplained mismatches) | the two class-(b) prose errors are the same items fixed for audit A |
| D — final adversarial | PHASE 1 CLOSURE PASS | recommended docs/05 §2 pointer applied (covered by B-1 fix) |

Final gate: `scripts/verify_phase1.sh` → **PHASE-1 CLOSURE GATE: PASS**.
