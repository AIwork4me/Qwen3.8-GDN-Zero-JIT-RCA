Independent subagent reviews (2026-10-07). Environment gate: PASS WITH NOTES
(all notes closed in commits 7c142c5..15ff117). The three gate reports below
are verbatim outputs; their findings were addressed in commit 4494ba9
(P5 cache-before restored from d865957; N1 result corrected; docs 04-07/09/10
mechanism + count corrections; prepare_run_env.sh warm mode made read-only).
The adversarial review ran AFTER those corrections.

## Verdicts

- environment gate: PASS WITH NOTES (notes closed before experiments)
- reproduction gate: FAIL initially (P5 cache-before clobbered by warm-mode
  harness bug + N1 wording) -> corrected in 4494ba9; underlying runs
  independently corroborated as genuine (distinct processes, cold-proof
  cache roots, correct envs, correct graph modes)
- call-graph gate: FAIL initially (5 doc discrepancies) -> all corrected;
  independent derivation otherwise agrees on every load-bearing edge
- compile-key gate: SUPPORTED -> claims tightened to what evidence proves
  (normal-mode runtime needs ⊆ normal-mode startup artifacts; w8a8 mechanism
  corrected to Triton M-specialization; capture sizes [1..64]; scope caveats)
- adversarial final review (after corrections): ROOT CAUSE CONFIRMED
  (disk-mtime proof cited as primary; envelope caveats folded into docs/09)
