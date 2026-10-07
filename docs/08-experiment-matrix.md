# 08 — Experiment Matrix (executed)

| id | run dir | model | mode | monitor | graphs | result |
|---|---|---|---|---|---|---|
| P1 | P1-cold-eager-warn | Qwen3.8-27B-FP8 | cold | warn (inactive by design) | eager | runtime inventory: 24 compile events (conv_update ×2, packed_decode ×2, w8a8 +4, infra); first-req +36 s |
| P2 | P2-cold-normal-error | Qwen3.8-27B-FP8 | cold | error | cuda-graph | **PASS**: 0 raises, 33/33 requests OK |
| P3 | P3-cold-normal-warn | Qwen3.8-27B-FP8 | cold (independent repeat) | warn | cuda-graph | 0 events; kernel set identical to P5 |
| P4 | P4-warm-reuse | Qwen3.8-27B-FP8 | warm (P5 cache) | warn | cuda-graph | 0 events; 0 cache entries touched; startup 3.0 min vs 8.5 cold |
| P5 | P5-cold-normal-warn | Qwen3.8-27B-FP8 | cold | warn+verbose | cuda-graph | **0 runtime JIT events**; 428 startup entries; flat latency |
| G1 | G1-cold-warn | Qwen3.5-0.8B | cold | warn (inactive) | eager | 35-kernel runtime inventory incl. all GDN families |
| G2 | G2-cold-normal-warn | Qwen3.5-0.8B | cold | warn+verbose | cuda-graph | **0 runtime JIT events** |
| G3 | G3-warm-reuse | Qwen3.5-0.8B | warm (G2 cache) | warn | cuda-graph | 0 events |
| N1 | N1-cold-warn | Qwen2.5-0.5B | cold | warn (inactive) | eager | battery flat; non-GDN control (harness validation) |

Notes:
- P2 was executed as cold/**normal**/error (not eager): `--jit-monitor-mode
  error` only arms when JIT warmup is enabled — eager disables the monitor
  (docs/07). This is the strongest scientifically valid form of the
  acceptance test.
- Original matrix rows P1/P2 'eager diagnostic' retained as P1 (eager
  inventory) + P2 (normal error) — protocol deviation documented.
