# 04 — Reproduction Record (Phase-1 primary + controls)

All runs: pinned vLLM main `31e2443c90542a33a4a4a293ea7186fba2796c67`
(editable, `/workspace/vllm-zero-jit-rca`), venv ROCm 7.14 stack,
gfx1100 W7900D, generic GDN path (no aiter), cold/warm isolated caches
(`/workspace/rca-caches/<RUN_ID>`, PREFLIGHT PASS recorded in each env.txt),
JIT monitor warn+verbose (except P2 error), greedy requests, token-count
battery {1,4,8,15,16,17,31,32,33,64,128} × batches {1,2,4} (27B: max_tokens
16; small: max_tokens 8). Evidence dirs: `evidence/runs/<RUN_ID>/`.

## Primary model runs (Qwen/Qwen3.8-27B-FP8, rev 017b9c7a)

| run | config | startup | runtime JIT events | first-req wall |
|---|---|---|---|---|
| P1 | cold, eager, warn | 6.5 min | monitor OFF (by design); **24 runtime compiles** (GDN decode + w8a8 + infra) | 62.7 s (then 68.3 s) vs 26.4 s steady |
| P5 | cold, normal, warn | 8.5 min (warmup 104 keys + cudagraph 1..512) | **0** | 26.13 s (== steady state) |
| P3 | cold, normal, warn (independent repeat) | ~8.5 min | **0**; triton kernel set identical to P5 (67 kernels) | 26.20 s |
| P2 | cold, normal, **error** | ~8.5 min | **0 raises**, 33/33 requests OK | — (acceptance PASS) |
| P4 | warm (reuse of P5 cache), normal | **3.0 min** | **0**; 0 triton entries touched after battery t0 | 26.14 s |

## Control runs

| run | model | config | runtime JIT |
|---|---|---|---|
| N1 | Qwen2.5-0.5B (dense, non-GDN) | cold eager warn | monitor OFF (eager); battery flat after harness fixes; inventory not compiled → see G/P runs for eager effect |
| G1 | Qwen3.5-0.8B (18 GDN layers) | cold eager warn | monitor OFF; **35-kernel runtime inventory** incl. conv1d_update ×2, packed decode ×2, layer_norm ×3, FLA chunk family |
| G2 | Qwen3.5-0.8B | cold normal warn | **0** (warmup 88 keys + capture covered all) |
| G3 | Qwen3.5-0.8B | warm (reuse G2 cache) normal | **0** |

## Reproducibility classification

- zero-runtime-JIT in normal mode: **deterministic** (P5, P3 identical; P2
  error-mode agrees; G2 agrees on the small model)
- runtime-JIT set in eager mode: **deterministic** (P1 single run + G1
  consistent kernel families; first-request latency spike reproducible)
- cold-startup cost: 8.5 min (normal cold) vs 3.0 min (warm) on this host —
  STARTUP JIT (timeline A), not runtime JIT
- no 43-minute first-inference JIT observed in ANY configuration; largest
  observed first-request compile cost: +42 s (P1 request #2, eager)
- no FP8 W8A8 23-minute startup autotune cliff observed in this
  configuration (startup profile run compiled 2/6 w8a8 variants eagerly;
  remaining variants compiled during capture/warmup or first requests
  depending on mode; engine ready in ≤ 8.5 min ≪ 600 s timeout)
