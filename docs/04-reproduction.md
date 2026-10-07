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
| P1 | cold, eager, warn | 6.5 min | monitor OFF (by design); **27 runtime compile entries** (GDN decode + w8a8 + infra) | 62.7 s (then 68.3 s) vs 26.4 s steady |
| P5 | cold, normal, warn | 8.5 min (warmup 104 keys + cudagraph capture sizes 1..64) | **0** | 26.13 s (== steady state) |
| P3 | cold, normal, warn (independent repeat) | ~8.5 min | **0**; triton kernel set identical to P5 (67 kernels) | 26.20 s |
| P2 | cold, normal, **error** | ~8.5 min | **0 raises**, 33/33 requests OK | — (acceptance PASS) |
| P4 | warm (reuse of P5 cache), normal | **3.0 min** | **0**; 0 triton entries touched after battery t0 | 26.14 s |

## Control runs

| run | model | config | runtime JIT |
|---|---|---|---|
| N1 | Qwen2.5-0.5B (dense, non-GDN) | cold eager warn | monitor OFF (eager); **80 post-battery Triton artifacts** (compile absorbed by the first, client-buggy battery's server-side execution — see N1 result.md caveat); flat latency in later batteries because kernels were already compiled |
| G1 | Qwen3.5-0.8B (18 GDN layers) | cold eager warn | monitor OFF; **23 runtime entries / 19 kernels** (decode-leg, matches P1 classes; prefill families compiled by the eager startup profile run); total cache inventory 35 kernels |
| G2 | Qwen3.5-0.8B | cold normal warn | **0** (warmup 88 keys + capture covered all) |
| G3 | Qwen3.5-0.8B | warm (reuse G2 cache) normal | **0** |

## Reproducibility classification

- zero-runtime-JIT in normal mode: **reproduced across independent runs** (P5, P3 identical cold sets; P2 error-mode agrees; G2 agrees on the small model) within the tested battery envelope
- runtime-JIT set in eager mode: **consistent with a stable mechanism** (single P1 run; G1 shows the same kernel families at 0.8B; first-request latency spike reproduced in P1 and G1)
- cold-startup cost: 8.5 min (normal cold) vs 3.0 min (warm) on this host —
  STARTUP JIT (timeline A), not runtime JIT
- no 43-minute first-inference JIT observed in ANY configuration; largest
  observed first-request compile cost: +42 s (P1 request #2, eager)
- no FP8 W8A8 23-minute startup autotune cliff observed in this
  configuration (startup profile run compiled 2/6 w8a8 variants eagerly;
  remaining variants compiled during capture/warmup or first requests
  depending on mode; engine ready in ≤ 8.5 min (510 s) < 600 s default
  timeout — below the limit, with only ~15% headroom)
