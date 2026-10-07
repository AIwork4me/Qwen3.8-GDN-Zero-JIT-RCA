# P4 — Qwen3.8-27B-FP8, warm reuse of P5's exact cache (normal, warn)

cache-before: 3375 triton files / 210.5 MB (+ vllm/xdg artifacts) from P5.
Startup 3.0 min (vs 8.5 min cold) = artifact reuse proven at startup.
Battery: 0 runtime JIT events (armed monitor; flat latency 26.14 s == cold
steady state). Note (adversarial review): for a WARM run "0 entries touched"
is weak evidence by itself (cache hits never rewrite mtimes) — the load-bearing
P4 evidence is the armed monitor + flat latency + 3.0-min startup (cache reuse).
