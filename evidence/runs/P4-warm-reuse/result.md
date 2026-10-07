# P4 — Qwen3.8-27B-FP8, warm reuse of P5's exact cache (normal, warn)

cache-before: 3375 triton files / 210.5 MB (+ vllm/xdg artifacts) from P5.
Startup 3.0 min (vs 8.5 min cold) = artifact reuse proven at startup.
Battery: 0 runtime JIT events; 0 triton entries touched after battery t0;
first-request 26.14 s == cold steady state. Runtime JIT events: 0.
