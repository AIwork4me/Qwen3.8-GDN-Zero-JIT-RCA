# P3 — Qwen3.8-27B-FP8, independent cold repeat (normal, warn)

Identical to P5 but fresh RUN_ID + empty caches. Result: 0 runtime JIT
events; triton kernel set byte-identical to P5 (67 kernels, diff clean);
first-request 26.20 s (no spike). Cold-run zero-JIT reproduced identically across two independent cold runs (P5, P3). runtime JIT events: 0.
