# P2 — Qwen3.8-27B-FP8, cold, ERROR mode, NORMAL (the #49349 acceptance gate)

- identical to P5 except `--jit-monitor-mode error`
- monitor activated (mode=error) after startup warmup+capture; battery:
  tokens {1,4,8,15,16,17,31,32,33,64,128} × batches {1,2,4}, max_tokens 16

## Result

```text
monitor raised:            0 times
failed requests:           0 / 33
inference succeeded:       yes (all requests OK)
first failing kernel:      NONE
```

0 JIT events recorded (monitor armed, error mode). `--jit-monitor-mode error` — the strongest zero-runtime-JIT acceptance test
from #49349 — **passes** for Qwen3.8-27B-FP8 on gfx1100 at pinned SHA
31e2443 in the default graph configuration with cold caches.
