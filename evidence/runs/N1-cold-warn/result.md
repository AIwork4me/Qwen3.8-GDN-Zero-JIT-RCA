# N1 — Qwen2.5-0.5B (non-GDN control), cold, warn, eager

- model: Qwen/Qwen2.5-0.5B (sha 060db6499f32faf8b98477b0a26969ef7d8b9987), dense full-attention
- vLLM: pinned main 31e2443c90542a33a4a4a293ea7186fba2796c67, editable, ROCm 7.14 venv
- flags: --enforce-eager --max-model-len 2048 --max-num-seqs 16
  --jit-monitor-mode warn --jit-monitor-verbose
- cold caches proven (cache-before.txt: all empty; PREFLIGHT PASS in env.txt)
- battery: prompt tokens {1,4,8,15,16,17,31,32,33,64,128} × batches {1,2,4};
  all requests OK; max_tokens=8, greedy

## Result

```text
runtime JIT events: 0 (as measured — see caveat)
```

**Caveat (corrected after independent audit):** in eager mode the JIT
monitor is NOT activated by upstream design (server.log line: "Enforce
eager set, disabling torch.compile, CUDAGraphs, and JIT kernel warmup";
gpu_worker.py:1056), so the absence of `jit_monitor` warnings is NOT
monitor-backed evidence. An earlier draft of this file wrongly cited
monitor/warmup activation — that text came from the FIRST (aborted,
non-eager) launch attempt of this run, which was discarded and relaunched
cold after the launcher arg bug fix; see harness notes below.

What the evidence does show: request latencies are flat from the first
battery request (0.094 s first vs 0.08 s median) — no first-request compile
spike (contrast G1: 4.78 s, P1: 62.7 s), consistent with the dense model
having no uncompiled kernels on its request path in this configuration.
Triton cache inventory for this run was not retained in the committed
evidence (subsumed by the G/P series which inventory the same generic
infra kernels).

## Harness fixes discovered during this smoke run (recorded)

1. launch_server.sh initially dropped extra vllm args (the aborted first
   attempt ran non-eager briefly before being killed and relaunched cold)
   — fixed; command.sh now embeds args.
2. /v1/completions rejects `prompt_token_ids`; use OpenAI-style
   `prompt: [ids]` — fixed in run_shape_battery.py.
3. token ids must stay below each model's vocab (Qwen2.5: 151936) — id
   generator clamped to [100, 100099] — fixed.

These are client-harness bugs, not vLLM behavior changes; the final N1
evidence reflects the fixed harness from a single cold launch.
