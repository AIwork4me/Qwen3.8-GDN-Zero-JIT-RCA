# N1 — Qwen2.5-0.5B (non-GDN control), cold, warn, eager

- model: Qwen/Qwen2.5-0.5B (sha 060db6499f32faf8b98477b0a26969ef7d8b9987), dense full-attention
- vLLM: pinned main 31e2443c90542a33a4a4a293ea7186fba2796c67, editable, ROCm 7.14 venv
- flags: --enforce-eager --max-model-len 2048 --max-num-seqs 16
  --jit-monitor-mode warn --jit-monitor-verbose
- cold caches proven (cache-before.txt: all empty; PREFLIGHT PASS in env.txt)
- battery: prompt tokens {1,4,8,15,16,17,31,32,33,64,128} × batches {1,2,4};
  all requests OK (after client-side fixes recorded below); max_tokens=8, greedy

## Result

```text
runtime JIT events (post engine-ready): 0
```

JIT monitor was active (observability config shows jit_monitor_mode='warn',
jit_monitor_verbose=True) and the startup JIT warmup ran
("JIT kernel warmup (108 compile keys)"). No `jit_monitor` warning appears
anywhere in server.log, including across the 16-token boundary (prompt
length 17) and batch sizes up to 4.

Interpretation: on this pinned SHA + host, generic dense-model vLLM serving
does not runtime-JIT under eager mode. Any runtime JIT observed for GDN
models in G*/P* runs is therefore not generic-vLLM noise.

## Harness fixes discovered during this smoke run (recorded)

1. launch_server.sh initially dropped extra vllm args (enforce_eager=False in
   first attempt) — fixed; command.sh now embeds args; N1 was relaunched cold.
2. /v1/completions rejects `prompt_token_ids`; use OpenAI-style
   `prompt: [ids]` — fixed in run_shape_battery.py.
3. token ids must stay below each model's vocab (Qwen2.5: 151936) — id
   generator clamped to [100, 100099] — fixed.

These are client-harness bugs, not vLLM behavior changes; the final N1
evidence reflects the fixed harness from a single cold launch.
