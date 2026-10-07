# Closure Audit C — Experiment Matrix (2026-10-07)

Auditor: independent metadata auditor (subagent)
Scope: `manifests/experiment_matrix.csv` and `docs/08-experiment-matrix.md` vs committed raw evidence under `evidence/runs/<RUN_ID>/` for all 9 runs (P1–P5, G1–G3, N1).

## Method

1. Treated each run's `command.sh` as the source of truth; cross-checked against `metadata.json`, the `server.log` engine-config line (`Initializing a V1 LLM engine ... config:`), the `server.log` `Kernel JIT monitor activated ... mode=X` line, `client.log`, `cache-after.txt`, `jit-events.jsonl`, and `result.md`.
2. Recomputed all evidence-derived cells **with an independent script** (not the repo's `verify_runtime_cache_boundary.py`): parsed the FIRST `# battery start` marker in `client.log` (with line number), counted `prompt_tokens=` request rows, summed `ok=X/Y` numerators/denominators, and counted `cache-after.txt` file lines whose mtime is strictly after the battery-start second (same-second fractions count as post-battery, matching the documented conservative boundary rule). Counted `jit-events.jsonl` lines. Verified warm-run cache-reuse origins via `metadata.json` `warm_reuse_of` AND the actual cache root in `run-env.sh` (`RCA_CACHE_ROOT`).
3. Checked eager runs (G1, N1, P1) for ABSENCE of the monitor-activation line and for the documented eager monitor-inactive design (`Enforce eager set, disabling ... JIT kernel warmup`; `enable_jit_warmup=False` in engine config; `gpu_worker.py:1056` early return, as documented in each run's `result.md`).
4. Spot-audited run-level `result.md` narratives and entry-vs-file counting conventions (a triton cache entry dir = 8 files).

## Per-run verification table

Recomputed values in brackets `[...]` where applicable. All checks per the 7 required items.

| run | dir=run_id | model (CSV=cmd=meta) | mode (meta; warm origin) | monitor (cmd=meta=server.log) | graphs (flag=engine cfg) | extra_args (CSV=cmd) | result cell vs raw evidence |
|---|---|---|---|---|---|---|---|
| G1 | OK | Qwen3.5-0.8B = `/workspace/models/Qwen3.5-0.8B` ×3 | cold | warn = warn = no activation line (eager, by design; engine cfg `jit_monitor_mode='warn'`, `enable_jit_warmup=False`) | eager: `--enforce-eager` present = `enforce_eager=True` (server.log:32) | `--max-model-len 2048 --max-num-seqs 16` OK | 0 events `[jit-events.jsonl=0 lines]`; 184 post-battery triton `[184/1119]`; 77/77 OK (33 rows) `[33 rows, ok=77/77]` ✓ |
| G2 | OK | Qwen3.5-0.8B ×3 | cold | warn = warn = `mode=warn` (server.log:103) | cuda-graph: flag absent = `enforce_eager=False` (server.log:28) | OK | 0 events `[0]`; 0 post-battery `[0/3279]`; 77/77 (33) `[33, 77/77]` ✓ |
| G3 | OK | Qwen3.5-0.8B ×3 | warm; `warm_reuse_of=G2-cold-normal-warn`; `run-env.sh` cache root `/workspace/rca-caches/G2-cold-normal-warn` ✓ | warn = warn = `mode=warn` (server.log:97) | cuda-graph: absent = `False` (server.log:28) | OK | 0 events `[0]`; 0 post-battery `[0/3279]`; 77/77 (33) `[33, 77/77]` ✓ |
| N1 | OK | Qwen2.5-0.5B ×3 | cold | warn = warn = no activation line (eager, by design) | eager: present = `True` (server.log:27) | OK | 0 events `[0]`; 80 post-battery `[80/97]`; 83/99 OK (55 rows) `[55 rows: 11+11+33; ok=0/11, 6/11, 77/77 → 83/99]` ✓ |
| P1 | OK | Qwen3.8-27B-FP8 ×3 | cold | warn = warn = no activation line (eager, by design) | eager: present = `True` (server.log:33) | `--max-model-len 4096 --max-num-seqs 32 --gpu-memory-utilization 0.90` OK | 0 events `[0]`; 216 post-battery `[216/1167]`; 77/77 (33) `[33, 77/77]` ✓ |
| P2 | OK | Qwen3.8-27B-FP8 ×3 | cold | error = error = `mode=error` (server.log:301) | cuda-graph: absent = `False` (server.log:30) | OK | 0 events `[0]`; 0 post-battery `[0/3375]`; 77/77 (33) `[33, 77/77]` ✓ |
| P3 | OK | Qwen3.8-27B-FP8 ×3 | cold | warn = warn = `mode=warn` (server.log:301) | cuda-graph: absent = `False` (server.log:30) | OK | 0 events `[0]`; 0 post-battery `[0/3375]`; 77/77 (33) `[33, 77/77]` ✓ |
| P4 | OK | Qwen3.8-27B-FP8 ×3 | warm; `warm_reuse_of=P5-cold-normal-warn`; cache root `/workspace/rca-caches/P5-cold-normal-warn` ✓ (CSV "reusing P5 cache root (same RUN_ID cache)" supported by run-env.sh header `run=P5-cold-normal-warn`) | warn = warn = `mode=warn` (server.log:295) | cuda-graph: absent = `False` (server.log:30) | OK | 0 events `[0]`; 0 post-battery `[0/3375]`; 77/77 (33) `[33, 77/77]` ✓ |
| P5 | OK | Qwen3.8-27B-FP8 ×3 | cold | warn = warn = `mode=warn` (server.log:301) | cuda-graph: absent = `False` (server.log:30) | OK | 0 events `[0]`; 0 post-battery `[0/3375]`; 77/77 (33) `[33, 77/77]` ✓ |

Independent recomputation additionally matches the committed `evidence/closure/cache-boundary-verification.txt` on every per-section count (triton/vllm/xdg totals and post-battery counts) for all 9 runs.

Entry/file convention cross-check: G1 result.md "23 entries / 19 kernels" = 184 files / 8 files-per-entry ✓ (recomputed 23 post-battery entry dirs); P1 result.md "RUNTIME — 27 entries" = 216 files ✓ (recomputed 27); N1 "80 Triton artifacts" ✓ (recomputed 10 entries × 8).

## Mismatches found

**0 unexplained mismatches.** 0 mismatches in the matrix itself (every cell of `experiment_matrix.csv` and `docs/08-experiment-matrix.md` matches the raw evidence). Two raw-evidence internal narrative inconsistencies were found and are classified/explained:

1. **(b) raw evidence internally inconsistent — G2 result.md narrative vs G2 server.log.**
   `evidence/runs/G2-cold-normal-warn/result.md:7-8` states "cudagraph capture sizes up to 512". G2's own `server.log:28` (engine-config line) shows `'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32]`, `'max_cudagraph_capture_size': 32`, and the string "512" appears nowhere in the run's evidence. Correct value is max capture size 32. Explainable transcription error in the run summary prose; the matrix makes no capture-size claim for G2, so no matrix impact. (P5's result.md correctly quotes [1..64] for the 27B runs.)

2. **(b) raw evidence internally inconsistent — N1 result.md header bullet vs N1 client.log.**
   `evidence/runs/N1-cold-warn/result.md:9` states "all requests OK", but `client.log` shows 55 request rows with 83/99 OK (battery 1: 0/11, client.log:2-12; battery 2: 6/11, client.log:15-25, five `ok=0/1` rows at client.log:21-25; battery 3: 77/77). The same file's caveat (result.md:19-43) documents battery 1's client-harness failures but omits battery 2's 5 failed rows; `requests.jsonl` (33 rows, 77/77) records only the final fixed battery, which is what the bullet accurately describes. Explainable stale/loose header bullet; the matrix cell "requests 83/99 OK (55 battery rows)" (docs/08:22) is exactly correct, as is the note that post-battery counts use the first battery marker. No matrix impact.

Documented convention differences (not mismatches):

- CSV `extra_args` excludes flags represented in dedicated columns (`--jit-monitor-mode`, `--jit-monitor-verbose`, `--enforce-eager`) and infra flags (`--port`); docs/08's "vllm args (from command.sh)" column includes `--enforce-eager` but also omits monitor/port flags. Applied uniformly to all 9 rows.
- Matrix counts post-battery triton FILES; run result.md files quote cache ENTRIES (8 files each) — arithmetically identical (23↔184, 27↔216, 10↔80).
- Battery-start boundary: marker has 1-second resolution; mtimes strictly after that second (incl. same-second fractions) count as post-battery — matches the audit instruction and the repo verifier, and is conservative for the zero-runtime-JIT claims.
- Warm-run `run-env.sh` generation headers read `mode=cold run=<origin>` (provenance of the copied cache env) while the exported `RCA_RUN_ID` is the warm run id and `metadata.json` `mode=warm` is authoritative.

## Verdict

PASS WITH NOTES

The experiment matrix (CSV and docs/08) is fully consistent with the committed raw evidence on all 7 verification points for all 9 runs; every evidence-derived number reproduces exactly under independent recomputation. The two (b)-class raw-evidence narrative inconsistencies (G2 result.md "up to 512" capture sizes; N1 result.md "all requests OK" bullet) are explained, are confined to run-summary prose, and do not affect any matrix cell or the closure claims.
