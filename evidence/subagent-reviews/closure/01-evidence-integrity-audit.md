# Closure Audit A — Evidence Integrity (2026-10-07)

Auditor: independent subagent (evidence-integrity scope). Repo state audited:
working tree at commit 82f7c9e plus uncommitted closure changes (modified
docs/STATUS/result.md/timeline.csv; untracked evidence/closure/, scripts).
Method: raw-evidence-first; every claim below was re-derived from committed
artifacts, not from doc summaries.

## Method

1. Enumerated the full tree; read every file under `evidence/environment/`
   and compared field-by-field against `manifests/environment.json`.
2. For all 9 runs (P1–P5, G1–G3, N1) read `metadata.json`, `git.txt`,
   `command.sh`, `env.txt`, `cache-before.txt`, `client.log`, and grepped
   `server.log` for the engine-config line (`enforce_eager=`, monitor mode,
   capture sizes), monitor activation, warmup-key counts, and metrics lines.
3. Recomputed cache-boundary counts and max/min mtimes directly from
   `cache-after.txt` manifests with sub-second string comparison (battery-t0
   second fractions counted as runtime, matching the documented conservative
   rule in `evidence/closure/cache-boundary-verification.txt`).
4. Ran `scripts/verify_phase1.sh` (repo closure gate) end-to-end.
5. Cross-checked docs claims (docs/04, 05, 07, 09, 10, 11, README, STATUS)
   against the raw evidence; spot-checked >10 load-bearing claims.
6. Tampering sweep: compared evidence-file filesystem mtimes against content
   timestamps in every run dir; compared committed cache manifests against
   the live `/workspace/rca-caches/` trees; verified `/workspace/vllm-zero-jit-rca`
   `git rev-parse HEAD`; verified vLLM source citations (vllm.py:1758,
   gpu_worker.py:1056, jit_monitor.py error-mode) in the pinned source tree;
   verified `manifests/model-qwen38.json` config sha256 against the live
   model config file.
7. Checked all 9 `timeline.csv` headers for the required provenance columns.

## Findings

1. **Environment identity — VERIFIED.** `manifests/environment.json` fields
   match raw evidence exactly: torch 2.14.1+rocm7.14 and torch.version.hip
   7.14.60850, gfx1100, W7900D/0x744b, driver 6.16.13
   (`evidence/environment/python-rocm714.txt:11-18`,
   `hardware.txt:194-213`); triton-rocm 3.8.0 with amd backend
   (`python-rocm714.txt:19-26`); devel-tree HIP 7.14.60850 / clang 23.0.0git
   toolchain isolated inside the venv with host /opt/rocm 7.2.1 explicitly
   labeled NOT used (`toolchain-rocm714.txt:2-8,55-61`); rocm-sdk 7.14.0
   wheel stack and editable vllm pin
   (`packages-rocm714.txt:172-176,204,218`); vLLM built from SHA 31e2443c
   with the devel hipcc (`vllm-build-rocm714.txt:10-16,38-42`). Gate evidence
   (`gate-check.txt:4-26`) shows 22 PASS lines incl. PATH/LD_LIBRARY_PATH/
   CMAKE_PREFIX_PATH free of /opt/rocm and "ENVIRONMENT GATE: PASS". The
   "COHERENT ROCm 7.14" classification is supported for the build+run stack;
   the host 7.2.1 userspace is disclosed as contamination-reference only.
2. **vLLM SHA identity — VERIFIED.** `vllm_commit` =
   31e2443c90542a33a4a4a293ea7186fba2796c67 in all 9 `metadata.json` files
   and in all 9 `git.txt` files; the live source tree at
   /workspace/vllm-zero-jit-rca resolves HEAD to the same SHA, and its
   `RCA_UPSTREAM_SHA` marker matches. Engine version string
   `v0.1.dev22529+g31e2443c9.d20261007` appears in every run's engine-config
   line. (RCA-repo `rca_commit` values differ between runs, correctly
   reflecting run-time history.)
3. **Cold-cache proofs — VERIFIED.** For P1, P2, P3, P5, G1, G2, N1:
   `cache-before.txt` lists all six sections (triton, torchinductor, vllm,
   xdg, triton-home, hf) as "0 files 0 bytes", and every `env.txt` ends with
   "PREFLIGHT PASS" bound to that run's isolated RCA_CACHE_ROOT. Warm runs
   P4/G3 correctly point at the P5/G2 cache roots respectively.
4. **Raw command identity — VERIFIED (9/9).** Each `command.sh` matches its
   `metadata.json` on model, port, jit-monitor-mode, and enforce-eager
   presence, and matches the run's own `server.log` engine-config line:
   enforce_eager=True for P1/G1/N1 (with `--enforce-eager` in command.sh,
   CompilationMode.NONE, enable_jit_warmup=False); enforce_eager=False for
   P2/P3/P5/P4/G2/G3 (no flag; CompilationMode.VLLM_COMPILE,
   capture sizes [1..64] for 27B, [1..32] for 0.8B). api_utils non-default
   args lines corroborate.
5. **P5 cache/battery separation — VERIFIED.** client.log line 1 battery
   start 03:14:40; monitor activation 03:07:22 (`server.log:301`); maximum
   cache-file mtime across ALL sections = 03:06:50.989 < 03:07:22; zero
   files with mtime after battery t0 (the only post-03:07:22 timestamp in
   cache-after.txt is the snapshot header line itself, 03:31:52).
   `verify_phase1.sh` independently re-derives and passes this boundary.
6. **P2 error-mode acceptance — VERIFIED.** `server.log:301` "Kernel JIT
   monitor activated; ... mode=error" at 03:47:06, before battery start
   03:48:01; client.log has exactly 33 request rows (11 token shapes ×
   batches {1,2,4}), every row fully OK (ok=1/1, 2/2, 4/4); jit-events.jsonl
   is 0 bytes. Error-mode semantics confirmed in pinned source
   (`vllm/utils/jit_monitor.py` — mode "error" raises on monitored compiles).
7. **P1 eager runtime artifacts — VERIFIED.** Post-battery cache files
   (mtime strictly after 02:32:48, same-second fractions counted): 217 total,
   216 in the triton section — satisfies ">200" and equals the "216
   post-battery Triton artifacts" cited in docs/09/STATUS (216 = 27 kernel
   entries × 8 files; 27 matches P1 result.md's audit-verified runtime-entry
   count). Runtime kernel families confirmed with post-battery mtimes:
   `_causal_conv1d_update_kernel` (02:33:25, 02:38:58) and
   `fused_recurrent_gated_delta_rule_packed_decode_kernel` (02:33:26,
   02:38:58), plus `_w8a8_triton_block_scaled_mm` at 02:32:50-53. Latency
   claim verified: 62.655 s / 68.319 s first two requests vs 26.402 s third
   (client.log + requests.jsonl) = +36/+42 s.
8. **Docs traceability — VERIFIED (>10 load-bearing claims re-derived).**
   (a) "zero runtime JIT, default mode, disk-level proof" — P5 and P3
   manifests: 0 post-battery files in any section (P3 last triton mtime
   04:12:53 < battery 04:14:30); (b) "104 compile keys" warmup — P5
   server.log:289 progress line "JIT kernel warmup (104 compile keys)";
   (c) capture sizes [1..64] — P5 engine-config line; (d) P2 "0 raises,
   33/33 OK" — finding 6; (e) eager monitor disabled by design — warning at
   vllm/config/vllm.py:~1758 present in P1/G1/N1 server logs and
   `_maybe_activate_jit_monitor` early-return confirmed at
   vllm/v1/worker/gpu_worker.py:1056 in the pinned tree; (f) N1 "80
   post-battery artifacts" — boundary verifier triton post_battery=80,
   first_post 02:01:01.7, consistent with raw manifest; (g) G1 decode
   families post-battery (184 triton files = 23 entries × 8) — verified;
   (h) "prefix-cache-hit 0.0% throughout / KV peaked 9%" — P5 metrics lines
   (97× "Prefix cache hit rate: 0.0%", max "GPU KV cache usage: 9.0%"); P1
   likewise 0.0%; (i) P5=P3 kernel-set identity — byte-identical
   triton-kernels.txt diff, 428 cache-entry dirs each ("428 entries / 67
   named kernels" in P5 result.md is correct at entry-directory granularity;
   the variant-name sum is 427 with a TOTAL_KERNELS=67 footer); (j) G2
   "warmup 88 keys" — G2 server.log; (k) model identity —
   manifests/model-qwen38.json config sha256 74227dd6… matches the live
   /workspace/models/Qwen3.8-27B-FP8/config.json and every P-run
   metadata.json.
9. **Timeline provenance — VERIFIED.** All 9 runs' timeline.csv carry header
   `timestamp,event,source_file,source_line_or_record,derivation,confidence`
   (+detail), with per-row source pointers (e.g., P5 timeline cites
   server.log:381 for jit_monitor_activated, cache-after.txt derived rows
   with explicit derivation labels). Spot-checked cited line numbers resolve
   correctly.
10. **Experiment matrix — VERIFIED.** `manifests/experiment_matrix.csv` rows
    match every run's committed command.sh flags/modes; the gate's
    regeneration check ("0 mismatches") passes.
11. **Tampering indicators — NONE FOUND (one disclosed restoration).**
    Evidence-file mtimes are contemporaneous with content for all raw run
    artifacts. Single exception: P5 `cache-before.txt` and `run-env.sh` carry
    filesystem mtime 05:12:25 vs content timestamp 02:59:01 — this is the
    *disclosed* restoration in commit 4494ba9 ("restore P5 cold cache-before
    (clobbered by warm-mode bug)"), acknowledged in
    evidence/subagent-reviews/README.md. Cold-start authenticity of P5 is
    independently corroborated: all 3375 triton files in P5's manifest have
    mtimes inside P5's own session window (min 02:59:43, after run-env
    prepared 02:59:01; max 03:06:50), i.e., no pre-existing artifacts could
    have been present. Committed manifests match the live
    /workspace/rca-caches trees exactly (P5 3375 files, P1 1167).
12. **NOTES (non-blocking) — internal contradictions / wording defects:**
    a. `evidence/runs/G2-cold-normal-warn/result.md` states "cudagraph
       capture sizes up to 512", contradicted by the run's own engine-config
       line (`cudagraph_capture_sizes: [1,2,4,8,16,24,32]`, max 32; no
       "512" anywhere in G2 server.log). Doc bug; G2's zero-runtime-JIT
       result is disk-proven and unaffected.
    b. `evidence/runs/N1-cold-warn/result.md` bullet says "all requests OK"
       while client.log batteries 1–2 contain ok=0/1 rows (client-harness
       API bug). The same file's "Caveat (corrected after independent
       audit)" discloses and correctly re-derives this, and docs/04 reflects
       it; the stale bullet remains a residual self-inconsistency.
    c. docs/09 line ~100 calls N1's 80 file-artifacts "80 entries" (loose
       unit wording; raw count is 80 files ≈ 10 kernel entries; N1
       result.md itself says "80 post-battery Triton artifacts").
    d. STATUS.md references "phase2/pr1/" which does not exist in this
       repository (dangling pointer).
    e. docs/11 records the historical gate FAIL (old host-env evidence);
       remediation verified done — superseded evidence quarantined under
       `evidence/environment/host-preexisting-environment/` and fresh
       rocm714 evidence now supports the manifest.
    f. `scripts/verify_phase1.sh` currently reports FAIL solely because
       closure audit files 01 (this file), 02, and 04 are not yet written;
       every evidence-level check in the gate passes.

## Verdict

PASS WITH NOTES

All eight audited dimensions verify against raw evidence: the environment
manifest is fully supported by committed rocm714 evidence with a passing
isolation gate; the pinned vLLM SHA 31e2443c is recorded consistently in all
9 runs and matches the live source tree; all 7 cold runs have empty
six-section cache-before manifests with PREFLIGHT PASS; commands, metadata,
and engine-config lines agree in all 9 runs; the P5 boundary (max artifact
03:06:50 < arming 03:07:22 < battery 03:14:40, zero post-battery files)
holds; P2 acceptance (mode=error armed pre-battery, 33/33 OK, empty events)
holds; P1's eager runtime JIT is real (216 post-battery triton files; GDN
decode kernel families with post-battery mtimes; +36/+42 s spikes); and the
load-bearing claims of docs/09 and STATUS.md trace line-level to raw
evidence. No tampering indicators beyond one disclosed, corroborated
restoration (P5 cache-before). Blocking notes: none. Non-blocking notes that
should be fixed before public upstream review: (1) G2 result.md "capture
sizes up to 512" contradicts its own engine config ([1..32]); (2) N1
result.md's stale "all requests OK" bullet contradicts its client.log and
its own caveat; (3) docs/09 "80 entries" wording for N1's 80 file-artifacts;
(4) STATUS.md dangling phase2/pr1/ reference; (5) the P5 cache-before
restoration should keep its disclosure prominent (it currently lives in a
commit message and the subagent README).
