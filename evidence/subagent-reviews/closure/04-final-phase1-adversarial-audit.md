# Closure Audit D — Final Phase-1 Adversarial Review (2026-10-07)

Reviewer mandate: attempt to FALSIFY the Phase-1 closure conclusion
(default-mode zero runtime JIT in the tested envelope; eager-mode runtime
JIT by upstream design; historical #52663 impact not reproducible at the
pinned stack). All seven mandated attack vectors were executed, plus
adversarial probing of the three closure scripts. Every check below was
re-derived independently from committed raw evidence AND, where the
experiment host state survives, from live disk state — not from the repo's
own verifier outputs.

## Method

1. **Independent boundary recomputation.** Re-parsed every
   `evidence/runs/<RUN_ID>/cache-after.txt` (all 6 sections) and every
   `client.log` battery marker with fresh code (not
   `scripts/verify_runtime_cache_boundary.py`) and recounted post-battery
   files per section per run.
2. **Live disk verification (stronger than manifest trust).** The
   experiment cache roots survive at `/workspace/rca-caches/<RUN_ID>/`.
   Recounted actual files and max mtimes per section per root directly on
   disk and compared against the committed manifests (counts must match;
   no file anywhere in a root may have an mtime after that run's battery
   start for default-mode runs).
3. **Out-of-root cache hunt.** Scanned `/root/.cache/triton`,
   `/root/.triton`, `/root/.cache/comgr`, `/root/.cache/tvm-ffi`,
   `/root/.cache/pip`, `/root/.config/vllm`, `/tmp/vllm_dist_*` for any
   write inside the experiment windows (2026-10-07 01:50–06:30).
4. **Config-line forensics.** Extracted `enforce_eager`,
   `cudagraph_capture_sizes`, `CUDAGraphMode`, `CompilationMode`,
   `enable_jit_warmup`, monitor-activation and "Enforce eager set,
   disabling…" lines from all 9 `server.log` files with line numbers.
5. **Clock-skew probe.** Compared client battery-start markers,
   server-side first-request watermarking lines, and file mtimes (all
   same host) for every run.
6. **Determinism probe.** Diffed the actual triton entry-dir sets of
   P5/P3/P2 (independent cold default runs) and P1 (eager) at disk level;
   verified the docs/07 eager-only packed_decode counterexample entry.
7. **Closure-script probing.** Ran
   `verify_runtime_cache_boundary.py` and `rebuild_experiment_matrix.py`
   and confirmed byte-identical regeneration; audited their comparison
   logic and oracle design; audited `build_derived_timeline.py`
   provenance rules against the committed `timeline.csv` files.
8. **Gate run.** Executed `scripts/verify_phase1.sh` end-to-end.

## Attack attempts and outcomes (7 vectors, each with evidence citations)

### Vector 1 — "Default mode still runtime-JITs" → FALSIFICATION FAILED

**Post-battery cache writes (manifested sections).** Independent
recomputation over all six sections (triton/torchinductor/vllm/xdg/
triton-home/hf) of P5/P3/P2/P4/G2/G3: **0 files with mtime after battery
start in any section of any default/graph run.** Matches
`evidence/closure/cache-boundary-verification.txt` lines 16–34, 54–92.
Post-battery counts for eager runs also reproduce exactly (P1: 216 triton
+ 1 xdg; G1: 184 + 2; N1: 80 + 1).

**Live-disk cross-check (not trusting manifests).** Actual on-disk state
of `/workspace/rca-caches/*` matches every manifest count exactly
(P5/P3/P2 triton 3375 files/428 entries each; P1 1167/152; G2 3279/416;
G1 1119/146; N1 97/13; vllm/xdg sections match too — no `head -4000`
truncation occurred, `scripts/stop_server.sh:29`). Maximum observed disk
mtimes vs battery starts (all local UTC, same host):

| run | last disk mtime (any section) | battery start | verdict |
|---|---|---|---|
| P5 | 03:06:50 (triton) | 03:14:40 | clean 7m50s gap |
| P3 | 04:12:53 (triton) | 04:14:30 | clean 1m37s gap |
| P2 | 03:46:34 (triton) | 03:48:01 | clean 1m27s gap |
| P4 | 04:32:31 (xdg; triton 03:06:50 = reused P5 root) | 04:38:25 | clean 5m54s gap |
| G2 | 02:16:34 (triton) | 02:23:14 | clean 6m40s gap |
| G3 | 04:56:33 (xdg; reused G2 root) | 04:59:05 | clean 2m32s gap |

Every cache-after snapshot header postdates its run's battery end
(coverage of the whole battery window), e.g. P5 snapshot 03:31:52 >
battery end 03:30:32 (`cache-after.txt:1`, `client.log:35`).

**Clock skew.** Client battery markers, server first-request lines and
file mtimes are same-host. Server-side "Watermarking is enabled for this
request" agrees with the client battery-start marker **to the exact
second in all six default runs** (P5 server.log:343 = 03:14:40 =
client.log:1; P2 :343 = 03:48:01; P3 :343 = 04:14:30; P4 :337 = 04:38:25;
G2 :143 = 02:23:14; G3 :137 = 04:59:05). No skew exists that could hide a
runtime write on either side of the boundary.

**Writes outside manifested roots.** The cache roots contain exactly the
six manifested section dirs (verified `ls` per root). Out-of-root probes:
`/root/.cache/triton` last write 2026-10-07T01:42:46 (pip-install era),
`/root/.triton` 01:40:33, `/root/.cache/comgr` 01:11:49 (build era),
`/root/.cache/tvm-ffi` 01:40:21 — **all quiescent during every run window
(first run N1 battery 01:58:49–…)**. Notably the comgr (hipRTC/comgr)
cache IS exercised during runs but lands under the redirected
`$XDG_CACHE_HOME/comgr` — visible on disk inside the run roots with
startup-window mtimes only (e.g. P5 root xdg/comgr latest 03:06:43 <
03:14:40; P4's reuse wrote llvmcache.timestamp 04:32:31 < 04:38:25). So
even comgr-level runtime compilation would have been captured by the
manifested xdg section — and none occurred post-battery.

**Env coverage.** `run-env.sh` exports TRITON_CACHE_DIR,
TORCHINDUCTOR_CACHE_DIR, VLLM_CACHE_ROOT, XDG_CACHE_HOME, TRITON_HOME,
HF_HOME (e.g. `P5-cold-normal-warn/run-env.sh:4-9`), enforced by
`scripts/preflight_cache_env.py:27-46` (requires the vars, forbids
/root/.cache, /home, /opt/venv, /tmp prefixes, requires the
/workspace/rca-caches/ isolation root); every `env.txt` ends
`PREFLIGHT PASS` with all six cache vars recorded (P5 env.txt:95-102).

**Instrument positive control.** The same measurement manifest demonstrably
*can* see runtime writes: the three eager runs show 216/184/80 post-battery
triton files through the identical harness. Absence in default mode is a
measurement result, not instrument blindness.

*Residual (non-blocking):* `HF_HUB_CACHE=/workspace/.cache/huggingface/hub`
is inherited unredirected (P5 env.txt:24) — only `HF_HOME` points into the
run root. Immaterial here (model loaded from local path; hf section empty;
hub cache stores no kernels), noted under residual risks.

### Vector 2 — "P5 contaminated by warm cache" → FALSIFICATION FAILED

- Cold proof: `P5-cold-normal-warn/cache-before.txt:1-7` records all six
  sections `0 files 0 bytes (must be empty)` at 02:59:01, produced by
  `scripts/prepare_run_env.sh` cold path which does `rm -rf "$ROOT"`
  (line 52) before creating empty dirs, with an inline `NONEMPTY`
  tripwire (line 75). `metadata.json` `mode=cold`. All 3375 triton
  files carry mtimes inside P5's own startup window (03:00–03:06).
- **3375 vs 1167 is the expected direction, not contamination**: normal
  mode compiles MORE at startup (428 entry dirs: 104-key JIT warmup
  [P5 server.log:288-290 "JIT kernel warmup (104 compile keys)"],
  cudagraph capture [1..64] [server.log:30 config; "Graph capturing
  finished" 03:07:22], V1 profile run), while eager defers decode-leg
  compilation to runtime (P1: 152 entries = 125 startup + 27 runtime).
  The small-model pair shows the same inversion: G2 normal 416 entries
  vs G1 eager 146. A P1-contaminated P5 would instead resemble P1's
  152-entry set with stale mtimes — the opposite of observed.
- **P3 independence is airtight**: different cache root
  (`/workspace/rca-caches/P3-cold-normal-warn`, P3 run-env.sh:3), own
  empty cache-before at 04:05:02, different EngineCore pid (97287 vs
  P5's 94059), artifacts written 04:07–04:12 (hours after P5's
  03:01–03:06), and the **triton entry-dir set is byte-identical to P5's
  (symmetric difference 0 across 428 entries, verified on live disk via
  `comm`)**. Contamination cannot produce a same-set, different-root,
  different-process, later-mtime repeat. P2 (third independent cold
  default run, pid 95755) is also set-identical.

### Vector 3 — "P2 monitor not active / armed late" → FALSIFICATION FAILED

- `P2-cold-normal-error/server.log:301`: "Kernel JIT monitor activated;
  … mode=error" at **03:47:06**, i.e. AFTER engine init (03:47:07 line
  302-adjacent "init engine … took 422.39 s" at 03:47:07) and **BEFORE
  battery start 03:48:01** (client.log:1; server first-request
  watermarking 03:48:01 at server.log:343).
- `jit-events.jsonl` = 0 lines (error mode would have raised → process
  would have failed a request); client.log shows 33/33 rows fully OK
  (ok=N/N per row; P2 result.md "0 raises, 33/33"). An armed error-mode
  monitor with zero raises while all requests succeed is the strongest
  available negative signal, and the ordering (armed → battery) is proven.

### Vector 4 — "P1 artifacts were startup artifacts" → FALSIFICATION FAILED

- Startup had demonstrably STOPPED writing before the battery: last
  pre-battery triton artifact 02:29:41.962 (`batch_memcpy_kernel`,
  P1 cache-after.txt) < "init engine … took 158.21 s" at **02:29:42**
  (P1 server.log:285) < "Application startup complete." (server.log:328,
  untimestamped uvicorn line, positioned between the 02:29:42 init block
  and the first request) < battery start **02:32:48** (client.log:1 ==
  server.log:329 watermarking 02:32:48) < first post-battery artifact
  **02:32:48.743** (`_apply_write_kernel`, sampler/KV infra). The
  3-minute-07-second idle gap contains ZERO triton writes — startup
  could not still have been writing at 02:32:48.
- The post-battery set is exactly the decode leg the eager startup never
  executes: `_apply_write_kernel`, `_gather_block_tables_kernel`,
  `precopy_mamba_align_fused_kernel`, `_w8a8_triton_block_scaled_mm`
  (cache-after.txt entries 02:32:48.7–02:39:26; inventory docs/05 §1:
  20 kernels / 27 entry dirs — independently recounted: 216 files /
  27 distinct entry dirs).
- Latency corroborates compile-at-request: P1 request #1 wall 62.655 s,
  #2 68.319 s, then flat 26.4 s (client.log rows 1-5) vs P5 flat
  26.13 s from request #1 — deltas +36.3/+41.9 s match the claimed
  "+36-42 s".

### Vector 5 — "Eager/default distinction is documentation fiction" → FALSIFICATION FAILED

Upstream's own engine-config lines differ exactly as claimed, in all 9 runs:

| run | enforce_eager | capture sizes | cudagraph_mode | compile_mode | jit_warmup | monitor |
|---|---|---|---|---|---|---|
| P1/G1/N1 | True | `[]` | NONE | NONE | False | never activated |
| P5/P3/P2/P4 | False | [1..64] | FULL_AND_PIECEWISE | VLLM_COMPILE | True | armed |
| G2/G3 | False | [1..32] | FULL_AND_PIECEWISE | VLLM_COMPILE | True | armed |

Citations: P1 server.log:33 (enforce_eager=True, `'cudagraph_capture_sizes': []`,
`enable_jit_warmup=False`), P5 server.log:30 (False, [1,2,4,…,64],
True); "Enforce eager set, disabling torch.compile, CUDAGraphs, and JIT
kernel warmup" present in P1 server.log:18 & :288, G1 :17 & :90, N1 :16
& :50, and absent from all default runs; "Kernel JIT monitor activated"
present in all six default runs (P5 :301, P3 :301, P2 :301, P4 :295,
G2 :103, G3 :97) and absent from all three eager runs. The distinction
is upstream-logged behavior, not narrative.

### Vector 6 — "ROCm 7.2 contaminated the experiment" → FALSIFICATION FAILED

- `evidence/environment/toolchain-rocm714.txt`: build/runtime toolchain =
  venv `_rocm_sdk_devel` (HIP 7.14.60850, AMD clang 23.0.0git,
  InstalledDir = venv devel tree); the host `/opt/rocm -> /opt/rocm-7.2.1`
  (HIP 7.2.53211, clang 22.0.0git) is recorded explicitly as
  "PRE-EXISTING host toolchain (NOT used; contamination reference)".
- `evidence/environment/gate-check.txt`: 21 PASS lines including
  "PASS PATH free of /opt/rocm", "PASS LD_LIBRARY_PATH free of /opt/rocm",
  "PASS cpp_extension ROCM_HOME is devel tree", "PASS vllm imports from
  pinned source", terminating "ENVIRONMENT GATE: PASS".
- Every run inherits the same contract via run-env.sh (ROCM_PATH/HIP_PATH/
  CMAKE_PREFIX_PATH/LD_LIBRARY_PATH/PATH all venv devel; P5 run-env.sh:13-17)
  and each env.txt records it (e.g. P5 env.txt:14,27,49,58,70).
- vLLM worktree at pinned SHA 31e2443c (verified live: `git rev-parse
  HEAD` = 31e2443c…); only non-code files modified (.claude/skills
  metadata, listed in each git.txt) — no runtime code divergence.

### Vector 7 — "Conclusion exceeds the tested envelope" → FALSIFICATION FAILED (one wording nit)

- `docs/09-root-cause.md:65-75` ("Scope envelope") lists the untested
  paths: prefix-cache-hit scheduling (hit rate 0.0% throughout),
  preemption/recompute (KV peaked 9%), non-greedy sampling, structured
  outputs, chat path, speculative decode, plus the precise battery
  envelope (≤128-token prompts × decode batches ≤4, max_tokens 16,
  greedy, max_num_seqs 32, capture ≤64, chunked-prefill 2048) and the
  note that the identical battery under eager compiled 27 entries (i.e.
  the battery demonstrably exercises every decode-leg/w8a8 bucket it can
  reach).
- AITER excluded and labeled: docs/09 "generic non-AITER GDN path";
  docs/10 H3 "AITER not in natural path" (`GDN_AITER_TRITON_AVAILABLE=False`).
  Single-GPU (tensor_parallel_size=1 in P1/P5 config lines) and
  gfx1100-generation pinning are stated in docs/09 Scope and the STATUS
  verdict wording ("single gfx1100"). Multi-GPU and other Radeon
  generations are thereby excluded from the claim, per the stated
  envelope. Large batches > capture envelope are bounded by max_num_seqs
  32 / capture 64 statement.
- Monitor coverage limits honestly recorded: docs/07:105-110 (inductor
  recompiles invisible to monitor; bounded by flat latency + empty
  runtime cache growth); claims scoped to the battery envelope.
- **Wording nit found (non-blocking):** `docs/05-runtime-jit-inventory.md`
  §2 header (line 44) still says "Coverage proof (P1-runtime ⊆ P5-startup,
  per-kernel variant counts)" with a packed_decode "2 → 2 YES" row, while
  `docs/07:76-81` explicitly corrects exactly this: one eager-runtime
  packed_decode entry is absent from P5. Verified on live disk: P1's
  runtime packed_decode entries = {AGERI7W4…, PDUA3AR2…}; P5's startup
  entries = {PDUA3AR2…, R5A55CP…} — AGERI7W4… is eager-only. docs/05's
  count table is numerically true but the "⊆" phrasing is the overstated
  form docs/07 disavows; the load-bearing conclusion (docs/07, docs/09)
  uses the corrected precise statement. Cosmetic inconsistency only.

### Closure-script probing (mandated addition)

**`scripts/verify_runtime_cache_boundary.py`** — sound.
- Boundary convention (lines 49-56, 80): second-resolution battery marker
  maps to `.000000000`; any same-second fraction counts as post-battery.
  For zero-claims this is strictly conservative (can only overcount
  runtime files, never hide them); for eager any_nonzero runs the counts
  (80-216) are far too large for the convention to be load-bearing.
  Marker is written before the first request is sent
  (`run_shape_battery.py:57-58`), so no runtime write can precede it.
- String comparison safety: `canon()` normalizes every timestamp to
  fixed-width `YYYY-MM-DDTHH:MM:SS.fffffffff`; lexicographic order equals
  chronological order for fixed-width same-era ISO strings; truncating
  longer ns fractions to 9 digits is order-preserving. All observed
  timestamps are same-format, same-day, same-TZ (host-local = UTC;
  proven by the exact-second client/server/mtime agreements above). No
  misordering case exists in this data.
- Oracle design (lines 29-44) encodes run DESIGN expectations only;
  counts are always recomputed from `cache-after.txt`; exit-nonzero on
  violation. Regeneration re-run during this audit: exit 0, output
  **byte-identical** to the committed
  `evidence/closure/cache-boundary-verification.txt`.

**`scripts/build_derived_timeline.py`** — provenance honest.
- raw/parsed/derived derivations are labeled per event; untimestamped
  uvicorn lines ("Application startup complete.") are documented as
  limitations rather than given fabricated times (docstring lines 16-19,
  notes at 225-226); the historical empty-timeline bug (superseded
  `extract_jit_events.py` regex mismatch) is disclosed rather than hidden
  (docstring lines 20-26); N1's triple-battery harness bug is carried as
  a note (lines 227-229). Spot-check of P1 timeline.csv: last_startup_
  triton 02:29:41.962 < engine_init_completed 02:29:42 < battery_start
  02:32:48 < first_runtime_triton 02:32:48.743 — every row traces to a
  real record (source_file + line cited). Nit: dead expression at line
  112 (`int(mon[:2]) if False else int(mon)`) — harmless.

**`scripts/rebuild_experiment_matrix.py`** — genuinely evidence-derived.
- Every technical column parsed from command.sh / metadata.json /
  server.log / client.log / cache-after.txt / jit-events.jsonl; hard
  cross-checks refuse output on any inconsistency, including
  command-vs-server-log eager mismatch (lines 148-155) and metadata
  run_id vs dir name (129-130). The historical hand-matrix mislabeling
  (P2/P3/P4/G2/G3 marked eager) is corrected FROM the raw commands and
  disclosed in the generated doc. Regeneration re-run during this audit:
  exit 0, both outputs **byte-identical** to committed
  `manifests/experiment_matrix.csv` and `docs/08-experiment-matrix.md`.
  Only the PLAN/purpose labels are hand-maintained, and they are clearly
  declared as such (lines 31-42).

## Residual risks / non-blocking notes

1. **Closure-audit persistence race (procedural, not evidentiary).** At
   audit time, `evidence/subagent-reviews/closure/` contained only
   `03-experiment-matrix-audit.md` (appeared 05:55 during this audit);
   audits 01 and 02 were not yet persisted, so `scripts/verify_phase1.sh`
   currently reports FAIL (3 missing-file checks: 01, 02, and this file
   04 before it was written). Every substantive gate step PASSES
   (cold proofs, P2 acceptance, P5 boundary, P1 eager artifacts, boundary
   verifier, matrix regeneration, timelines, environment evidence,
   wording hygiene). Re-run the gate after all four audit files exist.
   STATUS.md's "closure audits A-D … PASS" is anticipatory until then.
2. **docs/05 §2 header wording** retains the entry-level "P1-runtime ⊆
   P5-startup" phrasing corrected in docs/07:76-81 (disk-verified
   counterexample entry AGERI7W4…). Suggest a one-line pointer edit in
   docs/05 ("count-level coverage; entry-level statement per docs/07").
   Does not affect any verdict, which uses the precise form.
3. **HF_HUB_CACHE not run-isolated** (env.txt:24 points at the shared
   /workspace/.cache/huggingface/hub; only HF_HOME is per-run). No
   downloads occurred (local model path; hf section empty everywhere),
   but a future network-loaded model could write outside the manifested
   roots. Recommend redirecting HF_HUB_CACHE in prepare_run_env.sh.
4. **Non-Triton runtime code-object paths** (rocBLAS/Tensile lazy load,
   hipBLASLt) are outside both the monitor's hooks and the triton
   manifest; the comgr cache IS covered (via XDG redirect — verified at
   disk), and flat first-request latency (26.13 s == steady state in P5)
   bounds any such hidden cost to ~0. Claims are correctly scoped to
   "runtime JIT in the manifested cache sections".
5. **Single eager 27B run (P1)** — no independent eager cold repeat at
   27B; mitigated by G1 (small GDN eager, same families), N1 (dense
   eager, generic set), upstream design lines, and the latency signature.
6. **Concurrent GPU tenants during 2026-10-07 runs cannot be excluded
   post-hoc**; flat latencies, deterministic repeat (P3 == P5 set), and
   clean cache boundaries make interference immaterial to the verdict.
7. **P4/G3 run-env.sh header comments** carry the original cold run's
   generation stamp ("mode=cold run=P5-cold-normal-warn …") because the
   warm-reuse env files were derived by editing the exports; the exports
   themselves are correct (RCA_CACHE_ROOT at the reused root) and
   cache-before.txt documents the true warm state. Cosmetic provenance
   blemish only.
8. `/tmp/vllm_dist_*` dirs and `/root/.config/vllm/usage_stats.json`
   show run-window activity but are IPC/telemetry artifacts, not compile
   caches; irrelevant to the JIT boundary.

## Final verdict

All seven falsification vectors failed; the primary default-mode proof
was independently reproduced not only from the committed manifests but
from live disk state of the actual cache roots (counts identical, zero
post-battery writes in every section of every default/graph run, clean
gaps of 1m27s–7m50s between last startup artifact and battery start);
client/server/mtime clocks agree to the second; the eager/default
distinction is upstream-config-logged; the environment is provably free
of the host ROCm 7.2 toolchain; and the three closure scripts are
deterministic, evidence-derived, and honest about limitations. The single
wording nit (docs/05 §2 header vs docs/07 correction) and the pending
persistence of sibling closure audits are non-blocking.

PHASE 1 CLOSURE PASS
