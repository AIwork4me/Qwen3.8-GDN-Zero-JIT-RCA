# Phase-2 PR-1 — Test-Evidence Audit (2026-10-07)

Question under audit: can
`tests/jit_monitor/test_no_runtime_jit_rocm.py::test_qwen_gdn_no_runtime_jit_rocm`
(worktree `/workspace/vllm-rocm-gdn-zerojit-pr1`, base `68088ed3927e91bce8918db78f2efd9e644a4134`)
produce a FALSE GREEN — i.e., pass while the property it claims to protect
("Qwen GDN models must not JIT-compile during inference on ROCm", enforced via
`jit_monitor_mode="error"`) is actually violated?

## Method

1. Read the final diff (`phase2/pr1/diff.patch`) and the shipped test
   (`tests/jit_monitor/test_no_runtime_jit_rocm.py`) line by line.
2. Traced every mechanism the test relies on into vLLM source:
   - `vllm/utils/jit_monitor.py` (hook installation, `is_active()`, error mode)
   - `vllm/v1/worker/gpu_worker.py::compile_or_warm_up_model` (warmup →
     kernel_warmup → warmup_kernels → cudagraph capture → monitor activation
     ordering; `gpu_worker.py:912,917-919,922-924,1040,1056-1059`)
   - `tests/utils.py::create_new_process_for_each_test` (env propagation to the
     spawned child; `env = os.environ.copy()`)
   - `vllm/v1/executor/uniproc_executor.py::collective_rpc` (where
     `_worker_monitor_state` actually executes)
   - `vllm/model_executor/models/qwen3_5.py` (layer_types → module construction)
   - `tests/models/utils.py::dummy_hf_overrides` (in-place config mutation)
   - `vllm/config/compilation.py` (splitting ops incl. `vllm::qwen_gdn_attention_core`
     at :792; default cudagraph capture-size pattern at :714-718)
   - `vllm/model_executor/warmup/qwen_triton_warmup.py` (what the mutation disabled)
3. Cross-checked all four evidence artifacts (positive single test, positive full
   file, post-revert confirm, GDN-execution proof), the baseline log, the
   negative-control log, and the environment manifests
   (`upstream-base.json`, `manifests/environment.json`).
4. Attempted the six assigned attack vectors; each is reported below with the
   code/evidence that decides it.

## Attack attempts and outcomes (numbered, evidence-cited)

### 1. Cached artifacts hiding a regression — REFUTED (no false green)

**Env-var inheritance through the double spawn.** The chain is
pytest process → (spawn) test child → (spawn) EngineCore worker.

- The test sets `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`
  via `monkeypatch.setenv` in the pytest process
  (`test_no_runtime_jit_rocm.py:66-77`).
- `create_new_process_for_each_test` launches the child with
  `env = os.environ.copy()` (`tests/utils.py`, wrapper visible in the
  negative-control traceback at `tests/utils.py:1904`).
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` (set at
  `test_no_runtime_jit_rocm.py:73`) makes the EngineCore a spawned child of the
  test child, which inherits its environment.
- **Empirical proof of two-hop inheritance**: negative-control log line 170 —
  `[backbone] Using cache directory: /tmp/pytest-of-root/pytest-6/test_qwen_gdn_no_runtime_jit_r0/vllm/torch_compile_cache/...`
  — that is the `tmp_path` fixture value injected into the *pytest grandparent*,
  observed inside the *EngineCore grandchild*. Same for the AOT artifact path
  (line 174). The vars land where the compilations happen.

**Freshness.** Every JIT-relevant cache is redirected under the per-test
`tmp_path`: Triton (`TRITON_CACHE_DIR`), vLLM torch.compile cache including
`torch_aot_compile` (under `VLLM_CACHE_ROOT`, log lines 170/174), and inductor
(`TORCHINDUCTOR_CACHE_DIR`). `tmp_path` is unique per test per run
(`pytest-6/test_qwen_gdn_no_runtime_jit_r0`), so no run can read another run's
artifacts. Non-overridden caches (`HF_HOME`, `XDG_CACHE_HOME`) can hold only
configs/tokenizers/weights — never compiled kernel specializations, so they
cannot mask a warmup key.

**Strongest empirical refutation**: the negative control (09:38, mutation
applied) ran on the same machine *after* the positive runs (08:34, 08:42) and
still FAILED with a runtime Triton compile. If any shared/stale cache could
mask a missing GDN warmup key, that run — same machine, same venv, ~1 h later —
was the place it would have shown up as a pass. It did not. Additionally the
GDN-execution proof records `cache files: startup=2777 final=2777
created_after_battery=0` (gdn-execution-proof.txt:137): zero cache writes
during the battery, i.e., zero compiles, corroborating the hook-based result
with an artifact-level check.

In-process reuse *within* one engine (startup compiles reused by the battery)
is by design: the monitor is armed before the battery, so the first
post-activation compile raises regardless of cache hits afterwards.

### 2. Model avoiding GDN execution — REFUTED

**Truncation really builds a GDN layer.** `Qwen3_5Model.get_layer` selects the
module by `config.layer_types[extract_layer_index(prefix)]`
(`qwen3_5.py:256`), and `Qwen3_5DecoderLayer.__init__` constructs
`QwenGatedDeltaNetAttention` when `layer_type == "linear_attention"`
(`qwen3_5.py:147-154`). `_gdn_hf_overrides` *forces*
`layer_types=["linear_attention","full_attention"]` and `num_hidden_layers=2`
(test:30-42), so layer 0 is GDN regardless of upstream config drift.
`dummy_hf_overrides` mutates the passed `hf_config`/`text_config` in place and
returns `hf_config` itself (tests/models/utils.py:462-464, terminal
`return hf_config`), so the follow-up `text_config.num_hidden_layers = 2` /
`layer_types` assignments hit the same object the engine later reads — the
override composes correctly. Runtime confirmation: gdn-execution-proof.txt:85
(`worker_state`: `gdn_layers: 1, full_attn_layers: 1, layer_types:
["linear_attention","full_attention"]`).

**Dummy weights do not bypass the kernels.** The GDN path is shape-compiled
Triton; weight *values* are irrelevant to compilation and launch. Decisive
evidence: in the negative control (dummy weights, load_format="dummy"), the
inference traceback runs straight through the GDN core —
`qwen3_5.py:589 → … → qwen_gdn_linear_attn.py:1932 (qwen_gdn_attention_core)
→ :1424 (_forward_core) → fused_gdn_prefill_post_conv.py:215
(_fused_post_conv_kernel[grid])` (mutation log lines 265-271) — i.e., the GDN
kernels executed during the battery. Startup-side, the proof's Triton cache
inventory lists six GDN kernel families PRESENT
(`_causal_conv1d_update_kernel`, `fused_recurrent_gated_delta_rule_packed_decode_kernel`,
`_causal_conv1d_fwd_kernel`, `fused_sigmoid_gating_delta_rule_update_kernel`,
`chunk_fwd_kernel_o`, `layer_norm_fwd_kernel`; proof lines 139-144) and 0 files
created after the battery (line 137). The init log also shows the GDN module
arming its Triton path: `Using Triton/FLA GDN prefill kernel
(requested=auto, head_k_dim=128)` from `qwen_gdn_linear_attn.py:177`
(proof line 31; absent GDN modules could not emit it).

Minor evidence artifact (noted, not blocking): gdn-execution-proof.txt:85's
`attention_module_classes` list (`["Attention", "MMEncoderAttention",
"Qwen2_5_VisionAttention", "Qwen3NextAttention"]`) omits
`QwenGatedDeltaNetAttention` — evidently a filtering artifact of the throwaway
inspection script. The conclusion does not rest on that list; it rests on the
layer_types count, the init log, the cache inventory, and the negative-control
execution traceback.

### 3. Monitor inactive / compile hidden in a graph-replayed region — REFUTED as a false-green vector (one structural note)

**Monitor-off hole is closed by the pre-battery assertion.**
`_maybe_activate_jit_monitor` is called at the very end of
`compile_or_warm_up_model` (`gpu_worker.py:1040`), and silently *returns
without activating* if `kernel_config.enable_jit_warmup` is off
(`gpu_worker.py:1056-1059`). The test therefore asserts
`collective_rpc(_worker_monitor_state) == (True, "error")` for all workers
BEFORE the battery (test:59-60). In uniproc mode the RPC executes
synchronously in the EngineCore worker process (`uniproc_executor.py:102-106`),
reading that process's `jit_monitor._active` global (`jit_monitor.py:46-48`) and
its `observability_config.jit_monitor_mode`. Any config drift that disarms the
monitor turns the test red, not green. Runtime confirmation:
`worker_state: {"monitor": [true, "error"], …}` (proof line 85), logged after
`Kernel JIT monitor activated; … mode=error` in the EngineCore (proof line 77).

**A compile cannot hide inside a REPLAY.** Ordering first:
`kernel_warmup` (:912) → `warmup_kernels` (:917-919) → cudagraph
`capture_model()` (:922-924) → monitor activation (:1040). Log confirms:
`Graph capturing finished` then `Kernel JIT monitor activated` (proof lines
76-77). So compiles during CAPTURE are startup, by construction. During REPLAY,
no Python executes at all — Triton compilation requires the Python launcher
(`triton/runtime/jit.py: run → _do_compile → jit_post_compile_hook`; exactly
the chain in the negative-control traceback, log lines 272-283). There is no
mechanism by which a Triton kernel compiles "during" a replayed graph: whatever
kernels a replayed graph launches were compiled at capture (startup) or warmup.
So a runtime compile cannot be silently absorbed by graph replay.

**Where compiles CAN still happen at runtime — and are caught.**
`vllm::qwen_gdn_attention_core` is a splitting op
(`vllm/config/compilation.py:792`), so during PREFILL the GDN core runs eagerly
in Python between piecewise-compiled segments — every new specialization key
compiles through the armed hook. This is precisely what the negative control
demonstrated: battery item 1 (single 32-token prefill + decode) passed because
its keys were already compiled at startup capture; battery item 2 (varlen batch
with prompt lengths 4/32/128, log line 197) hit an uncaptured specialization
(`_fused_post_conv_kernel`) and the monitor raised (log line 289).

**Battery shapes vs. capture.** Default capture sizes follow
`[1, 2, 4] + range(8, 256, 8) + …` capped by max_num_seqs
(`compilation.py:714-718`); with `max_num_seqs=8` the FULL captures are
[1,2,4,8] (log: `FULL: 4/4`). All battery decode batches (1, 2, 3, 4) pad up
into captured sizes — none can escape into an eager decode that compiles
silently (an uncaptured decode would need batch > 8, impossible under
`max_num_seqs=8`; and even eager decode launches through Python, where the hook
fires). Prefill token counts (32; 4/32/128 varlen totaling 164; single 32) are
exact (no padding of varlen totals for the splitting ops) and all fall inside
the inductor compile range (1, 8192) (proof line 46), so no battery shape
escapes into an unplanned compile region. Two factual corrections to the
audit brief: the medium prompt is 32 tokens (`list(range(1, 33))`) and the long
prompt is 128 tokens (`list(range(1, 129))`), not 33/129.

**Structural note (not a false green).** Because decode always replays captured
FULL graphs, a *decode-only* warmup regression is structurally invisible to
this test — capture compiles those kernels at startup, so the test cannot
distinguish "warmup covered decode" from "capture covered decode". Under the
test's asserted property (no runtime JIT for what the battery runs under this
exact config), this is sound: production serving under the same config gets the
same capture-time compilation, i.e., no runtime JIT there either. The NOTE in
`test_no_runtime_jit.py:6-9` ("kernels fully covered by graphs captured during
warmup do not re-trigger the Python JIT hooks") describes exactly this
by-design masking. It becomes a real gap only under different configs
(enforce_eager, larger max_num_seqs), which are out of this test's scope — and
eager mode deliberately disables the monitor (`gpu_worker.py:1056-1059`).

### 4. Unmonitored torch.compile/inductor recompiles — out of scope (noted gap, not a false green of the stated property)

The monitor hooks Triton JIT/autotune, CuTeDSL, and TileLang only
(`jit_monitor.py:15-21`), and TileLang is skipped on ROCm due to broken
symbols (`jit_monitor.py:78-82`). torch.compile/inductor recompiles at runtime
would be invisible. Three reasons this does not falsify the test:

- The test's documented property (docstring, test:91-101) is the GDN Triton
  warmup/capture coverage, detected via `jit_monitor_mode="error"`; the
  monitor's own docstring scopes it the same way. The PR claims exactly what
  the hook can see.
- A runtime inductor recompile is precluded for battery shapes in practice:
  the backbone is compiled once over the dynamic range (1, 8192) (proof line
  46), covering every battery token count; `TORCHINDUCTOR_CACHE_DIR` is fresh
  per run so the startup compile cannot be partially skipped; and vLLM eagerly
  triggers inductor's once-per-process lazy inits during warmup rather than on
  a later cache miss (`gpu_worker.py:1020-1031`, `trigger_inductor_lazy_init`).
- No tilelang or CuTeDSL kernels exist on this path (AITER disabled via
  `VLLM_ROCM_USE_AITER=0`, test:72; `cuteDSL … not available` in the log, proof
  line 68), so those hook gaps are empty set here.

Residual (noted): a *silent* runtime inductor recompile (e.g., a guard miss
introduced by a future refactor) would not fail this test. That is a monitor
scope limitation inherited by every `jit_monitor_mode` consumer, not specific
to this PR; flagging it here so nobody reads "no runtime JIT" more broadly
than "no monitored runtime JIT".

### 5. Negative-control validity — CONFIRMED VALID

`mutation1-gdn-warmup-noop.txt`:

- The mutation no-ops exactly the GDN warmup: early `return` inserted after the
  `_QWEN_MODEL_TYPES` check in `qwen_triton_warmup` (log lines 5-17);
  consequently the mutated run's startup log omits `Warming up Qwen GDN Triton
  kernels for model_type=qwen3_5_text` (present in the positive proof at line
  64, absent in the mutation run's otherwise-identical warmup sequence, log
  lines 180-184 — the generic 72-key warmup, VL, M-RoPE and Mamba warmups all
  still ran).
- The failure IS the monitor raising, not an unrelated crash: the EngineCore
  traceback shows `triton/runtime/jit.py:904 → _call_hook → hook(…) →
  vllm/utils/jit_monitor.py:270 (_on_jit_compile) → :230
  (_log_triton_jit_compile) → :134 (_handle_jit_event) → raise RuntimeError`
  with message `Triton kernel JIT compilation during inference:
  _fused_post_conv_kernel. This causes a latency spike; consider extending
  warmup to cover this shape/config.` (log lines 283-289). The engine had fully
  booted (monitor activated, log line 187), battery item 1 completed (log
  lines 326-331), and item 2's varlen prefill (prompt lens 4/32/128, log line
  197) triggered the compile. The `EngineDeadError` seen by the test is the
  propagation envelope, not the cause.
- The kernel that fired is a GDN prefill kernel
  (`third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py:215`,
  reached from `qwen_gdn_linear_attn.py:1424`), i.e., the failure demonstrates
  sensitivity to exactly the protected property — GDN warmup coverage — and to
  the GDN execution path itself (dummy weights notwithstanding).

Caveat (noted): the mutation removes the *entire* GDN warmup; sensitivity to a
*single-key* regression on a shape outside the battery (e.g., a lone
non-divisible-16 single-sequence prefill length, or a chunked-prefill boundary)
is not proven by this control and is inherently bounded by battery coverage.
Within the battery's shapes, the varlen miss proves the detection end-to-end:
compile → hook → error-mode raise → engine death → subprocess exit 1 → test
FAILED (log line 348). Post-revert confirm (post-revert-confirm.txt: `1 passed`,
263 s) closes the loop.

### 6. Determinism / flakiness risks — no false-green vectors found; failure modes are loud

- `collective_rpc(..., timeout=30)` (test:59): in uniproc non-blocking=False
  mode the call is synchronous in the EngineCore process
  (`uniproc_executor.py:102-106`); the engine is idle at that point (battery
  not started). Effectively no timeout flake surface; a wedged engine would
  fail red.
- `shutdown(timeout=30)` (test:63): observed ~1 s clean shutdowns
  (proof lines 145-154). A hang would raise after 30 s — red, not green.
- `gpu_memory_utilization=0.03` + `kv_cache_memory_bytes=256MiB`: because the
  explicit KV bytes config is set, memory profiling is skipped and the util
  fraction is not enforced (log line 53: `… skipped memory profiling. This
  does not respect the gpu_memory_utilization config`). Ran green 3× on the
  48 GB W7900D (277 s / 263 s / 239 s; init 174-177 s). On MI355 (larger
  free memory) headroom only grows; on smaller GPUs the failure mode is a loud
  OOM, not a silent pass.
- HF hub availability: `load_format="dummy"` avoids weights, but config,
  tokenizer and the multimodal processor for `Qwen/Qwen3.5-0.8B` are fetched
  (proof lines 7-19: arch resolution, chat-template detection, 14.7 s
  multi-modal warmup). An offline CI without a warm HF cache fails loudly at
  engine construction — red, not green.
- Cross-GPU determinism of the *kernel set*: the evidence is gfx1100-only
  (W7900D, triton-rocm 3.8.0, ROCm 7.14 — manifests/environment.json). The
  `Cannot use ROCm custom paged attention kernel, falling back to Triton`
  warning (proof line 69) is platform-specific; MI355 may select different
  backends and hence different compile keys. If MI355 coverage were missing,
  the test fails RED there — the correct direction — but conversely the
  positive evidence does not yet prove MI355 greenness; first CI run on the
  MI355 pool is still untested territory (coverage note, not a validity flaw).
- Seeded engine (`seed=0`) and fixed token-id prompts (no tokenizer dependence
  in the battery itself, test_no_runtime_jit.py:52-61) make the executed shapes
  deterministic; the sampler battery uses seeded params.

## Blocking findings (or none)

**None.** All six attack vectors were either refuted on mechanism plus
empirical evidence (1, 2, 3, 5), or shown to be outside the test's documented
property with no path to a false green (4, 6).

Non-blocking notes for the record:

1. Decode-side GDN warmup regressions are structurally masked by FULL-graph
   capture (attack 3) — consistent with the test's config-scoped property and
   the NOTE in `test_no_runtime_jit.py:6-9`, but worth knowing when reading a
   green run.
2. The test does not itself assert that the GDN Triton path executed (no
   in-test check of GDN module presence or GDN families in the startup cache).
   If a future refactor silently rerouted `linear_attention` away from the
   Triton GDN path, this test would stay green while guarding nothing. Today
   that risk is covered only by the local proof artifact, which is not durable
   in CI. A cheap `collective_rpc` module-introspection assert would close it.
3. Monitor blind spots inherited from `jit_monitor.py`: inductor recompiles
   everywhere, TileLang on ROCm (:78-82), autotune re-benchmarks (print-only).
   Empty sets for this model/path today.
4. `gdn-execution-proof.txt:85` omits `QwenGatedDeltaNetAttention` from
   `attention_module_classes` (inspection-script artifact); the conclusion
   rests on the layer_types count, init log, cache inventory, and the
   negative-control execution traceback instead.
5. Battery facts corrected: medium prompt = 32 tokens, long = 128 (not 33/129).
   Single-sequence non-divisible-16 prefill lengths and chunked-prefill
   boundaries are not exercised (the varlen 164-token batch is the only
   non-divisible total).
6. MI355 (gfx95xx) behavior is unproven; all evidence is gfx1100. First CI run
   remains the real test of portability.

## Verdict: PASS WITH NOTES

The test cannot produce a false green for the property it asserts. Caches are
provably isolated across both spawn hops and empirically non-masking (the
negative control failed even when run after the positives on the same machine);
the truncated model provably constructs and executes the GDN Triton path under
dummy weights; the monitor is asserted armed in the very worker that runs the
battery, is activated strictly after all startup compilation and graph capture,
and no compilation can occur inside a graph replay; the one negative control
fails through the monitor's own error path on a GDN kernel; and every identified
flakiness/scope limitation fails red or lies outside the asserted property.
