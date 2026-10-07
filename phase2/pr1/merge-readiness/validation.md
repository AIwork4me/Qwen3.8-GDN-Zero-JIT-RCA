# Merge-Readiness Validation Record (current main @ 3ca00a82)

Environment (revalidated, identical stack to Phase-1/2):

- GPU: AMD Radeon Pro W7900D, gfx1100 (driver 6.16.13)
- venv: /workspace/venv-qwen-gdn-rca — torch 2.14.1+rocm7.14 (hip 7.14.60850),
  triton-rocm 3.8.0, pip rocm-sdk 7.14.0 stack (rocm-sdk-core/devel/device-gfx1100/
  libraries 7.14.0); host /opt/rocm 7.2 NOT used (ROCM_PATH/HIP_PATH pinned to the
  venv devel tree by the canonical build/run scripts)
- vLLM: current upstream main `3ca00a8261c7790ba7487e2001ab9004ad8065f2` built
  editable from /workspace/vllm-rocm-gdn-zerojit-final (branch
  rocm-gdn-zero-runtime-jit-test-main) — `import vllm` resolves to that tree
- Build deviation: same as historical — CMake FetchContent of ROCm/triton
  (triton_kernels @ 669b31ac, pin unchanged on current main) cannot clone through
  the egress proxy; satisfied via TRITON_KERNELS_SRC_DIR pointing at the pinned-sha
  local checkout's python/triton_kernels directory. No other deviations.
- First build attempt failed: TRITON_KERNELS_SRC_DIR pointed at the repo root
  (repo-root CMakeLists requires TRITON_CACHE_PATH) and ambient ROCM_PATH=/opt/rocm
  polluted the env; corrected to the documented python-dir path + canonical env.
  Evidence: evidence/local/build.log (final, successful run).
- Build: `pip install -e . --no-build-isolation`, ~15 min,
  `vllm-0.1.dev1+g3ca00a826...rocm714`, BUILD_SCRIPT_DONE rc=0.
- Test-only pip addition: ruff 0.14.0 (lint parity with historical). No stack changes.

## Results (all on current main, fresh worktree, isolated caches)

| step | command | result | runtime |
|---|---|---|---|
| environment gate | hipcc --version / torch / triton / GPU / ROCM_HOME | PASS (HIP 7.14.60850, devel-tree toolchain) | — |
| build current main | pip install -e . (canonical env) | OK; import path = final worktree | ~15 min |
| baseline existing test | pytest -v -s ...::test_v2_sampler_warmup_rocm | 1 passed | 130.75 s |
| new test (exact) | pytest -v -s ...::test_qwen_gdn_no_runtime_jit_rocm | 1 passed | 278.56 s |
| full file | pytest -v -s tests/jit_monitor/test_no_runtime_jit_rocm.py | 2 passed | 394.14 s |
| GDN-execution proof | temporary in-process harness (deleted after run) | PASS | 247.80 s |
| negative control | GDN-warmup no-op mutation → same test | 1 FAILED via monitor raise | 271.90 s |
| revert | git checkout -- warmup file | clean (0 markers, only 2 intended files modified) | — |
| post-revert positive | pytest -v -s ...::test_qwen_gdn_no_runtime_jit_rocm | 1 passed | 267.02 s |
| lint | ruff check + ruff format --check (changed file, ruff 0.14.0) | clean | — |
| yaml | python yaml.safe_load on jit_monitor.yaml | OK | — |
| collect | pytest --collect-only (changed file) | 2 tests collected | — |
| diff | git diff --check / --stat upstream/main | 2 files, tests+CI only | — |

Note: the GDN-execution proof's first run failed for a harness-only reason
(collective_rpc needs VLLM_ALLOW_INSECURE_SERIALIZATION=1, which the real test sets
via _isolate_rocm_jit_caches but the proof harness initially didn't); fixed in the
harness, not the patch. The temporary harness file was deleted; `git status` shows
only the 2 intended files.

## Monitor-armed proof (current main)

- test asserts `collective_rpc(_worker_monitor_state) == [(True, "error")]` before
  the battery — exercised in all passing runs;
- in-process proof recorded `monitor=(True, 'error')` in the worker.

## GDN execution proof (current main, in-process harness)

```
PROOF arch=['Qwen3_5ForConditionalGeneration'] model_type=qwen3_5_text
PROOF layer_types=['linear_attention', 'full_attention']
PROOF gdn_layers=1 monitor=(True, 'error')
PROOF cache files after startup: 2777
PROOF battery complete with monitor=error and ZERO raises
PROOF cache files: startup=2777 final=2777 created_after_battery=0
PROOF family _causal_conv1d_update_kernel: PRESENT (96 mentions)
PROOF family fused_recurrent_gated_delta_rule_packed_decode_kernel: PRESENT (51)
PROOF family _causal_conv1d_fwd_kernel: PRESENT (48)
PROOF family fused_sigmoid_gating_delta_rule_update_kernel: PRESENT (51)
PROOF family chunk_fwd_kernel_o: PRESENT (1728)
PROOF family layer_norm_fwd_kernel: PRESENT (153)
GDN-PROOF: PASS
```

The in-test GDN assertion (`_worker_gdn_layer_count` via
`_iter_qwen_gdn_layers`) is exercised by every GDN-test run (`expect_gdn=True`)
and independently confirmed by the proof harness (gdn_layers=1).

## Negative control (current main)

Mutation (disposable, reverted, never committed/pushed):

```diff
+    if True:  # NEGATIVE-CONTROL MUTATION (disposable): skip GDN warmup
+        return
```
(inserted after the model-type guard in `qwen_triton_warmup`)

Result: test FAILS with

```
RuntimeError: Triton kernel JIT compilation during inference: _fused_post_conv_kernel.
This causes a latency spike; consider extending warmup to cover this shape/config.
```
(raised from jit_monitor._handle_jit_event during the battery — the same kernel and
mechanism as the historical control). Revert → clean → test PASS.

## CI budget reassessment

Current-main runtimes are statistically identical to historical (131/279/394 s vs
137/263-278/392 s). No evidence-based reason to change the 600 s per-test / 25 min
job budget. First upstream MI355 run remains the authoritative hardware signal.

## Evidence files

- evidence/local/environment-gate.txt
- evidence/local/build.log
- evidence/local/baseline-dense-test.txt
- evidence/local/new-gdn-test.txt
- evidence/local/full-file.txt
- evidence/local/gdn-execution-proof.txt
- evidence/negative-control/mutation1-gdn-warmup-noop.txt
- evidence/negative-control/post-revert-confirm.txt
