# Phase-2 PR-1 — Validation Record

Environment (validated Phase-1 stack, unchanged):

- GPU: AMD Radeon Pro W7900D, gfx1100, 48 GB (driver 6.16.13)
- venv: /workspace/venv-qwen-gdn-rca — torch 2.14.1+rocm7.14,
  torch.version.hip 7.14.60850, triton-rocm 3.8.0, rocm-sdk* 7.14.0
  (devel-tree hipcc; host /opt/rocm 7.2 NOT used)
- vLLM: upstream main @ 68088ed3927e91bce8918db78f2efd9e644a4134 built
  from source (editable) in the worktree
  /workspace/vllm-rocm-gdn-zerojit-pr1, branch rocm-gdn-zero-runtime-jit-test
- Deviation note: current main added a CMake FetchContent for
  ROCm/triton @ 669b31ac (triton_kernels) which cannot clone through the
  egress proxy; satisfied via TRITON_KERNELS_SRC_DIR pointing at the
  codeload tarball of the same pinned SHA (documented in upstream-base.json).
- Test-only pip additions: tblib (tests/conftest.py import), ruff 0.14.0
  (lint). No stack changes.

## Results summary

| step | command | result | runtime |
|---|---|---|---|
| environment gate | scripts/verify_environment_gate.py | PASS | — |
| build current main | pip install -e . (worktree) | OK, imports | ~40 min |
| baseline existing test | pytest -q tests/jit_monitor/test_no_runtime_jit_rocm.py | 1 passed | 137 s (+23 s overhead) |
| new test (exact) | pytest -q ...::test_qwen_gdn_no_runtime_jit_rocm | 1 passed | 263–278 s |
| full file | pytest -q tests/jit_monitor/test_no_runtime_jit_rocm.py | 2 passed | 392 s |
| GDN-execution proof | in-process battery w/ module+cache inspection | PASS (see below) | 239 s |
| negative control | GDN-warmup no-op mutation → same test | 1 FAILED via monitor raise | 255–265 s |
| revert | git checkout -- warmup file; grep no marker | clean | — |
| post-revert positive | pytest -q ...::test_qwen_gdn_no_runtime_jit_rocm | 1 passed | 263 s |
| lint | ruff check + ruff format --check (changed py file) | clean | — |
| yaml check | parse .buildkite/test_areas/jit_monitor.yaml | OK | — |

## Monitor-armed proof

- test asserts `collective_rpc(_worker_monitor_state) == [(True, "error")]`
  before the battery and the assertion is exercised in the passing runs;
- in-process proof recorded `monitor: [True, "error"]` in the worker.

## GDN-exercised proof (evidence/positive/gdn-execution-proof.txt)

- arch `Qwen3_5ForConditionalGeneration`; 2 decoder layers built;
  `layer_types == ["linear_attention", "full_attention"]`;
- module inspection: 1 `QwenGatedDeltaNetAttention` + 1 `Qwen3NextAttention`;
- startup Triton cache contains `_causal_conv1d_update_kernel`,
  `fused_recurrent_gated_delta_rule_packed_decode_kernel`,
  `_causal_conv1d_fwd_kernel`, `fused_sigmoid_gating_delta_rule_update_kernel`,
  `chunk_fwd_kernel_o` (FLA), `layer_norm_fwd_kernel`;
- 2 777 startup cache files, **0 created after battery start** (disk-level
  zero-runtime-JIT confirmation, Phase-1 methodology).

## Negative control (evidence/negative-control/mutation1-gdn-warmup-noop.txt)

Disposable uncommitted mutation: early `return` inserted in
`vllm/model_executor/warmup/qwen_triton_warmup.py::qwen_triton_warmup`
(GDN warmup skipped; JIT warmup key count drops 104 → 72 in the engine log).
Effect: `test_qwen_gdn_no_runtime_jit_rocm` FAILS with

```text
RuntimeError: Triton kernel JIT compilation during inference:
_fused_post_conv_kernel. This causes a latency spike; consider extending
warmup to cover this shape/config.
```

raised by `vllm/utils/jit_monitor.py:134` (`_handle_jit_event`, error mode)
— i.e. the failure is exactly the protected property (runtime JIT detected
by the armed monitor), not an unrelated crash. Mutation fully reverted
(`git checkout --`; no marker remains; `git diff` clean except the test +
CI files). Post-revert run passes.

Sensitivity conclusion: the test would catch a GDN-warmup coverage
regression on this path.

## Resource measurements (W7900, 48 GB)

- new test wall: ~4.4 min (boot+capture+warmup ≈ 3.5 min, battery < 1 min)
- gpu_memory_utilization=0.03 (≈1.44 GB) sufficient for the truncated model
- MI355 CI lane: per-test timeout bumped 300 s → 600 s; job 10 → 25 min
  (two boots ≈ 7 min measured; margin for slower CI cold starts)

## Final push record

- local worktree commit: 6ae52736229cb6246645eb9f5980c97538cbd8ae
- pushed fork commit: 37363be53af1528426e1e3cb1dc4249ac4b4b564
  (supersedes 80fb4493, same tree; ref history rewritten once on our own
  branch to reformat the commit message to natural line breaks — the
  original fixed-width wrapping was replaced at author preference before
  any PR was opened)
- both commits carry the identical git tree
  66911b2809a324ad7b832a2a3485d6f92c8c1290 (cryptographic content
  equality); the sha differs only because the fork commit was created
  through the GitHub git-data API (direct git push uploads were
  consistently truncated by the lab egress proxy)
- fork branch: https://github.com/AIwork4me/vllm/tree/rocm-gdn-zero-runtime-jit-test
- fork main was first fast-forwarded to upstream main via the
  merge-upstream API so the base commit 68088ed3 exists on the fork
