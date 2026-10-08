# Upstream Drift Analysis — fresh-image refresh (execution-time)

- PR: vllm-project/vllm#60395 (head `9e370fac11bd6fe3a5a92d762318b674e80af827`, branch `rocm-gdn-zero-runtime-jit-test-main`)
- Previously validated base: `3ca00a8261c7790ba7487e2001ab9004ad8065f2` (validated 2026-10-07 ~10:55Z)
- Execution-time upstream main: `d6fe5dca686b3f6c5f018d399f4f119fedf842f1` (2026-10-07T23:21:51Z, verified identical to GitHub API at retrieval)
- Prompt informational SHA `4ac5a36a8e...` is inside this drift range; superseded by `d6fe5dca` at execution time.
- Drift: 33 commits, merge-base = previously validated base `3ca00a826` (PR branch is strictly ahead-by-one on that base).

## Watched load-bearing files

| file | commits | verdict |
|---|---|---|
| tests/jit_monitor/test_no_runtime_jit_rocm.py | 0 | UNCHANGED |
| tests/jit_monitor/test_no_runtime_jit.py | 0 | UNCHANGED |
| tests/jit_monitor/conftest.py | 0 | UNCHANGED |
| tests/models/utils.py | 0 | UNCHANGED |
| tests/utils.py | 0 | UNCHANGED |
| .buildkite/test_areas/jit_monitor.yaml | 0 | UNCHANGED |
| vllm/model_executor/warmup/qwen_triton_warmup.py | 0 | UNCHANGED |
| vllm/model_executor/warmup/kernel_warmup.py | 1 | CHANGED — fba462e008 (#58165): FlashInfer autotune bucket rounding (round_up=True + speculator max-M). Confined to `_run_flashinfer_autotune_dummy_runs`; FlashInfer-only, not the generic ROCm Triton warmup path used by this test (AITER=0, no FlashInfer). |
| vllm/utils/jit_monitor.py | 0 | UNCHANGED |
| vllm/platforms/rocm.py | 0 | UNCHANGED |
| vllm/envs.py | 0 | UNCHANGED |

## Qwen3.5 / GDN adjacent drift (33-commit range)

- `548597367e` [ROCm][Perf] Add AITER FlyDSL GDN prefill backend (#57560) — touches `vllm/v1/attention/backends/gdn_attn.py` and `qwen_gdn_linear_attn.py`. Verified: `_resolve_gdn_prefill_backend` on ROCm still returns `"triton"` for auto/default; `aiter_flydsl` requires explicit `gdn_prefill_backend` additional_config opt-in plus AITER availability (CDNA3+, VLLM_ROCM_USE_AITER=1). Test path (VLLM_ROCM_USE_AITER=0) semantics preserved.
- `3d6853b32c` hybrid mamba state index seed on prefix-cache hits — state-index seeding; test uses no prefix cache; indirect only.
- `d64a18832b` MiniMax-M3 Triton tensor-descriptor migration — different model family.
- `240ee8de9e` MoRI-IO KV block-offset (MTP) — KV connector/spec-decode only.
- `a8260cbc5c` Bump Transformers to 5.19.0 — dependency bump; affects requirements/common.txt installed at build time; Qwen3.5 config support retained.
- qwen3_5_mtp.py / qwen3_next_mtp.py — MTP head loader changes; test does not use MTP.
- CI-only: 947dd62c07, b47ec440bc, 71f9a74920, efb8d85f24, c741bfca70(docs), a8260cbc5c(CI).

## Classification

**LOW** — no directly edited load-bearing file; the only watched-file change is FlashInfer-scoped; the GDN AITER FlyDSL addition is verified opt-in and cannot alter the generic Triton GDN path selected by this test. Risk is indirect (transformers 5.19.0 bump, hybrid-mamba state seeding). Fresh local validation on `d6fe5dca` remains mandatory and is performed in this run.

## Refresh cycle 2 (execution-time, 2026-10-08T01:1xZ)

During cycle-1 validation upstream advanced 9 commits (d6fe5dca → fdfc171a).
One touches a watched file: `a58bdd0d4e` #58685 — `vllm/utils/jit_monitor.py`
TileLang hook now wraps `JITImpl.compile` instead of `__call__` (perf; hot path
untouched). The Triton monitor path used by this test is unchanged; none of the
9 commits touch the PR's own files, `qwen_triton_warmup.py`, GDN model paths,
or the CI yaml. Branch was re-rebased onto `fdfc171a` (clean cherry-pick, new
head `90b1a661`), rebuilt (v0.1.dev22635+g90b1a6614.rocm714) and the full ROCm
jit-monitor file revalidated: 2 passed in 425.19s with the load-bearing log
markers present (jit_monitor_mode='error', Warming up Qwen GDN Triton kernels,
Using Triton/FLA GDN prefill kernel). Drift classification for cycle 2: LOW.
