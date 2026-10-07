# Audit 01 — Upstream Drift (merge-readiness refresh)

- retrieved_at_utc: 2026-10-07T10:55:48Z
- previous_phase2_base: `68088ed3927e91bce8918db78f2efd9e644a4134` (2026-10-06)
- current_upstream_main: `3ca00a8261c7790ba7487e2001ab9004ad8065f2` ([CI] Skip EmbeddingGemma2 pooling tests..., #60371, 2026-10-07T10:22:31Z)
- drift: **60 commits ahead, 0 behind, 282 files changed**
- method: `git fetch --depth=1 upstream main` succeeded; full/deepened fetches still fail
  through the egress proxy (pack truncation). Drift enumerated via GitHub compare API.
  Evidence: `evidence/upstream-drift-commits.txt`, `evidence/upstream-drift-files.txt`.

## Watched-file review

| file | status | relevance to patch |
|---|---|---|
| `tests/jit_monitor/test_no_runtime_jit_rocm.py` | UNCHANGED | patch target; no conflict |
| `tests/jit_monitor/test_no_runtime_jit.py` | UNCHANGED | sibling dense test; no conflict |
| `tests/models/utils.py` | UNCHANGED | `dummy_hf_overrides` import; no conflict |
| `.buildkite/test_areas/jit_monitor.yaml` | UNCHANGED | CI target; no conflict |
| `vllm/model_executor/warmup/qwen_triton_warmup.py` | UNCHANGED | GDN warmup mechanism intact |
| `vllm/model_executor/warmup/kernel_warmup.py` | CHANGED | FlashInfer autotune cache persistence only (per-rank files, dirty-flag save); Qwen Triton warmup untouched; our ROCm/dummy-weight run does not reach FlashInfer autotune |
| `vllm/model_executor/warmup/deep_gemm_warmup.py` | CHANGED | CUDA DeepGEMM; not on ROCm path |
| `vllm/platforms/rocm.py` | CHANGED | UltraQuant 4-bit KV backend selection (new ULTRAQUANT enum + layout checks); no jit/warmup/monitor lines; our run uses default bf16 KV cache |
| `vllm/config/vllm.py` | CHANGED | spec-decode unsupported-features list (ngram_gpu moved to supported; heterogeneous-vocab blocked); no spec decode in our test |
| `vllm/envs.py` | CHANGED | `VLLM_ROCM_AITER_MLA_DCP_VERIFY` default `segmented`→`auto` (gfx950 MLA DCP); irrelevant to GDN path with `VLLM_ROCM_USE_AITER=0` |
| `vllm/model_executor/models/registry.py` | CHANGED | adds EmbeddingGemma2Model only |
| `vllm/model_executor/models/config.py` | CHANGED | EmbeddingGemma2 + XPU/CUDA cache-dtype defaults; no Qwen3.5 change |
| `vllm/v1/worker/mamba_utils.py`, `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` | CHANGED | #60210 "Index Mamba align state copies by request slot in MRV2" — hybrid linear-attention state machinery, spec-decode align context only; our battery is non-speculative but runs MRV2 |
| Qwen3.5 / GDN / FLA kernel paths | UNCHANGED | no `qwen*`, `fla`, `gated_delta`, `linear_attention` model-path changes in drift |

Notable drift commits considered and dismissed as non-duplicates/non-blockers:
- `b267fe48` [ROCm][Perf] W4A16 gfx11 stride padding — quantized GEMM path; we use dummy bf16
- `29f955cc` [ROCm] MLA dual RMSNorm + FP8 DeepSeek-R1 — MLA, not GDN
- `23423437` [ROCm][DCU] MLA DCP round-robin asm decode — MLA DCP, AITER
- `02d8a94e` [CI][ROCm] skip DSv4.1 decoder replay graph test — different test area
- `4b4eb627` [CI] remove AMD mirror of CPU-only rust cargo steps — `test-amd.yaml`, not `jit_monitor.yaml`

## Classification

**LOW-MEDIUM**

- Both directly edited files unchanged; Qwen GDN execution path (model code, FLA kernels,
  `qwen_triton_warmup`) unchanged; CI yaml unchanged.
- Residual risk is indirect: MRV2 hybrid-state machinery saw a spec-decode bugfix (#60210),
  and the FlashInfer warmup area moved adjacent to our warmup registry. Neither is expected
  to alter the GDN zero-JIT contract, but both make **fresh local validation on current main
  mandatory** (planned: baseline dense test, new GDN test, full file, negative control).

## Consequence for patch application

The historical diff applies conceptually to identical upstream file contents for both edited
files. Manual reapplication on a fresh worktree at `3ca00a82` is expected to be conflict-free.
