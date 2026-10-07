# Final Duplicate-Work Gate (merge-readiness)

- search_time_utc: 2026-10-07 ~11:05Z
- upstream_main: `3ca00a8261c7790ba7487e2001ab9004ad8065f2`
- searcher 1: primary agent (gh pr list / gh api search across open+draft PRs, merged PRs, issues, current main)
- searcher 2: independent general-purpose subagent (fresh context, full report saved verbatim below)

## Queries covered (union)

`Qwen GDN runtime JIT`, `Qwen3.5 GDN jit monitor`, `ROCm GDN zero runtime JIT`,
`test_no_runtime_jit_rocm`, `qwen_triton_warmup test`, `QwenGatedDeltaNetAttention JIT`,
`jit_monitor_mode`, `jit monitor`, `zero runtime JIT`, `GDN warmup`, `GDN zero-JIT`,
`GatedDeltaNet warmup`, full-text `no_runtime_jit`, exact test name
`test_qwen_gdn_no_runtime_jit_rocm`, plus the 100 newest PRs (created desc, through
2026-10-07 ~11:00Z) and the contents of `tests/jit_monitor/` at the current-main ref.

## Key findings

- Exact test name: **0 full-text hits** repo-wide.
- `tests/jit_monitor/test_no_runtime_jit_rocm.py` at current main contains exactly one
  test (`test_v2_sampler_warmup_rocm`, dense Qwen3-0.6B, landed by #58092 on 2026-09-23).
  No GDN test exists anywhere in the directory.
- Nearest PRs, each inspected in depth and ruled NON-duplicates:
  - **#56693** (open draft) enable `--jit-monitor-mode error` CI for DSv4/Gemma4/gpt-oss —
    CUDA-side, touches the CUDA `test_no_runtime_jit.py`, no ROCm, no GDN.
  - **#58091** (merged) `[ROCm][Test]` GDN prefill numerics — kernel numerics test, not
    zero-JIT e2e coverage.
  - **#57560 / #58802 / #60250** (open) ROCm AITER/FlyDSL/RMSNorm GDN perf — runtime work,
    not regression coverage; mission explicitly excludes AITER PRs as duplicates.
  - **#48363** (open) Mamba2 jit warmup fix — Mamba2, no test files.
  - **#54797 / #54251 / #45000** (merged) Qwen Triton warmup extensions/fixes — warmup
    *fixes*; their existence is motivation (the regression class is real: #45000 fixed a
    GDN KKT warmup regression on RDNA that CI did not catch), not duplication.
  - **#50109** (merged) created the CUDA `tests/jit_monitor/` e2e CI — CUDA analog only.
  - **#59137** (merged) reshaped the MI355 mirror block in `jit_monitor.yaml` — the only
    shared touchpoint; our yaml change is additive (label + timeout) on top of the same
    block, complementary, and the file is blob-identical between old base and current main.
- No equivalent coverage appeared in the last-24h newest-PR sweep.

## Verdict

```text
CLEAR TO PROCEED
```

## Independent subagent report (verbatim)

> ## 1. Queries run and notable results
>
> **Open PRs** (`gh pr list --state open --search`):
>
> | Query | Notable results |
> |---|---|
> | "Qwen GDN runtime JIT" | #45819 GDN batch invariance, #57560 AITER FlyDSL GDN prefill, #26807 GDN APC — none jit-monitor tests |
> | "Qwen3.5 GDN jit monitor" | #37429 hybrid KV-cache sizing only |
> | "ROCm GDN zero runtime JIT" | #59523 cuMem pool (draft), #45819, #26807 — none relevant |
> | "test_no_runtime_jit_rocm" | **empty** |
> | "qwen_triton_warmup test" | **empty** |
> | "QwenGatedDeltaNetAttention JIT" | #57294 CPU backend, #45819 — none relevant |
> | "jit_monitor_mode ROCm" | #59581 XPU Triton routing (draft), #47438, #45288 — none relevant |
> | "GDN zero-JIT" | #50380 GDN ucache (draft), #48363 Mamba2 jit warmup, #52190 torch.compile ROCm — none relevant |
> | "GatedDeltaNet warmup" | #57039 FlashInfer GDN copies, #26807 — none relevant |
>
> **Merged PRs (last ~90d)**: hits were all warmup/perf/bugfix work — #54797 (Extend Qwen Triton warmup), #54251 (Warm up Qwen GDN gated RMSNorm), #49903, #36599, #58092, #58465, #39822, #45000 (RDNA GDN KKT warmup fix). Zero test-coverage PRs.
>
> **Issue/PR full-text API**: `jit_monitor GDN` → only warmup PRs; `jit_monitor ROCm` → no test PRs; `no_runtime_jit` → **0 results**; `test_qwen_gdn_no_runtime_jit_rocm` exact-name → **0 results**; `jit-monitor/jit_monitor in:title` → #58685, #59356, #56693, #48363, #58590, #46215, #50109, #42165, #40137 — all monitor semantics, warmups, or CUDA-side CI.
>
> **Newest PRs (created desc, 100 newest through 2026-10-07 10:59Z, #60388–#60317)**: no GDN/zero-JIT/ROCm-test equivalents in the last ~24h.
>
> ## 2. Candidate PRs inspected in depth
>
> | PR | State | Files | Duplicate? |
> |---|---|---|---|
> | **#56693** | open, draft | `tests/jit_monitor/test_no_runtime_jit.py`, `vllm/utils/jit_monitor.py`, `vllm/config/observability.py`, etc. | **No** — CUDA-side jit-monitor CI enablement for dense DSv4/Gemma4/gpt-oss in the *CUDA* test; no ROCm, no GDN, no new test |
> | **#48363** | open | `vllm/model_executor/warmup/hybrid_mamba_warmup.py`, `kernel_warmup.py` | **No** — Mamba2 warmup *fix*; no test files, no ROCm zero-JIT regression |
> | **#58091** | merged | `tests/kernels/mamba/test_gdn_rocm_layout_dispatch.py`, buildkite yamls | **No** — kernel numerics test, not e2e zero-JIT |
> | **#57560** | open | `tests/v1/attention/test_gdn_aiter_flydsl.py`, aiter ops, gdn backend | **No** — perf backend + its own kernel test; orthogonal (contribution tests the *generic* Triton path with AITER off) |
> | **#58802** | open, draft | `tests/kernels/mamba/test_gdn_output_projection.py`, aiter ops | **No** — perf fusion + unit test |
> | **#60250** | open | `tests/compile/passes/test_fusion.py`, `layernorm.py`, `platforms/rocm.py` | **No** — perf enablement; no jit-monitor test |
> | **#58092** | merged 2026-09-23 | created `tests/jit_monitor/test_no_runtime_jit_rocm.py` + ROCm mirror job in `jit_monitor.yaml` | **No — this is the baseline the contribution extends**: dense Qwen3-0.6B sampler-warmup zero-JIT test; no GDN test |
> | #54797 | merged | `tests/model_executor/test_qwen_triton_warmup.py` + warmup code | No — warmup unit tests, not jit_monitor_mode="error" e2e; no ROCm |
> | #54251 | merged | `tests/kernels/test_fla_layernorm_guard.py` + warmup code | No — warmup fix, CUDA-side guard test |
> | #50109 | merged | created `tests/jit_monitor/` dir, `test_no_runtime_jit.py` (CUDA), `jit_monitor.yaml` | No — CUDA-side CI enablement; the ROCm GDN analog doesn't exist there |
> | #58465, #59356, #58685, #49903, #36599, #60244, #50519 | merged/open | infra/monitor/warmup only | No |
> | **#59137** | merged | touches `.buildkite/test_areas/jit_monitor.yaml` — renames mirror label to "(MI355 DPX)" and moves `device: mi355_dpx` | **No — minor touchpoint overlap**: additive (label/timeout bump), not conflicting |
>
> ## 3. Contents of `tests/jit_monitor/` @ 3ca00a82
>
> `__init__.py`, `conftest.py`, `test_hooks.py`, `test_hooks_gpu.py`, `test_no_runtime_jit.py`, `test_no_runtime_jit_rocm.py` — the ROCm file contains exactly one test, `test_v2_sampler_warmup_rocm`, booting dense **Qwen/Qwen3-0.6B** with `jit_monitor_mode="error"`. **No Qwen3.5/GDN test exists anywhere in the directory.**
>
> ## FINAL VERDICT
>
> **CLEAR**
>
> No open, draft, or recently merged PR — nor anything in the 100 newest PRs through 2026-10-07 — adds ROCm Qwen3.5/GDN e2e zero-JIT regression coverage; the exact test name returns zero full-text hits and `tests/jit_monitor/` at ref 3ca00a82 contains only the dense Qwen3-0.6B ROCm sampler-warmup test. All nearby GDN PRs (#48363, #57560, #58091, #58802, #60250, #54797, #54251) are warmup fixes, perf backends, or numerics tests without `jit_monitor_mode="error"` regression coverage, and #56693/#50109 are CUDA-side jit-monitor CI enablement. The only shared touchpoint is `.buildkite/test_areas/jit_monitor.yaml`, where the contribution's additive timeout/label bump on the MI355 mirror job (last reshaped by merged #59137) is complementary, not conflicting.
