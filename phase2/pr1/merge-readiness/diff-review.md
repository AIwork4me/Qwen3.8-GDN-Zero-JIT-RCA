# Final diff review (merge-readiness, current main @ 3ca00a82)

Diff: `git diff upstream/main` — 2 files, tests + CI only.

```
 .buildkite/test_areas/jit_monitor.yaml        |  8 ++++----
 tests/jit_monitor/test_no_runtime_jit_rocm.py | 62 ++++++++++++++++-----------
 2 files changed, 58 insertions(+), 13 deletions(-)
```

## Line-by-line disposition

| hunk | purpose | verdict |
|---|---|---|
| module docstring | file now covers dense + GDN; old text would be false | justified (3 lines) |
| `_worker_gdn_layer_count` | closes the GDN-less false-green hole using the warmup's own predicate | justified (7 lines) |
| `_gdn_hf_overrides` | two-layer hybrid truncation w/ layer_types pin | justified |
| `_run_rocm_shape_battery(model, hf_overrides, expect_gdn=False)` | parameterization; only way to share the boot block without duplication | justified; dense defaults preserved |
| `if expect_gdn:` count assert | GDN presence oracle | justified |
| `_isolate_rocm_jit_caches` extraction | 7 env vars moved verbatim, same order | justified; dense behavior identical |
| dense test body | docstring byte-identical to upstream (delete+re-add around extraction) | no churn |
| new `test_qwen_gdn_no_runtime_jit_rocm` | the contribution | scoped docstring ("standard inference battery", AITER out of scope) |
| yaml label/comment | step now runs MRV2 + GDN | justified |
| yaml timeout 10→25 min, --timeout 300→600 | second engine boot per file | justified (W7900 full file 394 s; MI355 is the authoritative signal) |

## Hygiene

- `git diff --check` clean; `git status --short` shows only the 2 files
- no debug prints / local paths / evidence references / temporary asserts / markers
  (temporary GDN-proof harness used during validation was deleted; verified)
- `ruff check` + `ruff format --check` (ruff 0.14.0) PASS on the changed py file
- yaml parses; `pytest --collect-only` collects 2 tests
- dense test semantics verified unchanged vs upstream (same model/kwargs/env/order/spawn/asserts)

## Independent minimality subagent verdict (verbatim)

> **Findings per checkpoint:**
>
> 1. **File scope** — Exactly 2 files: `tests/jit_monitor/test_no_runtime_jit_rocm.py` (+62/−13) and `.buildkite/test_areas/jit_monitor.yaml` (+4/−4). No production/runtime/kernel/warmup code touched (the test only *imports* `_iter_qwen_gdn_layers`; nothing under `vllm/` modified). `git status --untracked-files=all` shows exactly the 2 modified files, zero strays.
>
> 2. **Hygiene** — Grep of the diff for `print(`, `/workspace`, evidence/proof/TODO/FIXME/HACK/DEBUG/pdb/logging/artifact strings: clean. No commented-out code, no temp asserts (both asserts are the test's real oracle plus a non-vacuity guard), no local paths (`/vllm-workspace/` in yaml is pre-existing context).
>
> 3. **Hunk justification** — All hunks serve the contribution: Module docstring required — old text would be false; dense-test docstring text byte-identical to upstream (delete+re-add around helper extraction); `_gdn_hf_overrides`/`_worker_gdn_layer_count` minimal scaffolding; parameterization + cache-isolation extraction is the only way to add a second test without duplicating the block; no whitespace churn.
>
> 4. **Dense test semantics preserved** — Verified line-by-line vs upstream: same model, same `dummy_hf_overrides` reference, identical LLM kwargs, identical 7 env vars in the same order, same spawn decorator, same monitor-state assert, same `_run_shape_battery`, same shutdown. `expect_gdn=False` default means the new RPC never runs for the dense test. No weakening.
>
> 5. **YAML minimality** — 4 changed values only: label (+matching comment), job timeout 10→25, pytest `--timeout` 300→600. All directly implied by going from one to two e2e engine spin-ups.
>
> **FINAL VERDICT: MINIMAL / MAINTAINER-FRIENDLY**
>
> Reductions required: (none)
