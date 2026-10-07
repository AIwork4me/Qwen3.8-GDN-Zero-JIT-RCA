# 07 — Compile-Key Analysis (core RCA; corrected after independent audit)

Question: why did startup fail (eager) or succeed (normal) in producing the
exact compile artifacts the first real requests needed?

## The runtime keys (eager, cold — P1)

27 triton cache entries mtime-attributed after battery t0 (audit-verified),
covering: `_causal_conv1d_update_kernel` (2), 
`fused_recurrent_gated_delta_rule_packed_decode_kernel` (2),
`_w8a8_triton_block_scaled_mm` (+4), `fused_sigmoid_gating_delta_rule_update_kernel`
(+1), `kernel_paged_attention_2d` (1), `reshape_and_cache_kernel_flash` (1),
`_apply_write_kernel` (3), and 13 further sampler/KV/mamba-align kernels
(1 each).

## Mechanism per kernel family

### Decode-leg kernels (conv1d_update, packed decode, paged attention, sampler) — class H (+F)

In **normal mode** these compile during STARTUP because:
1. cudagraph capture executes real decode steps at sizes
   [1,2,4,8,16,24,32,40,48,56,64] (`force_attention=FULL`,
   gpu_model_runner.py:6881/:6031-6046) → Triton compiles each
   per-size specialization (conv1d_update: 6 variants);
2. the 104-key JIT warmup launches compile-only variants (registry from
   #50174).

In **eager mode** `--enforce-eager` sets `kernel_config.enable_jit_warmup=False`
and disables cudagraph capture (vllm/config/vllm.py:1746-1764; P1 log has
the exact message; engine config shows `cudagraph_capture_sizes: []`,
`enable_jit_warmup=False`), and the V1 profile run never executes a decode
step: with `attn_metadata=None` (gpu_model_runner.py:5956; metadata only
under force_attention/FULL :5978) the GDN core returns via
`_warmup_prefill_kernels` (qwen_gdn_linear_attn.py:1272-1274) before the
decode branch. The startup phase therefore produces ZERO decode-leg
artifacts; the first real decode request compiles them → runtime JIT.
Classification: **H (decode path never exercised during startup)** — by
upstream design in eager — plus **F** (warmup-vs-runtime configuration
difference).

### `_w8a8_triton_block_scaled_mm` — Triton dynamic-arg specialization on M (class B-adjacent)

Corrected mechanism (audit re-derived from .ttir signatures): gfx1100 has
**no** tuned W8A8 config JSONs in-tree (0 of 205 device-keyed files), so
`get_w8a8_block_fp8_configs` returns None and the **default config
(BLOCK_SIZE_M=64) is used for every M; the m-bucket config lookup
(fp8_utils.py:963-966) is inert here**. All 6 variants share identical
launch options (num_warps=4, shared=25088). The variants differ by
**Triton's dynamic-argument specialization on M**: startup (profile/capture)
compiles the `%M {tt.divisibility=16}` specialization; runtime requests
with M values that are constant-foldable or non-divisible produce
additional specializations. In normal mode the capture sizes + warmup
produce the full needed set (6) at startup; in eager, 4 of the 6 first
appear at runtime.

### `layer_norm_fwd_kernel` — class I (already fixed at this SHA)

The 3 startup variants come from `warmup_layer_norm_fwd`'s
M-specialization enumeration (layernorm_guard.py:270-331; PR #54251
lineage). The runtime needed 0 more — **even in eager mode** — so the
historical #49349 ROWS_PER_BLOCK gap is closed at this SHA for this
geometry.

## Coverage proof (normal mode) — precise claim

Decisive evidence for normal mode is behavioral + artifact-level:

1. **zero runtime compiles**: P5's 428 triton entries all mtime-attributed
   BEFORE battery t0 (audit re-verified), zero jit_monitor events across
   P5/P3/P2/G2/G3, flat first-request latency (26.13 s == steady state);
2. **deterministic**: P3 independent cold run reproduced an identical
   kernel set (symmetric difference 0 by audit's name normalization);
3. per-kernel variant counts: every kernel that runtime-compiled in eager
   has ≥ that count among P5's startup variants (20/20 table, docs/05 §2).

Scope caveat (audit): a literal entry-level superset claim vs EAGER is
overstated — one eager-runtime packed_decode entry is absent from P5
(an eager-only, unbatched specialization normal mode never executes). The
proven statement is: **normal-mode runtime needs ⊆ normal-mode startup
artifacts** (evidenced by zero runtime compiles in armed-monitor runs), not
a bitwise superset of eager's keys.

## Conclusion

For the DEFAULT (normal) configuration at pinned SHA 31e2443 on gfx1100
(within the tested envelope: prompts ≤128 tokens, decode batches ≤4,
max_tokens 16, greedy, max_num_seqs 32, capture ≤64):

```text
compile-key cause of runtime JIT: NONE — no gap observed
```

For DIAGNOSTIC (eager) configuration:

```text
runtime JIT root cause = startup produces no decode-leg artifacts and
only the startup-shaped w8a8 specializations, because enforce_eager
disables cudagraph capture AND the JIT warmup registry
(vllm/config/vllm.py:1746-1764; gpu_worker.py:1056), while the V1
profile run never executes a decode step (attn_metadata=None →
_warmup_prefill_kernels); all decode-leg Triton kernels therefore
compile at the first real decode request.
```

Measurement-coverage limits (audit): the JIT monitor hooks Triton
(jit+autotune) and CuTeDSL only; Tilelang is skipped on ROCm
(jit_monitor.py:81-82); a hypothetical torch.compile/inductor runtime
recompile would be invisible to the monitor — bounded here only by flat
first-request latency and empty runtime-side cache growth. Claims are
scoped to the battery envelope above.
