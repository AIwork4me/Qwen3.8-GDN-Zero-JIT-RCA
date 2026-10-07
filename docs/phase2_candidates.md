# Phase-2 Candidates (post-RCA; NOT implemented)

Ranked by (root-cause coverage, upstream generality, regression risk, merge
probability, AMD-specific vs generic value).

1. **Eager-mode diagnostic JIT inventory UX** — when `enforce_eager` +
   jit-monitor flags are requested, log explicitly (INFO) at startup:
   "monitor disabled; runtime JIT expected in eager mode" (one line in
   `_maybe_activate_jit_monitor` / arg parsing). Tiny risk, prevents the
   exact misreading this RCA had to unravel. Coverage: diagnostic clarity.
2. **Decode-key warmup for eager/diagnostic mode (opt-in)** — an env-gated
   `qwen_triton_warmup` extension compiling the decode-leg keys
   (`causal_conv1d_update`, packed decode, decode paged-attn, sampler
   batches) via compile-only descriptors, so diagnostic runs can also be
   zero-runtime-JIT when desired. Generic; moderate merge probability
   (aligns with #49349 workstream; must not change dispatch).
3. **w8a8 m-bucket startup coverage in eager** — profile-run loop over the
   config m-buckets {1,4,16,32,64,128,256,512} for block-FP8 GEMMs (or
   reuse the warmup registry); removes the +4 runtime variants seen in P1.
   Generic (all backends); guarded by enable_jit_warmup.
4. **`--jit-monitor-mode error` CI job on ROCm gfx1100 for a GDN model**
   (mirrors #50109 for AMD): cold-cache engine + battery; would have caught
   nothing here (already green) but locks the property in. Low risk,
   moderate merge probability (needs AMD CI capacity).
5. **Ship RDNA3 (`AMD_Radeon_Graphics`/RX 7900 XTX) W8A8 tuned configs**
   (#52663 ask; community JSONs already validated in-issue) — addresses the
   STARTUP autotune cliff on fresh hosts, not runtime JIT. AMD-specific,
   high merge probability, zero runtime risk.
6. **Document `VLLM_ENGINE_READY_TIMEOUT_S` for cold FP8 starts on
   non-Instinct devices** — doc-only; mirrors #52663 secondary ask.

Explicitly NOT candidates (would change runtime behavior under study):
registering new warmups inside model forward, altering VllmJitKernel
dispatch, changing kernel launch configs.
