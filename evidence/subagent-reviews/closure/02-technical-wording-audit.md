# Closure Audit B — Technical Wording (2026-10-07)

Auditor: independent technical-language auditor (Subagent B, closure).
Scope: README.md, STATUS.md, docs/00,01,02,03,04,05,06,07,09,10,11, phase2_candidates,
evidence/runs/*/result.md (all 9), evidence/subagent-reviews/README.md — current
working-tree state.

## Method

1. Read every in-scope file end to end.
2. Searched for the overstatement patterns: "fixed", "disproved", "resolved",
   "root cause", "all/every compile keys", "all decode keys", "no/zero runtime JIT",
   "deterministic", "not causal", "≪", universal quantifiers.
3. Cross-checked every quantitative or causal wording against raw evidence:
   `evidence/closure/cache-boundary-verification.txt` (per-run post-battery file
   counts: P1=216, G1=184, N1=80 triton files; all default/graph runs 0),
   `evidence/runs/*/triton-kernels.txt` (per-kernel variant counts),
   per-run result.md latency claims, and the internal docs/07 compile-key
   correction (entry-level eager superset retracted).
4. Applied the established fact set: default/graph runs show zero cache files and
   zero monitor events after battery start; eager runs show post-battery Triton
   artifacts with first-request spikes (P1/G1); the JIT monitor is intentionally
   inactive in eager (upstream design); #52663 reported ~43-min first-inference
   JIT on vllm 0.27.1+rocm723; tested envelope = prompts ≤128 tokens, batches ≤4,
   max_tokens 16, greedy, max_num_seqs 32, capture sizes through 64, single
   gfx1100, generic non-AITER GDN path; N1 (dense non-GDN) ALSO runtime-JITs in
   eager.
5. Severity rule (per charter): BLOCKING = materially overstates evidence and
   could mislead an upstream maintainer. Other findings = apply/recommended.

## Findings

### B-1 (BLOCKING) — docs/05-runtime-jit-inventory.md:44 (+ table row :49, summary :52)

Current text (line 44):
> capture sizes [1,2,4,8,16,24,32,40,48,56,64] + V1 profile run. Coverage proof (P1-runtime ⊆ P5-startup, per-kernel variant counts):

Current text (line 49, table row):
> fused_recurrent_gated_delta_rule_packed_decode_kernel          2 → 2  YES

Current text (line 52):
> (all 20 P1-runtime kernels covered; 51 extra startup-only kernels)

Problem: asserts the literal subset "P1-runtime ⊆ P5-startup", which
docs/07:76-81 explicitly retracts ("a literal entry-level superset claim vs EAGER
is overstated — one eager-runtime packed_decode entry is absent from P5 (an
eager-only, unbatched specialization normal mode never executes)"). The table row
"2 → 2 YES" presents count equality as entry identity for exactly the kernel
docs/07 flags. Additionally, "51 extra startup-only kernels" is arithmetically
inconsistent with the adjacent totals (67 named kernels − 20 shared = 47, or 70 −
20 = 50 under the stated audit normalization — neither yields 51). An upstream
maintainer would conclude normal-mode startup is a bitwise superset of eager's
runtime keys; it is not.

Suggested replacement (line 44):
> capture sizes [1,2,4,8,16,24,32,40,48,56,64] + V1 profile run. Per-kernel-family coverage (every P1-runtime kernel family present in P5-startup with ≥ per-kernel variant count; NOT an entry-level subset — one eager-only packed_decode specialization is absent from P5, see docs/07 scope caveat):

Suggested replacement (line 49):
> fused_recurrent_gated_delta_rule_packed_decode_kernel          2 → 2  YES*

Suggested replacement (line 52):
> (all 20 P1-runtime kernel families present at ≥ per-kernel variant count; the remaining P5 kernels are startup-only)
> (*count-level coverage only: per docs/07, one eager-runtime packed_decode entry — an unbatched specialization normal mode never executes — is absent from P5; the proven statement is normal-mode runtime needs ⊆ normal-mode startup artifacts, evidenced by 0 runtime compiles in armed-monitor runs.)

### B-2 (BLOCKING) — docs/10-falsification.md:6 (H2 row, "Experiment" column)

Current text:
> P1-runtime ⊆ P5-startup superset proof

Problem: same retracted entry-level subset/superset claim as B-1, stated as the
proof artifact in the falsification matrix. Contradicts docs/07's corrected
claim ("normal-mode runtime needs ⊆ normal-mode startup artifacts ... not a
bitwise superset of eager's keys").

Suggested replacement:
> P1-runtime ⊆ P5-startup at per-kernel variant-count level (not entry-level; one eager-only packed_decode specialization absent from P5 — docs/07 scope caveat)

### B-3 (BLOCKING) — docs/06-call-graph.md:45-48

Current text:
> - `layer_norm_fwd_kernel` (3 variants) compiled only during startup: the
>   variants come from `qwen_triton_warmup`'s `warmup_layer_norm_fwd`
>   M-specialization enumeration (qwen_triton_warmup.py:148-168 →
>   layernorm_guard.py:270-331), not from a profile-run forward.

Problem: contradicts committed run evidence. The observed 3 startup layer_norm
variants are documented in the eager runs (P1, G1), where the JIT warmup registry
is disabled (`enable_jit_warmup=False`) and `qwen_triton_warmup` cannot have run.
P1 result.md:46-47 attributes them to "the V1 profile-run forward"; G1
result.md:37-39 lists layer_norm among families "compiled during the eager
STARTUP profile run". The phrase "not from a profile-run forward" is therefore
false for the very runs that supplied the observation; the warmup-module
attribution can only hold for normal-mode runs. A maintainer would misderive the
eager startup mechanism from this bullet.

Suggested replacement:
> - `layer_norm_fwd_kernel` (3 variants) compiled only during startup. Attribution differs by mode: in EAGER (P1/G1) the JIT warmup registry is disabled, so the startup variants come from the V1 profile-run forward (P1/G1 result.md); in NORMAL mode the same 3 M-specializations are produced by `qwen_triton_warmup`'s `warmup_layer_norm_fwd` enumeration (qwen_triton_warmup.py:148-168 → layernorm_guard.py:270-331).

### B-4 (BLOCKING) — docs/07-compile-key-analysis.md:58-60

Current text:
> The 3 startup variants come from `warmup_layer_norm_fwd`'s
> M-specialization enumeration (layernorm_guard.py:270-331; PR #54251
> lineage). The runtime needed 0 more — **even in eager mode** — so the
> historical #49349 ROWS_PER_BLOCK gap is closed at this SHA for this
> geometry.

Problem: same attribution error as B-3 in the section whose load-bearing evidence
is "even in eager mode": in eager the warmup module does not run, so the 3 eager
startup variants cannot come from `warmup_layer_norm_fwd`; P1/G1 result.md
attribute them to the V1 profile run. The conclusion (0 extra runtime variants
even in eager; gap closed for the tested geometry) is evidence-backed and stays;
the attribution sentence overstates mechanism knowledge.

Suggested replacement:
> In normal mode the 3 startup variants come from `warmup_layer_norm_fwd`'s M-specialization enumeration (layernorm_guard.py:270-331; PR #54251 lineage); in eager (warmup disabled) they come from the V1 profile-run forward (P1/G1 result.md). Either way the runtime needed 0 more — **even in eager mode** — so the historical #49349 ROWS_PER_BLOCK gap is closed at this SHA for this geometry.

### B-5 (BLOCKING) — docs/04-reproduction.md:43 and docs/09-root-cause.md:93-94

Current text (docs/04:43):
> depending on mode; engine ready in ≤ 8.5 min ≪ 600 s timeout)

Current text (docs/09:93-94):
> - The 600 s engine-ready timeout class of #52663: not observed (cold
>   startup ≤ 8.5 min ≪ 600 s default here), noting config differences

Problem: 8.5 min = 510 s, i.e. 85% of the 600 s default — below the limit but
not "much less than" (≪) it. The symbol materially overstates the safety margin
(~15% headroom) of the "timeout class not observed" conclusion; a maintainer
could reasonably conclude default-timeout headroom on gfx1100 cold starts is
comfortable when it is thin (and #52663 reported 1474 s on a different config).

Suggested replacement (docs/04:43):
> depending on mode; engine ready in ≤ 8.5 min (510 s) < 600 s default timeout — below the limit, with only ~15% headroom)

Suggested replacement (docs/09:93-94):
> - The 600 s engine-ready timeout class of #52663: not observed (cold startup ≤ 8.5 min = 510 s < 600 s default here — a modest margin), noting config differences

### B-6 (apply) — docs/04-reproduction.md:32-33

Current text:
> - zero-runtime-JIT in normal mode: **deterministic** (P5, P3 identical; P2
>   error-mode agrees; G2 agrees on the small model)

Problem: "deterministic" is a universal property claim established from n=2
identical cold runs (plus corroborating error-mode/small-model runs), and the
bullet lacks the battery-envelope bound.

Suggested replacement:
> - zero-runtime-JIT in normal mode: **reproduced across independent runs** (P5, P3 identical cold sets; P2 error-mode agrees; G2 agrees on the small model) within the tested battery envelope

### B-7 (apply) — docs/04-reproduction.md:34-35

Current text:
> - runtime-JIT set in eager mode: **deterministic** (P1 single run + G1
>   consistent kernel families; first-request latency spike reproducible)

Problem: self-undermining — a single P1 run cannot establish determinism, and the
parenthetical admits it ("P1 single run").

Suggested replacement:
> - runtime-JIT set in eager mode: **consistent with a stable mechanism** (single P1 run; G1 shows the same kernel families at 0.8B; first-request latency spike reproduced in P1 and G1)

### B-8 (apply) — evidence/runs/P3-cold-normal-warn/result.md:5

Current text:
> first-request 26.20 s (no spike). Cold-run zero-JIT is DETERMINISTIC. runtime JIT events: 0.

Problem: "DETERMINISTIC" from two cold runs (P5, P3) overstates; determinism is
not establishable from n=2.

Suggested replacement:
> first-request 26.20 s (no spike). Cold-run zero-JIT reproduced identically across two independent cold runs (P5, P3). runtime JIT events: 0.

### B-9 (apply) — docs/07-compile-key-analysis.md:71-72

Current text:
> 2. **deterministic**: P3 independent cold run reproduced an identical
>    kernel set (symmetric difference 0 by audit's name normalization);

Problem: same n=2 "deterministic" overstatement; the evidence supports
"reproducible", not a determinism property.

Suggested replacement:
> 2. **reproducible**: P3 independent cold run reproduced an identical kernel set (symmetric difference 0 by audit's name normalization);

### B-10 (apply) — evidence/runs/P5-cold-normal-warn/result.md:23-27

Current text:
> Every kernel family that runtime-compiled in eager P1 —
> `_causal_conv1d_update_kernel`, `fused_recurrent_gated_delta_rule_packed_decode_kernel`,
> `_w8a8_triton_block_scaled_mm` (all 6 variants), `kernel_paged_attention_2d`,
> sampler/KV infra — was fully compiled during startup in normal mode
> (cudagraph capture sizes [1..64] + 104-key JIT warmup).

Problem: "Every ... was fully compiled" invites the entry-level superset reading
that docs/07:76-81 retracts (one eager-runtime packed_decode entry is absent from
P5). The proven fact is per-family/variant-count coverage plus zero normal-mode
runtime compiles in armed-monitor runs.

Suggested replacement:
> Every kernel family that runtime-compiled in eager P1 —
> `_causal_conv1d_update_kernel`, `fused_recurrent_gated_delta_rule_packed_decode_kernel`,
> `_w8a8_triton_block_scaled_mm` (6 startup variants, ≥ eager's 6), `kernel_paged_attention_2d`,
> sampler/KV infra — was compiled during startup in normal mode with ≥ per-kernel
> variant count (cudagraph capture sizes [1..64] + 104-key JIT warmup). Coverage is
> at kernel-family/variant-count level, not entry-level identity: one eager-only
> packed_decode specialization is absent from P5 (docs/07 scope caveat).

### B-11 (apply) — evidence/runs/G2-cold-normal-warn/result.md:18-23

Current text:
> Every GDN kernel family that runtime-compiled in EAGER G1 —
> `_causal_conv1d_update_kernel` (6 variants here vs 2 in eager),
> `fused_recurrent_gated_delta_rule_packed_decode_kernel` (2),
> `_causal_conv1d_fwd_kernel` (2), `layer_norm_fwd_kernel`, FLA chunk
> autotune set, `batch_memcpy_kernel` — was fully compiled during STARTUP
> (warmup + cudagraph capture) in normal mode.

Problem: same "fully compiled" family-level phrasing as B-10; without the
entry-level caveat it supports the retracted superset reading (the docs/07
caveat was derived for P1/P5 and not analyzed for G1/G2, so entry identity is
simply unproven here).

Suggested replacement:
> Every GDN kernel family that runtime-compiled in EAGER G1 —
> `_causal_conv1d_update_kernel` (6 variants here vs 2 in eager),
> `fused_recurrent_gated_delta_rule_packed_decode_kernel` (2),
> `_causal_conv1d_fwd_kernel` (2), `layer_norm_fwd_kernel`, FLA chunk
> autotune set, `batch_memcpy_kernel` — was compiled during STARTUP
> (warmup + cudagraph capture) in normal mode, with per-kernel variant counts
> ≥ eager's (family-level coverage; entry-level eager superset not claimed).

### B-12 (apply) — evidence/runs/G2-cold-normal-warn/result.md:30-32

Current text:
> The
> historical #49349 Qwen-GDN report (layer_norm_fwd ROWS_PER_BLOCK mismatch,
> main d9dabfa, 2026-08-28) is **no longer reproducible for this model/shape
> battery at this SHA** (fixed by #54251/#54797 lineage).

Problem: the parenthetical "fixed by #54251/#54797 lineage" is unqualified —
"fixed" implies the general historical bug is fixed; the main clause's
model/battery/SHA qualification does not extend into the parenthetical, and the
PR attribution is source-lineage-derived, not experimentally isolated. Correct
framing (used in docs/01/07/09) is "gap closed at this SHA for the tested
geometry".

Suggested replacement:
> The
> historical #49349 Qwen-GDN report (layer_norm_fwd ROWS_PER_BLOCK mismatch,
> main d9dabfa, 2026-08-28) is **no longer reproducible for this model/shape
> battery at this SHA** (gap closed at this SHA for the tested geometry; warmup
> coverage traces to the #54251/#54797 lineage).

### B-13 (apply) — evidence/runs/P5-cold-normal-warn/result.md:31-35

Current text:
> - The historical "first inference triggers ~43 min Triton JIT" (#52663
>   comment, vllm 0.27.1+rocm723, Aug 2026) is **not reproducible** on pinned
>   current main in the default (graph) configuration: the decode-path kernels
>   compile during cudagraph capture at startup, and the 104-key warmup covers
>   the rest.

Problem: "not reproducible on pinned current main" lacks the tested-envelope and
single-host bound; the correct framing is "not reproduced within the tested
battery envelope on this stack/host; the eager-mode mechanism remains
reproducible" (the next paragraph covers eager, but this sentence alone can be
quoted unqualified).

Suggested replacement:
> - The historical "first inference triggers ~43 min Triton JIT" (#52663
>   comment, vllm 0.27.1+rocm723, Aug 2026) was **not reproduced** on pinned
>   current main in the default (graph) configuration within the tested battery
>   envelope (prompts ≤128 tokens, decode batches ≤4, max_tokens 16) on this
>   gfx1100 host: the decode-path kernels compile during cudagraph capture at
>   startup, and the 104-key warmup covers the rest.

### B-14 (apply) — evidence/runs/P2-cold-normal-error/result.md:16-18

Current text:
> 0 JIT events recorded (monitor armed, error mode). `--jit-monitor-mode error` — the strongest zero-runtime-JIT acceptance test
> from #49349 — **passes** for Qwen3.8-27B-FP8 on gfx1100 at pinned SHA
> 31e2443 in the default graph configuration with cold caches.

Problem: acceptance claim states config and caches but omits the battery
envelope; quoted alone it reads as a general zero-runtime-JIT pass for the
model/stack.

Suggested replacement:
> 0 JIT events recorded (monitor armed, error mode). `--jit-monitor-mode error` — the strongest zero-runtime-JIT acceptance test
> from #49349 — **passes** for Qwen3.8-27B-FP8 on gfx1100 at pinned SHA
> 31e2443 in the default graph configuration with cold caches, for the tested
> battery (prompts ≤128 tokens, decode batches ≤4, max_tokens 16, greedy).

### B-15 (apply) — docs/10-falsification.md:5 (H1 row, "Evidence for" cell)

Current text:
> in NORMAL mode capture+warmup compile all decode keys at startup (P5 428 entries, 0 runtime compiles (coverage))

Problem: "all decode keys" without the battery qualifier — proven only for keys
the tested battery exercises.

Suggested replacement:
> in NORMAL mode capture+warmup compile at startup all decode keys the tested battery exercises (P5 428 entries, 0 runtime compiles (coverage))

### B-16 (apply) — docs/09-root-cause.md:109-111

Current text:
> - FP8 causal to runtime JIT: NO (FP8 w8a8 compiles are startup-timeline in
>   normal mode; in eager the extra w8a8 runtime variants ride along with the
>   same startup-coverage mechanism, they are not GDN-specific).

Problem: categorical "NO" without envelope qualifier, and it omits the
dispositive control evidence (non-FP8 G1/N1 also runtime-JIT in eager). Read as
a headline it can be misquoted as "FP8 kernels never compile at runtime", which
is false in eager (P1: +4 w8a8 runtime variants — though the parenthetical does
disclose this).

Suggested replacement:
> - FP8 causal to the runtime-JIT mechanism: NO, within the tested envelope —
>   non-FP8 models runtime-JIT in eager too (G1, N1), and in normal mode FP8
>   w8a8 compiles are startup-timeline (0 runtime FP8 compiles). In eager, 4
>   extra w8a8 M-specialization variants do compile at runtime, riding the same
>   missing-startup-coverage mechanism (not GDN-specific).

### B-17 (note) — docs/05-runtime-jit-inventory.md:3-4

Current text:
> Sources: P1 (27B eager, mtime-attributed triton cache + latency), G1 (0.8B
> eager), P5/P3 (27B normal, startup-side), G2 (0.8B normal).

Problem: the runtime-JIT inventory sources omit N1 (dense non-GDN eager control,
80 post-battery artifacts); §1 therefore documents runtime compiles only for GDN
models and could be read as GDN-specific. docs/09 handles this correctly; docs/05
should cross-reference.

Suggested replacement:
> Sources: P1 (27B eager, mtime-attributed triton cache + latency), G1 (0.8B
> eager), P5/P3 (27B normal, startup-side), G2 (0.8B normal); N1 (dense
> non-GDN eager control: 80 post-battery artifacts, generic infra kernels —
> eager runtime JIT is configuration-generic, see docs/09).

### B-18 (note) — README.md:19-21, STATUS.md:27-31, docs/09-root-cause.md:15-17 (eager verdict clause)

Current text (README:19-21):
> Enforce-eager: CONFIRMED — runtime JIT still occurs under --enforce-eager
> because graph capture and JIT warmup are intentionally disabled by
> current upstream design.

Problem: accurate as written, but incomplete — the JIT monitor is ALSO
intentionally inactive in eager (the exact misreading this RCA had to unravel;
cf. phase2_candidates item 1). Adding the monitor clause to the verdict block
prevents "eager + 0 monitor events" from being misread as zero JIT.

Suggested replacement (README:19-21; mirror in STATUS/docs/09):
> Enforce-eager: CONFIRMED — runtime JIT still occurs under --enforce-eager
> because graph capture and JIT warmup are intentionally disabled (and the
> JIT monitor is intentionally inactive) by current upstream design.

### B-19 (note) — docs/09-root-cause.md:27-28 and 37-39

Current text (line 27-28):
> startup JIT warmup and graph capture cover all compile keys required by the
> tested Qwen3.8/GDN inference envelope.

Problem: qualified by "tested ... envelope", but the same battery under eager
required one compile key (unbatched packed_decode specialization) that P5 startup
does not contain. Tighten to "required ... in normal-mode execution" so the
"all compile keys" phrase cannot be read across modes.

Suggested replacement (line 27-28; mirror at 37-39):
> startup JIT warmup and graph capture cover all compile keys required by the
> tested Qwen3.8/GDN inference envelope in normal-mode execution (one eager-only
> specialization is not covered, by design — docs/07).

### B-20 (note, no change required) — STATUS.md:8 and README.md:1

STATUS.md:8 "P5/P3 zero runtime JIT" is unqualified in the table row (the verdict
block below restates the full envelope — acceptable, optionally append "(tested
envelope)"). README.md:1 title "Zero Runtime JIT" is the investigation codename
and is qualified by the verdict block on the same page — acceptable as-is.

## Verdict

**FAIL**

Blocking findings (materially overstate evidence; would mislead an upstream
maintainer if quoted):

- B-1 (docs/05:44,49,52) — "P1-runtime ⊆ P5-startup" coverage proof and "all 20
  ... kernels covered" retain the entry-level superset claim that docs/07
  explicitly retracted; "51 extra" figure inconsistent with stated totals.
- B-2 (docs/10:6) — same retracted "⊆ ... superset proof" in the H2 row.
- B-3 (docs/06:45-48) — layer_norm startup variants attributed to the JIT warmup
  module "not from a profile-run forward", contradicting P1/G1 result.md for the
  eager runs where warmup cannot have run.
- B-4 (docs/07:58-60) — same eager-mode attribution error in the section whose
  evidence is "even in eager mode".
- B-5 (docs/04:43, docs/09:93-94) — "≤ 8.5 min ≪ 600 s" mischaracterizes a 510 s
  vs 600 s (~15% headroom) margin as "much less than".

Apply-before-close (non-blocking but recommended): B-6 through B-16 (determinism
wording from n≤2; "fully compiled" family-level phrasing without the entry-level
caveat in P5/G2 result.md; unqualified "fixed" in G2; missing envelope qualifiers
in P5/P2 result.md; "all decode keys" in H1; "FP8 causal ... NO" headline).
Notes: B-17 through B-20.

Everything else audited (README verdict block, STATUS verdict block, docs/00, 01,
02, 03, 09 root-cause framing, 10 H3/H4/H6-H10, 11, phase2_candidates, P1/P4/G1/
G3/N1 result.md) correctly carries the envelope/mechanism-vs-impact qualifiers and
is consistent with the raw cache and latency evidence.
