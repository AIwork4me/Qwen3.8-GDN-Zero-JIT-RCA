# 11 — External Review Findings (Phase-1 Gate Review)

Date: 2026-10-07
Reviewer input: external review of the repository at checkpoint `3d331dc69aa5fa70a03f80057d65bc2faeebbc75`.

## Gate status as of review

```text
Phase-1 Gate   = FAIL
Root Cause     = UNRESOLVED
Phase 2        = NOT AUTHORIZED
```

## Blocker A — committed environment evidence is inconsistent

`manifests/environment.json` declares the experiment stack
(torch 2.14.1+rocm7.14, HIP 7.14.x, Triton 3.8.x, ROCm SDK 7.14.0),
but the committed raw evidence `evidence/environment/python.txt` was
collected from a *different, pre-existing host environment*
(`/opt/venv`, torch 2.9.1+git, HIP 7.2, Triton 3.5.1, vLLM 0.16.1.dev).

=> The manifest is currently **not supported by raw evidence**.

Remediation (this session):
- preserve old evidence under a clearly-labeled name
  (`evidence/environment/host-preexisting-environment/`);
- collect fresh raw evidence from `/workspace/venv-qwen-gdn-rca`
  (`python-rocm714.txt`, `packages-rocm714.txt`, `toolchain-rocm714.txt`,
  `vllm-build-rocm714.txt`);
- rebuild `manifests/environment.json` from the new raw evidence only.

## Blocker B — mixed ROCm 7.2 host toolchain vs ROCm 7.14 Python stack

Host evidence (`evidence/environment/env-sanitized.txt`) shows:

```text
/opt/rocm            -> ROCm 7.2.1
hipcc --version      -> 7.2.53211
ROCM_PATH=/opt/rocm
LD_LIBRARY_PATH includes /opt/rocm/lib
```

while the experiment target is the PyTorch ROCm 7.14 wheel stack
(pip `rocm-sdk-core/libraries/device-gfx1100` 7.14.0 in the venv).

Risk: vLLM native source build silently compiled with ROCm 7.2 headers /
`hipcc` from `/opt/rocm` against a ROCm 7.14 PyTorch runtime.

Remediation (this session):
- discover the *actual* ROCm 7.14 toolchain (hipcc, clang, headers,
  libs, CMake configs) shipped inside the venv pip packages;
- classify the source-build toolchain as
  `COHERENT ROCm 7.14 | MIXED / INVALID | UNRESOLVED`
  in `docs/02-environment.md`;
- never let `/opt/rocm` 7.2 compilers or headers enter the vLLM build
  without explicit evidence-backed justification.

## Blocker C — cache isolation script does not persist environment variables

`scripts/reset_caches.sh` exports cache variables inside its own child
shell. Called as `./scripts/reset_caches.sh cold RUN_ID`, the exports do
NOT propagate to the shell that later launches vLLM => fake "cold" runs
are possible (server silently using `/root/.cache/triton`).

Remediation (this session):
- add `scripts/prepare_run_env.sh` which writes
  `evidence/runs/<RUN_ID>/run-env.sh` with all cache env exports;
- every launcher must `source` that file and record
  `env | grep -E 'TRITON_CACHE|TORCHINDUCTOR_CACHE|VLLM_CACHE_ROOT|XDG_CACHE_HOME'`;
- preflight assertions terminate the run if any expected cache var is
  absent or points at a default location;
- `cache-before.txt` / `cache-after.txt` manifests recorded per run.

## Session-start state (raw facts)

- RCA repo at `main` @ `3d331dc69aa5fa70a03f80057d65bc2faeebbc75`; two
  untracked upstream evidence files present (`issue-52663.json`,
  `issue-49349.json`) — preserved and committed as evidence.
- GPU confirmed gfx1100 (Radeon PRO W7900 class, card model 0x744b,
  48 GB), driver 6.16.13.
- vLLM source worktree `/workspace/vllm-zero-jit-rca` at pinned SHA
  `31e2443c90542a33a4a4a293ea7186fba2796c67` (partial clone; worktree
  extracted from tarball `downloads/vllm-main.tar.gz`, sha256 recorded
  in `evidence/upstream/vllm-main-31e2443-tarball.sha256`).
  Git index was found broken (all entries staged-deleted); repaired
  offline via `git read-tree` (no worktree content touched).
- venv `/workspace/venv-qwen-gdn-rca`: torch 2.14.1(+rocm7.14),
  torchvision 0.29.1+rocm7.14, triton-rocm 3.8.0, rocm-sdk* 7.14.0
  (fresh verification captured this session in
  `evidence/environment/python-rocm714.txt`).
- Egress proxy intermittently returns 503 for git smart-HTTP; retries
  or API-based fallbacks required for pushes (helper:
  `scripts/git_retry.sh`).
