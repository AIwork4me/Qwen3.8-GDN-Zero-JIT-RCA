#!/usr/bin/env python3
"""Preflight assertion: cache env vars must be set and isolated.

Run AFTER sourcing evidence/runs/<RUN_ID>/run-env.sh and BEFORE any
vLLM/Triton process launch. Exits non-zero on contamination.
"""
import os
import sys

REQUIRED = [
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "VLLM_CACHE_ROOT",
    "XDG_CACHE_HOME",
    "RCA_RUN_ID",
    "RCA_CACHE_ROOT",
]

FORBIDDEN_PREFIXES = (
    "/root/.cache",
    "/home/",
    "/opt/venv",
    "/tmp/",  # /tmp is allowed only under the dedicated /tmp/opencode; caches must not live there
)

fail = 0
for key in REQUIRED:
    val = os.environ.get(key)
    print(f"{key}={val}")
    if not val:
        print(f"FAIL: {key} is not set (did you source run-env.sh in THIS shell?)")
        fail = 1

for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT", "XDG_CACHE_HOME"):
    val = os.environ.get(key, "")
    if not val:
        continue
    if val.startswith(FORBIDDEN_PREFIXES) or val == os.path.expanduser("~/.cache"):
        print(f"FAIL: {key}={val} points at a default/shared location")
        fail = 1
    if not val.startswith("/workspace/rca-caches/"):
        print(f"FAIL: {key}={val} outside /workspace/rca-caches/ isolation root")
        fail = 1
    if not os.path.isdir(val):
        print(f"FAIL: {key}={val} does not exist as a directory")
        fail = 1

print("PREFLIGHT", "FAIL" if fail else "PASS")
sys.exit(fail)
