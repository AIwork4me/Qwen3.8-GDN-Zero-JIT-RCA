#!/bin/bash
# Cache isolation helpers. Usage:
#   reset_caches.sh cold  <RUN_ID>   # fresh empty cache dirs
#   reset_caches.sh warm  <RUN_ID>   # verify+reuse existing dirs
#   reset_caches.sh manifest <RUN_ID> <out-file>  # list cache contents
set -euo pipefail
MODE="${1:?cold|warm|manifest}"
RUN_ID="${2:?run id}"
ROOT="/workspace/rca-caches/${RUN_ID}"
export TRITON_CACHE_DIR="$ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/torchinductor"
export VLLM_CACHE_ROOT="$ROOT/vllm"
export XDG_CACHE_HOME="$ROOT/xdg"
case "$MODE" in
  cold)
    rm -rf "$ROOT"
    mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"
    echo "COLD cache prepared at $ROOT"
    ;;
  warm)
    [ -d "$TRITON_CACHE_DIR" ] || { echo "missing $TRITON_CACHE_DIR"; exit 1; }
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"
    echo "WARM cache reused at $ROOT"
    ;;
  manifest)
    OUT="${3:?output file}"
    : > "$OUT"
    for d in triton torchinductor vllm xdg; do
      echo "## $d" >> "$OUT"
      find "$ROOT/$d" -type f -printf '%P\t%s\t%TY-%Tm-%Td %TH:%TM:%TS\n' 2>/dev/null | sort >> "$OUT"
    done
    echo "manifest written to $OUT"
    ;;
  *) echo "bad mode"; exit 1;;
esac
