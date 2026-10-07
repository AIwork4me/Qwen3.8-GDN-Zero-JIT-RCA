#!/bin/bash
# Retry wrapper for git network operations through the flaky egress proxy.
# Usage: git_retry.sh <max_attempts> <delay_seconds> -- <git args...>
set -u
MAX="${1:?max attempts}"; DELAY="${2:?delay s}"; shift 2; shift # drop --
n=0
until git "$@"; do
  n=$((n+1))
  echo "[git_retry] attempt $n/$MAX failed (exit $?)" >&2
  if [ "$n" -ge "$MAX" ]; then echo "[git_retry] giving up" >&2; exit 1; fi
  sleep "$DELAY"
done
