#!/bin/bash
# Stop the server for a run and record the cache manifest.
# Usage: stop_server.sh <RUN_ID>
set -euo pipefail
RUN_ID="${1:?run id}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNDIR="$REPO_ROOT/evidence/runs/$RUN_ID"
[ -f "$RUNDIR/server.pid" ] || { echo "no server.pid"; exit 1; }
PID=$(cat "$RUNDIR/server.pid")
if kill -0 "$PID" 2>/dev/null; then
  # graceful: SIGINT the process group of vllm serve, then wait, then escalate
  PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
  kill -INT -- -"$PGID" 2>/dev/null || kill -INT "$PID" 2>/dev/null || true
  for i in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
  kill -0 "$PID" 2>/dev/null && kill -TERM -- -"$PGID" 2>/dev/null || true
  sleep 3
  kill -0 "$PID" 2>/dev/null && kill -KILL -- -"$PGID" 2>/dev/null || true
fi
echo "server stopped (pid $PID)"
# cache-after manifest
source "$RUNDIR/run-env.sh"
{
  echo "# cache-after $(date -Is) run=$RUN_ID"
  for d in triton torchinductor vllm xdg triton-home hf; do
    p="$RCA_CACHE_ROOT/$d"
    n=$(find "$p" -type f 2>/dev/null | wc -l)
    b=$(du -sb "$p" 2>/dev/null | awk '{print $1}')
    echo "## $d  files=$n bytes=$b"
    find "$p" -type f -printf '%P\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort | head -4000
  done
} > "$RUNDIR/cache-after.txt"
echo "cache-after written"
