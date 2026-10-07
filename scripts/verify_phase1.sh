#!/bin/bash
# Verify a run directory is complete per the Phase-1 evidence contract.
# Usage: verify_phase1.sh <run_dir>
set -u
D="${1:?run dir}"
miss=0
for f in metadata.json command.sh run-env.sh env.txt git.txt cache-before.txt server.log client.log jit-events.jsonl timeline.csv cache-after.txt result.md; do
  if [ ! -f "$D/$f" ]; then echo "MISSING: $f"; miss=$((miss+1)); fi
done
# empty jit-events.jsonl is valid evidence (zero events) and must be justified in result.md
if [ ! -s "$D/jit-events.jsonl" ] && ! grep -qiE '0 JIT events|runtime JIT events.*: 0' "$D/result.md"; then
  echo "EMPTY jit-events.jsonl without 0-event justification in result.md"; miss=$((miss+1))
fi
# cache-before must prove emptiness for cold runs
if grep -q '"mode": "cold"' "$D/metadata.json" 2>/dev/null; then
  grep -qE 'NONEMPTY' "$D/cache-before.txt" && { echo "COLD-RUN CACHE NOT EMPTY"; miss=$((miss+1)); }
fi
# preflight line in env.txt must have passed
grep -q 'PREFLIGHT PASS' "$D/env.txt" || { echo "PREFLIGHT NOT PASS IN env.txt"; miss=$((miss+1)); }
# server must have reached ready (unless the experiment intentionally failed early -> result.md must say so)
if ! grep -qE 'Application startup complete|Uvicorn running on|engine ready' "$D/server.log" 2>/dev/null; then
  echo "SERVER-READY MARKER ABSENT (justify in result.md)"; 
fi
echo "verify_phase1: $([ $miss -eq 0 ] && echo PASS || echo FAIL) ($D)"
exit $miss
