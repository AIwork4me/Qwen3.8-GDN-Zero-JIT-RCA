#!/bin/bash
# Phase-1 closure gate (repo-level). Verifies docs, manifests, raw evidence,
# cold-cache proofs, P2 acceptance, monitor-event evidence, cache-boundary
# conditions, experiment-matrix consistency, and closure audit presence.
#
# Usage: verify_phase1.sh [repo_root]   (default: repo containing this script)
#
# Environment gate note: scripts/verify_environment_gate.py validates the LIVE
# interpreter/GPU and is hardware-dependent; this gate instead checks the
# COMMITTED environment evidence (gate-check.txt) plus manifest consistency.
# Run the live gate separately on the experiment host:
#   /workspace/venv-qwen-gdn-rca/bin/python scripts/verify_environment_gate.py
set -u
ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
fail=0
step() { printf '\n== %s ==\n' "$1"; }
ok()   { echo "PASS $1"; }
bad()  { echo "FAIL $1"; fail=$((fail+1)); }

step "docs present"
for d in 00-scope 01-upstream-landscape 02-environment 03-model-architecture \
         04-reproduction 05-runtime-jit-inventory 06-call-graph \
         07-compile-key-analysis 08-experiment-matrix 09-root-cause \
         10-falsification 11-external-review-findings phase2_candidates; do
  [ -s "docs/$d.md" ] && ok "docs/$d.md" || bad "docs/$d.md missing/empty"
done

step "manifests present"
for m in experiment_matrix.csv environment.json upstream.json model-qwen38.json; do
  [ -s "manifests/$m" ] && ok "manifests/$m" || bad "manifests/$m missing/empty"
done

step "primary run dirs + raw command files"
for r in P1-cold-eager-warn P2-cold-normal-error P3-cold-normal-warn \
         P4-warm-reuse P5-cold-normal-warn G1-cold-warn G2-cold-normal-warn \
         G3-warm-reuse N1-cold-warn; do
  d="evidence/runs/$r"
  [ -d "$d" ] || { bad "run dir $d missing"; continue; }
  ok "run dir $r"
  # note: jit-events.jsonl may legitimately be EMPTY (armed monitor + zero
  # events, or monitor inactive by design in eager) — existence is required,
  # semantic emptiness is validated per-run below and in result.md
  for f in metadata.json command.sh run-env.sh env.txt git.txt cache-before.txt \
           cache-after.txt server.log client.log requests.jsonl jit-events.jsonl \
           timeline.csv result.md; do
    [ -e "$d/$f" ] || bad "$r/$f missing"
  done
  grep -q 'PREFLIGHT PASS' "$d/env.txt" || bad "$r/env.txt PREFLIGHT PASS absent"
done

step "cold-cache proofs (cache-before empty)"
for r in P1-cold-eager-warn P2-cold-normal-error P3-cold-normal-warn \
         P5-cold-normal-warn G1-cold-warn G2-cold-normal-warn N1-cold-warn; do
  if grep -qE 'NONEMPTY' "evidence/runs/$r/cache-before.txt"; then
    bad "$r cache-before NONEMPTY"
  else
    ok "$r cache-before empty"
  fi
done

step "P2 error-mode acceptance (33/33 rows OK, 0 raises)"
P2ROWS=$(grep -cE 'ok=[0-9]+/[0-9]+' evidence/runs/P2-cold-normal-error/client.log)
P2OK=$(grep -E 'ok=' evidence/runs/P2-cold-normal-error/client.log | awk -F'ok=' '{split($2,a,"/"); if (a[1]==a[2]) c++} END{print c+0}')
[ "$P2ROWS" -eq 33 ] && ok "P2 battery has 33 request rows" || bad "P2 rows=$P2ROWS (want 33)"
[ "$P2OK" -eq 33 ] && ok "P2 all 33 rows fully OK (0 failed requests)" || bad "P2 fully-OK rows=$P2OK/33"
grep -q 'Kernel JIT monitor activated' evidence/runs/P2-cold-normal-error/server.log \
  && ok "P2 monitor armed (server.log)" || bad "P2 monitor-armed line absent"
grep -q 'mode=error' evidence/runs/P2-cold-normal-error/server.log \
  && ok "P2 monitor mode=error" || bad "P2 mode=error absent"
! [ -s evidence/runs/P2-cold-normal-error/jit-events.jsonl ] \
  && ok "P2 jit-events empty (0 raises)" || bad "P2 jit-events nonempty (monitor raised)"

step "armed-monitor zero-event evidence (P5, P3)"
for r in P5-cold-normal-warn P3-cold-normal-warn; do
  ! [ -s "evidence/runs/$r/jit-events.jsonl" ] && ok "$r jit-events empty" || bad "$r jit-events nonempty"
  grep -q 'Kernel JIT monitor activated' "evidence/runs/$r/server.log" && ok "$r monitor armed" || bad "$r monitor-armed line absent"
done

step "P5 monitor armed AFTER all cache artifacts (primary boundary)"
python3 - <<'PY' || fail=$((fail+1))
import re, sys
from pathlib import Path
root = Path('.')
log = (root/'evidence/runs/P5-cold-normal-warn/server.log').read_text(errors='replace')
m = re.search(r'INFO 10-\d+ (\d{2}:\d{2}:\d{2}) \[jit_monitor\.py:\d+\] Kernel JIT monitor activated', log)
assert m, "monitor activation line not found"
armed = '2026-10-07T' + m.group(1)
last = None
for line in (root/'evidence/runs/P5-cold-normal-warn/cache-after.txt').read_text().splitlines():
    if line.startswith('#') or '\t' not in line: continue
    ts = line.rsplit('\t',1)[1]
    if last is None or ts > last: last = ts
b0 = re.search(r'# battery start (\S+)', (root/'evidence/runs/P5-cold-normal-warn/client.log').read_text()).group(1)
assert last < armed < b0, f"boundary violated: last={last} armed={armed} battery={b0}"
print(f"PASS P5 boundary: last-artifact {last} < armed {armed} < battery {b0}")
PY

step "P1 eager runtime artifacts present (mechanism reproducible)"
python3 - <<'PY' || fail=$((fail+1))
import re
from pathlib import Path
b0 = re.search(r'# battery start (\S+)', Path('evidence/runs/P1-cold-eager-warn/client.log').read_text()).group(1)
n = sum(1 for line in Path('evidence/runs/P1-cold-eager-warn/cache-after.txt').read_text().splitlines()
        if '\t' in line and not line.startswith('#') and line.rsplit('\t',1)[1] > b0)
n_triton = None
sec = None; t = 0
for line in Path('evidence/runs/P1-cold-eager-warn/cache-after.txt').read_text().splitlines():
    if line.startswith('## '): sec = line[3:].split()[0]
    elif '\t' in line and not line.startswith('#') and sec == 'triton' and line.rsplit('\t',1)[1] > b0: t += 1
n_triton = t
assert n > 200, f"expected >200 post-battery cache files in P1, got {n}"
assert n_triton >= 216, f"expected >=216 post-battery triton files in P1, got {n_triton}"
print(f"PASS P1 post-battery cache files: {n} total / {n_triton} triton (eager runtime JIT reproducible)")
PY

step "cache-boundary verifier (all runs)"
python3 scripts/verify_runtime_cache_boundary.py >/dev/null 2>&1 \
  && ok "verify_runtime_cache_boundary.py PASS" || bad "verify_runtime_cache_boundary.py FAIL"
[ -s evidence/closure/cache-boundary-verification.txt ] \
  && ok "cache-boundary-verification.txt exists" || bad "cache-boundary-verification.txt missing"

step "experiment matrix matches committed evidence (regenerability)"
if python3 - "$ROOT" <<'PY'
import subprocess, sys, tempfile, shutil, os
from pathlib import Path
root = Path(sys.argv[1])
tmp = Path(tempfile.mkdtemp())
try:
    bak_csv = tmp / "experiment_matrix.csv"
    bak_doc = tmp / "08-experiment-matrix.md"
    shutil.copy(root / "manifests/experiment_matrix.csv", bak_csv)
    shutil.copy(root / "docs/08-experiment-matrix.md", bak_doc)
    rc = subprocess.run([sys.executable, "scripts/rebuild_experiment_matrix.py"],
                        cwd=root, capture_output=True).returncode
    same = ((root / "manifests/experiment_matrix.csv").read_bytes() == bak_csv.read_bytes()
            and (root / "docs/08-experiment-matrix.md").read_bytes() == bak_doc.read_bytes())
    # restore originals regardless (byte-identical when consistent)
    shutil.copy(bak_csv, root / "manifests/experiment_matrix.csv")
    shutil.copy(bak_doc, root / "docs/08-experiment-matrix.md")
    sys.exit(0 if (rc == 0 and same) else 1)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
PY
then ok "matrix regeneration deterministic and consistent (0 mismatches)"
else bad "matrix out of sync with evidence — rerun scripts/rebuild_experiment_matrix.py and commit"
fi

step "derived timelines present with provenance"
for r in P1-cold-eager-warn P2-cold-normal-error P3-cold-normal-warn \
         P5-cold-normal-warn G1-cold-warn G2-cold-normal-warn G3-warm-reuse; do
  f="evidence/runs/$r/timeline.csv"
  [ -s "$f" ] && grep -q 'derivation' "$f" && [ "$(wc -l < "$f")" -ge 5 ] \
    && ok "$r timeline" || bad "$r timeline missing/too thin"
done

step "committed environment evidence"
grep -q 'ENVIRONMENT GATE: PASS' evidence/environment/gate-check.txt \
  && ok "gate-check.txt records PASS" || bad "gate-check.txt does not record PASS"
python3 - <<'PY' || fail=$((fail+1))
import json
env = json.load(open('manifests/environment.json'))
assert env['torch'] == '2.14.1+rocm7.14', env['torch']
assert env['gfx'] == 'gfx1100', env['gfx']
assert '7.14' in env['rocm_compiler_version'], env['rocm_compiler_version']
assert env['source_build_rocm_toolchain_classification'] == 'COHERENT ROCm 7.14'
print('PASS environment manifest: coherent ROCm 7.14 / gfx1100 / torch 2.14.1+rocm7.14')
PY
python3 - <<'PY' || fail=$((fail+1))
import json, hashlib
h = hashlib.sha256(open('manifests/model-qwen38.json','rb').read()).hexdigest()
up = json.load(open('manifests/upstream.json'))
assert up['sha'] == '31e2443c90542a33a4a4a293ea7186fba2796c67', up['sha']
print('PASS upstream pin manifest: ' + up['sha'])
PY

step "closure audits present"
for a in 01-evidence-integrity-audit 02-technical-wording-audit \
         03-experiment-matrix-audit 04-final-phase1-adversarial-audit; do
  [ -s "evidence/subagent-reviews/closure/$a.md" ] && ok "$a.md" || bad "$a.md missing"
done

step "verdict wording hygiene"
if grep -rniE 'historical root cause disproved' README.md STATUS.md docs/ 2>/dev/null | grep -qv '^docs/11'; then
  bad "oversimplified 'historical root cause disproved' wording still present"
else
  ok "no unqualified 'historical root cause disproved' wording"
fi

printf '\n==========================================\n'
if [ "$fail" -eq 0 ]; then
  echo "PHASE-1 CLOSURE GATE: PASS"
else
  echo "PHASE-1 CLOSURE GATE: FAIL ($fail failed checks)"
fi
exit "$fail"
