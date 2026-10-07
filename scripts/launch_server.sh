#!/bin/bash
# Launch an isolated vLLM server for one RCA experiment run.
#
# Usage:
#   launch_server.sh <RUN_ID> <MODE=cold|warm> <MONITOR=warn|error> [extra vllm args...]
#
# Produces evidence/runs/<RUN_ID>/{metadata.json,command.sh,run-env.sh,env.txt,git.txt,cache-before.txt,server.log}
# Server runs in background; caller polls /health then runs clients, then
# stop_server.sh <RUN_ID>.
set -euo pipefail
RUN_ID="${1:?run id}"
MODE="${2:?cold|warm}"
MONITOR="${3:?warn|error}"
shift 3
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNDIR="$REPO_ROOT/evidence/runs/$RUN_ID"
MODEL_PATH="${RCA_MODEL:-/workspace/models/Qwen3.8-27B-FP8}"
PORT="${RCA_PORT:-8137}"

bash "$REPO_ROOT/scripts/prepare_run_env.sh" "$MODE" "$RUN_ID" >/dev/null

# record git/source identity
{
  echo "rca repo:"; git -C "$REPO_ROOT" rev-parse HEAD; git -C "$REPO_ROOT" log --oneline -1
  echo "vllm source dir: /workspace/vllm-zero-jit-rca"
  git -C /workspace/vllm-zero-jit-rca rev-parse HEAD
  git -C /workspace/vllm-zero-jit-rca status --short | head -10
  echo "worktree-vs-tarball-check: extracted from downloads/vllm-main.tar.gz sha256 recorded in evidence/upstream"
} > "$RUNDIR/git.txt" 2>&1

CMD_FILE="$RUNDIR/command.sh"
cat > "$CMD_FILE" <<EOF
#!/bin/bash
# generated for run $RUN_ID
source "$RUNDIR/run-env.sh"
source /workspace/venv-qwen-gdn-rca/bin/activate
export VLLM_ENGINE_READY_TIMEOUT_S=\${VLLM_ENGINE_READY_TIMEOUT_S:-7200}
exec vllm serve "$MODEL_PATH" \\
  --port $PORT \\
  --jit-monitor-mode $MONITOR --jit-monitor-verbose \\
  $*
EOF
chmod +x "$CMD_FILE"

# env snapshot AFTER sourcing run-env (this is what the server inherits, minus shell funcs)
(
  source "$RUNDIR/run-env.sh"
  echo "# env captured $(date -Is) for $RUN_ID"
  env | sort | sed -E 's/(TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL|COOKIE)=.*/\1=<REDACTED>/I'
  echo "# cache preflight:"
  python3 "$REPO_ROOT/scripts/preflight_cache_env.py" || exit 9
) > "$RUNDIR/env.txt" 2>&1

# metadata
python3 - "$RUN_ID" "$MODE" "$MONITOR" "$MODEL_PATH" "$PORT" "$@" > "$RUNDIR/metadata.json" <<'PY'
import json, subprocess, sys, time, hashlib
run_id, mode, monitor, model_path, port = sys.argv[1:6]
extra = sys.argv[6:]
meta = {
    "run_id": run_id,
    "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "mode": mode,
    "jit_monitor_mode": monitor,
    "jit_monitor_verbose": True,
    "model_path": model_path,
    "port": port,
    "extra_vllm_args": extra,
    "rca_commit": subprocess.run(["git","-C","/workspace/qwen-gdn-rca","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
    "vllm_commit": subprocess.run(["git","-C","/workspace/vllm-zero-jit-rca","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
    "model_config_sha256": hashlib.sha256(open(model_path+"/config.json","rb").read()).hexdigest(),
}
print(json.dumps(meta, indent=1))
PY

# launch (nohup; log inside run dir)
nohup bash "$CMD_FILE" > "$RUNDIR/server.log" 2>&1 &
echo $! > "$RUNDIR/server.pid"
echo "launched $RUN_ID (pid $(cat $RUNDIR/server.pid)) port $PORT monitor=$MONITOR mode=$MODE"
echo "logs: $RUNDIR/server.log"
