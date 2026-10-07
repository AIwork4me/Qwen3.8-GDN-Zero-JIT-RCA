#!/usr/bin/env python3
"""Extract JIT events + timeline from a run's server.log.

Usage: extract_jit_events.py <run_dir>

Produces:
  jit-events.jsonl  one record per runtime JIT event
  timeline.csv      startup vs inference phase markers
"""
import json
import os
import re
import sys

# example lines:
# WARNING 10-07 01:23:45 [jit_monitor.py:141] Triton kernel JIT compilation during inference: NAME. This causes ... (+ verbose block)
# INFO ... Engine hit rendezvous / "Application startup complete" / "Uvicorn running on"
JIT_RE = re.compile(
    r"^(?P<ts>\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+\S+\s+\[jit_monitor(?:\.py)?:\d+\]\s+"
    r"(?P<kind>Triton kernel|CuTeDSL|torch\.compile|Inductor)\s+JIT compilation during inference:\s+(?P<kernel>\S+?)\.(?=\s|This)"
)
STARTUP_RE = re.compile(
    r"^(?P<ts>\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"(?P<what>JIT monitor|engine ready|Engine core is ready|Application startup complete|Uvicorn running on|Adding requests|Started server process|Warmup|autotun)"
)
CONSTEXPR_RE = re.compile(r"constexprs=\{(?P<c>[^}]*)\}")
SIGNATURE_RE = re.compile(r"signature=\{(?P<s>[^}]*)\}")
GENERIC_DETAIL = re.compile(r"(?P<k>constexprs|signature|extra)=\{(?P<v>[^}]*)\}")


def main(run_dir: str) -> int:
    log = os.path.join(run_dir, "server.log")
    events_path = os.path.join(run_dir, "jit-events.jsonl")
    timeline_path = os.path.join(run_dir, "timeline.csv")
    n_events = 0
    with open(log, errors="replace") as f, open(events_path, "w") as ev, open(
        timeline_path, "w"
    ) as tl:
        tl.write("timestamp,kind,detail\n")
        pending = None
        for line in f:
            m = JIT_RE.match(line)
            if m:
                rec = {
                    "ts": m.group("ts"),
                    "kind": m.group("kind"),
                    "kernel": m.group("kernel"),
                }
                pending = rec
                events_flat = json.dumps(rec)
                continue
            if pending is not None:
                d = GENERIC_DETAIL.search(line)
                if d:
                    pending[d.group("k")] = d.group("v")
                    continue
                # flush on first non-detail line
                ev.write(json.dumps(pending) + "\n")
                tl.write(f'{pending["ts"]},jit,{pending["kernel"]}\n')
                n_events += 1
                pending = None
            s = STARTUP_RE.match(line)
            if s:
                what = s.group("what")
                if what == "JIT monitor":
                    detail = "activated"
                else:
                    detail = line.strip()[:160]
                tl.write(f'{s.group("ts")},phase,{detail}\n')
        if pending is not None:
            ev.write(json.dumps(pending) + "\n")
            n_events += 1
    print(f"extracted {n_events} JIT events -> {events_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
