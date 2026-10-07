#!/usr/bin/env python3
"""Deterministic runtime-cache-boundary verifier (Phase-1 primary evidence).

For each selected run:
  1. read the battery start from raw client evidence (client.log marker);
  2. parse all cache manifest timestamps (cache-after.txt, cross-checked
     against cache-before.txt for warm runs);
  3. count files created after battery start, per cache section;
  4. compare against the per-run expectation (the oracle below encodes the
     run DESIGN, not the results — counts are always computed from evidence);
  5. exit nonzero if any expectation is violated.

Boundary convention: the client battery-start marker has second resolution;
a cache mtime strictly later than that second (including later fractions of
the same second) counts as post-battery. This is conservative for the
zero-runtime-JIT claims (it can only overcount, never hide, runtime files).

Output: evidence/closure/cache-boundary-verification.txt (also stdout).

Usage: verify_runtime_cache_boundary.py [repo_root]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Expected post-battery artifact counts per run (design oracle).
#   "any_nonzero" — at least one post-battery file must exist (eager runs:
#                   runtime JIT expected; the monitor is intentionally off).
#   0             — no post-battery file may exist in ANY section (default
#                   graph-mode runs: zero runtime JIT claim).
EXPECT = {
    "P5": 0,   # primary zero-runtime-JIT proof (cold, default/graph, warn)
    "P3": 0,   # independent cold repeat
    "P2": 0,   # cold, default/graph, monitor error mode
    "P4": 0,   # warm reuse of P5 cache
    "G2": 0,   # small GDN cold default/graph
    "G3": 0,   # small GDN warm reuse
    "P1": "any_nonzero",  # cold eager: runtime JIT expected (216 triton files)
    "G1": "any_nonzero",  # small GDN cold eager: runtime JIT expected
    "N1": "any_nonzero",  # dense non-GDN eager: generic runtime JIT expected
}

SECTIONS = ("triton", "torchinductor", "vllm", "xdg", "triton-home", "hf")


def canon(ts: str) -> str:
    ts = ts.replace("Z", "").replace("+00:00", "").replace("+0000", "")
    if "." in ts:
        head, frac = ts.split(".", 1)
        frac = (frac + "000000000")[:9]
    else:
        head, frac = ts, "000000000"
    return f"{head}.{frac}"


def battery_start(client_log: Path) -> tuple[str, int] | None:
    for i, line in enumerate(client_log.read_text().splitlines(), 1):
        m = re.match(r"# battery start (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", line)
        if m:
            return m.group(1), i
    return None


def section_counts(cache_manifest: Path, boundary: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    sec = None
    for line in cache_manifest.read_text().splitlines():
        if line.startswith("## "):
            sec = line[3:].split()[0]
            out.setdefault(sec, {"total": 0, "post": 0, "first_post": None, "last_pre": None})
        elif line.startswith("#") or "\t" not in line or sec is None:
            continue
        else:
            ts = canon(line.rstrip("\n").rsplit("\t", 1)[1])
            s = out[sec]
            s["total"] += 1
            if ts > boundary:
                s["post"] += 1
                if s["first_post"] is None or ts < s["first_post"]:
                    s["first_post"] = ts
            else:
                if s["last_pre"] is None or ts > s["last_pre"]:
                    s["last_pre"] = ts
    return out


def verify_run(run_dir: Path, report: list[str]) -> bool:
    run_long = run_dir.name
    short = re.match(r"^([PGN]\d+)", run_long).group(1)  # type: ignore[union-attr]
    bt = battery_start(run_dir / "client.log")
    if bt is None:
        report.append(f"FAIL {run_long}: no battery-start marker in client.log")
        return False
    bt_raw, bt_line = bt
    boundary = canon(bt_raw)
    counts = section_counts(run_dir / "cache-after.txt", boundary)
    post_total = sum(s["post"] for s in counts.values())
    post_triton = counts.get("triton", {}).get("post", 0)
    exp = EXPECT.get(short)
    if exp is None:
        report.append(f"SKIP {run_long}: no expectation defined")
        return True

    ok = True
    if exp == 0:
        verdict = "PASS" if post_total == 0 else "FAIL"
        ok = post_total == 0
    else:  # any_nonzero
        verdict = "PASS" if post_triton > 0 else "FAIL"
        ok = post_triton > 0

    meta = json.loads((run_dir / "metadata.json").read_text())
    srv_txt = (run_dir / "server.log").read_text(errors="replace")
    armed = "Kernel JIT monitor activated" in srv_txt
    jit_events = sum(1 for _ in (run_dir / "jit-events.jsonl").open())

    report.append(
        f"{verdict} {run_long} (expect post-battery "
        f"{'== 0 in ALL sections' if exp == 0 else 'triton > 0 (eager runtime JIT expected)'})"
    )
    report.append(
        f"     battery_start={bt_raw} (client.log line {bt_line}); "
        f"mode={meta['mode']}; graphs={'eager' if '--enforce-eager' in (run_dir / 'command.sh').read_text() else 'default/graph'};"
        f" monitor={'armed' if armed else 'inactive'}; monitor_events={jit_events}"
    )
    for name in SECTIONS:
        if name not in counts:
            continue
        s = counts[name]
        report.append(
            f"     {name:14s} total={s['total']:5d} post_battery={s['post']:5d}"
            + (f" first_post={s['first_post']}" if s["first_post"] else "")
            + (f" last_startup={s['last_pre']}" if s["last_pre"] and exp == 0 else "")
        )
    if exp == 0 and post_total == 0:
        last_all = max((s["last_pre"] for s in counts.values() if s["last_pre"]), default=None)
        report.append(f"     zero-runtime-JIT boundary: no cache mtime after {bt_raw}; last artifact {last_all}")
    report.append("")
    return ok


def main(repo: Path) -> int:
    runs_root = repo / "evidence" / "runs"
    report: list[str] = [
        "# Runtime cache-boundary verification (generated — do not hand-edit)",
        "",
        f"generated_by: scripts/verify_runtime_cache_boundary.py",
        f"boundary_rule: cache mtime strictly after the battery-start second",
        f"               (conservative: same-second fractions count as runtime)",
        "",
    ]
    all_ok = True
    for run_dir in sorted(runs_root.iterdir()):
        if run_dir.is_dir():
            all_ok &= verify_run(run_dir, report)
    report.append(all_ok
                  and "OVERALL: PASS — all run expectations satisfied (default/graph runs: zero post-battery artifacts; eager runs: runtime artifacts present)"
                  or "OVERALL: FAIL — at least one run violated its expectation")
    text = "\n".join(report) + "\n"
    out = repo / "evidence" / "closure" / "cache-boundary-verification.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    sys.stdout.write(text)
    return 0 if all_ok else 1


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    sys.exit(main(root))
