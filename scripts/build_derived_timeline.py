#!/usr/bin/env python3
"""Build per-run derived timelines with provenance, from raw evidence only.

For every run directory under evidence/runs/<RUN_ID>/ this script derives a
timeline.csv with columns:

    timestamp,event,source_file,source_line_or_record,derivation,confidence

Rules (honesty contract):
  - `raw`      the timestamp is printed verbatim in the cited record.
  - `parsed`   the timestamp is parsed from a raw log record (timezone/
               year normalization only; server.log uses UTC local time,
               year taken from metadata.json).
  - `derived`  the timestamp is computed from raw records (e.g. max cache
               mtime, first request epoch t0 converted to UTC ISO).
No event is invented: markers that vLLM logs without a timestamp (e.g.
"Application startup complete." from uvicorn) are documented as limitations
in docs/09 rather than given fabricated times.

This script supersedes scripts/extract_jit_events.py for timeline purposes:
that script's regexes anchored at line start and never matched vLLM's
"(APIServer pid=...) INFO MM-DD HH:MM:SS [...]" prefix, which is why the
committed timeline.csv files were historically empty (header-only stubs).
JIT event extraction (jit-events.jsonl) is unaffected: in all committed runs
the monitor was either armed with zero events (normal mode) or intentionally
inactive (eager mode), so those files legitimately remain empty.

Usage: build_derived_timeline.py [repo_root] [RUN_ID ...]
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
import sys
from pathlib import Path

VLLM_TS = re.compile(r"INFO (\d{2})-(\d{2}) (\d{2}:\d{2}:\d{2}) \[([^\]]+)\] (.*)")

# server.log markers: (regex on the message+source, event name)
SERVER_MARKERS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Initializing a V1 LLM engine"), "engine_init_start"),
    (re.compile(r"Initial profiling/warmup run took"), "profile_run_completed"),
    (re.compile(r"JIT kernel warmup starting"), "jit_warmup_starting"),
    (re.compile(r"JIT kernel warmup finished in"), "jit_warmup_finished"),
    (re.compile(r"Warming up Qwen GDN Triton kernels"), "qwen_gdn_triton_warmup"),
    (re.compile(r"Kernel JIT monitor activated"), "jit_monitor_activated"),
    (re.compile(r"init engine \(profile, create kv cache, warmup model\) took"), "engine_init_completed"),
    (re.compile(r"Watermarking is enabled for this request"), "first_server_side_request_seen"),
    (re.compile(r"Cannot use ROCm custom paged attention kernel"), "rocm_paged_attn_fallback"),
]

ISO = "%Y-%m-%dT%H:%M:%SZ"


def canon_ts(ts: str) -> str:
    """Canonical comparable form: YYYY-MM-DDTHH:MM:SS.fffffffff (9 digits).

    Accepts 'YYYY-MM-DDTHH:MM:SSZ', '...+00:00', or '...SS.fffffffff'.
    Boundary convention: a second-resolution battery-start marker maps to
    '.000000000', so cache mtimes within the same second count as post-
    battery (conservative for zero-runtime-JIT claims).
    """
    ts = ts.replace("Z", "").replace("+00:00", "").replace("+0000", "")
    if "." in ts:
        head, frac = ts.split(".", 1)
        frac = (frac + "000000000")[:9]
    else:
        head, frac = ts, "000000000"
    return f"{head}.{frac}"


def iso_utc(*, epoch: float | None = None, ymd_hms: tuple | None = None) -> str:
    if epoch is not None:
        return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).strftime(ISO)
    y, m, d, hh, mm, ss = ymd_hms  # type: ignore[misc]
    return dt.datetime(y, m, d, hh, mm, ss, tzinfo=dt.timezone.utc).strftime(ISO)


def meta_year(meta: dict) -> int:
    return int(meta["created"][:4])


def battery_markers(client_log: Path) -> list[tuple[str, str, int]]:
    """All battery start/end markers: (iso_ts, kind, line_no)."""
    out = []
    for i, line in enumerate(client_log.read_text().splitlines(), 1):
        m = re.match(r"# battery (start|end) (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", line)
        if m:
            ts = m.group(2).replace("T", " ").replace("-", "-")  # keep raw shape
            out.append((iso_utc(ymd_hms=tuple(map(int, [ts[0:4], ts[5:7], ts[8:10], ts[11:13], ts[14:16], ts[17:19]]))),
                        f"battery_{m.group(1)}", i))
    return out


def server_events(server_log: Path, year: int) -> list[tuple[str, str, str, str]]:
    """(iso_ts, event, source_ref, raw_excerpt) parsed from timestamped vLLM lines."""
    out = []
    seen: set[str] = set()
    for i, line in enumerate(server_log.read_text(errors="replace").splitlines(), 1):
        m = VLLM_TS.search(line)
        if not m:
            continue
        mon, day, hms, src, msg = m.groups()
        for pat, event in SERVER_MARKERS:
            if pat.search(msg) or pat.search(src + " " + msg):
                key = event
                if event in seen and event not in ("first_server_side_request_seen",):
                    continue  # first occurrence only, except per-request markers
                seen.add(key)
                ts = iso_utc(ymd_hms=(year, int(mon[:2]) if False else int(mon), int(day),
                                      int(hms[0:2]), int(hms[3:5]), int(hms[6:8])))
                # month string is "10"; map month abbrev not needed: vLLM uses MM already
                out.append((ts, event, f"server.log:{i}", line.strip()[:160]))
                break
    return out


def cache_section_bounds(cache_after: Path, battery_t0: str | None):
    """Per-section (max_mtime<=t0, min_mtime>t0, count_post) from cache-after.txt."""
    sections: dict[str, dict] = {}
    sec = None
    for line in cache_after.read_text().splitlines():
        if line.startswith("## "):
            sec = line[3:].split()[0]
            sections.setdefault(sec, {"files": 0, "post": [], "pre": []})
        elif line.startswith("#") or "\t" not in line or sec is None:
            continue
        else:
            ts = canon_ts(line.rstrip("\n").rsplit("\t", 1)[1])
            sections[sec]["files"] += 1
            if battery_t0 and ts > battery_t0:
                sections[sec]["post"].append(ts)
            else:
                sections[sec]["pre"].append(ts)
    return sections


def build(run_dir: Path) -> tuple[list[dict], list[str]]:
    meta = json.loads((run_dir / "metadata.json").read_text())
    year = meta_year(meta)
    rel = lambda p: f"evidence/runs/{run_dir.name}/{p}"  # noqa: E731
    notes: list[str] = []
    ev: list[dict] = []

    def add(ts, event, src_file, src_ref, derivation, confidence, detail=""):
        ev.append({"timestamp": ts, "event": event, "source_file": src_file,
                   "source_line_or_record": src_ref, "derivation": derivation,
                   "confidence": confidence, "detail": detail})

    # metadata: run created (raw)
    created = meta["created"]  # e.g. 2026-10-07T02:59:01+0000
    add(created[:19] + "Z" if "+" in created else created, "run_metadata_created",
        rel("metadata.json"), "created", "raw", "high")

    # run-env.sh generation marker (raw)
    run_env = run_dir / "run-env.sh"
    if run_env.exists():
        first = run_env.read_text().splitlines()[0]
        m = re.search(r"at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", first)
        if m:
            add(m.group(1) + "Z", "run_env_prepared", rel("run-env.sh"), "line 1", "raw", "high")

    # cache-before snapshot header (raw)
    cb = run_dir / "cache-before.txt"
    if cb.exists():
        first = cb.read_text().splitlines()[0]
        m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", first)
        if m:
            add(m.group(1) + "Z", "cache_before_snapshot", rel("cache-before.txt"), "header", "raw", "high",
                detail=first)

    # server.log parsed events
    for ts, event, ref, excerpt in server_events(run_dir / "server.log", year):
        add(ts, event, rel("server.log"), ref, "parsed", "high", detail=excerpt)

    # client.log battery markers (raw)
    markers = battery_markers(run_dir / "client.log")
    for ts, kind, lineno in markers:
        add(ts, kind if len(markers) <= 2 else kind + f"_{sum(1 for t, k, _ in markers if k == kind and t <= ts)}",
            rel("client.log"), f"line {lineno}", "raw", "high")

    # requests.jsonl first/last request (parsed epoch)
    req = run_dir / "requests.jsonl"
    if req.exists():
        lines = [json.loads(x) for x in req.open() if x.strip()]
        if lines:
            add(iso_utc(epoch=lines[0]["t0"]), "first_recorded_request_t0",
                rel("requests.jsonl"), "record[0].t0", "parsed", "high",
                detail=f"prompt_tokens={lines[0].get('prompt_tokens')} batch={lines[0].get('batch')}")
            last = lines[-1]
            add(iso_utc(epoch=last["t0"] + last.get("wall_s", 0)), "last_recorded_request_end",
                rel("requests.jsonl"), f"record[{len(lines)-1}].t0+wall_s", "parsed", "high")

    # derived cache boundary events (from cache-after mtimes vs first battery t0)
    first_bt = next((t for t, k, _ in markers if k == "battery_start"), None)
    if first_bt:
        secs = cache_section_bounds(run_dir / "cache-after.txt", canon_ts(first_bt))
        for name in sorted(secs):
            s = secs[name]
            if s["pre"]:
                add(max(s["pre"]), f"last_startup_artifact_{name}",
                    rel("cache-after.txt"), f"max mtime <= battery_start ({s['files'] - len(s['post'])} files)",
                    "derived", "high")
            if s["post"]:
                add(min(s["post"]), f"first_runtime_artifact_{name}",
                    rel("cache-after.txt"), f"min mtime > battery_start ({len(s['post'])} files)",
                    "derived", "high")
        if not any(secs[s]["post"] for s in secs):
            add(first_bt, "zero_post_battery_artifacts_all_sections",
                rel("cache-after.txt"), "derived: 0 files with mtime > battery_start", "derived", "high")

    # cache-after snapshot header (raw)
    ca = run_dir / "cache-after.txt"
    if ca.exists():
        first = ca.read_text().splitlines()[0]
        m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", first)
        if m:
            add(m.group(1) + "Z", "cache_after_snapshot", rel("cache-after.txt"), "header", "raw", "high")

    ev.sort(key=lambda e: (e["timestamp"], e["event"]))

    # limitations note
    notes.append("server.log 'Application startup complete.' / 'Uvicorn running on' are logged by "
                 "uvicorn WITHOUT timestamps; they are intentionally absent from this timeline.")
    if run_dir.name == "N1-cold-warn":
        notes.append("N1: three battery markers exist (harness-bug retries); the first battery's "
                     "requests failed client-side but executed server-side (see result.md caveat).")
    return ev, notes


def main(repo: Path, only: list[str]) -> int:
    runs_root = repo / "evidence" / "runs"
    dirs = sorted(d for d in runs_root.iterdir() if d.is_dir() and (not only or d.name in only or any(d.name.startswith(o) for o in only)))
    for run_dir in dirs:
        events, notes = build(run_dir)
        out = run_dir / "timeline.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "event", "source_file",
                                              "source_line_or_record", "derivation",
                                              "confidence", "detail"], lineterminator="\n")
            w.writeheader()
            for e in events:
                w.writerow(e)
        print(f"{run_dir.name}: {len(events)} events")
        for n in notes:
            print(f"    note: {n}")
    return 0


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith(("P", "G", "N")) else Path.cwd()
    args = [a for a in sys.argv[1:] if a.startswith(("P", "G", "N"))]
    sys.exit(main(root or Path(__file__).resolve().parent.parent, args))
