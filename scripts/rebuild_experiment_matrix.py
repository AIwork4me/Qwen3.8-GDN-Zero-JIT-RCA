#!/usr/bin/env python3
"""Rebuild the experiment matrix from committed raw evidence.

Sources of truth (per run directory under evidence/runs/<RUN_ID>/):
  - command.sh      the actually-executed vllm serve command (authoritative)
  - metadata.json   run bookkeeping (mode, monitor, model, port, args)
  - server.log      engine config line (enforce_eager cross-check)
  - client.log      request battery outcome counts
  - cache-after.txt post-battery artifact counts (vs battery t0)
  - jit-events.jsonl monitor event count

The script derives every technical column from these files and hard-fails on
any internal inconsistency (command vs metadata vs server log). The only
hand-maintained inputs are the human planning labels (plan/purpose) which are
not properties of the evidence.

Outputs (overwritten):
  manifests/experiment_matrix.csv
  docs/08-experiment-matrix.md

Usage: rebuild_experiment_matrix.py [repo_root]
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

# Planning labels (not evidence properties; the single hand-maintained input).
PLAN = {
    "P1": ("primary", "Untouched-main cold eager reproduction with JIT monitor warn+verbose (runtime-JIT inventory)"),
    "P2": ("primary", "Monitor error-mode acceptance test (#49349 gate): cold default/graph config"),
    "P3": ("primary", "Cold-repeat determinism (>=2 independent cold default-mode runs)"),
    "P4": ("primary", "Warm-cache control reusing P5 cache root (same RUN_ID cache)"),
    "P5": ("primary", "Primary zero-runtime-JIT test: cold default/graph configuration"),
    "G1": ("small-GDN", "Small GDN eager runtime-JIT inventory control"),
    "G2": ("small-GDN", "Small GDN cold default/graph zero-runtime-JIT control"),
    "G3": ("small-GDN", "Small GDN warm reuse after G2 (same cache root)"),
    "N1": ("non-GDN", "Dense non-GDN model control (harness validation + eager generic-kernel baseline)"),
}

VLLM_LOG_TS = re.compile(r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
ENFORCE_EAGER_CFG = re.compile(r"enforce_eager=(True|False)")


def parse_command_sh(path: Path) -> dict:
    text = path.read_text()
    joined = " ".join(line.rstrip("\\ \n") for line in text.splitlines())
    out = {}
    m = re.search(r"vllm serve\s+(\S+)", joined)
    out["model_path"] = m.group(1).strip('"') if m else None
    m = re.search(r"--port\s+(\d+)", joined)
    out["port"] = m.group(1) if m else None
    m = re.search(r"--jit-monitor-mode\s+(\S+)", joined)
    out["jit_monitor_mode"] = m.group(1) if m else None
    out["enforce_eager"] = "--enforce-eager" in joined
    extras = []
    for flag in ("--max-model-len", "--max-num-seqs", "--gpu-memory-utilization"):
        m = re.search(re.escape(flag) + r"\s+(\S+)", joined)
        if m:
            extras.extend([flag, m.group(1)])
    out["extra_args"] = " ".join(extras)
    return out


def parse_client_log(path: Path) -> dict:
    batteries, ok, total, cur_ok, cur_total, rows = 0, 0, 0, 0, 0, 0
    first_t0 = None
    for line in path.read_text().splitlines():
        if line.startswith("# battery start"):
            batteries += 1
            m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", line)
            if batteries == 1 and m:
                first_t0 = m.group(1)
        m = re.match(r"prompt_tokens=\d+ batch=\d+ wall=\S+ ok=(\d+)/(\d+)", line)
        if m:
            ok += int(m.group(1))
            total += int(m.group(2))
            rows += 1
    return {"batteries": batteries, "battery_t0": first_t0, "ok": ok, "total": total, "rows": rows}


def parse_cache_after(path: Path, battery_t0: str | None) -> dict:
    """Per-section file counts; post-battery counts vs the FIRST battery t0."""
    sections: dict[str, dict] = {}
    sec = None
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            sec = line[3:].split()[0]
            sections.setdefault(sec, {"files": 0, "post_battery": 0})
        elif line.startswith("#") or "\t" not in line or sec is None:
            continue
        else:
            sections[sec]["files"] += 1
            ts = line.rstrip("\n").rsplit("\t", 1)[1]
            if battery_t0 and ts > battery_t0:
                sections[sec]["post_battery"] += 1
    return sections


def parse_server_log(path: Path) -> dict:
    enforce = set()
    monitor_activated = None
    for line in path.read_text(errors="replace").splitlines():
        for m in ENFORCE_EAGER_CFG.finditer(line):
            enforce.add(m.group(1))
        if "Kernel JIT monitor activated" in line:
            m = VLLM_LOG_TS.search(line)
            mode = "error" if "mode=error" in line else "warn" if "mode=warn" in line else "?"
            monitor_activated = (m.group(1) if m else None, mode)
    return {
        "enforce_eager_values": enforce,
        "monitor_activated": monitor_activated,
    }


def short_model(path_str: str | None) -> str:
    if not path_str:
        return "?"
    return path_str.rstrip("/").split("/")[-1]


def build_run_row(run_dir: Path, errors: list[str]) -> dict:
    run_id_long = run_dir.name
    meta = json.loads((run_dir / "metadata.json").read_text())
    run_id = meta["run_id"]
    if run_id != run_id_long:
        errors.append(f"{run_id_long}: metadata run_id {run_id} != dir name")
    short = re.match(r"^([PGN]\d+)", run_id_long)
    if not short or short.group(1) not in PLAN:
        errors.append(f"{run_id_long}: no planning entry for run")
        short = short.group(1) if short else run_id_long
    else:
        short = short.group(1)

    cmd = parse_command_sh(run_dir / "command.sh")

    # --- consistency cross-checks (raw command is the source of truth) ---
    if cmd["model_path"] != meta["model_path"]:
        errors.append(f"{run_id}: model {cmd['model_path']!r} != metadata {meta['model_path']!r}")
    if cmd["jit_monitor_mode"] != meta["jit_monitor_mode"]:
        errors.append(f"{run_id}: monitor {cmd['jit_monitor_mode']!r} != metadata {meta['jit_monitor_mode']!r}")
    if str(meta["port"]) != str(cmd["port"]):
        errors.append(f"{run_id}: port {cmd['port']!r} != metadata {meta['port']!r}")

    srv = parse_server_log(run_dir / "server.log")
    eager_cfg = cmd["enforce_eager"]
    if eager_cfg:
        if srv["enforce_eager_values"] - {"True"}:
            errors.append(f"{run_id}: command is eager but server.log shows non-eager config")
    else:
        if "False" not in srv["enforce_eager_values"] or "True" in srv["enforce_eager_values"]:
            errors.append(f"{run_id}: command is default/graph but server.log enforce_eager={srv['enforce_eager_values']}")

    cli = parse_client_log(run_dir / "client.log")
    cache = parse_cache_after(run_dir / "cache-after.txt", cli["battery_t0"])
    jit_events = sum(1 for _ in (run_dir / "jit-events.jsonl").open()) if (run_dir / "jit-events.jsonl").exists() else 0

    post_total = sum(s["post_battery"] for s in cache.values())
    post_triton = cache.get("triton", {}).get("post_battery", 0)

    plan, purpose = PLAN[short]
    graphs = "eager" if eager_cfg else "cuda-graph"
    model = short_model(cmd["model_path"])
    label = f"{model} {plan} {meta['mode']} {meta['jit_monitor_mode']} {graphs}"

    return {
        "run_id": short,
        "run_dir": run_id_long,
        "label": label,
        "model": model,
        "mode": meta["mode"],
        "monitor": meta["jit_monitor_mode"],
        "graphs": graphs,
        "extra_args": cmd["extra_args"],
        "purpose": purpose,
        # evidence-derived result fields (for the markdown doc)
        "requests_ok": cli["ok"],
        "requests_total": cli["total"],
        "battery_rows": cli["rows"],
        "batteries": cli["batteries"],
        "battery_t0": cli["battery_t0"],
        "post_battery_total": post_total,
        "post_battery_triton": post_triton,
        "cache_sections": cache,
        "jit_monitor_events": jit_events,
        "monitor_activated": srv["monitor_activated"],
        "enforce_eager": eager_cfg,
    }


def result_cell(r: dict) -> str:
    bits = []
    ma = r["monitor_activated"]
    if r["enforce_eager"]:
        bits.append("monitor inactive (eager, by design)")
    elif ma:
        bits.append(f"monitor armed mode={ma[1]}")
    else:
        bits.append("monitor NOT armed (UNEXPECTED)")
    bits.append(f"{r['jit_monitor_events']} monitor events")
    bits.append(f"{r['post_battery_triton']} post-battery triton files")
    bits.append(f"requests {r['requests_ok']}/{r['requests_total']} OK ({r['battery_rows']} battery rows)")
    if r["run_id"] == "P5":
        bits.append("**primary zero-runtime-JIT proof**")
    if r["run_id"] == "P1":
        bits.append("**eager runtime-JIT inventory**")
    return "; ".join(bits)


def main(repo: Path) -> int:
    runs_root = repo / "evidence" / "runs"
    errors: list[str] = []
    rows = [build_run_row(d, errors) for d in sorted(runs_root.iterdir()) if d.is_dir()]
    rows.sort(key=lambda r: r["run_id"])

    if errors:
        print("EVIDENCE INCONSISTENCIES FOUND — refusing to write outputs:")
        for e in errors:
            print(f"  - {e}")
        return 1

    # ---- manifests/experiment_matrix.csv ----
    csv_path = repo / "manifests" / "experiment_matrix.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["run_id", "label", "model", "mode", "monitor", "graphs", "extra_args", "purpose"])
        for r in rows:
            w.writerow([
                r["run_id"], r["label"], r["model"], r["mode"], r["monitor"],
                r["graphs"], r["extra_args"], r["purpose"],
            ])

    # ---- docs/08-experiment-matrix.md ----
    lines = [
        "# 08 — Experiment Matrix (executed; generated)",
        "",
        "GENERATED by `scripts/rebuild_experiment_matrix.py` from committed raw",
        "evidence (`command.sh`, `metadata.json`, `server.log`, `client.log`,",
        "`cache-after.txt`, `jit-events.jsonl`). Do not hand-edit; regenerate with:",
        "",
        "```bash",
        "python3 scripts/rebuild_experiment_matrix.py",
        "```",
        "",
        "The raw `command.sh` files are the source of truth for each row. The",
        "historical hand-maintained matrix (pre-closure) mislabeled P2/P3/P4 and",
        "G2/G3 as `eager`; the committed commands prove those runs were executed",
        "in the DEFAULT (graph) configuration. P5 was the primary cold default-",
        "mode run; P1 was the only primary eager run.",
        "",
        "| id | run dir | model | mode | monitor | graphs | vllm args (from command.sh) | evidence-derived result |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        args = r["extra_args"] + (" --enforce-eager" if r["enforce_eager"] else "")
        mode = r["mode"] + (f" (reuse of {json.loads((runs_root / r['run_dir'] / 'metadata.json').read_text()).get('warm_reuse_of', '?')})" if r["mode"] == "warm" else "")
        lines.append(
            f"| {r['run_id']} | {r['run_dir']} | {r['model']} | {mode} | {r['monitor']} | {r['graphs']} | `{args}` | {result_cell(r)} |"
        )
    lines += [
        "",
        "Notes:",
        "- Battery = prompts {1,4,8,15,16,17,31,32,33,64,128} tokens x batches",
        "  {1,2,4} (33 request rows = 77 individual requests), greedy;",
        "  max_tokens 16 (27B) / 8 (small models).",
        "- \"post-battery triton files\" counts cache entries in `cache-after.txt`",
        "  whose mtime is after the FIRST battery-start marker in `client.log`",
        "  (deterministic verifier: `scripts/verify_runtime_cache_boundary.py`).",
        "- N1's first battery (client harness bug: rejected `prompt_token_ids`)",
        "  still executed server-side inference — see the N1 run `result.md`",
        "  caveat; post-battery counts for N1 use the first battery marker and",
        "  therefore include those compile artifacts.",
        "- P1/P2 'eager diagnostic' rows from the original planning matrix were",
        "  superseded at execution time (P2 ran cold/default/error); the raw",
        "  command files in each run dir document the deviation.",
        "",
    ]
    (repo / "docs" / "08-experiment-matrix.md").write_text("\n".join(lines))

    print(f"wrote {csv_path}")
    print(f"wrote {repo / 'docs' / '08-experiment-matrix.md'}")
    for r in rows:
        print(f"  {r['run_id']:3s} {r['mode']:5s} {r['monitor']:5s} {r['graphs']:10s} post-battery triton={r['post_battery_triton']:3d} requests={r['requests_ok']}/{r['requests_total']}")
    return 0


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    sys.exit(main(root))
