#!/usr/bin/env python3
"""Shape battery client: exercise prefill/decode shapes against a running server.

Usage: run_shape_battery.py <run_dir> [--port 8137] [--base-url http://127.0.0.1]

Sends controlled prompts (token-count targeted via explicit token ids) and
records per-request timing + completion to client.log / requests.jsonl.
"""
import argparse
import json
import os
import time
import urllib.request

API_TOKENS = "/v1/completions"


def post(url, payload, timeout=1800):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return time.time() - t0, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--port", type=int, default=8137)
    ap.add_argument("--host", default="http://127.0.0.1")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument(
        "--battery",
        default="1,4,8,15,16,17,31,32,33,64,128",
        help="comma list of prompt token counts",
    )
    ap.add_argument("--batches", default="1", help="comma list of batch sizes (concurrent)")
    args = ap.parse_args()
    url = f"{args.host}:{args.port}{API_TOKENS}"

    # tokenizer-independent token control: send token ids directly.
    # ids within [100, 100099]: valid for every battery model's vocab
    # (Qwen2.5 151936, Qwen3.x 248320); pairwise distinct for n <= 127.
    def ids_for(n, seed=0):
        return [(seed * 9973 + i * 7919) % 100000 + 100 for i in range(n)]

    reqs_path = os.path.join(args.run_dir, "requests.jsonl")
    log_path = os.path.join(args.run_dir, "client.log")
    battery = [int(x) for x in args.battery.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    import threading

    with open(reqs_path, "w") as rf, open(log_path, "a") as lf:
        lf.write(f"# battery start {time.strftime('%FT%T')} url={url}\n")
        for b in batches:
            for n in battery:
                payload = {
                    "prompt": ids_for(n, seed=b),
                    "max_tokens": args.max_tokens,
                    "temperature": 0.0,
                    "ignore_eos": True,
                }
                t0 = time.time()
                results = [None] * b

                def worker(i):
                    try:
                        lat, body = post(url, payload)
                        results[i] = {"ok": True, "latency": lat, "body": body}
                    except Exception as e:  # noqa: BLE001
                        results[i] = {"ok": False, "error": repr(e)}

                ths = [threading.Thread(target=worker, args=(i,)) for i in range(b)]
                for t in ths:
                    t.start()
                for t in ths:
                    t.join()
                wall = time.time() - t0
                oks = sum(1 for r in results if r and r.get("ok"))
                rec = {
                    "t0": t0,
                    "prompt_tokens": n,
                    "batch": b,
                    "wall_s": round(wall, 3),
                    "ok": oks,
                    "total": b,
                    "results": [
                        {
                            "ok": r.get("ok"),
                            "latency": r.get("latency"),
                            "error": r.get("error"),
                            "n_tokens": (
                                (r["body"]["choices"][0].get("token_ids") or [0] * 999)
                                and len(r["body"]["choices"][0]["text"].split())
                                if r and r.get("ok")
                                else None
                            ),
                        }
                        for r in results
                    ],
                }
                rf.write(json.dumps(rec) + "\n")
                rf.flush()
                lf.write(
                    f"prompt_tokens={n} batch={b} wall={wall:.3f}s ok={oks}/{b}\n"
                )
                lf.flush()
        lf.write(f"# battery end {time.strftime('%FT%T')}\n")
    print("battery complete ->", reqs_path)


if __name__ == "__main__":
    main()
