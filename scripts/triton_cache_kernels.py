#!/usr/bin/env python3
"""List kernel names compiled into a Triton cache dir.

Triton 3.8 cache layout: <TRITON_CACHE_DIR>/<hash>/ with json metadata
containing the kernel name. Usage: triton_cache_kernels.py <cache_dir>
"""
import json
import os
import sys


def main(cache_dir: str) -> int:
    if not os.path.isdir(cache_dir):
        print(f"not a dir: {cache_dir}")
        return 1
    kernels = {}
    for entry in sorted(os.listdir(cache_dir)):
        d = os.path.join(cache_dir, entry)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".json"):
                p = os.path.join(d, fn)
                try:
                    meta = json.load(open(p))
                except Exception:
                    continue
                name = (
                    meta.get("name")
                    or meta.get("kernel_name")
                    or meta.get("src_name")
                )
                if name:
                    key = f"{name}"
                    rec = kernels.setdefault(
                        key, {"entries": set(), "signature": None}
                    )
                    rec["entries"].add(entry)
                    if rec["signature"] is None and meta.get("signature"):
                        rec["signature"] = str(meta["signature"])
    for name in sorted(kernels):
        rec = kernels[name]
        print(f"{name}\tvariants={len(rec['entries'])}")
        for e in sorted(rec["entries"]):
            print(f"  {e}")
    print(f"TOTAL_KERNELS={len(kernels)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
