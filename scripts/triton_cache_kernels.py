#!/usr/bin/env python3
"""List kernel names compiled into a Triton cache dir (triton 3.x layout:
each hashed entry dir contains files named <kernel_name>.<ext>)."""
import json
import os
import sys


def main(cache_dir: str) -> int:
    if not os.path.isdir(cache_dir):
        print(f"not a dir: {cache_dir}")
        return 1
    kernels: dict[str, dict] = {}
    for entry in sorted(os.listdir(cache_dir)):
        d = os.path.join(cache_dir, entry)
        if not os.path.isdir(d):
            continue
        names = set()
        sig = None
        for fn in os.listdir(d):
            base = fn.rsplit(".", 1)[0]
            if fn.endswith((".json", ".ttir", ".ttgd", ".hsaco", ".cubin", ".llir", ".ptx")):
                # group files look like __grp__RealName__...; strip markers
                b = base
                if b.startswith("__grp__"):
                    b = b[len("__grp__"):]
                b = b.rsplit("__", 1)[0] if "__" in b and not b.endswith("_kernel") else b
                names.add(b)
            if fn.endswith(".json"):
                try:
                    j = json.load(open(os.path.join(d, fn)))
                    if isinstance(j, dict):
                        sig = j.get("signature") or sig
                except Exception:
                    pass
        for n in names:
            if n.startswith("__"):
                continue
            rec = kernels.setdefault(n, {"entries": set()})
            rec["entries"].add(entry)
    for n in sorted(kernels):
        print(f"{n}\tvariants={len(kernels[n]['entries'])}")
    print(f"TOTAL_KERNELS={len(kernels)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
