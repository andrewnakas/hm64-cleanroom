"""List the ROM's top-level segments from the decomp's splat yaml (no ROM access).

python -m games.hm64.census <pristine> [out.json]
Prints one line per asset group (dir prefix) with count and bytes.
"""
import json
import sys
from collections import defaultdict

import yaml


def segments(pristine):
    y = yaml.safe_load(open(f"{pristine}/config/us/splat.us.yaml"))
    segs = y["segments"]
    out = []
    for i, s in enumerate(segs):
        if isinstance(s, list):
            continue
        if isinstance(s, dict):
            nxt = segs[i + 1]
            end = nxt[0] if isinstance(nxt, list) else nxt["start"]
            out.append(dict(name=s["name"], type=s.get("type"), start=s["start"], end=end,
                            dir=s.get("dir", ""), vram=s.get("vram"),
                            subs=len(s.get("subsegments", []) or [])))
    return out


def main():
    pristine = sys.argv[1]
    segs = segments(pristine)
    groups = defaultdict(lambda: [0, 0, set()])
    for s in segs:
        key = "/".join(s["dir"].split("/")[:3]) or s["name"]
        g = groups[key]
        g[0] += 1
        g[1] += s["end"] - s["start"]
        g[2].add(s["type"])
    for k, (n, b, t) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        print(f"{k:45s} {n:5d} {b / 1024:9.1f} KB  {','.join(sorted(map(str, t)))}")
    if len(sys.argv) > 2:
        json.dump(segs, open(sys.argv[2], "w"), indent=0)


if __name__ == "__main__":
    main()
