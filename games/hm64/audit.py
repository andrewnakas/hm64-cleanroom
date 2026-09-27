"""DIRTY (dev check): classify the bytes of each sprite asset the spec does NOT regenerate.

    python -m games.hm64.audit <rom> <pristine> <spec dir>
Expected leftovers: frame headers (8 B), palette offset tables and 4-byte palette headers/trailers, animation
metadata, the sprite-to-palette table. Anything else non-zero is reported (it could be art we miss).
"""
import json
import os
import sys

import numpy as np


def main():
    rom = open(sys.argv[1], "rb").read()
    sys.path.insert(0, os.path.join(sys.argv[2], "tools"))
    from libhm64.common import rom as R
    R.set_rom_path(sys.argv[1])
    from libhm64.sprites import addresses as A
    spec = json.load(open(os.path.join(sys.argv[3], "textures.json")))
    cov = np.zeros(len(rom), bool)
    for t in spec["textures"]:
        cov[t["off"] - 8:t["off"] + t["n"]] = True        # data + its 8-byte header
    for p in spec["palettes"]:
        cov[p["off"] - 4:p["off"] + 2 * p["n"] + 4] = True  # colours + header/trailer
    u32 = lambda o: int.from_bytes(rom[o:o + 4], "big")
    tot = {"sheet": 0, "palsec": 0, "anim": 0, "s2p": 0, "tail": 0}
    worst = []
    for info in A.get_all_sprites():
        base = info.addr_base
        ao = A.get_asset_offsets(info.addr_index)
        # palette section: offset table is structure
        if ao[1]:
            first = u32(base + ao[1])
            if 0 < first <= 0x1000:
                cov[base + ao[1]:base + ao[1] + first] = True
        regions = [("sheet", base, base + ao[0] if ao[0] else base + ao[1]),
                   ("palsec", base + ao[1], base + ao[2]), ("anim", base + ao[2], base + ao[3]),
                   ("s2p", base + ao[3], base + ao[4])]
        for name, a, b in regions:
            if not (0 <= a < b <= len(rom)) or b - a > 0x400000:
                continue
            seg = np.frombuffer(rom[a:b], np.uint8)
            left = (~cov[a:b]) & (seg != 0)
            if name in ("anim", "s2p"):
                continue
            tot[name] += int(left.sum())
            if left.sum() > 64:
                worst.append((int(left.sum()), info.label, name))
    print("non-zero bytes not regenerated (excluding animation metadata and s2p tables):", tot)
    for n, lab, name in sorted(worst, reverse=True)[:12]:
        print(f"  {lab:28s} {name:7s} {n} B")


if __name__ == "__main__":
    main()
