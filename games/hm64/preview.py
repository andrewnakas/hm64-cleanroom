"""Decode frames from a (clean) ROM to PNGs and a contact sheet.

    python -m games.hm64.preview <rom.z64> <spec dir> <match,...> <out.png> [--scale 2] [--width 1600]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from cleanroom.gfx import png
from games.hm64.extract_spec import indices, rgba5551


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rom")
    ap.add_argument("spec")
    ap.add_argument("match")
    ap.add_argument("out")
    ap.add_argument("--scale", default="2")
    ap.add_argument("--width", default="1600")
    a = ap.parse_args()
    rom = open(a.rom, "rb").read()
    t = json.load(open(os.path.join(a.spec, "textures.json")))
    keys = a.match.split(",")
    d = tempfile.mkdtemp()
    for x in t["textures"]:
        if not any(k in x["id"] for k in keys):
            continue
        p = t["palettes"][x["pal"]]
        cols = rgba5551(rom[p["off"]:p["off"] + 2 * p["n"]])
        idx = np.minimum(indices(rom[x["off"]:x["off"] + x["n"]], x["w"], x["h"], x["fmt"]), len(cols) - 1)
        png.write(os.path.join(d, x["id"].replace("/", "__") + ".png"), cols[idx])
    subprocess.run([sys.executable, "-m", "games.hm64.sheet", d, "", a.out, "--scale", a.scale, "--width", a.width])
    shutil.rmtree(d)


if __name__ == "__main__":
    main()
