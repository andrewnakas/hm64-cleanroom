"""Clean-side audio check: decode every regenerated sample from a clean ROM and compare with the spec.

    python -m games.hm64.audio_check <clean.z64> <spec dir>
Prints: tonal samples within 25 / 50 cents of the kept median pitch, octave errors, silent or clipped samples,
and loop seams (jump at the loop point relative to the signal's RMS).
"""
import json
import os
import sys

import numpy as np

from cleanroom.audio import vadpcm
from cleanroom.audio.pitch import median_f0
from games.hm64.audio_gen import tonal


def main():
    rom = open(sys.argv[1], "rb").read()
    sp = json.load(open(os.path.join(sys.argv[2], "samples.json")))
    ok25 = ok50 = octave = n_ton = silent = clipped = bad_loop = 0
    worst = []
    for s in sp["samples"]:
        bo = s["book"]
        book = np.frombuffer(rom[bo + 8:bo + 8 + 2 * s["order"] * s["npred"] * 8], ">i2").tolist()
        pcm = vadpcm.decode(rom[s["data"]:s["data"] + s["n"]], {"order": s["order"], "npred": s["npred"], "book": book},
                            s["nframes"]).astype(np.float64)
        rms = np.sqrt((pcm ** 2).mean()) if len(pcm) else 0
        if rms < 30:
            silent += 1
        if np.abs(pcm).max(initial=0) >= 32700:
            clipped += 1
        lp = s.get("loop")
        if lp and lp["count"] and 16 < lp["start"] < lp["end"] <= len(pcm):
            jump = abs(pcm[lp["end"] - 1] - pcm[lp["start"] - 1]) if lp["start"] > 0 else 0
            if rms and jump > 3 * rms:
                bad_loop += 1
        if tonal(s):
            n_ton += 1
            f = median_f0((pcm / 32768).astype(np.float32), s["rate"])
            if f:
                c = 1200 * np.log2(f / s["f0"])
                ok25 += abs(c) <= 25
                ok50 += abs(c) <= 50
                octave += abs(abs(c) - 1200) < 60
                worst.append((abs(c), s["i"], round(c)))
    worst.sort(reverse=True)
    print(f"samples {len(sp['samples'])}: tonal {n_ton}, within 25c {ok25}, within 50c {ok50}, octave errors {octave}; "
          f"silent {silent}, clipped {clipped}, loop seams {bad_loop}")
    print("worst tonal (cents):", [(i, c) for _, i, c in worst[:8]])


if __name__ == "__main__":
    main()
