"""Taint scan: every regenerated range of the clean ROM vs every retail one (dev check, reads retail).

    python -m games.hm64.taint_report <retail.z64> <clean.z64> <spec dir>

Streams per slot: texture index bytes, decoded RGBA (with each ROM's own palette), palette words,
font cells, ADPCM bytes and decoded PCM. A window of cleanroom.taint.WINDOW bytes shared with retail
(non-trivial) is a hit; runs >= FAIL_RUN fail. Prints '<n> failing'.
"""
import json
import os
import sys

import numpy as np

from cleanroom import taint
from games.hm64.extract_spec import rgba5551, indices

FAIL_RUN = getattr(taint, "FAIL_RUN", 32)


def streams(rom, spec_dir):
    t = json.load(open(os.path.join(spec_dir, "textures.json")))
    pals = t["palettes"]
    for x in t["textures"]:
        raw = rom[x["off"]:x["off"] + x["n"]]
        yield "tex:" + x["id"], raw
        p = pals[x["pal"]]
        cols = rgba5551(rom[p["off"]:p["off"] + 2 * p["n"]])
        idx = np.minimum(indices(raw, x["w"], x["h"], x["fmt"]), len(cols) - 1)
        yield "rgba:" + x["id"], cols[idx].tobytes()
    for p in pals:
        yield "pal:" + p["id"], rom[p["off"]:p["off"] + 2 * p["n"]]
    f = t["font"]
    yield "font", rom[f["off"]:f["end"]]
    s = json.load(open(os.path.join(spec_dir, "samples.json")))
    sys.path.insert(0, os.environ.get("HM64_PRISTINE", "D:/n64work/hm64/pristine") + "/tools")
    from cleanroom.audio import vadpcm
    for x in s["samples"]:
        raw = rom[x["data"]:x["data"] + x["n"]]
        yield f"adpcm:{x['i']}", raw
        bo = x["book"]
        book = np.frombuffer(rom[bo + 8:bo + 8 + 2 * x["order"] * x["npred"] * 8], ">i2").tolist()
        pcm = vadpcm.decode(raw, {"order": x["order"], "npred": x["npred"], "book": book}, x["nframes"])
        yield f"pcm:{x['i']}", np.asarray(pcm, ">i2").tobytes()


PX_WIN, PX_DISTINCT, PX_FAIL = 8, 4, 16


def px_hashes(buf):
    """Texel-aware windows for decoded RGBA: PX_WIN pixels, ignored when they hold fewer than
    PX_DISTINCT distinct colours (flat or two-tone runs every encoder makes) or repeat with period <= 2."""
    a = np.frombuffer(buf[:len(buf) // 4 * 4], ">u4").astype(np.uint64)
    n = len(a) - PX_WIN + 1
    if n <= 0:
        return np.zeros(0, np.uint64), np.zeros(0, bool)
    h = np.zeros(n, np.uint64)
    prime = np.uint64(1099511628211)
    with np.errstate(over="ignore"):
        for k in range(PX_WIN):
            h = (h ^ a[k:k + n]) * prime
    win = np.lib.stride_tricks.sliding_window_view(a, PX_WIN)
    srt = np.sort(win, axis=1)
    distinct = 1 + (srt[:, 1:] != srt[:, :-1]).sum(axis=1)
    per2 = (win[:, 2:] == win[:, :-2]).all(axis=1)
    return h, (distinct < PX_DISTINCT) | per2


def px_scan(index, streams):
    out = []
    for label, s in streams:
        h, low = px_hashes(s)
        if not len(h):
            continue
        pos = np.minimum(np.searchsorted(index, h), max(0, len(index) - 1))
        m = (index[pos] == h) & ~low if len(index) else np.zeros(len(h), bool)
        if m.any():
            d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
            run = int((np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0]).max()) + PX_WIN - 1
            out.append((label, int(np.nonzero(m)[0][0]), int(m.sum()), run * 4))
    return out


def main():
    retail = open(sys.argv[1], "rb").read()
    clean = open(sys.argv[2], "rb").read()
    spec = sys.argv[3]
    bad, nhits, kinds = [], 0, {}
    # one index per stream kind keeps memory low (cross-kind matches are meaningless anyway)
    for kind in ("tex", "rgba", "pal", "font", "adpcm", "pcm"):
        parts = []
        for label, s in streams(retail, spec):
            if label.split(":")[0] == kind:
                h, per = (px_hashes if kind == "rgba" else taint._hashes)(s)
                parts.append(h[~per])
        index = np.concatenate(parts) if parts else np.zeros(0, np.uint64)
        del parts
        index.sort()
        sel = ((l, s) for l, s in streams(clean, spec) if l.split(":")[0] == kind)
        hits = px_scan(index, sel) if kind == "rgba" else taint.scan(index, sel)
        del index
        nhits += len(hits)
        b = [h for h in hits if h[3] >= (PX_FAIL * 4 if kind == "rgba" else FAIL_RUN)]
        kinds[kind] = len(b)
        bad += b
    bad.sort(key=lambda h: -h[3])
    print(f"taint: {nhits} slots with short coincidental matches; {len(bad)} failing (run >= {FAIL_RUN} B) {kinds}")
    for label, off, n, run in bad[:10]:
        print(f"  FAIL {label} run {run} B at {off}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
