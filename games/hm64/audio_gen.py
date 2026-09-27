"""CLEAN ROOM: the libmus sample bank from spec/samples.json, in place.

Each sample is resynthesised from its outline (cleanroom.audio.descriptor), held at the kept median
pitch when it is tonal, encoded with our own 4-predictor VADPCM book (same shape as the slot), and
its loop state comes from our own decoded stream. Offsets and sizes stay exactly as in the bank.
"""
import hashlib
import json
import struct

import numpy as np

from cleanroom.audio import descriptor, vadpcm


def _seed(*parts):
    return int.from_bytes(hashlib.sha1("/".join(map(str, parts)).encode()).digest()[:4], "little")


def fit_predictors(x, n):
    """n order-2 predictors fitted to our own signal (k-means over per-frame least-squares fits)."""
    x = np.asarray(x, np.float64)
    fits = []
    for s in range(2, len(x) - 16, 16):
        y, p1, p2 = x[s:s + 16], x[s - 1:s + 15], x[s - 2:s + 14]
        if (y ** 2).sum() < 1e3:
            continue
        a, *_ = np.linalg.lstsq(np.stack([p1, p2], 1), y, rcond=None)
        fits.append(a)
    defaults = [(1.8, -0.82), (1.0, 0.0), (1.95, -0.96), (0.0, 0.0)]
    if len(fits) < n * 2:
        return defaults[:n]
    f = np.clip(np.asarray(fits), [-1.95, -0.98], [1.95, 0.98])
    order = np.argsort(f[:, 0])
    c = f[order[np.linspace(0, len(f) - 1, n).astype(int)]].copy()
    for _ in range(12):
        lab = np.argmin(((f[:, None, :] - c[None]) ** 2).sum(-1), 1)
        for k in range(n):
            if (lab == k).any():
                c[k] = f[lab == k].mean(0)
    out = []
    for a1, a2 in c:
        a2 = float(np.clip(a2, -0.98, 0.98))
        out.append((float(np.clip(a1, -(1 - a2) + 0.02, (1 - a2) - 0.02)), a2))
    return out


def tonal(s):
    fr = s["desc"]["frames"]
    return s.get("f0", 0) > 20 and fr and np.median([f["h"] for f in fr]) > 0.45


def sample_pcm(s):
    desc = json.loads(json.dumps(s["desc"]))
    ton = tonal(s)
    if ton:
        for f in desc["frames"]:
            if f["h"] > 0.3:
                f["f0"] = s["f0"]
    n, rate = s["nframes"], s["rate"]
    x = np.asarray(descriptor.synthesize(desc, n, rate, seed=_seed("hm64", s["i"])), np.float64)
    if ton:
        t = np.arange(n) / rate
        env = np.sqrt(np.convolve(x ** 2, np.ones(256) / 256, "same"))
        x = x + 0.6 * np.sqrt(2) * env * np.sin(2 * np.pi * s["f0"] * t)
    lp = s.get("loop")
    if lp and lp["end"] <= n and lp["end"] - lp["start"] > 32:
        x = descriptor.make_loop_seamless(x, lp["start"], lp["end"])
    peak = np.abs(x).max()
    if peak > 0.99:
        x *= 0.99 / peak
    dither = np.random.default_rng(_seed("dither", s["i"])).integers(-1, 2, n)
    return np.clip(np.round(x * 32000) + dither, -32768, 32767).astype(np.int64)


def build(spec, put):
    """put(offset, bytes) for every sample, book and loop state."""
    for s in spec["samples"]:
        pcm = sample_pcm(s)
        preds = fit_predictors(pcm, s["npred"])
        book = vadpcm.make_book(preds)
        data, book, dec = vadpcm.encode(pcm, book)
        data = data[:s["n"]] + bytes(max(0, s["n"] - len(data)))
        put(s["data"], data)
        hdr = struct.pack(">II", s["order"], s["npred"])
        put(s["book"], hdr + struct.pack(f">{len(book['book'])}h", *book["book"]))
        lp = s.get("loop")
        if lp:
            st = vadpcm.loop_state(dec, lp["start"]) if lp["count"] else [0] * 16
            put(lp["off"] + 12, struct.pack(">16h", *st))
