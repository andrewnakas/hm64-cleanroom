"""CLEAN ROOM: every regenerated asset byte from spec/ + our own code.

    python -m games.hm64.generate <spec dir> <out patch file> [--only textures|audio|font] [--sheet N]

Writes a patch file of (offset, bytes) records that pack.py lays over the kept-facts base image.
Textures: each frame painted from its facts (paint.py, or an override PNG in overrides/textures/),
then all frames that share a palette are quantised together into our own palette (k-means).
"""
import argparse
import json
import os
import struct
import time
from collections import defaultdict

import numpy as np
from scipy.cluster.vq import kmeans2

from games.hm64 import chibi, labels, paint, photos, portraits

HERE = os.path.dirname(os.path.abspath(__file__))
OVR = os.path.join(HERE, "overrides", "textures")


def to5551(rgb, a):
    r, g, b = (np.clip(np.round(rgb[:, i] * 31 / 255), 0, 31).astype(np.uint16) for i in range(3))
    return (r << 11) | (g << 6) | (b << 1) | (a > 0).astype(np.uint16)


def pack_ci(idx, fmt):
    idx = idx.ravel().astype(np.uint8)
    if fmt == "ci4":
        if len(idx) % 2:
            idx = np.append(idx, 0)
        return ((idx[0::2] << 4) | (idx[1::2] & 15)).astype(np.uint8).tobytes()
    return idx.tobytes()


def override(t):
    p = os.path.join(OVR, t["id"].replace("/", "__") + ".png")
    if not os.path.exists(p):
        return None
    from PIL import Image
    im = Image.open(p).convert("RGBA")
    if im.size != (t["w"], t["h"]):
        im = im.resize((t["w"], t["h"]), Image.LANCZOS)
    return np.asarray(im).copy()


def image(t):
    img = override(t)
    if img is None:
        img = portraits.image(t)
    if img is None:
        img = chibi.image(t)
    if img is None:
        img = photos.image(t)
    if img is None:
        img = labels.image(t)
    return img if img is not None else paint.paint(t)


def quantise(imgs, n, seed):
    """Shared palette of n colours for a list of RGBA images. Entry 0 is transparent when any texel is."""
    trans = any((im[..., 3] < 128).any() for im in imgs)
    px = np.concatenate([im.reshape(-1, 4) for im in imgs])
    op = px[px[:, 3] >= 128][:, :3].astype(np.float32)
    k0 = 1 if trans else 0
    k = n - k0
    pal = np.zeros((n, 3), np.float32)
    if len(op):
        uniq = np.unique((op // 8).astype(np.int32), axis=0)
        kk = min(k, len(uniq))
        rng = np.random.default_rng(seed)
        sample = op[rng.choice(len(op), min(len(op), 24000), replace=False)] if len(op) > 24000 else op
        if kk >= len(uniq):
            cent = uniq.astype(np.float32) * 8 + 4
        else:
            cent, _ = kmeans2(sample, kk, iter=8, minit="++", seed=rng)
        pal[k0:k0 + len(cent)] = cent
        k = len(cent)
    alpha = np.ones(n, bool)
    if trans:
        alpha[0] = False
    out = []
    for im in imgs:
        flat = im.reshape(-1, 4)
        idx = np.zeros(len(flat), np.uint8)
        o = flat[:, 3] >= 128
        if o.any():
            c = pal[k0:k0 + k]
            f = flat[o][:, :3].astype(np.float32)
            best = np.empty(len(f), np.int64)
            for s in range(0, len(f), 8192):
                d = ((f[s:s + 8192, None, :] - c[None]) ** 2).sum(-1)
                best[s:s + 8192] = d.argmin(1)
            idx[o] = best + k0
        if not trans:
            idx[~o] = 0
        out.append(idx.reshape(im.shape[:2]))
    words = to5551(pal, alpha)
    if trans:
        words[0] = 0
    return out, words


def textures(spec, put, only_ids=None):
    by_pal = defaultdict(list)
    for t in spec["textures"]:
        by_pal[t["pal"]].append(t)
    pals = spec["palettes"]
    made = {}
    for pi, ts in by_pal.items():
        if only_ids and not any(t["id"] in only_ids for t in ts):
            continue
        p = pals[pi]
        n, k = p["n"], p.get("k", p["n"])
        imgs = [image(t) for t in ts]
        idxs, words = quantise(imgs, k, paint.h32("pal", p["id"]))
        words = np.concatenate([words, np.zeros(n - len(words), words.dtype)])
        made[pi] = words
        put(p["off"], words.astype(">u2").tobytes())
        for t, idx in zip(ts, idxs):
            b = pack_ci(idx, t["fmt"])[:t["n"]]
            put(t["off"], b + bytes(t["n"] - len(b)))
    return made


def runtime_palettes(spec, put, made):
    """Palettes the game switches to at runtime: our sibling palette (same index layout), shifted by the
    difference of the two kept mean colours, alpha bits kept from our sibling."""
    n_out = 0
    for p in spec["palettes"]:
        if "sibling" not in p or p["sibling"] not in made:
            continue
        w = made[p["sibling"]].astype(np.uint32)
        rgb = np.stack([(w >> 11) & 31, (w >> 6) & 31, (w >> 1) & 31], 1).astype(np.float32) * 255 / 31
        shift = np.asarray(p["mean"], np.float32) - np.asarray(p["sibling_mean"], np.float32)
        rgb = np.clip(rgb + shift, 0, 255)
        alpha = (w & 1).astype(bool)
        out = to5551(rgb, alpha)
        out[~alpha] = 0
        out = np.resize(out, p["n"]) if len(out) != p["n"] else out
        put(p["off"], out.astype(">u2").tobytes())
        n_out += 1
    return n_out


class Patch:
    def __init__(self):
        self.recs = []

    def put(self, off, b):
        self.recs.append((int(off), bytes(b)))

    def save(self, path):
        with open(path, "wb") as f:
            for off, b in sorted(self.recs):
                f.write(struct.pack(">II", off, len(b)))
                f.write(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("out")
    ap.add_argument("--only", default="")
    ap.add_argument("--match", default="", help="comma list of id substrings: only palettes used by matching frames")
    a = ap.parse_args()
    t0 = time.time()
    spec = json.load(open(os.path.join(a.spec, "textures.json")))
    P = Patch()
    if a.only in ("", "textures"):
        ids = None
        if a.match:
            keys = a.match.split(",")
            ids = {t["id"] for t in spec["textures"] if any(k in t["id"] for k in keys)}
        made = textures(spec, P.put, ids)
        print(f"runtime palettes: {runtime_palettes(spec, P.put, made)}")
        print(f"textures: {len(spec['textures'])} frames, {len(spec['palettes'])} palettes ({time.time() - t0:.0f} s)")
    if a.only in ("", "font") and not a.match:
        from games.hm64 import font
        font.build(spec["font"], P.put)
        print("font: done")
    if a.only in ("", "audio") and not a.match:
        from games.hm64 import audio_gen
        sp = json.load(open(os.path.join(a.spec, "samples.json")))
        audio_gen.build(sp, P.put)
        print(f"audio: {len(sp['samples'])} samples ({time.time() - t0:.0f} s)")
    P.save(a.out)
    print(f"patch: {len(P.recs)} records, {sum(len(b) for _, b in P.recs) / 1e6:.2f} MB -> {a.out}")


if __name__ == "__main__":
    main()
