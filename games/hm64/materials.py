"""CLEAN ROOM: map tile textures as tiling materials chosen from the kept colour grid.

Each grid cell's kept colour is classified (grass, soil, wood, stone, roof, plaster, glass, water, dark);
every class has a tiling pattern (our own procedural code) that modulates the kept colour. Class weights
are interpolated between cells so neighbouring materials blend like the grid's own colour layout.
"""
import colorsys

import numpy as np

from games.hm64 import paint

CLASSES = ["grass", "soil", "wood", "stone", "roof", "plaster", "glass", "water", "dark"]


def classify(rgb):
    r, g, b = [v / 255.0 for v in rgb[:3]]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h *= 360
    if l < 0.12:
        return "dark"
    if l > 0.8 and s < 0.35:
        return "plaster"
    if 60 <= h < 160 and s > 0.2:
        return "grass"
    if 180 <= h < 250 and s > 0.25:
        return "glass" if l > 0.55 else ("water" if s > 0.45 else "roof")
    if (h >= 330 or h < 15) and s > 0.3:
        return "roof"
    if s < 0.14:
        return "stone"
    if 15 <= h < 60:
        return "wood" if l < 0.36 else "soil"
    return "soil"


def _rng(t, name):
    return np.random.default_rng(paint.h32("mat", name, t["id"], paint.reseed(t["id"])))


def _wrapdist(a, b, n):
    d = np.abs(a - b) % n
    return np.minimum(d, n - d)


def pattern(name, t, w, h):
    """Luminance modulation in about [-1, 1] that tiles at (w, h)."""
    rng = _rng(t, name)
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    if name == "grass":
        m = paint.periodic_noise(rng.integers(1 << 30), w, h, cell=2.0, octaves=2) * 0.6
        for _ in range(max(4, w * h // 12)):
            x, y = rng.integers(0, w), rng.integers(0, h)
            ln = rng.integers(2, 4)
            v = rng.choice([-1.0, 1.0])
            for k in range(ln):
                m[(y - k) % h, (x + (k // 2) * int(rng.integers(-1, 2))) % w] += 0.9 * v
        return np.clip(m, -1.2, 1.2)
    if name == "soil":
        m = paint.periodic_noise(rng.integers(1 << 30), w, h, cell=3.0, octaves=2) * 0.7
        for _ in range(max(2, w * h // 40)):
            x, y = rng.integers(0, w), rng.integers(0, h)
            m[y % h, x % w] -= 1.0
            m[y % h, (x + 1) % w] += 0.5
        return m
    if name == "wood":
        ph = max(4, h // max(1, round(h / 6)))                          # plank height dividing h
        row = (Y // ph).astype(int)
        m = paint.periodic_noise(rng.integers(1 << 30), w, h, cell=6.0, octaves=1) * 0.3
        m += 0.35 * np.sin(2 * np.pi * (X / w * max(1, w // 8) + row * 0.37))  # grain along the plank
        m[(Y % ph) == 0] -= 1.2                                          # gaps between planks
        for r in range(h // ph + 1):                                     # butt joints, staggered
            x = int((r * 0.61 % 1) * w)
            m[r * ph:(r + 1) * ph, x % w] -= 0.8
        return m
    if name == "stone":
        n = max(2, (w * h) // 64)
        px, py = rng.uniform(0, w, n), rng.uniform(0, h, n)
        d1 = np.full((h, w), 1e9, np.float32)
        d2 = np.full((h, w), 1e9, np.float32)
        tone = np.zeros((h, w), np.float32)
        tones = rng.uniform(-0.4, 0.4, n)
        for k in range(n):
            d = np.hypot(_wrapdist(X, px[k], w), _wrapdist(Y, py[k], h))
            closer = d < d1
            d2 = np.where(closer, d1, np.minimum(d2, d))
            tone = np.where(closer, tones[k], tone)
            d1 = np.minimum(d1, d)
        m = tone + 0.3 * paint.periodic_noise(rng.integers(1 << 30), w, h, cell=2.0, octaves=1)
        m[(d2 - d1) < 1.0] -= 1.3                                       # grout
        return m
    if name == "roof":
        rh = max(3, h // max(1, round(h / 5)))
        tw = max(4, w // max(1, round(w / 6)))
        row = (Y // rh).astype(int)
        xo = (X + (row % 2) * tw / 2) % tw
        m = -0.8 * ((Y % rh) / rh) + 0.4                                 # each shingle row darker at its bottom
        m[(Y % rh) == rh - 1] -= 0.8
        m[np.abs(xo - tw / 2) < 0.5] -= 0.6
        return m
    if name == "plaster":
        return paint.periodic_noise(rng.integers(1 << 30), w, h, cell=4.0, octaves=2) * 0.25
    if name == "glass":
        m = np.zeros((h, w), np.float32)
        m[(X % max(4, w // 2)) < 1] -= 1.0
        m[(Y % max(4, h // 2)) < 1] -= 1.0
        m += 0.6 * (((X + Y) % max(6, w // 2)) < 2)                      # glints
        return m
    if name == "water":
        return 0.6 * np.sin(2 * np.pi * (X / w * 2 + np.sin(2 * np.pi * Y / h) * 0.5)) * \
            paint.periodic_noise(rng.integers(1 << 30), w, h, cell=4.0, octaves=1)
    return paint.periodic_noise(rng.integers(1 << 30), w, h, cell=3.0, octaves=1) * 0.2


AMOUNT = {"grass": 0.16, "soil": 0.13, "wood": 0.2, "stone": 0.22, "roof": 0.22, "plaster": 0.06,
          "glass": 0.25, "water": 0.12, "dark": 0.1}


def modulation(t):
    w, h = t["w"], t["h"]
    n = int(round(len(t["grid"]) ** 0.5))
    cls = [classify(c) if c[3] >= 128 else "plaster" for c in t["grid"]]
    mod = np.zeros((h, w), np.float32)
    for c in sorted(set(cls)):
        onehot = [[255.0 * (k == c)] * 4 for k in cls]
        wt = np.clip(paint.upsample_grid(onehot, n, w, h)[..., 0] / 255.0, 0, 1)
        mod += wt * pattern(c, t, w, h) * AMOUNT[c]
    return mod


def cutout(t):
    """Map object with a kept silhouette: the cel cutout, textured by material."""
    img = paint.cutout(t).astype(np.float32)
    img[..., :3] *= (1.0 + modulation(t))[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def tile(t):
    w, h = t["w"], t["h"]
    n = int(round(len(t["grid"]) ** 0.5))
    cls = [classify(c) for c in t["grid"]]
    # per-class weight maps: one-hot per cell, bilinearly interpolated like the colours
    used = sorted(set(cls))
    img = paint.base_colour(t)
    mod = np.zeros((h, w), np.float32)
    for c in used:
        onehot = [[255.0 * (k == c)] * 4 for k in cls]
        wt = np.clip(paint.upsample_grid(onehot, n, w, h)[..., 0] / 255.0, 0, 1)
        mod += wt * pattern(c, t, w, h) * AMOUNT[c]
    img[..., :3] *= (1.0 + mod)[..., None]
    img[..., 3] = paint.alpha(t)
    return np.clip(img, 0, 255).astype(np.uint8)
