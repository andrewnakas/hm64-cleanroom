"""CLEAN ROOM: character sprites (entitySprites) painted from our own part briefs (sprite_briefs.json).

A brief names a character's part colours; zones along the frame's own silhouette (kept alpha) say which
parts may appear where (hat, head, torso, legs, shoes), and the kept colour grid picks between them per
texel. The head's skin region tells which way the character faces; we draw eyes on it. Colours far from
every part (tools, props held in the frame) keep the grid colour.

brief: {"hat": rgb|null, "hair": rgb, "skin": rgb, "top": rgb, "bottom": rgb, "shoes": rgb,
        "extra": [rgb, ...] (accents allowed on the torso), "hands": rgb (default skin),
        "head": 0.45 (head fraction of height incl. hat), "hat_h": 0.2, "legs": 0.28, "eyes": rgb}
"""
import json
import os

import numpy as np
from scipy import ndimage

from games.hm64 import paint

HERE = os.path.dirname(os.path.abspath(__file__))
_B = None


def briefs():
    global _B
    if _B is None:
        p = os.path.join(HERE, "sprite_briefs.json")
        _B = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    return _B


def brief_for(t):
    b = briefs()
    name = t["group"].rsplit("/", 1)[-1]
    br = b.get(name)
    if isinstance(br, str):
        br = b.get(br)
    return br


_F = None


def facing_of(t):
    global _F
    if _F is None:
        p = os.path.join(HERE, "spec", "facing.json")
        _F = json.load(open(p)) if os.path.exists(p) else {}
    e = _F.get(t["group"], {}).get(str(int(t["id"].rsplit("/", 1)[1])))
    return e["dir"] if e else None


def role_of(t):
    e = (_F or {}).get(t["group"], {}).get(str(int(t["id"].rsplit("/", 1)[1])))
    return e["role"] if e else "body"


def _near(c, a, b):
    """How much more c looks like a than like b (positive: a)."""
    c = np.asarray(c, np.float32)
    return float(((c - b) ** 2).sum() - ((c - a) ** 2).sum())


def render(t, br):
    w, h = t["w"], t["h"]
    a = paint.alpha(t)
    solid = a >= 128
    if not solid.any() or min(w, h) < 8:
        return paint.cutout(t)
    grid_rgb = paint.base_colour(t)[..., :3]
    ys, xs = np.nonzero(solid)
    y0, y1 = ys.min(), ys.max() + 1
    H = max(1, y1 - y0)
    yy = (np.arange(h)[:, None] - y0 + 0.5) / H * np.ones((1, w))
    f = lambda k, d: np.asarray(br.get(k, d), np.float32)
    skin, hair, top, bottom = f("skin", [248, 200, 160]), f("hair", [120, 80, 50]), f("top", [200, 60, 60]), f("bottom", [60, 90, 200])
    shoes, hands = f("shoes", [70, 50, 40]), f("hands", br.get("skin", [248, 200, 160]))
    hat = f("hat", [0, 0, 0]) if br.get("hat") else None
    head_h, hat_h, legs = br.get("head", 0.45), br.get("hat_h", 0.2) if br.get("hat") else 0.0, br.get("legs", 0.28)
    img = np.zeros((h, w, 3), np.float32)
    # zones
    zhead = solid & (yy < head_h)
    ztorso = solid & (yy >= head_h) & (yy < 1 - legs)
    zlegs = solid & (yy >= 1 - legs)
    img[ztorso] = top
    img[zlegs] = bottom
    img[solid & (yy >= 0.9)] = shoes
    # head: hat band, then hair with a face region where the grid says skin
    img[zhead] = hair
    if hat is not None:
        img[zhead & (yy < hat_h)] = hat
    face_zone = zhead & (yy >= hat_h + (head_h - hat_h) * (0.15 if hat is not None else 0.35))
    eyes = []
    if face_zone.any():
        fy, fx = np.nonzero(face_zone)
        fx0, fx1 = fx.min(), fx.max() + 1
        fdir = facing_of(t)
        cx_face = None
        mid = (fx0 + fx1) / 2
        if fdir == "S":
            cx_face, span, n_eyes = mid, 0.36, 2
        elif fdir in ("SW", "SE"):
            sgn = -1 if fdir == "SW" else 1
            cx_face, span, n_eyes = mid + sgn * (fx1 - fx0) * 0.1, 0.34, 2
        elif fdir in ("W", "E"):
            sgn = -1 if fdir == "W" else 1
            cx_face, span, n_eyes = mid + sgn * (fx1 - fx0) * 0.22, 0.26, 1
        if cx_face is not None:
            fyc = fy.min() + (fy.max() - fy.min()) * 0.55
            ry = (fy.max() - fy.min() + 1) * 0.55
            rx = (fx1 - fx0) * span
            Y, X = np.mgrid[0:h, 0:w]
            face = face_zone & (((X + 0.5 - cx_face) / max(rx, 1)) ** 2 + ((Y + 0.5 - fyc) / max(ry, 1)) ** 2 <= 1)
            img[face] = skin
            ey = int(round(fyc - ry * 0.05))
            if n_eyes == 2:
                off = max(1.0, rx * 0.45)
                eyes = [(ey, int(round(cx_face - off - 0.5))), (ey, int(round(cx_face + off - 0.5)))]
            else:
                side = -1 if cx_face < mid else 1
                eyes = [(ey, int(round(cx_face + side * rx * 0.35 - 0.5)))]
            eyes = [(y, x) for y, x in eyes if 0 <= y < h and 0 <= x < w and face[y, x]]
    # hands: the outer tips of arms (rows of the torso wider than the torso core)
    rows = [r for r in range(h) if ztorso[r].any()]
    if rows:
        widths = np.array([ztorso[r].sum() for r in rows])
        core = np.median(widths)
        for r, wd in zip(rows, widths):
            if wd > core + 3:
                cols = np.nonzero(ztorso[r])[0]
                img[r, cols[:2]] = hands
                img[r, cols[-2:]] = hands
    # collar accent just under the head
    if br.get("collar") is not None:
        c = np.asarray(br["collar"], np.float32)
        band = ztorso & (yy < head_h + 0.07)
        bx = np.nonzero(band.any(0))[0]
        if len(bx):
            mid = (bx.min() + bx.max()) / 2
            img[band & (np.abs(np.arange(w)[None, :] + 0.5 - mid) < (bx.max() - bx.min()) * 0.25)] = c
    # props (tools): low-saturation grid colours far from every part keep the grid colour
    parts = np.stack([p for p in (skin, hair, top, bottom, shoes, hands) + ((hat,) if hat is not None else ())])
    dmin = ((grid_rgb[:, :, None, :] - parts[None, None]) ** 2).sum(-1).min(-1)
    sat = grid_rgb.max(-1) - grid_rgb.min(-1)
    prop = solid & ~zhead & (dmin > 105.0 ** 2) & (sat < 30)
    img[prop] = grid_rgb[prop]
    lab = np.zeros((h, w), np.int64)
    flat = img.reshape(-1, 3)
    _, inv = np.unique(np.round(flat).astype(np.int32), axis=0, return_inverse=True)
    lab = inv.reshape(h, w)
    # shading, borders, outline
    inside = ndimage.distance_transform_edt(solid)
    sh = np.zeros_like(inside)
    sh[1:, 1:] = inside[:-1, :-1]
    light = np.clip((inside - sh) * 0.5, -1, 1)
    img *= (0.9 + 0.14 * light)[..., None]
    lum = img @ np.array([0.3, 0.59, 0.11], np.float32)
    border = np.zeros((h, w), bool)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nb = np.roll(np.roll(lab, dy, 0), dx, 1)
        nl = np.roll(np.roll(lum, dy, 0), dx, 1)
        border |= (nb != lab) & (lum < nl)
    img[border & solid] *= 0.72
    img[solid & (inside <= 1.0)] *= 0.42
    ec = np.asarray(br.get("eyes", [30, 24, 30]), np.float32)
    for y, x in eyes:
        img[y, x] = ec
        if y + 1 < h and zhead[y + 1, x] and H >= 30:
            img[y + 1, x] = ec
    grain = paint.periodic_noise(paint.h32("chibi", t["id"], paint.reseed(t["id"])), w, h, cell=3.0, octaves=1)
    img *= (1.0 + 0.025 * grain)[..., None]
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = np.clip(img, 0, 255)
    out[..., 3] = np.where(solid, 255, 0)
    return out


def image(t):
    if not t["group"].startswith("sprite/entitySprites/") or "alpha2" not in t:
        return None
    br = brief_for(t)
    if not br:
        return None
    facing_of(t)
    if role_of(t) == "part":
        return None
    if "cel" in br:
        return paint.cel(t, np.asarray(br["cel"], np.float32), br.get("bias", 0.5))
    return render(t, br)
