"""CLEAN ROOM: album photos composed from our own clean character sprites (photo_briefs.json).

Brief: {"cast": [[sprite group, facing "S"|"SW"|"SE"|"W"|"E", x centre, feet y, height], ...] (normalised),
        "bg": "grid" (the photo's kept colour layout, softened) | [top rgb, bottom rgb],
        "ops": [{"e": [cx, cy, rx, ry], "c": rgb} | {"rect": [x0, y0, x1, y1], "c": rgb} | {"dots": n, "c": rgb}],
        "tint": "sepia" | null, "border": true}
The kept alpha outline stays the photo's shape; the kept grid colours the background.
"""
import json
import os

import numpy as np
from PIL import Image

from games.hm64 import chibi, paint

HERE = os.path.dirname(os.path.abspath(__file__))
_B = None
_T = None


def briefs():
    global _B
    if _B is None:
        p = os.path.join(HERE, "photo_briefs.json")
        _B = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    return _B


def textures():
    global _T
    if _T is None:
        _T = {t["id"]: t for t in json.load(open(os.path.join(HERE, "spec", "textures.json")))["textures"]}
    return _T


def frame_for(group, facing):
    """A body frame of that character facing that way (mirrored from the opposite side if needed)."""
    chibi.facing_of({"group": "", "id": "x/0"})
    F = chibi._F
    key = next((g for g in F if g.rsplit("/", 1)[-1] == group), None)
    if key is None:
        return None, False
    want, flip = facing, False
    cand = [int(k) for k, v in F[key].items() if v["dir"] == want and v["role"] == "body"]
    if not cand:
        mirror = {"E": "W", "W": "E", "SE": "SW", "SW": "SE"}.get(want)
        cand = [int(k) for k, v in F[key].items() if v["dir"] == mirror and v["role"] == "body"]
        flip = bool(cand)
    if not cand:
        cand = [int(k) for k, v in F[key].items() if v["role"] == "body"]
    if not cand:
        return None, False
    tid = f"{key}/{min(cand):03d}"
    t = textures().get(tid)
    if t is None:
        return None, False
    img = chibi.image(t)
    if img is None:
        img = paint.paint(t)
    return img, flip


def render(t, br):
    w, h = t["w"], t["h"]
    a = paint.alpha(t)
    bg = br.get("bg", "grid")
    if bg == "cutout":
        img = paint.cutout(t).astype(np.float32)[..., :3]
    elif bg == "grid":
        img = paint.base_colour(t)[..., :3]
        img *= (1 + 0.05 * paint.periodic_noise(paint.h32("photo", t["id"]), w, h, cell=5.0, octaves=2))[..., None]
    else:
        top, bot = np.asarray(bg[0], np.float32), np.asarray(bg[1], np.float32)
        f = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        img = np.broadcast_to(top * (1 - f) + bot * f, (h, w, 3)).copy()
    canvas = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGBA")
    Y, X = np.mgrid[0:h, 0:w]
    for op in br.get("ops", []):
        arr = np.asarray(canvas).copy()
        c = op.get("c", [255, 255, 255])
        if "e" in op:
            cx, cy, rx, ry = op["e"]
            m = ((X + 0.5) / w - cx) ** 2 / rx ** 2 + ((Y + 0.5) / h - cy) ** 2 / ry ** 2 <= 1
        elif "rect" in op:
            x0, y0, x1, y1 = op["rect"]
            m = ((X + 0.5) / w >= x0) & ((X + 0.5) / w < x1) & ((Y + 0.5) / h >= y0) & ((Y + 0.5) / h < y1)
        elif "dots" in op:
            rng = np.random.default_rng(paint.h32("dots", t["id"]))
            m = np.zeros((h, w), bool)
            m[rng.integers(0, h, op["dots"]), rng.integers(0, w, op["dots"])] = True
        else:
            continue
        arr[m, :3] = c
        canvas = Image.fromarray(arr)
    for group, facing, cx, fy, ht in br.get("cast", []):
        spr, flip = frame_for(group, facing)
        if spr is None:
            continue
        im = Image.fromarray(spr)
        bb = im.getbbox()
        if bb:
            im = im.crop(bb)
        if flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        th = max(4, int(round(ht * h)))
        tw = max(2, int(round(im.width * th / im.height)))
        im = im.resize((tw, th), Image.NEAREST if th >= im.height else Image.LANCZOS)
        canvas.alpha_composite(im, (int(round(cx * w - tw / 2)), int(round(fy * h - th))))
    out = np.asarray(canvas).astype(np.float32)
    if br.get("tint") == "sepia":
        l = out[..., :3] @ np.array([0.3, 0.59, 0.11], np.float32)
        out[..., 0], out[..., 1], out[..., 2] = l * 1.07 + 18, l * 0.95 + 8, l * 0.78
    if br.get("border", True):
        solid = a >= 128
        inside = paint.ndimage.distance_transform_edt(solid)
        out[solid & (inside <= 3), :3] = [246, 244, 236]
        out[solid & (inside <= 1), :3] = [190, 186, 176]
    out[..., 3] = np.where(a >= 128, 255, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def image(t):
    br = briefs().get(t["id"])
    return render(t, br) if br else None
