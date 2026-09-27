"""CLEAN ROOM: dialogue portraits (characterAvatars, 48x48) painted from our own briefs.

portrait_briefs.json:
  "characters": {name: {"hair": rgb, "skin": rgb, "iris": rgb, "face": [cx, cy, rx, ry],
                        "eyes": [[x, y], [x, y]], "eye_r": [rx, ry], "mouth": [x, y],
                        "fringe": y, "bangs": n, "glasses": rgb|null, "beard": rgb|null, "extra": [facepaint ops]}}
  "frames": {"000": {"who": name, "eyes": "open|closed|happy|wink|sad|angry|wide|surprised|tired",
                     "mouth": "smile|open|grin|flat|frown|o|laugh|pout|none", "blush": bool, "tears": bool,
                     "sweat": bool, "dx": shift, "dy": shift, ...character keys may be overridden}}
Coordinates are normalised (0..1, y down). The kept 2-bit alpha outline is the silhouette and the kept
colour grid colours hair/clothes outside the face; everything in the face is drawn here.
"""
import json
import math
import os

import numpy as np

from cleanroom.gfx.facepaint import Canvas
from games.hm64 import paint

HERE = os.path.dirname(os.path.abspath(__file__))
_B = None


def briefs():
    global _B
    if _B is None:
        p = os.path.join(HERE, "portrait_briefs.json")
        _B = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"characters": {}, "frames": {}}
    return _B


def _mix(a, b, t):
    return [a[i] * (1 - t) + b[i] * t for i in range(3)]


def _dark(c, k=0.55):
    return [v * k for v in c[:3]]


def eye(cv, x, y, rx, ry, style, iris, lash, flip=False):
    """One anime eye centred at (x, y) with half-size (rx, ry)."""
    lw = ry * 0.28
    if style in ("closed", "tired_closed"):
        d = cv.seg_dist([(x - rx, y), (x - rx * 0.4, y + ry * 0.35), (x + rx * 0.4, y + ry * 0.35), (x + rx, y)])
        cv.paint(d < lw, lash)
        return
    if style == "dot":            # small bead eyes (babies, animals)
        cv.paint(cv.ell(x, y, rx * 0.55, ry * 0.7) <= 1, (24, 18, 22))
        cv.paint(cv.ell(x - rx * 0.15, y - ry * 0.25, rx * 0.2, ry * 0.2) <= 1, (255, 255, 255))
        return
    if style == "happy":
        d = cv.seg_dist([(x - rx, y + ry * 0.3), (x, y - ry * 0.45), (x + rx, y + ry * 0.3)])
        cv.paint(d < lw, lash)
        return
    if style == "shut":           # squeezed shut (> <)
        s = -1 if flip else 1
        d = cv.seg_dist([(x - rx * s, y - ry * 0.5), (x + rx * 0.7 * s, y), (x - rx * s, y + ry * 0.5)])
        cv.paint(d < lw, lash)
        return
    scale_i = {"wide": 0.55, "surprised": 0.45}.get(style, 1.0)
    # sclera
    cv.paint(cv.ell(x, y + ry * 0.05, rx, ry) <= 1, (255, 255, 252))
    # iris + pupil (looking slightly toward the viewer's left, the 3/4 view direction)
    ix, iy = x - rx * 0.12, y + ry * 0.12
    irx, iry = rx * 0.72 * scale_i, ry * 0.82 * scale_i
    clipm = cv.ell(x, y + ry * 0.05, rx, ry) <= 1
    cv.paint((cv.ell(ix, iy, irx, iry) <= 1) & clipm, _dark(iris, 0.75))
    cv.paint((cv.ell(ix, iy + iry * 0.2, irx * 0.75, iry * 0.65) <= 1) & clipm, iris)
    cv.paint((cv.ell(ix, iy, irx * 0.42, iry * 0.48) <= 1) & clipm, (24, 18, 22))
    cv.paint(cv.ell(ix - irx * 0.35, iy - iry * 0.35, rx * 0.22, ry * 0.2) <= 1, (255, 255, 255))
    # upper lash line, heavier toward the outer corner
    top = [(x - rx * 1.05, y - ry * 0.35), (x - rx * 0.5, y - ry * 0.95), (x + rx * 0.4, y - ry * 1.0),
           (x + rx * 1.08, y - ry * 0.55)]
    cv.paint(cv.seg_dist(top) < lw * 1.2, lash)
    if style == "angry":
        s = 1 if not flip else -1
        cv.paint(cv.seg_dist([(x - rx * 1.1 * s, y - ry * 1.25), (x + rx * 1.0 * s, y - ry * 0.75)]) < lw * 1.3, lash)
    if style == "sad":
        s = 1 if not flip else -1
        cv.paint(cv.seg_dist([(x - rx * 1.0 * s, y - ry * 0.85), (x + rx * 1.0 * s, y - ry * 1.35)]) < lw * 1.1, lash)
    if style == "tired":
        cv.paint((cv.y < y - ry * 0.15) & (cv.ell(x, y, rx * 1.15, ry * 1.2) <= 1), _dark(lash, 1.0))


def mouth(cv, x, y, w, style, skin):
    ink = (110, 40, 40)
    lw = 0.018
    if style == "none":
        return
    if style == "smile":
        cv.paint(cv.seg_dist([(x - w, y - w * 0.3), (x, y + w * 0.35), (x + w, y - w * 0.3)]) < lw, ink)
    elif style == "flat":
        cv.paint(cv.seg_dist([(x - w * 0.8, y), (x + w * 0.8, y)]) < lw, ink)
    elif style == "frown":
        cv.paint(cv.seg_dist([(x - w, y + w * 0.35), (x, y - w * 0.25), (x + w, y + w * 0.35)]) < lw, ink)
    elif style == "pout":
        cv.paint(cv.seg_dist([(x - w * 0.5, y), (x + w * 0.5, y + w * 0.2)]) < lw, ink)
    elif style in ("open", "laugh", "grin", "o"):
        ry = {"open": w * 0.8, "laugh": w * 1.0, "grin": w * 0.55, "o": w * 0.75}[style]
        rx = {"open": w * 0.8, "laugh": w * 1.1, "grin": w * 1.2, "o": w * 0.55}[style]
        m = cv.ell(x, y, rx, ry) <= 1
        if style in ("laugh", "grin"):
            m = m & (cv.y > y - ry * 0.3)
        cv.paint(m, (120, 30, 40))
        cv.paint(m & (cv.ell(x, y + ry * 0.55, rx * 0.7, ry * 0.45) <= 1), (230, 110, 120))
        if style == "grin":
            cv.paint(m & (cv.y < y), (255, 255, 250))


def _op(cv, op, dx, dy):
    c = op.get("c", (0, 0, 0))
    if "e" in op:
        x, y, rx, ry = op["e"]
        cv.paint(cv.ell(x + dx, y + dy, rx, ry, op.get("rot", 0)) <= 1, c)
    elif "line" in op:
        cv.paint(cv.seg_dist([(x + dx, y + dy) for x, y in op["line"]]) < op.get("w", 0.015), c)
    elif "poly" in op:
        from cleanroom.gfx.facepaint import _poly_mask
        cv.paint(_poly_mask(cv, [(x + dx, y + dy) for x, y in op["poly"]]), c)


def render(t, fb):
    B = briefs()
    ch = dict(B["characters"].get(fb.get("who", ""), {}))
    ch.update({k: v for k, v in fb.items() if k != "who"})
    w, h = t["w"], t["h"]
    a = paint.alpha(t)
    solid = a >= 128
    base = paint.cutout(t).astype(np.float32)[..., :3]
    cv = Canvas(w, h, (0, 0, 0))
    cv.img = base.repeat(4, 0).repeat(4, 1)
    dx, dy = ch.get("dx", 0.0), ch.get("dy", 0.0)
    skin = ch.get("skin", [248, 204, 160])
    hair = ch.get("hair", [120, 80, 50])
    iris = ch.get("iris", [90, 60, 40])
    lash = _dark(hair, 0.35) if sum(hair) > 200 else (20, 14, 16)
    fx, fy, frx, fry = ch.get("face", [0.45, 0.66, 0.24, 0.26])
    fx, fy = fx + dx, fy + dy
    sil = np.asarray(solid, np.float32).repeat(4, 0).repeat(4, 1) > 0.5
    inside = paint.ndimage.distance_transform_edt(sil) / (4.0 * max(w, h))
    # hair: flat colour, darker toward the outline, a shine band across the crown, a few strands
    cv.paint(sil, hair)
    cv.paint(sil & (inside < 0.05), _dark(hair, 0.78))
    shine = cv.ell(fx + 0.02, fy - fry * 1.25, frx * 1.35, fry * 0.75)
    cv.paint(sil & (shine <= 1) & (shine >= 0.62) & (cv.y < fy - fry * 1.0), _mix(hair, (255, 255, 240), 0.35))
    ns = int(ch.get("strands", 3))
    for k in range(ns):
        ang = -0.7 + 1.4 * k / max(1, ns - 1)
        x0, y0 = fx + 0.02 + 0.08 * ang, fy - fry * 1.75
        pts = [(x0 + math.sin(ang) * r * 0.7, y0 + r) for r in np.linspace(0.06, 0.26, 4)]
        cv.paint(sil & (cv.seg_dist(pts) < 0.006), _dark(hair, 0.8))
    # clothes: the band below the chin
    cy0 = ch.get("clothes_y", fy + fry * 0.95)
    clothes = ch.get("clothes")
    if clothes is None:
        g = np.asarray(t["grid"], np.float32)
        n = int(round(len(g) ** 0.5))
        row = [c for c in g.reshape(n, n, 4)[-1] if c[3] >= 128]
        clothes = list(np.mean(row, 0)[:3]) if row else [90, 110, 170]
    cv.paint(sil & (cv.y > cy0), clothes)
    cv.paint(sil & (np.abs(cv.y - cy0) < 0.012), _dark(clothes, 0.6))
    # face: an oval with a narrower chin
    fm = cv.ell(fx, fy, frx, fry)
    chin = cv.ell(fx - frx * 0.1, fy + fry * 0.35, frx * 0.72, fry * 0.72)
    face = np.minimum(fm, np.where(cv.y > fy, chin, fm)) <= 1
    cv.paint(face, _mix(skin, (255, 255, 255), 0.05))
    # soft cheek shading on the far side
    cv.paint(face & (cv.ell(fx + frx * 0.75, fy + fry * 0.2, frx * 0.5, fry * 0.8) <= 1), _mix(skin, (200, 120, 90), 0.18))
    # fringe: hair over the top of the face with n bangs
    fr = ch.get("fringe", fy - fry * 0.35) + dy
    n = max(1, int(ch.get("bangs", 4)))
    xs = np.clip((cv.x - (fx - frx * 1.1)) / (frx * 2.2), 0, 1)
    wave = fr + 0.045 * np.abs(np.sin(xs * math.pi * n))
    cv.paint(face & (cv.y < wave), hair)
    cv.paint(face & (np.abs(cv.y - wave) < 0.012), _dark(hair, 0.6))
    if ch.get("beard"):
        cv.paint(face & (cv.y > fy + fry * 0.3), ch["beard"])
    # eyes
    if ch.get("hat"):
        hc, hy = ch["hat"]
        cv.paint(sil & (cv.y < hy + dy), hc)
        cv.paint(sil & (np.abs(cv.y - hy - dy) < 0.012), _dark(hc, 0.6))
    ex = ch.get("eyes_at", [[fx - frx * 0.42, fy - 0.02], [fx + frx * 0.38, fy - 0.03]])
    erx, ery = ch.get("eye_r", [0.06, 0.06] if ch.get("male") else [0.07, 0.08])
    style = ch.get("eyes", "open")
    for k, (x, y) in enumerate(ex):
        s = style
        if style == "wink":
            s = "closed" if k == 1 else "open"
        # the far eye (viewer's right in the 3/4 view) is narrower
        sx = erx * (0.85 if k == 1 else 1.0)
        eye(cv, x + dx, y + dy, sx, ery, s, iris, lash, flip=(k == 1))
        brow_y = y + dy - ery * 1.7
        bs = ch.get("brows", {"angry": "angry", "sad": "sad", "surprised": "raised"}.get(style, "normal"))
        tilt = {"angry": 0.03, "sad": -0.03, "raised": 0.0, "normal": 0.0}[bs] * (1 if k == 0 else -1)
        lift = -0.02 if bs == "raised" else 0.0
        if brow_y > wave.min() if isinstance(wave, np.ndarray) else True:
            cv.paint(cv.seg_dist([(x + dx - sx, brow_y + lift - tilt), (x + dx + sx, brow_y + lift + tilt)]) < (0.013 if ch.get("male") else 0.008),
                     _dark(hair, 0.5))
    if ch.get("glasses"):
        g = ch["glasses"]
        for k, (x, y) in enumerate(ex):
            sx = erx * (0.85 if k == 1 else 1.0)
            ring = cv.ell(x + dx, y + dy, sx * 1.45, ery * 1.25)
            cv.paint((ring <= 1) & (ring >= 0.72), g)
        cv.paint(cv.seg_dist([(ex[0][0] + dx + erx * 1.4, ex[0][1] + dy), (ex[1][0] + dx - erx * 1.2, ex[1][1] + dy)]) < 0.012, g)
    if ch.get("sunglasses"):
        g = ch["sunglasses"]
        for k, (x, y) in enumerate(ex):
            cv.paint(cv.ell(x + dx, y + dy, erx * 1.5, ery * 1.1) <= 1, g)
            cv.paint(cv.ell(x + dx - erx * 0.5, y + dy - ery * 0.4, erx * 0.35, ery * 0.2) <= 1, _mix(g, (255, 255, 255), 0.5))
        cv.paint(cv.seg_dist([(ex[0][0] + dx, ex[0][1] + dy), (ex[1][0] + dx, ex[1][1] + dy)]) < 0.012, g)
    if ch.get("blush"):
        for (x, y) in ex:
            cv.paint(cv.ell(x + dx, y + dy + ery * 1.6, erx * 0.8, ery * 0.35) <= 1, _mix(skin, (240, 90, 100), 0.45))
    if ch.get("tears"):
        x, y = ex[0]
        cv.paint(cv.ell(x + dx - erx * 0.3, y + dy + ery * 1.6, erx * 0.3, ery * 0.6) <= 1, (140, 200, 255))
    if ch.get("sweat"):
        cv.paint(cv.ell(fx + frx * 0.9, fy - fry * 0.4, 0.03, 0.05) <= 1, (160, 210, 255))
    # nose hint and mouth
    mx, my = ch.get("mouth_at", [fx - frx * 0.05, fy + fry * 0.62])
    cv.paint(cv.seg_dist([(mx - 0.01 + dx, my + dy - fry * 0.32), (mx + dx, my + dy - fry * 0.26)]) < 0.01, _mix(skin, (150, 80, 60), 0.5))
    if ch.get("mustache"):
        mc = ch["mustache"]
        for sgn in (-1, 1):
            cv.paint(cv.ell(mx + dx + sgn * 0.045, my + dy - fry * 0.12, 0.05, 0.028, sgn * 12) <= 1, mc)
    mouth(cv, mx + dx, my + dy, ch.get("mouth_w", 0.045), ch.get("mouth", "smile"), skin)
    for op in ch.get("extra", []):
        _op(cv, op, dx, dy)
    img = np.zeros((h, w, 4), np.float32)
    img[..., :3] = cv.result()
    img[..., 3] = np.where(solid, 255, 0)
    # keep the dark silhouette edge from the cutout
    edge = solid & (paint.ndimage.distance_transform_edt(solid) <= 1.0)
    img[edge, :3] = base[edge] if False else img[edge, :3] * 0.55
    return np.clip(img, 0, 255).astype(np.uint8)


def image(t):
    if "characterAvatars" not in t["id"]:
        return None
    fb = briefs()["frames"].get(t["id"].rsplit("/", 1)[1])
    return render(t, fb) if fb else None
