"""CLEAN ROOM: re-typeset text-bearing sprites (labels.json: our own transcriptions).

Entry: {"text": "Notebook" | ["line 1", "line 2"], "mode": "plate" | "text",
        "ink": [r,g,b] (default: contrast with the plate), "outline": [r,g,b] | null,
        "box": [x0, y0, x1, y1] (default: opaque bounding box, inset), "align": "center" | "left",
        "weight": stroke thickness factor (default 1)}
plate: the sprite's plate is painted from its facts (paint.cutout) and the text drawn on it.
text:  the sprite is only text: alpha = our text + outline (the kept outline is not used).
"""
import json
import os

import numpy as np
from scipy import ndimage

from cleanroom.gfx import strokefont
from games.hm64 import paint

HERE = os.path.dirname(os.path.abspath(__file__))
_LABELS = None


def labels():
    global _LABELS
    if _LABELS is None:
        _LABELS = {}
        for name in ("labels_auto.json", "labels.json"):     # hand entries override the batch transcriptions
            p = os.path.join(HERE, name)
            if os.path.exists(p):
                _LABELS.update(json.load(open(p, encoding="utf-8")))
        _LABELS = {k: v for k, v in _LABELS.items() if not k.startswith("_")}
    return _LABELS


def _line(text, h, w, weight):
    th = max(0.55, h * 0.085) * weight
    m = strokefont.render_line(text, h, thickness=th)
    # trim empty columns
    cols = np.nonzero(m.max(0) > 0.05)[0]
    if len(cols):
        m = m[:, cols[0]:cols[-1] + 1]
    if m.shape[1] > w:
        xs = np.linspace(0, m.shape[1] - 1, w)
        x0 = np.floor(xs).astype(int)
        x1 = np.minimum(x0 + 1, m.shape[1] - 1)
        f = (xs - x0)[None, :]
        m = m[:, x0] * (1 - f) + m[:, x1] * f
    return m


def text_mask(lines, w, h, box, align="center", weight=1.0, sizes=None):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    n = len(lines)
    sizes = sizes or [1.0] * n
    tops = np.concatenate([[0], np.cumsum(sizes)]) / sum(sizes) * bh
    out = np.zeros((h, w), np.float32)
    for i, t in enumerate(lines):
        lh = tops[i + 1] - tops[i]
        hh = max(5, int(round(lh)))
        m = _line(t, hh, bw, weight)
        top = int(round(y0 + tops[i] + (lh - hh) / 2))
        left = x0 + (0 if align == "left" else (bw - m.shape[1]) // 2)
        ys, xs = slice(max(0, top), min(h, top + hh)), slice(max(0, left), min(w, left + m.shape[1]))
        sub = m[ys.start - top:ys.stop - top, xs.start - left:xs.stop - left]
        out[ys, xs] = np.maximum(out[ys, xs], sub)
    return out


def default_box(t, alpha, mode):
    solid = alpha >= 128
    if not solid.any():
        return [0, 0, t["w"] - 1, t["h"] - 1]
    ys, xs = np.nonzero(solid)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    if mode == "text":       # the text sits where the kept outline is, less the outline ring
        return [int(x0 + 1), int(y0 + 1), int(x1 - 1), int(y1 - 1)]
    ix, iy = max(1, (x1 - x0) // 12), max(1, (y1 - y0) // 7)
    return [int(x0 + ix), int(y0 + iy), int(x1 - ix), int(y1 - iy)]


def render(t, lab):
    lines = lab["text"] if isinstance(lab["text"], list) else [lab["text"]]
    mode = lab.get("mode", "plate")
    a = paint.alpha(t)
    box = lab.get("box") or default_box(t, a, mode)
    m = text_mask(lines, t["w"], t["h"], box, lab.get("align", "center"), lab.get("weight", 1.0), lab.get("sizes"))
    if mode == "text":
        img = np.zeros((t["h"], t["w"], 4), np.float32)
        ink = lab.get("ink", "grid")
        if ink == "grid":        # the frame's own kept colour (mean of opaque grid cells), lifted if dark
            g = np.asarray([c for c in t["grid"] if c[3] >= 128] or t["grid"], np.float32)[:, :3].mean(0)
            ink = g * 1.25 + 20 if g @ [0.3, 0.59, 0.11] < 90 else g
        ink = np.clip(np.asarray(ink, np.float32), 0, 255)
        ol = lab.get("outline", [40, 28, 20])
        core = m > 0.45
        if ol is not None:
            ring = ndimage.binary_dilation(core, iterations=1) & ~core
            img[ring, :3] = ol
            img[ring, 3] = 255
        img[core, :3] = ink
        img[core, 3] = 255
        return img.astype(np.uint8)
    img = paint.cutout(t).astype(np.float32)
    solid = img[..., 3] > 0
    if lab.get("plate") in ("wood", "stone", "roof"):
        from games.hm64 import materials
        img[..., :3] *= (1.0 + 0.22 * materials.pattern(lab["plate"], t, t["w"], t["h"]))[..., None]
    plate = img[solid][:, :3].mean(0) if solid.any() else np.array([128, 128, 128])
    lum = plate @ [0.3, 0.59, 0.11]
    ink = np.asarray(lab.get("ink", [48, 32, 20] if lum > 150 else [255, 252, 240]), np.float32)
    ol = lab.get("outline", None if lum > 150 else [36, 24, 16])
    core = m > 0.45
    if ol is not None:
        ring = ndimage.binary_dilation(core, iterations=1) & ~core & solid
        img[ring, :3] = ol
    # soften: blend by coverage for the anti-aliased edge
    cov = np.clip(m, 0, 1)[..., None]
    img[..., :3] = np.where(solid[..., None], img[..., :3] * (1 - cov) + ink * cov, img[..., :3])
    return np.clip(img, 0, 255).astype(np.uint8)


def image(t):
    lab = labels().get(t["id"])
    return render(t, lab) if lab else None
