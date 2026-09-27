"""Clean-room painters: RGBA images from spec facts only (grid, 2-bit alpha outline, size)."""
import hashlib

import numpy as np
from scipy import ndimage

from cleanroom.decomp.gen import upsample_grid, unpack_alpha2


_RESEED = None


def reseed(tid):
    """Re-roll count for a frame (reseed.json): changes only its grain seed, recorded for audit."""
    global _RESEED
    if _RESEED is None:
        import json
        import os
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reseed.json")
        _RESEED = json.load(open(p)) if os.path.exists(p) else {}
    return _RESEED.get(tid, 0)


def h32(*parts):
    return int.from_bytes(hashlib.sha1("/".join(map(str, parts)).encode()).digest()[:4], "little")


def periodic_noise(seed, w, h, cell=4.0, octaves=2):
    """Value noise that wraps at the texture edges (map tiles repeat)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        c = max(1.0, cell / (2 ** o))
        gw, gh = max(1, int(round(w / c))), max(1, int(round(h / c)))
        lat = rng.standard_normal((gh, gw)).astype(np.float32)
        ys = np.arange(h, dtype=np.float32) * gh / h
        xs = np.arange(w, dtype=np.float32) * gw / w
        y0, x0 = ys.astype(int), xs.astype(int)
        fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
        fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        y1, x1 = (y0 + 1) % gh, (x0 + 1) % gw
        v = (lat[y0][:, x0] * (1 - fx) + lat[y0][:, x1] * fx) * (1 - fy) + \
            (lat[y1][:, x0] * (1 - fx) + lat[y1][:, x1] * fx) * fy
        out += amp * v
        tot += amp
        amp *= 0.5
    return out / tot


def base_colour(t):
    n = int(round(len(t["grid"]) ** 0.5))
    g = np.asarray(t["grid"], np.float32)
    return upsample_grid(g.tolist(), n, t["w"], t["h"])


def alpha(t):
    if "alpha2" not in t:
        return np.full((t["h"], t["w"]), 255, np.float32)
    return unpack_alpha2(t["alpha2"], t["w"], t["h"])


def tile(t):
    """Map tile / opaque texture: kept colour layout + wrapping grain."""
    w, h = t["w"], t["h"]
    img = base_colour(t)
    n = periodic_noise(h32("tile", t["id"], reseed(t["id"])), w, h, cell=max(2.0, min(w, h) / 8), octaves=3)
    img[..., :3] *= (1.0 + 0.10 * n)[..., None]
    img[..., 3] = alpha(t)
    return np.clip(img, 0, 255).astype(np.uint8)


def cutout(t):
    """Sprite: kept silhouette + colour layout, painted like a cel: dark outline, light from the top left."""
    w, h = t["w"], t["h"]
    a = alpha(t)
    solid = a >= 128
    img = base_colour(t)
    rgb = img[..., :3]
    if solid.any() and min(w, h) >= 6:
        inside = ndimage.distance_transform_edt(solid)
        # light from the top left: compare the distance to the edge up-left vs down-right
        sh = np.zeros_like(inside)
        sh[1:, 1:] = inside[:-1, :-1]
        light = np.clip((inside - sh) * 0.5, -1, 1)
        depth = np.clip(inside / max(2.0, min(w, h) / 6), 0, 1)
        rgb *= (0.86 + 0.14 * depth + 0.10 * light)[..., None]
        if "overlayScreens" in t["group"]:
            # UI panels: outline only the outer silhouette, not holes inside it (dash marks, cut-outs)
            outer = ndimage.distance_transform_edt(ndimage.binary_fill_holes(solid))
            edge = solid & (outer <= 1.0)
        else:
            edge = solid & (inside <= 1.0)
        rgb[edge] *= 0.45
    grain = periodic_noise(h32("cut", t["id"], reseed(t["id"])), w, h, cell=3.0, octaves=1)
    rgb *= (1.0 + 0.04 * grain)[..., None]
    img[..., 3] = np.where(solid, 255, 0)
    return np.clip(img, 0, 255).astype(np.uint8)


_GROUP_PAL = {}


def group_palette(group, frames, k=7):
    """A character's colour set: k-means over the kept grid colours of all its frames (opaque cells)."""
    if group in _GROUP_PAL:
        return _GROUP_PAL[group]
    from scipy.cluster.vq import kmeans2
    cols = np.asarray([c[:3] for f in frames for c in f["grid"] if c[3] >= 160], np.float32)
    if len(cols) < k:
        _GROUP_PAL[group] = None
        return None
    rng = np.random.default_rng(h32("gp", group))
    cent, _ = kmeans2(cols, k, iter=12, minit="++", seed=rng)
    _GROUP_PAL[group] = cent
    return cent


def cel(t, pal, main_bias=1.0):
    """Sprite drawn as cel regions: every texel snaps to the character's colour set, regions are
    cleaned with a mode filter, and region borders get a darker line (plus the silhouette outline)."""
    w, h = t["w"], t["h"]
    a = alpha(t)
    solid = a >= 128
    rgb = base_colour(t)[..., :3]
    # rendered sprites are shaded: match on colour normalised to the frame's own brightest cells
    g = np.asarray([c[:3] for c in t["grid"] if c[3] >= 128] or [c[:3] for c in t["grid"]], np.float32)
    top = max(40.0, float(np.percentile(g @ np.array([0.3, 0.59, 0.11], np.float32), 90)))
    ptop = float(np.max(pal @ np.array([0.3, 0.59, 0.11], np.float32)))
    rgbn = rgb * (ptop / top)
    d = ((rgbn[:, :, None, :] - pal[None, None]) ** 2).sum(-1)
    d[..., 0] *= main_bias          # the first colour is the body colour: mixtures lean toward it
    lab = d.argmin(-1)
    if min(w, h) >= 8:
        # mode filter: each texel takes the most common label in its 3x3 neighbourhood (inside the silhouette)
        votes = np.zeros((h, w, len(pal)), np.float32)
        for k in range(len(pal)):
            votes[..., k] = ndimage.uniform_filter(((lab == k) & solid).astype(np.float32), 3)
        lab = np.where(solid, votes.argmax(-1), lab)
    img = np.zeros((h, w, 4), np.float32)
    img[..., :3] = pal[lab]
    if solid.any() and min(w, h) >= 6:
        inside = ndimage.distance_transform_edt(solid)
        sh = np.zeros_like(inside)
        sh[1:, 1:] = inside[:-1, :-1]
        light = np.clip((inside - sh) * 0.5, -1, 1)
        img[..., :3] *= (0.92 + 0.12 * light)[..., None]
        # region borders: the darker side of a label change gets a line
        lum = (pal @ np.array([0.3, 0.59, 0.11], np.float32))[lab]
        border = np.zeros((h, w), bool)
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nb = np.roll(np.roll(lab, dy, 0), dx, 1)
            nl = np.roll(np.roll(lum, dy, 0), dx, 1)
            border |= (nb != lab) & (lum <= nl)
        img[border & solid, :3] *= 0.72
        img[solid & (inside <= 1.0), :3] *= 0.45
    grain = periodic_noise(h32("cel", t["id"], reseed(t["id"])), w, h, cell=3.0, octaves=1)
    img[..., :3] *= (1.0 + 0.03 * grain)[..., None]
    img[..., 3] = np.where(solid, 255, 0)
    return np.clip(img, 0, 255).astype(np.uint8)


def paint(t, frames=None):
    if t["group"].startswith("sprite/entitySprites") and "alpha2" in t and frames:
        pal = group_palette(t["group"], frames)
        if pal is not None:
            return cel(t, pal)
    if t["group"].startswith("map/") or t["group"].startswith("sprite/mapObjects"):
        from games.hm64 import materials
        if t["group"].endswith("/tile") or "alpha2" not in t:
            return materials.tile(t)
        return materials.cutout(t)
    if "alpha2" in t:
        return cutout(t)
    return tile(t)
