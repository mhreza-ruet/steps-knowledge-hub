"""VLM spec + pixel detection -> data. No per-figure settings.

pipeline(image, spec) does:
  1. frame, ticks and tick-label positions (detect.py)
  2. axis calibration from the VLM's tick values (calibrate.py)
  3. text / legend detection inside the plot -> areas to ignore, and the
     exact colour of every series read from its legend sample
  4. per series: colour mask -> markers (blob centres) or a traced line
  5. pixels -> data values
"""
from __future__ import annotations

import colorsys

import numpy as np
from scipy import ndimage as ndi

from . import detect as D
from .calibrate import AxisMap, calibrate_axis, map_for_pixel

# ----------------------------------------------------------------- colour names
HUES = {"red": 0, "crimson": 350, "orange": 30, "brown": 25, "gold": 48, "yellow": 55, "olive": 60, "lime": 100,
        "green": 120, "dark green": 120, "teal": 175, "cyan": 185, "turquoise": 175, "light blue": 200, "sky blue": 200,
        "blue": 230, "navy": 230, "dark blue": 230, "royal blue": 225, "indigo": 260, "purple": 280, "violet": 275,
        "magenta": 300, "fuchsia": 300, "pink": 320, "hot pink": 325, "rose": 340,
        # less common names VLMs use
        "coral": 8, "salmon": 8, "tomato": 9, "scarlet": 5, "maroon": 0, "burgundy": 345, "wine": 345,
        "peach": 28, "apricot": 28, "tan": 34, "beige": 38, "ochre": 40, "amber": 45, "mustard": 50, "khaki": 54,
        "chartreuse": 90, "yellow-green": 85, "yellowgreen": 85, "yellow green": 85, "lime green": 100,
        "mint": 150, "sea green": 146, "emerald": 140, "forest green": 120, "aqua": 180, "azure": 210,
        "cobalt": 220, "lavender": 270, "lilac": 280, "mauve": 290, "plum": 300, "orchid": 300}
ACHROMATIC = ("black", "gray", "grey", "dark gray", "dark grey", "charcoal")


def colour_target(name):
    n = (name or "").lower().strip()
    if any(a == n or n.startswith(a) or n.endswith(a) for a in ACHROMATIC) and not any(h in n for h in ("blue", "green", "red")):
        return {"kind": "dark", "name": n}
    best = None
    for k, h in HUES.items():
        if k in n and (best is None or len(k) > len(best[0])):
            best = (k, h)
    if best is None:
        return {"kind": "dark", "name": n}
    return {"kind": "hue", "hue": float(best[1]), "name": n}


def hsv_arrays(img):
    rgb = img.astype(float) / 255.0
    mx, mn = rgb.max(-1), rgb.min(-1)
    delta = mx - mn
    sat = np.where(mx > 0, delta / np.maximum(mx, 1e-9), 0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    dd = np.maximum(delta, 1e-9)
    hue = np.where(mx == r, ((g - b) / dd) % 6, np.where(mx == g, (b - r) / dd + 2, (r - g) / dd + 4)) * 60
    return hue, sat, mx


def hue_dist(a, b):
    return np.abs((np.asarray(a) - b + 180) % 360 - 180)


# --------------------------------------------------------------- plot region
def plot_region(frame, shape, inset=None):
    H, W = shape
    lw = frame["line_width"]
    k = int(np.ceil(lw / 2)) + 1 if inset is None else inset
    m = np.zeros((H, W), bool)
    L, R, T, B = (int(round(frame[k_])) for k_ in ("left", "right", "top", "bottom"))
    m[max(0, T + k):max(0, B - k + 1), max(0, L + k):max(0, R - k + 1)] = True
    return m


# ------------------------------------------------------------------ legends
def find_legend(img, frame, lines, hue, sat, val):
    """Legend entries = text lines with a colour/marker sample just to their left.
    Returns (entries in reading order, list of legend bounding boxes)."""
    H, W = img.shape[:2]
    textmask = np.zeros((H, W), bool)
    for l in lines:
        textmask[l["y0"]:l["y1"], l["x0"]:l["x1"]] = True
    chrom = (sat >= 0.3) & (val >= 0.3)
    dark = D.dark_mask(img, 120, 0.3)
    ents = []
    for l in lines:
        h = l["y1"] - l["y0"]
        y0, y1 = max(0, l["y0"] - 3), min(H, l["y1"] + 3)
        x1 = l["x0"] - 2
        x0 = max(int(frame["left"]) + 2, x1 - int(max(60, 4 * h)))
        if x1 - x0 < 8:
            continue
        reg = np.zeros((H, W), bool)
        reg[y0:y1, x0:x1] = True
        reg &= ~textmask
        c = _largest_blob(reg & chrom)
        d = _largest_blob(reg & dark)
        nc, nd = int(c.sum()), int(d.sum())
        if nc >= 10 and nc >= 0.25 * nd:
            ys, xs = np.where(c)
            hh = hue[c]
            # circular mean hue
            ang = np.deg2rad(hh)
            mh = (np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) + 360) % 360
            ents.append(dict(text=l, kind="hue", hue=float(mh), sample=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                             rgb=np.median(img[c], 0).tolist()))
        elif nd >= 10:
            ys, xs = np.where(d)
            ents.append(dict(text=l, kind="dark", sample=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                             rgb=np.median(img[d], 0).tolist()))
    # keep entries that form a legend: >= 2 entries stacked tightly (vertical
    # legend) or on one row (horizontal legend)
    groups = []
    used = set()
    for i, e in enumerate(ents):
        if i in used:
            continue
        g = [i]
        h = e["text"]["y1"] - e["text"]["y0"]
        for j, f in enumerate(ents):
            if j <= i or j in used:
                continue
            same_col = abs((f["sample"][0] + f["sample"][2]) - (e["sample"][0] + e["sample"][2])) / 2 <= 10 and abs(f["text"]["y0"] - ents[g[-1]]["text"]["y0"]) <= 3.2 * max(h, 10)
            same_row = abs((f["text"]["y0"] + f["text"]["y1"]) - (e["text"]["y0"] + e["text"]["y1"])) / 2 <= 4
            if same_col or same_row:
                g.append(j)
        if len(g) >= 2:
            used.update(g)
            groups.append([ents[k] for k in g])
    groups.sort(key=lambda g: min(e["text"]["y0"] for e in g))
    ordered, boxes = [], []
    for g in groups:
        g.sort(key=lambda e: (round(e["text"]["y0"] / 6), e["text"]["x0"]))
        ordered.extend(g)
        xs0 = min(min(e["sample"][0], e["text"]["x0"]) for e in g)
        ys0 = min(min(e["sample"][1], e["text"]["y0"]) for e in g)
        xs1 = max(max(e["sample"][2], e["text"]["x1"]) for e in g)
        ys1 = max(max(e["sample"][3], e["text"]["y1"]) for e in g)
        boxes.append((xs0, ys0, xs1, ys1))
    return ordered, boxes


def expand_legend_boxes(img, boxes, lines, entries, max_search=30):
    """A legend box built from the detected entries misses (a) entries whose text
    was not recognised (superscripts, e.g. PO4^3-) and (b) the legend frame and
    its drop shadow. Grow each box: absorb text lines aligned with the entries,
    then push every edge out to the frame line and through a thick shadow."""
    H, W = img.shape[:2]
    ink = img.mean(-1) < 200
    out = []
    for (x0, y0, x1, y1) in boxes:
        tx0 = [e["text"]["x0"] for e in entries if x0 <= e["text"]["x0"] <= x1 and y0 <= e["text"]["y0"] <= y1]
        ref = float(np.median(tx0)) if tx0 else None
        lh = float(np.median([l["y1"] - l["y0"] for l in lines])) if lines else 15
        changed = True
        while changed and ref is not None:
            changed = False
            for l in lines:
                if abs(l["x0"] - ref) <= 12 and not (y0 <= l["y0"] and l["y1"] <= y1):
                    gap = max(y0 - l["y1"], l["y0"] - y1)
                    if gap <= 1.5 * lh:
                        y0, y1, x1 = min(y0, l["y0"]), max(y1, l["y1"]), max(x1, l["x1"])
                        changed = True
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        def push(edge, step, along_rows):
            # move outward until a frame line (dense ink across the box), then through the shadow
            hit = None
            for k in range(1, max_search + 1):
                e = edge + step * k
                if not (0 <= e < (H if along_rows else W)):
                    break
                frac = ink[e, x0:x1 + 1].mean() if along_rows else ink[y0:y1 + 1, e].mean()
                if frac >= 0.6:
                    hit = e
                elif hit is not None:
                    break
            return hit if hit is not None else edge
        y0n, y1n = push(y0, -1, True), push(y1, +1, True)
        x0n, x1n = push(x0, -1, False), push(x1, +1, False)
        out.append((x0n, y0n, x1n, y1n))
    return out


def _largest_blob(m):
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def exclusion_mask(shape, lines, legend_boxes, pad_text=3, pad_legend=1):
    H, W = shape
    m = np.zeros((H, W), bool)
    for l in lines:
        m[max(0, l["y0"] - pad_text):l["y1"] + pad_text, max(0, l["x0"] - pad_text):l["x1"] + pad_text] = True
    for (x0, y0, x1, y1) in legend_boxes:
        m[max(0, y0 - pad_legend):y1 + pad_legend, max(0, x0 - pad_legend):x1 + pad_legend] = True
    return m


# ------------------------------------------------------------- series masks
def series_mask(img, target, hue, sat, val, region, other_hues=()):
    if target["kind"] == "dark":
        g = img.mean(-1)
        lvl = target.get("gray_level")
        m = (sat <= 0.25) & (g <= (min(lvl + 45, 150) if lvl is not None else 115))
        return m & region
    h = target["hue"]
    tol = 22.0
    for o in other_hues:  # don't overlap a neighbouring series' hue
        dd = float(hue_dist(o, h))
        if 0 < dd < 2 * tol:
            tol = max(8.0, dd / 2 - 1)
    m = (hue_dist(hue, h) <= tol) & (sat >= 0.2) & (val >= 0.3)
    return m & region


def refine_hue(target, hue, sat, val, region, window=35):
    """Snap a colour-name hue to the actual hue peak in the plot (e.g. 'blue'
    -> 226 deg for a slate blue)."""
    if target["kind"] != "hue":
        return target
    m = region & (sat >= 0.35) & (val >= 0.3) & (hue_dist(hue, target["hue"]) <= window)
    if m.sum() < 15:
        return target
    hist, edges = np.histogram(hue[m], bins=72, range=(0, 360))
    k = int(np.argmax(hist))
    return dict(target, hue=float((edges[k] + edges[k + 1]) / 2))


def unclaimed_hue(target, claimed, hue, sat, val, region, min_px=15):
    """The VLM named a colour that is not in the plot (e.g. 'purple' for a
    blue line): take the nearest strong hue peak no other series claims."""
    m = region & (sat >= 0.35) & (val >= 0.3)
    if (hue_dist(hue, target["hue"]) <= 22)[m].sum() >= min_px or m.sum() < min_px:
        return target
    hist, edges = np.histogram(hue[m], bins=72, range=(0, 360))
    centres = (edges[:-1] + edges[1:]) / 2
    peaks = [(hist[i], centres[i]) for i in range(72)
             if hist[i] >= min_px and hist[i] >= hist[i - 1] and hist[i] >= hist[(i + 1) % 72]]
    peaks = [p for p in peaks if all(hue_dist(p[1], c) > 25 for c in claimed)]
    if not peaks:
        return target
    best = min(peaks, key=lambda p: hue_dist(p[1], target["hue"]))
    return dict(target, hue=float(best[1]), note=f"colour '{target['name']}' not found; used nearest unclaimed hue")


# ------------------------------------------------------------------ markers
def _fill_small_holes(m, max_hole=250):
    filled = ndi.binary_fill_holes(m)
    holes = filled & ~m
    lab, n = ndi.label(holes)
    if n:
        sizes = ndi.sum(holes, lab, range(1, n + 1))
        small = np.isin(lab, np.where(sizes <= max_hole)[0] + 1)
        m = m | small
    return m


def find_markers(mask):
    """Marker centres from a colour mask that may also contain the series'
    line, error bars, etc. Thin strokes are removed with a distance-transform
    threshold chosen from the mask itself (markers are thicker than lines)."""
    if mask.sum() < 6:
        return [], {}
    m0 = ndi.binary_closing(mask, np.ones((2, 2)))
    m = _fill_small_holes(m0)
    holes = m & ~m0
    dt = ndi.distance_transform_edt(m)
    lm = (dt == ndi.maximum_filter(dt, size=3)) & (dt > 0)
    vals = np.sort(dt[lm])[::-1]
    # marker radius ~ the largest local thickness; median of the 10 largest is
    # robust to a single overlap. Strokes (lines, error bars) are thinner.
    marker_r = float(np.median(vals[:10]))
    t = max(1.6, 0.6 * marker_r)
    core = dt >= t
    lab, n = ndi.label(core)
    if n == 0:
        return [], {"dt_threshold": t}
    blobs = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        blobs.append((xs, ys))
    areas = np.array([len(b[0]) for b in blobs])
    typical = float(np.median(areas))
    pts = []
    mpx = 2 * (np.sqrt(typical / np.pi) + t)   # marker diameter estimate
    grow = int(np.ceil(t)) + 1
    for (xs, ys), a in zip(blobs, areas):
        if a < 0.3 * typical:
            continue
        k = min(3, int(round(a / typical))) if typical > 0 else 1   # >3 fused 'markers' is a thick line, not markers
        while k >= 2:
            cs = _kmeans_split(xs, ys, k)
            # a single odd-shaped marker (triangle, star) is not 2-3 markers:
            # the split centres of real fused markers are about a marker apart
            dmin = min(np.hypot(p[0] - q[0], p[1] - q[1]) for i_, p in enumerate(cs) for q in cs[i_ + 1:])
            if dmin >= 0.5 * mpx:
                break
            k -= 1
        if k >= 2:
            for cx, cy in cs:
                pts.append((cx, cy, a / k))
        else:
            # centre of the full marker = core bbox centre (the core is the marker shrunk evenly)
            pts.append(((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, a))
    # hollow or filled? share of each marker's area that is an enclosed hole
    hf = []
    R_ = int(np.ceil(marker_r)) + 1
    for cx, cy, _ in pts:
        y0_, x0_ = max(0, int(cy) - R_), max(0, int(cx) - R_)
        win_f, win_h = m[y0_:int(cy) + R_ + 1, x0_:int(cx) + R_ + 1], holes[y0_:int(cy) + R_ + 1, x0_:int(cx) + R_ + 1]
        if win_f.sum():
            hf.append(win_h.sum() / win_f.sum())
    return sorted(pts), {"dt_threshold": t, "typical_core_area": typical, "marker_px": mpx, "marker_r": marker_r,
                         "hole_fraction": float(np.median(hf)) if hf else 0.0}


def _merge_near(pts, d):
    """One marker split into 2-3 cores (triangles, markers crossed by a line)
    gives centres a few px apart: merge centres closer than d (greedy, by area)."""
    out = []
    for x, y, a in sorted(pts, key=lambda p: -p[2]):
        for o in out:
            cx, cy = o[0] / o[2], o[1] / o[2]
            if (x - cx) ** 2 + (y - cy) ** 2 < d * d:
                o[0] += x * a; o[1] += y * a; o[2] += a
                break
        else:
            out.append([x * a, y * a, a])
    return sorted((o[0] / o[2], o[1] / o[2], o[2]) for o in out)


def _kmeans_split(xs, ys, k):
    P = np.c_[xs, ys].astype(float)
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    proj = (P - c) @ vt[0]
    centers = np.quantile(proj, (np.arange(k) + 0.5) / k)
    for _ in range(25):
        a = np.argmin(np.abs(proj[:, None] - centers[None]), 1)
        centers = np.array([proj[a == j].mean() if (a == j).any() else centers[j] for j in range(k)])
    out = []
    for j in range(k):
        sel = a == j
        if sel.any():
            out.append(((xs[sel].min() + xs[sel].max()) / 2, (ys[sel].min() + ys[sel].max()) / 2))
    return out


# -------------------------------------------------------------------- lines
def trace_line(mask, min_cc=6):
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n:
        sizes = ndi.sum(mask, lab, range(1, n + 1))
        mask = np.isin(lab, np.where(sizes >= min_cc)[0] + 1)
    xs, ys = [], []
    for x in np.where(mask.any(0))[0]:
        rows = np.where(mask[:, x])[0]
        xs.append(float(x)); ys.append(float(np.median(rows)))
    return np.array(xs), np.array(ys)


def _stitch(chains):
    """Join chain ends, nearest ends first, until one path remains."""
    chains = [list(c) for c in chains if c]
    while len(chains) > 1:
        best = None
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                for ei in (0, -1):
                    for ej in (0, -1):
                        a, b = chains[i][ei], chains[j][ej]
                        d = np.hypot(a[0] - b[0], a[1] - b[1])
                        if best is None or d < best[0]:
                            best = (d, i, j, ei, ej)
        _, i, j, ei, ej = best
        ci = chains[i] if ei == -1 else chains[i][::-1]
        cj = chains[j] if ej == 0 else chains[j][::-1]
        chains = [c for t, c in enumerate(chains) if t not in (i, j)] + [ci + cj]
    return chains[0]


def split_same_style(pts, k, x_tol, sep_px=6.0):
    """Two series drawn identically (e.g. adsorption / desorption branches).
    Each point looks at its neighbours within a small radius. The direction the
    neighbourhood runs in (its main axis) is the local direction of the curves;
    if the neighbours lie clearly on two sides ACROSS that direction, the point
    goes to the upper/left side or the lower/right side. Measuring across the
    local direction, not just in height, keeps the branches apart where they
    run steeply (near-vertical parts, where both branches have almost the same
    x). Where the branches coincide, points alternate between them.
    Returns k lists ordered bottom to top, each in drawing order."""
    pts = sorted(pts)
    if k != 2 or len(pts) < 4:
        return [pts] + [[] for _ in range(k - 1)]
    P = np.array([(p[0], p[1]) for p in pts], float)
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    nn = np.sort(D, 1)[:, 1]
    R = max(x_tol, 3.0 * float(np.median(nn)))
    lower, upper, amb = [], [], []
    for i, p in enumerate(pts):
        nb = P[D[i] <= R]
        if len(nb) < 3:
            amb.append(p); continue
        c = nb - nb.mean(0)
        t = np.linalg.svd(c, full_matrices=False)[2][0]       # local direction
        if t[0] < 0 or (t[0] == 0 and t[1] > 0):
            t = -t                                              # left -> right
        nrm = np.array([t[1], -t[0]])                           # points up (or left on rising parts)
        q = nb @ nrm
        if q.max() - q.min() > sep_px:
            (upper if P[i] @ nrm > (q.max() + q.min()) / 2 else lower).append(p)
        else:
            amb.append(p)
    for j, p in enumerate(amb):
        (lower if j % 2 == 0 else upper).append(p)
    return [path_order(lower), path_order(upper)]


def path_order(pts):
    """points of one curve in drawing order: nearest ends joined first, so a
    curve that runs steeply or bends back is not zig-zagged by sorting on x"""
    if len(pts) < 3:
        return sorted(pts)
    path = _stitch([[p] for p in pts])
    return path if path[0][0] <= path[-1][0] else path[::-1]


def _reversals(pts, min_step):
    """how often the path, read left to right, turns between going up and going down
    (steps smaller than min_step are ignored)"""
    p = sorted(pts)
    signs = [np.sign(b[1] - a[1]) for a, b in zip(p, p[1:]) if abs(b[1] - a[1]) >= min_step]
    return int(sum(1 for a, b in zip(signs, signs[1:]) if a != b))


def two_branches(pts, x_tol, sep_px, min_rev=6, min_pts=4):
    """Is this one marker set really two curves of the same style? Read left
    to right, two interleaved curves zig-zag between them (many up/down
    reversals); each separated curve is smooth. Returns [lower, upper] or None."""
    if len(pts) < 2 * min_pts + 2:
        return None
    r_all = _reversals(pts, sep_px / 2)
    if r_all < min_rev:
        return None
    lo, up = split_same_style(pts, 2, x_tol, sep_px)
    if len(lo) < min_pts or len(up) < min_pts:
        return None
    if _reversals(lo, sep_px / 2) + _reversals(up, sep_px / 2) <= 0.4 * r_all:
        return [lo, up]
    return None


def recover_hidden(out, raw, X, min_evidence=0.06):
    """Markers hidden under another series' marker. In most plots the series
    are measured at the same x values, so the x positions found in ALL marker
    series form a shared grid. Where a series has no marker at a grid x but
    another series has one there, and a little of this series' colour peeks
    out around that marker (corners of a larger marker, the joining line),
    the hidden marker is put at the visible marker's position."""
    names = [n for n, r in raw.items() if r["kind"] == "marker" and not r["same_style"]
             and out["series"].get(n, {}).get("px")]
    if len(names) < 2:
        return
    allx = sorted(x for n in names for x in out["series"][n]["px"])
    tol = 0.5 * min(raw[n]["marker_px"] for n in names)
    grid = []
    for x in allx:                                   # cluster the x positions of all markers
        if grid and x - grid[-1][-1] <= tol:
            grid[-1].append(x)
        else:
            grid.append([x])
    grid = [float(np.mean(g)) for g in grid]
    for n in names:
        rec, r = out["series"][n], raw[n]
        H, W = r["mask"].shape
        h = 0.8 * r["marker_px"]
        need = min_evidence * np.pi * (r["marker_px"] / 2) ** 2
        add = []
        for gx in grid:
            if any(abs(gx - x) <= tol for x in rec["px"]):
                continue
            best = None
            for o in names:                           # visible markers of other series at this x
                if o == n:
                    continue
                for ox, oy in zip(out["series"][o]["px"], out["series"][o]["py"]):
                    if abs(ox - gx) > tol:
                        continue
                    y0, y1 = int(max(0, oy - h)), int(min(H, oy + h + 1))
                    x0, x1 = int(max(0, ox - h)), int(min(W, ox + h + 1))
                    ev = int(r["mask"][y0:y1, x0:x1].sum())
                    if ev >= need and (best is None or ev > best[2]):
                        best = (ox, oy, ev)
            if best is not None:
                add.append(best)
        if not add:
            continue
        px = np.r_[rec["px"], [a[0] for a in add]]
        py = np.r_[rec["py"], [a[1] for a in add]]
        maps = r["maps"]
        vx = X.to_value(px)
        vy = np.array([float(map_for_pixel(maps, p).to_value(p)) for p in py])
        o_ = np.argsort(vx)
        rec.update(x=vx[o_].round(6).tolist(), y=vy.round(8)[o_].tolist(),
                   px=px[o_].round(1).tolist(), py=py[o_].round(1).tolist())
        if len(maps) > 1:
            rec["segment"] = [int(maps.index(map_for_pixel(maps, p))) for p in py[o_]]
        rec["recovered_hidden"] = len(add)
        out["warnings"].append(f"'{n}': {len(add)} marker(s) hidden under another series' markers recovered "
                               f"at the shared x positions")


def simplify_line(pts, tol=0.5):
    """Ramer-Douglas-Peucker on (x, y, w) points sorted by x."""
    if len(pts) < 3:
        return list(pts)
    P = np.array([(p[0], p[1]) for p in sorted(pts)], float)
    keep = np.zeros(len(P), bool); keep[0] = keep[-1] = True
    stack = [(0, len(P) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        d = P[b] - P[a]
        nrm = np.hypot(*d) or 1.0
        dist = np.abs(d[0] * (P[a + 1:b, 1] - P[a, 1]) - d[1] * (P[a + 1:b, 0] - P[a, 0])) / nrm
        k = int(np.argmax(dist))
        if dist[k] > tol:
            j = a + 1 + k
            keep[j] = True
            stack += [(a, j), (j, b)]
    return [(float(x), float(y), 1.0) for x, y in P[keep]]


def connection_fraction(mask, px, py, marker_px, step=1.0):
    """Median, over consecutive markers (left to right), of the fraction of the
    gap between them that is inked in this series' mask (dilated by 1 px)."""
    mk = ndi.binary_dilation(mask, np.ones((3, 3)))
    H, W = mk.shape
    r = 0.7 * marker_px
    fr = []
    for x0, y0, x1, y1 in zip(px[:-1], py[:-1], px[1:], py[1:]):
        d = np.hypot(x1 - x0, y1 - y0)
        if d <= 2 * r + 3:
            continue                      # markers touch: nothing to measure
        ts = np.arange(r, d - r, step) / d
        xs = np.clip(np.round(x0 + ts * (x1 - x0)).astype(int), 0, W - 1)
        ys = np.clip(np.round(y0 + ts * (y1 - y0)).astype(int), 0, H - 1)
        fr.append(mk[ys, xs].mean())
    return float(np.median(fr)) if fr else 1.0


def legend_layout(legend, boxes, match, series):
    """For redrawing: which series sit in which legend box, in how many columns, at what text size."""
    inv = {j: i for i, j in match.items()}
    out = []
    for b in boxes:
        idx = [j for j, e in enumerate(legend)
               if b[0] - 2 <= e["text"]["x0"] <= b[2] + 2 and b[1] - 2 <= e["text"]["y0"] <= b[3] + 2]
        if not idx:
            continue
        cy = [(legend[j]["text"]["y0"] + legend[j]["text"]["y1"]) / 2 for j in idx]
        ncol = len(idx) if (max(cy) - min(cy)) <= 4 else 1
        out.append({"box": [int(v) for v in b], "ncol": ncol,
                    "series": [series[inv[j]]["name"] for j in idx if j in inv],
                    # one text size for all legends: the smallest box-wise median (shadows / superscripts inflate boxes)
                    "text_px": float(np.median([legend[j]["text"]["y1"] - legend[j]["text"]["y0"] for j in idx]))})
    if out:
        tp = min(l["text_px"] for l in out)
        for l in out:
            l["text_px"] = tp
    return out


def match_legend(series, legend, max_hue=45.0):
    """Pair each VLM series with a legend sample by COLOUR (the VLM's series
    order often differs from the legend order). Same-colour entries fall back
    to order. Returns {series index: legend index}."""
    if not legend or not series:
        return {}
    from scipy.optimize import linear_sum_assignment
    BIG = 1e6
    C = np.full((len(series), len(legend)), BIG)
    for i, s in enumerate(series):
        t = colour_target(s["color"])
        for j, e in enumerate(legend):
            order = 0.5 * abs(i - j)                  # tie-break: keep the order
            if t["kind"] == "dark" and e["kind"] == "dark":
                C[i, j] = order
            elif t["kind"] == "hue" and e["kind"] == "hue":
                d = float(hue_dist(t["hue"], e["hue"]))
                if d <= max_hue:
                    C[i, j] = d + order
    r, c = linear_sum_assignment(C)
    return {int(i): int(j) for i, j in zip(r, c) if C[i, j] < BIG}


# --------------------------------------------------------------- the pipeline
def calibrate_all(img, frame, spec):
    cal, info = {}, {}
    for side, key, along in (("bottom", "x_axis", "x"), ("left", "y_axis", "y"), ("right", "y_axis_right", "y")):
        ax = spec.get(key)
        if not ax or (side == "right" and not frame["has_right"]):
            continue
        ticks = D.detect_ticks(img, frame, side)
        labels = D.tick_label_blobs(img, frame, side, ticks)
        maps, inf = calibrate_axis(ax["tick_labels"], [w["c"] for w in labels], [t["pos"] for t in ticks if t.get("major")],
                                   ax["scale"], ax.get("broken", False), along,
                                   label_widths=[w["x1"] - w["x0"] for w in labels])
        inf["ticks"] = ticks
        inf["labels"] = [(w["x0"], w["y0"], w["x1"], w["y1"]) for w in labels]
        if maps is None and key != "y_axis_right":
            # no numeric ticks (e.g. FTIR transmittance): use % of the frame
            if along == "x":
                a, b = frame["left"], (frame["right"] - frame["left"]) / 100
            else:
                a, b = frame["bottom"], -(frame["bottom"] - frame["top"]) / 100
            maps = [AxisMap(a, b)]
            inf["method"] = "no tick values: relative units (0-100 % of the plot frame)"
            inf["relative"] = True
        if maps is not None:
            cal[key] = maps
        info[key] = inf
    return cal, info


def _dispatch(image_path, spec):
    """the extractor for the chart type the VLM reported"""
    from .spec_types import canonical_type
    ct = canonical_type(spec.get("chart_type"))
    img = D.load_rgb(image_path)
    if ct == "pie":
        from .pie import pipeline_pie
        return pipeline_pie(img, spec)
    if ct == "bar" and str(spec.get("orientation", "")).startswith("h") or ct == "stacked_bar":
        from .stacked import pipeline_stacked
        return pipeline_stacked(img, spec)
    if ct == "bar":
        from .bars import pipeline_bars
        return pipeline_bars(img, spec)
    if ct == "histogram":
        from .hist import pipeline_hist
        return pipeline_hist(img, spec)
    if ct == "box":
        from .boxplot import pipeline_box
        return pipeline_box(img, spec)
    if ct == "heatmap":
        from .heatmap import pipeline_heatmap
        return pipeline_heatmap(img, spec)
    if ct == "contour":
        from .heatmap import pipeline_field
        return pipeline_field(img, spec)
    if ct == "polar":
        from .polar import pipeline_polar
        return pipeline_polar(img, spec)
    if ct in ("bar3d", "scatter3d", "surface3d"):
        from . import plot3d
        return getattr(plot3d, f"pipeline_{ct}")(img, spec)
    if ct == "other":
        H, W = img.shape[:2]
        return {"chart_type": "other", "size": (W, H), "spec": spec, "series": {}, "warnings": [],
                "reject": f"unsupported chart type ({spec.get('vlm_chart_type') or 'other'})"}
    try:
        return _pipeline_xy(image_path, spec)
    except RuntimeError as e:
        # no axes in the image at all: most likely a pie the VLM did not recognise
        # (the pie's own self-checks reject anything that is not a pie)
        from .pie import pipeline_pie
        out = pipeline_pie(img, {**spec, "chart_type": "pie", "slices": spec.get("slices") or []})
        out["warnings"].insert(0, f"no axes found ({e}) - read as a pie chart")
        out["spec"] = {**out["spec"], "chart_type": "pie"}
        return out


def pipeline(image_path, spec):
    """Dispatch on the chart type the VLM reported, then the rejection gates (reject.py):
    every result carries status 'ok' | 'flagged' | 'rejected' and the reasons."""
    from .reject import assess
    try:
        out = _dispatch(image_path, spec)
    except Exception as e:                       # an extractor that crashes = this image is rejected, not the run
        out = {"chart_type": spec.get("chart_type"), "spec": spec, "series": {}, "warnings": [],
               "reject": f"extraction failed: {type(e).__name__}: {e}"}
    out["image"] = str(image_path)
    out.setdefault("warnings", [])
    out.setdefault("series", {})
    img = D.load_rgb(image_path)
    if "size" not in out:
        out["size"] = (img.shape[1], img.shape[0])
    status, reasons, checks = assess(img, spec, out)
    out["status"], out["status_reasons"], out["checks"] = status, reasons, checks
    return out


def _pipeline_xy(image_path, spec):
    img = D.load_rgb(image_path)
    H, W = img.shape[:2]
    frame = D.detect_frame(img)
    cal, cal_info = calibrate_all(img, frame, spec)
    hue, sat, val = hsv_arrays(img)
    region = plot_region(frame, (H, W))
    lines = D.text_lines_inside(img, frame)
    legend, legend_boxes = find_legend(img, frame, lines, hue, sat, val)
    legend_boxes = expand_legend_boxes(img, legend_boxes, lines, legend)
    excl = exclusion_mask((H, W), lines, legend_boxes)
    area = region & ~excl

    series = spec["series"]
    targets = []
    match = match_legend(series, legend)          # series index -> legend entry (by colour, not by order)
    for i, s in enumerate(series):
        t = colour_target(s["color"])
        if i in match:
            e = legend[match[i]]
            if e["kind"] == "hue":
                t = {"kind": "hue", "hue": e["hue"], "name": s["color"], "from": "legend sample"}
            else:
                t = {"kind": "dark", "name": s["color"], "gray_level": float(np.mean(e["rgb"])), "from": "legend sample"}
        else:
            t = refine_hue(t, hue, sat, val, area)
            t["from"] = "colour name"
        targets.append(t)
    for i, t in enumerate(targets):
        if t["kind"] == "hue" and t["from"] == "colour name":
            claimed = [u["hue"] for j, u in enumerate(targets) if j != i and u["kind"] == "hue"]
            targets[i] = unclaimed_hue(t, claimed, hue, sat, val, area)
    hues = [t["hue"] for t in targets if t["kind"] == "hue"]

    # group series with an identical look (same colour target + marker + fill)
    groups = []
    for i, (s, t) in enumerate(zip(series, targets)):
        key = (t["kind"], round(t.get("hue", -1) / 10), s["marker"], s["marker_fill"])
        for g in groups:
            if g["key"] == key:
                g["idx"].append(i); break
        else:
            groups.append({"key": key, "idx": [i]})

    out = {"image": str(image_path), "size": (W, H), "frame": frame, "spec": {**spec, "series": [dict(q) for q in spec["series"]]},
           "calibration": {k: [m.as_dict() for m in v] for k, v in cal.items()},
           "calibration_info": {k: {kk: vv for kk, vv in v.items() if kk not in ("ticks", "labels")} for k, v in cal_info.items()},
           "legend": [{"kind": e["kind"], "hue": e.get("hue"), "rgb": e["rgb"], "text_box": (e["text"]["x0"], e["text"]["y0"], e["text"]["x1"], e["text"]["y1"])} for e in legend],
           "legend_boxes": legend_boxes, "legend_layout": legend_layout(legend, legend_boxes, match, series), "text_boxes": [(l["x0"], l["y0"], l["x1"], l["y1"]) for l in lines],
           "series": {}, "styles": {}, "warnings": []}
    if "x_axis" not in cal or "y_axis" not in cal:
        out["warnings"].append("axis calibration failed - no data extracted")
        return out
    X = cal["x_axis"][0]
    raw = {}
    for g in groups:
        i0 = g["idx"][0]
        s, t = series[i0], targets[i0]
        others = [h for h in hues if t["kind"] != "hue" or abs(h - t["hue"]) > 1e-6]
        if t["kind"] == "hue" and s["marker"] != "none":
            # coloured markers may sit ON the axis lines (x = 0, y = 0 points drawn unclipped):
            # look a little beyond the frame - nothing else outside the frame is coloured
            ext = plot_region(frame, (H, W), inset=-8) & ~excl
            m = series_mask(img, t, hue, sat, val, ext, others)
            # if an axis line itself has this colour (blue-grey frames), it is not data
            lw2 = int(np.ceil(frame["line_width"] / 2)) + 1
            L_, R_, T_, B_ = (int(round(frame[k_])) for k_ in ("left", "right", "top", "bottom"))
            for band in (np.s_[T_:B_ + 1, L_ - lw2:L_ + lw2 + 1], np.s_[T_:B_ + 1, R_ - lw2:R_ + lw2 + 1],
                         np.s_[T_ - lw2:T_ + lw2 + 1, L_:R_ + 1], np.s_[B_ - lw2:B_ + lw2 + 1, L_:R_ + 1]):
                sub = m[band]
                if sub.size and sub.any(axis=1 if sub.shape[0] > sub.shape[1] else 0).mean() > 0.5:
                    m[band] = False
        else:
            m = series_mask(img, t, hue, sat, val, area, others)
        if t["kind"] == "dark" and s["marker"] != "none":
            # dark markers crossed by coloured lines: re-attach the coloured pixels touching them
            chrom = region & ~excl & (sat >= 0.3)
            m = m | (chrom & ndi.binary_dilation(m, np.ones((5, 5))))
        rgb = np.median(img[m], 0).tolist() if m.sum() else [0, 0, 0]
        if s["marker"] != "none":
            pts, minfo = find_markers(m)
            kind = "marker"
        else:
            xs, ys = trace_line(m)
            pts, minfo = list(zip(xs, ys, np.ones(len(xs)))), {}
            if len(xs):
                dtl = ndi.distance_transform_edt(m)
                minfo["line_px"] = float(2 * np.median(dtl[np.round(ys).astype(int), np.round(xs).astype(int)]))
            kind = "line"
        k = len(g["idx"])
        x_tol_ = 0.015 * (frame["right"] - frame["left"])
        parts = (split_same_style(pts, k, x_tol=x_tol_, sep_px=0.8 * minfo.get("marker_px", 8))
                 if k > 1 else [pts])
        if k > 1:
            # order the names bottom -> top using the VLM's position hints
            order = sorted(g["idx"], key=lambda j: {"lower": 0, "bottom": 0, "middle": 1, "upper": 2, "top": 2}.get(series[j]["position"] or "", 1))
            todo = [(series[j], p) for j, p in zip(order, parts)]
        else:
            todo = [(s, pts)]
            if kind == "marker":
                br = two_branches(pts, x_tol=0.015 * (frame["right"] - frame["left"]), sep_px=0.8 * minfo.get("marker_px", 8))
                if br is not None:
                    # the VLM listed one series but the pixels hold two interleaved curves
                    # (e.g. adsorption / desorption branches drawn identically)
                    lo = dict(s, name=f"{s['name']} (lower branch)", position="lower")
                    up = dict(s, name=f"{s['name']} (upper branch)", position="upper")
                    todo = [(lo, br[0]), (up, br[1])]
                    i_ = [q["name"] for q in out["spec"]["series"]].index(s["name"])
                    out["spec"]["series"][i_:i_ + 1] = [lo, up]
                    out["warnings"].append(f"'{s['name']}' holds two interleaved curves of the same style - split into lower / upper branch")
        if kind == "line":
            # a traced curve has one point per pixel column; keep only the points
            # needed to redraw it within half a pixel (Ramer-Douglas-Peucker)
            todo = [(sj, simplify_line(part, tol=0.5)) for sj, part in todo]
        for sj, part in todo:
            ykey = "y_axis_right" if sj["y_axis"] == "right" and "y_axis_right" in cal else "y_axis"
            maps = cal[ykey]
            raw[sj["name"]] = {"mask": m, "marker_px": minfo.get("marker_px", 8), "same_style": len(todo) > 1,
                               "kind": kind, "maps": maps, "ykey": ykey}
            px = np.array([p[0] for p in part], float)
            py = np.array([p[1] for p in part], float)
            if len(px) == 0:
                out["warnings"].append(f"no pixels found for series '{sj['name']}'")
                out["series"][sj["name"]] = {"x": [], "y": [], "y_axis": ykey, "kind": kind}
                continue
            vx = X.to_value(px)
            vy = np.array([float(map_for_pixel(maps, p).to_value(p)) for p in py])
            o = np.arange(len(vx)) if len(todo) > 1 else np.argsort(vx)    # same-style branches keep their path order
            seg = [int(maps.index(map_for_pixel(maps, p))) for p in py[o]] if len(maps) > 1 else None
            rec = {"x": vx[o].round(6).tolist(), "y": vy.round(8)[o].tolist(), "y_axis": ykey, "kind": kind,
                   "px": px[o].round(1).tolist(), "py": py[o].round(1).tolist()}
            if seg is not None:
                rec["segment"] = seg
            if kind == "marker" and len(px) >= 2:
                # does a line join the markers? measured, not taken from the VLM
                ink = m if not sj.get("line_color") else (m | (region & ~excl & ((img.mean(-1) < 200) | (sat > 0.25))))
                frac = connection_fraction(ink, px[o], py[o], minfo.get("marker_px", 8))
                line_px = "solid" if frac >= 0.7 else ("dashed" if frac >= 0.3 else "none")
                rec["line_fraction"] = round(frac, 2)
                fill_px = "hollow" if minfo.get("hole_fraction", 0) >= 0.08 else "filled"
                for q in out["spec"]["series"]:
                    if q["name"] == sj["name"] and q["marker_fill"] != fill_px:
                        out["warnings"].append(f"'{sj['name']}': VLM said '{q['marker_fill']}' markers, pixels show '{fill_px}' "
                                               f"({minfo.get('hole_fraction', 0):.0%} of the marker is an enclosed hole) - pixels used")
                        q["marker_fill"] = fill_px
                    if q["name"] == sj["name"] and q["line"] != line_px:
                        out["warnings"].append(f"'{sj['name']}': VLM said line '{q['line']}', pixels show '{line_px}' "
                                               f"({frac:.0%} of the gaps between markers are inked) - pixels used")
                        q["line"] = line_px
            out["series"][sj["name"]] = rec
            out["styles"][sj["name"]] = {"rgb": rgb, "target": {k_: v for k_, v in t.items()}, **minfo}
    recover_hidden(out, raw, X)
    return out
