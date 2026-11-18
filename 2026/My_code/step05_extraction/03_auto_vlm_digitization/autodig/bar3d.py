"""3-D bar charts by analysis-by-synthesis: guess the bar heights, DRAW the bars with the
camera found from the axes, compare the drawing with the picture, improve the guess.

Why: in a 3-D bar chart the bars hide each other. Reading each visible top face on its
own fails for bars that are partly hidden, and a face cut by a bar in front gives a wrong
height. Drawing the whole scene (back to front, like the plotting program does) explains
every visible pixel at once, so a partly hidden bar is fitted by what is left of it.

What is assumed (generic for 3-D bar charts):
  * one bar per grid cell: the x and y tick labels (read by the VLM) give the grid - as in
    the front row / column, every row and column has the same number of bars
  * all bars have the same square footprint (width w, measured) and stand on the floor
  * the lighting gives each face direction its own shade: top / x-facing side / y-facing side

Steps:
  1. camera (axes3d.py) -> project any 3-D point
  2. observed face map: bar pixels (coloured), split per colour family into 3 brightness
     levels; which level is the top / which a side is learned from a first drawing
  3. heights: coordinate search per cell (coarse grid, then fine), 3 sweeps; bar width: the
     footprint that explains the picture best
  4. a cell whose height changes nothing in the drawing is HIDDEN (behind taller bars):
     reported as missing, not guessed
  5. bar colour per cell: the colour of its visible faces
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from .axes3d import project, viewer


def camera_side(P):
    """camera centre (perspective) or viewing direction (affine) from P = null space"""
    _, _, Vt = np.linalg.svd(P)
    C = Vt[-1]
    if abs(C[3]) > 1e-9 * np.abs(C[:3]).max():
        return "point", C[:3] / C[3]
    v = C[:3] / np.linalg.norm(C[:3])
    return "dir", (v if v[2] > 0 else -v)            # the viewer looks from above


def box_faces(i, j, h, w, zb):
    x0, x1, y0, y1 = i - w / 2, i + w / 2, j - w / 2, j + w / 2
    z0, z1 = zb, max(h, zb)
    return [  # (label, normal, corners)
        (1, (0, 0, 1), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]),
        (2, (-1, 0, 0), [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)]),
        (2, (1, 0, 0), [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]),
        (3, (0, -1, 0), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]),
        (3, (0, 1, 0), [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]),
    ]


class Scene:
    def __init__(self, P, shape, nx, ny, zb, crop, limits, near_xy, aspect=None):
        self.P, self.shape, self.nx, self.ny, self.zb = P, shape, nx, ny, zb
        self.x0, self.y0, self.x1, self.y1 = crop
        self.kind, self.cam = viewer(P, limits, near_xy)
        # depth order must be judged in the drawn box's own proportions, not in data units
        # (a z-axis running to 40 against an x-axis running to 4 would distort distances)
        self.lo = np.array([limits["x"][0], limits["y"][0], limits["z"][0]], float)
        span = np.array([limits["x"][1], limits["y"][1], limits["z"][1]], float) - self.lo
        asp = np.asarray(aspect if aspect is not None else (4.0, 4.0, 3.0), float)
        self.scale = asp / np.maximum(span, 1e-12)
        if self.kind == "point":
            self.cam_n = (self.cam - self.lo) * self.scale
        else:
            v = self.cam * self.scale
            self.cam_n = v / np.linalg.norm(v)

    def _depth(self, X):
        Xn = (np.asarray(X, float) - self.lo) * self.scale
        return float(np.linalg.norm(Xn - self.cam_n)) if self.kind == "point" else -float(Xn @ self.cam_n)

    def _visible(self, normal, centre):
        n = np.asarray(normal, float)
        to_cam = (self.cam - centre) if self.kind == "point" else self.cam
        return float(n @ to_cam) > 0

    def render(self, H, w, with_cells=False):
        """H: (nx, ny) heights (nan = no bar). -> face-label image (0 bg, 1 top, 2 x-side, 3 y-side)"""
        W_, H_ = self.x1 - self.x0, self.y1 - self.y0
        lab = Image.new("L", (W_, H_), 0)
        dl = ImageDraw.Draw(lab)
        cel = Image.new("I", (W_, H_), 0) if with_cells else None
        dc = ImageDraw.Draw(cel) if with_cells else None
        cells = [(i, j) for i in range(self.nx) for j in range(self.ny) if H[i, j] == H[i, j] and H[i, j] > self.zb]
        cells.sort(key=lambda c: -self._depth((c[0], c[1], 0.5 * (H[c] + self.zb))))      # far first
        for (i, j) in cells:
            faces = box_faces(i, j, H[i, j], w, self.zb)
            for labv, nrm, cs in faces:
                cs = np.array(cs, float)
                if not self._visible(nrm, cs.mean(0)):
                    continue
                px = project(self.P, cs) - [self.x0, self.y0]
                poly = [tuple(p) for p in px]
                dl.polygon(poly, fill=labv)
                if with_cells:
                    dc.polygon(poly, fill=1 + i * self.ny + j)
        out = np.asarray(lab)
        return (out, np.asarray(cel)) if with_cells else out


def observed_shades(img, bars, crop, n_levels=3):
    """bar pixels -> brightness rank per colour family (0 darkest .. 2 brightest), -1 = not bar"""
    x0, y0, x1, y1 = crop
    sub = img[y0:y1, x0:x1].astype(float)
    m = bars[y0:y1, x0:x1]
    mx, mn = sub.max(-1), sub.min(-1)
    hue = np.zeros(mx.shape)
    d = np.maximum(mx - mn, 1e-9)
    r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
    hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    rank = np.full(mx.shape, -1, int)
    # colour families: hue peaks among bar pixels
    hh = hue[m]
    hist, edges = np.histogram(hh, bins=36, range=(0, 360))
    peaks = [k for k in range(36) if hist[k] >= 0.03 * m.sum() and hist[k] == max(hist[(k - 1) % 36], hist[k], hist[(k + 1) % 36])]
    centres = np.array([(edges[k] + edges[k + 1]) / 2 for k in peaks]) if peaks else np.array([float(np.median(hh))])
    fam = np.argmin(np.abs((hue[..., None] - centres[None, None, :] + 180) % 360 - 180), -1)
    val = mx / 255.0
    families = []
    for f in range(len(centres)):
        sel = m & (fam == f)
        if sel.sum() < 30:
            continue
        v = val[sel]
        # 3 brightness levels (k-means in 1-D)
        c = np.percentile(v, [15, 50, 85])
        for _ in range(20):
            k = np.argmin(np.abs(v[:, None] - c[None, :]), 1)
            c = np.array([v[k == q].mean() if (k == q).any() else c[q] for q in range(n_levels)])
        order = np.argsort(c)
        rk = np.empty(n_levels, int); rk[order] = np.arange(n_levels)
        r_ = np.full(sel.sum(), -1)
        r_ = rk[np.argmin(np.abs(v[:, None] - c[None, :]), 1)]
        rank[sel] = r_
        families.append({"hue": float(centres[f]), "levels": c[order].round(3).tolist(), "pixels": int(sel.sum())})
    return rank, families


def fit_bars(img, cam, bars, zb, z_top, init=None, sweeps=3):
    P = cam["P"]
    nx, ny = len(cam["x_values"]), len(cam["y_values"])
    Hh, Ww = img.shape[:2]
    # work only around the axes box
    (xl0, xl1), (yl0, yl1), (zl0, zl1) = cam["limits"]["x"], cam["limits"]["y"], cam["limits"]["z"]
    corners = project(P, np.array([[a, b, c] for a in (xl0, xl1) for b in (yl0, yl1) for c in (zl0, max(zl1, z_top))]))
    crop = (max(0, int(corners[:, 0].min()) - 5), max(0, int(corners[:, 1].min()) - 5),
            min(Ww, int(corners[:, 0].max()) + 6), min(Hh, int(corners[:, 1].max()) + 6))
    rank, families = observed_shades(img, bars, crop)
    view = cam.get("view") or {}
    lim_v = view.get("limits") or cam["limits"]
    scene = Scene(P, img.shape, nx, ny, zb, crop, lim_v, (cam["x_edge"], cam["y_edge"]), view.get("box_aspect"))
    if init is None or np.all(np.isnan(init)):
        H = np.full((nx, ny), 0.5 * (zb + z_top))
    else:
        H = np.where(np.isnan(init), float(np.nanmedian(init)), init)
    w = 0.6
    # which brightness rank is the top / which a side: learned from a first drawing
    lab0 = scene.render(H, w)
    obs_bar = rank >= 0
    C = np.zeros((3, 3))
    for rk in range(3):
        for lb in range(1, 4):
            C[rk, lb - 1] = np.sum((rank == rk) & (lab0 == lb))
    r_, c_ = linear_sum_assignment(-C)
    rank2label = {int(a): int(b) + 1 for a, b in zip(r_, c_)}
    # two side shades that are hard to tell apart (light from the front) count as one 'side' class
    gaps = [np.diff(f["levels"]).min() for f in families if len(f["levels"]) == 3]
    merge_sides = bool(gaps) and min(gaps) < 0.04
    obs = np.zeros(rank.shape, np.uint8)
    for rk, lb in rank2label.items():
        obs[rank == rk] = 2 if (merge_sides and lb == 3) else lb

    def cost(Hc, wc):
        r = scene.render(Hc, wc)
        if merge_sides:
            r = np.where(r == 3, 2, r)
        return int(np.count_nonzero(r != obs))
    coarse = np.linspace(zb, z_top, 41)
    step = coarse[1] - coarse[0]
    best = cost(H, w)
    for sweep in range(sweeps):
        if sweep == 1:            # bar width: the footprint that explains the picture best
            ws = np.arange(0.3, 0.95, 0.05)
            cs = [cost(H, wv) for wv in ws]
            w = float(ws[int(np.argmin(cs))]); best = min(cs)
        # nearest bars first: they hide the others, so they are the best constrained
        order = sorted(((i, j) for i in range(nx) for j in range(ny)), key=lambda c: scene._depth((c[0], c[1], 0.5 * (zb + z_top))))
        for (i, j) in order:
                cand = list(coarse) + list(np.clip(H[i, j] + np.linspace(-step, step, 11), zb, z_top))
                vals = []
                for hv in cand:
                    H2 = H.copy(); H2[i, j] = hv
                    vals.append(cost(H2, w))
                k = int(np.argmin(vals))
                H[i, j] = cand[k]; best = vals[k]
    lab, cells = scene.render(H, w, with_cells=True)
    # a bar that shows (almost) no pixels at its fitted height is hidden behind taller bars:
    # any height below the bars in front fits the picture equally well -> not reported
    hidden, bounds = [], {}
    for i in range(nx):
        for j in range(ny):
            if np.count_nonzero(cells == 1 + i * ny + j) < 25:
                hidden.append((i, j))
    # a bar must EXPLAIN its pixels: removing it (floor height) must cost clearly more than it
    # gains. A bar made up to explain a few stray pixels (an edge of a neighbour, a grid line)
    # where the true bar is hidden shows pixels, but adds almost nothing to the fit.
    base = cost(H, w)
    evidence = {}
    for i in range(nx):
        for j in range(ny):
            if (i, j) in hidden:
                continue
            vis = int(np.count_nonzero(cells == 1 + i * ny + j))
            H2 = H.copy(); H2[i, j] = zb
            gain = cost(H2, w) - base
            evidence[(i, j)] = (vis, gain)
            if gain < max(25, 0.08 * vis):
                hidden.append((i, j))
    for (i, j) in hidden:
        # the tallest bar that would still stay hidden: an upper bound for the reader
        ub = zb
        for hv in coarse:
            H2 = H.copy(); H2[i, j] = hv
            _, c2 = scene.render(np.nan_to_num(H2, nan=zb), w, with_cells=True)
            if np.count_nonzero(c2 == 1 + i * ny + j) >= 25:
                break
            ub = hv
        bounds[(i, j)] = float(ub)
    for (i, j) in hidden:
        H[i, j] = np.nan
    lab, cells = scene.render(np.nan_to_num(H, nan=zb), w, with_cells=True)
    # colour of each bar: median of its visible top-face pixels in the picture
    sub = img[crop[1]:crop[3], crop[0]:crop[2]]
    colours = {}
    for i in range(nx):
        for j in range(ny):
            sel = (cells == 1 + i * ny + j) & (obs > 0)
            if sel.sum() >= 5:
                top = sel & (lab == 1)
                colours[(i, j)] = np.median(sub[top if top.sum() >= 5 else sel], 0).tolist()
    info = {"bar_evidence_px": {f"{i},{j}": v for (i, j), v in evidence.items()}, "hidden_cells": hidden, "hidden_upper_bound": bounds, "bar_width_grid_units": w, "pixels_unexplained": best, "pixels_compared": int(obs.size),
            "fit_share_explained": round(1 - best / max(1, int((obs > 0).sum() + (lab > 0).sum())), 3),
            "shade_to_face": {str(k): ["", "top", "x-side", "y-side"][v] for k, v in rank2label.items()},
            "side_shades_merged": merge_sides,
            "colour_families": families, "crop": crop}
    return H, w, colours, info
