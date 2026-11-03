"""3-D bar charts, 3-D scatter and 3-D surfaces - on top of the camera from axes3d.py.

A picture of a 3-D plot loses one coordinate: a pixel is a whole line of sight.
Data are recovered only where the picture holds the missing piece:

  bar3d      a bar stands on a known floor cell (x, y category) -> its top face
             gives z. Top faces are the bar shade whose patches have no vertical
             edges; each top face is paired with the floor cell that explains it
             best (one cell per face, Hungarian).
  scatter3d  colour = z (a colour bar whose title is the z-axis title)  -> z from
             the colour, (x, y) from the camera;  stems (drop lines) -> the stem
             foot on the floor gives (x, y), the marker gives z.  Without such a
             cue the depth is unknown: the points are reported at the middle of
             their line of sight inside the box and FLAGGED (kept out of the scores).
  surface3d  colour bar -> z from the (shading-tolerant) colour, (x, y) from the
             camera, for every visible surface pixel on a grid. Without a colour
             bar: FLAGGED as above.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import detect as D
from .axes3d import calibrate_3d, project, solve_xy_given_z, solve_z_given_xy
from .colorbar import read_colorbar, colorbar_exclusion
from .extract import colour_target, hsv_arrays, refine_hue, series_mask, find_markers


def _base(spec, img, kind):
    H, W = img.shape[:2]
    return {"chart_type": kind, "size": (W, H), "spec": spec, "warnings": [], "checks": [], "series": {}}


def _camera(img, spec, out, excl=None):
    cam, inf = calibrate_3d(img, spec, excl)
    out["camera_info"] = {k: v for k, v in inf.items() if k != "hypotheses"}
    if cam is None:
        out["reject"] = f"3-D axes: {inf.get('reason')}"
        return None
    out["camera"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cam.items()}
    rms = inf.get("reprojection_rms_px", 99)
    out["checks"].append(("3-D camera reprojection of the ticks (px)", round(rms, 2), "<= 2.5", rms <= 2.5))
    return cam


def _limits(cam):
    xs, ys, zs = cam["x_values"], cam["y_values"], cam["z_values"]
    pad = lambda v: (min(v) - 0.6 * (max(v) - min(v)) / max(1, len(v) - 1), max(v) + 0.6 * (max(v) - min(v)) / max(1, len(v) - 1))
    return pad(xs), pad(ys), pad(zs)


def _ray_midpoint(P, px, lim):
    """depth unknown: the middle of the pixel's line of sight inside the axis box"""
    (x0, x1), (y0, y1), (z0, z1) = lim
    pts = []
    for z in np.linspace(z0, z1, 41):
        try:
            x, y = solve_xy_given_z(P, px, z)
        except np.linalg.LinAlgError:
            continue
        if x0 <= x <= x1 and y0 <= y <= y1:
            pts.append((x, y, z))
    if not pts:
        return None
    a = np.array(pts)
    return a.mean(0), a.min(0), a.max(0)


def _plot_area(img, cam, excl):
    """pixels inside the projected 3-D box"""
    H, W = img.shape[:2]
    (x0, x1), (y0, y1), (z0, z1) = _limits(cam)
    corners = project(cam["P"], np.array([[a, b, c] for a in (x0, x1) for b in (y0, y1) for c in (z0, z1)]))
    from matplotlib.path import Path
    from scipy.spatial import ConvexHull
    hull = ConvexHull(corners)
    path = Path(corners[hull.vertices])
    yy, xx = np.mgrid[:H, :W]
    m = path.contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(H, W)
    if excl is not None:
        m &= ~excl
    return m


# ------------------------------------------------------------------ 3-D bars
def pipeline_bar3d(img, spec):
    out = _base(spec, img, "bar3d")
    cam = _camera(img, spec, out)
    if cam is None:
        return out
    P = cam["P"]
    hue, sat, val = hsv_arrays(img)
    area = _plot_area(img, cam, None)
    bars = area & (sat >= 0.25) & (val >= 0.12)
    if bars.sum() < 50:
        out["reject"] = "3-D bars: no coloured bars inside the axes"; return out
    # faces: regions of one flat colour (lighting gives every face of a bar its own shade)
    q = (img // 10).astype(np.int32)
    key = q[..., 0] * 10000 + q[..., 1] * 100 + q[..., 2]
    keys, cnt = np.unique(key[bars], return_counts=True)
    regions = []
    for kk in keys[cnt >= 25]:
        m = ndi.binary_opening(bars & (key == kk), np.ones((2, 2)))
        lab, n = ndi.label(m)
        for i_, sl in enumerate(ndi.find_objects(lab), start=1):
            if sl is None:
                continue
            reg = lab[sl] == i_
            if reg.sum() >= 20:
                regions.append((sl, reg))
    xs, ys = cam["x_values"], cam["y_values"]
    cells = [(i, j) for i in range(len(xs)) for j in range(len(ys))]
    from matplotlib.path import Path

    def face_score(sl, reg, x, y, z, w):
        corners = project(P, [[x - w / 2, y - w / 2, z], [x + w / 2, y - w / 2, z], [x + w / 2, y + w / 2, z], [x - w / 2, y + w / 2, z]])
        y0, x0 = sl[0].start, sl[1].start
        yy, xx = np.mgrid[y0:sl[0].stop, x0:sl[1].stop]
        inside = Path(corners).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(reg.shape)
        prec = (inside & reg).sum() / reg.sum()
        e1, e2 = corners[1] - corners[0], corners[3] - corners[0]
        poly_area = abs(e1[0] * e2[1] - e1[1] * e2[0])
        cover = (inside & reg).sum() / max(poly_area, 1)
        return prec * min(1.0, cover) ** 0.3
    # a top face = a region that the projected top square of some cell explains
    best_w, cand = None, []
    for w in (0.4, 0.5, 0.6, 0.7, 0.8):
        sc = []
        for ri, (sl, reg) in enumerate(regions):
            yy, xx = np.where(reg)
            c = (xx.mean() + sl[1].start, yy.mean() + sl[0].start)
            for b, (i, j) in enumerate(cells):
                z, res = solve_z_given_xy(P, c, xs[i], ys[j])
                if res > 25:
                    continue
                sc.append((face_score(sl, reg, xs[i], ys[j], z, w), ri, b, z))
        tot = sum(sorted((t_[0] for t_ in sc), reverse=True)[:len(cells)])
        if best_w is None or tot > best_w[0]:
            best_w, cand = (tot, w), sc
    w = best_w[1]
    out["bar_width_grid_units"] = w
    from scipy.optimize import linear_sum_assignment
    C = np.zeros((len(regions), len(cells))); Z = np.zeros_like(C)
    for sc_, ri, b, z in cand:
        if sc_ > C[ri, b]:
            C[ri, b], Z[ri, b] = sc_, z
    r, c = linear_sum_assignment(-C)
    got = {}
    for a, b in zip(r, c):
        if C[a, b] >= 0.6:
            # refine z: move the top square up / down until it explains the region best
            sl, reg = regions[a]
            i, j = cells[b]
            zs = Z[a, b] + np.linspace(-0.08, 0.08, 17) * (max(cam["z_values"]) - min(cam["z_values"]))
            scs = [face_score(sl, reg, xs[i], ys[j], z, w) for z in zs]
            got[cells[b]] = (float(zs[int(np.argmax(scs))]), float(max(scs)))
    rows = []
    for (i, j), (z, res) in sorted(got.items()):
        rows.append({"x_category": cam["x_labels"][i], "y_category": cam["y_labels"][j], "height": round(z, 6), "face_fit": round(res, 2)})
    out["table"] = rows
    out["series"]["height"] = {"cells": [[r_["x_category"], r_["y_category"]] for r_ in rows], "values": [r_["height"] for r_ in rows], "kind": "bar3d"}
    n_cells = len(cells)
    out["checks"] += [("bars read / grid cells", f"{len(rows)}/{n_cells}", ">= 50 %", len(rows) >= 0.5 * n_cells)]
    if len(rows) < n_cells:
        out["warnings"].append(f"{n_cells - len(rows)} grid cells without a visible bar top (hidden behind taller bars, or zero)")
    return out


# ------------------------------------------------------------------ 3-D scatter
def pipeline_scatter3d(img, spec):
    out = _base(spec, img, "scatter3d")
    cb = spec.get("colorbar")
    scale, cinfo = (read_colorbar(img, cb) if cb else (None, {}))
    excl = colorbar_exclusion(cinfo, img.shape, 8) if cinfo.get("found") else None
    cam = _camera(img, spec, out, excl)
    if cam is None:
        return out
    P = cam["P"]
    lim = _limits(cam)
    hue, sat, val = hsv_arrays(img)
    area = _plot_area(img, cam, excl)
    zt = str((spec.get("z_axis") or {}).get("title", "")).strip().lower()
    colour_is_z = scale is not None and (str((cb or {}).get("same_as", "")).lower() == "z" or
                                         (zt and zt == str((cb or {}).get("title", "")).strip().lower()))
    stems = bool(spec.get("stems"))
    z_lo = cam["z_floor"]
    z_base = 0.0 if min(cam["z_values"]) <= 0 <= max(cam["z_values"]) else z_lo
    flagged = False
    for q in spec.get("series") or []:
        if colour_is_z:
            vals_, dist = scale.value(img[area])
            m = np.zeros(area.shape, bool); m[area] = (dist <= 22) & (sat[area] >= 0.15)
        else:
            t = refine_hue(colour_target(q.get("color", "")), hue, sat, val, area)
            m = series_mask(img, t, hue, sat, val, area, [])
        pts, info = find_markers(m) if m.sum() >= 6 else ([], {})
        if stems and not colour_is_z:
            # thin anti-aliased stems are paler than the markers: follow them on a softer colour mask
            from .extract import hue_dist
            m_soft = area & (sat >= 0.08) & ((hue_dist(hue, t["hue"]) <= 25) if t["kind"] == "hue" else (img.mean(-1) < 170))
        if not pts:
            out["warnings"].append(f"no markers of '{q['name']}' found"); continue
        if stems and not colour_is_z:
            # a 'marker' hanging on another marker's stem (same column, joined by the line) is a stem piece
            keep = []
            for k, (cx, cy, a_) in enumerate(pts):
                above = [p_ for p_ in pts if p_[1] < cy - 3 and abs(p_[0] - cx) <= 3]
                hang = any(m[int(p_[1]):int(cy), max(0, int(round(cx)) - 1):int(round(cx)) + 2].any(1).mean() > 0.9 for p_ in above)
                if not hang:
                    keep.append((cx, cy, a_))
            pts = keep
        rec = {"x": [], "y": [], "z": [], "px": [], "kind": "marker3d", "depth": None}
        n_try = len(pts)
        for (cx, cy, _a) in pts:
            if colour_is_z:
                r_ = int(max(1, 0.3 * info.get("marker_px", 6)))
                core = img[int(cy) - r_:int(cy) + r_ + 1, int(cx) - r_:int(cx) + r_ + 1].reshape(-1, 3)
                z = float(np.median(scale.value(core)[0]))
                x, y = solve_xy_given_z(P, (cx, cy), z)
                rec["depth"] = "colour = z (colour bar)"
            elif stems:
                foot = _stem_foot(m_soft, cx, cy, cam, info.get("marker_px", 6))
                if foot is None:
                    continue
                x, y = solve_xy_given_z(P, foot, z_base)
                z, res = solve_z_given_xy(P, (cx, cy), x, y)
                if res > 3.0:            # the marker is not straight above that foot: a wrong stem was followed
                    continue
                rec["depth"] = "stem foot on the floor"
            else:
                mid = _ray_midpoint(P, (cx, cy), lim)
                if mid is None:
                    continue
                (x, y, z), _, _ = mid
                rec["depth"] = "UNKNOWN - middle of the line of sight inside the axis box"
                flagged = True
            rec["x"].append(round(float(x), 6)); rec["y"].append(round(float(y), 6)); rec["z"].append(round(float(z), 6))
            rec["px"].append([round(cx, 1), round(cy, 1)])
        out["series"][q["name"]] = rec
        if stems and not colour_is_z:
            out["checks"].append((f"'{q['name']}': markers whose stem foot was found", f"{len(rec['x'])}/{n_try}", ">= 70 %",
                                  len(rec["x"]) >= 0.7 * max(1, n_try)))
    if stems and not colour_is_z and not flagged and out["series"]:
        out["flag"] = ("depth from stem feet (experimental): a stem hidden behind another marker or stem shortens it "
                       "and the point moves - check these points; kept out of the accuracy scores")
    if flagged:
        out["flag"] = ("depth not recoverable: no colour bar for z and no stems - x, y, z are guesses along each "
                       "marker's line of sight; kept out of the accuracy scores")
    out["table"] = [{"series": n, "x": a, "y": b, "z": c, "depth": d["depth"]} for n, d in out["series"].items()
                    for a, b, c in zip(d["x"], d["y"], d["z"])]
    n = len(out["table"])
    out["checks"].append(("points extracted", n, ">= 1", n >= 1))
    return out


def _stem_foot(m, cx, cy, cam, mpx):
    """follow a drop line from the marker downwards (image direction of -z) to its end"""
    P = cam["P"]
    xs, ys = cam["x_values"], cam["y_values"]
    xm, ym = np.mean(xs), np.mean(ys)
    zs = cam["z_values"]
    d = project(P, [xm, ym, min(zs)])[0] - project(P, [xm, ym, max(zs)])[0]
    d = d / np.linalg.norm(d)
    H, W = m.shape
    t, last, gap = 0.6 * mpx, None, 0
    while gap <= 8 and t < 2 * H:
        x, y = cx + t * d[0], cy + t * d[1]
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < W and 0 <= yi < H):
            break
        if m[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2].any():
            last, gap = (x, y), 0
        else:
            gap += 1
        t += 1
    if last is None or np.hypot(last[0] - cx, last[1] - cy) < mpx:
        return None
    return last


# ------------------------------------------------------------------ 3-D surface
def pipeline_surface3d(img, spec, step=3):
    out = _base(spec, img, "surface3d")
    cb = spec.get("colorbar")
    scale, cinfo = (read_colorbar(img, cb) if cb else (None, {}))
    excl = colorbar_exclusion(cinfo, img.shape, 8) if cinfo.get("found") else None
    cam = _camera(img, spec, out, excl)
    if cam is None:
        return out
    P = cam["P"]
    lim = _limits(cam)
    area = _plot_area(img, cam, excl)
    g = img.mean(-1)
    sat = (img.max(-1).astype(float) - img.min(-1)) / np.maximum(img.max(-1), 1)
    # the panes and grid are light grey: not the surface
    pane = (sat < 0.06) & (g > 200)
    H, W = g.shape
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    sel = area[yy, xx] & ~pane[yy, xx]
    py, px = yy[sel], xx[sel]
    xs, ys, zs, flags = [], [], [], []
    if scale is not None:
        v, dist = scale.value(img[py, px].astype(float), shaded=True)
        ok = dist <= 25
        for a, b, z in zip(px[ok], py[ok], v[ok]):
            try:
                x, y = solve_xy_given_z(P, (a, b), z)
            except np.linalg.LinAlgError:
                continue
            if lim[0][0] <= x <= lim[0][1] and lim[1][0] <= y <= lim[1][1]:
                xs.append(x); ys.append(y); zs.append(float(z))
        out["depth"] = "colour = z (colour bar), shading-tolerant colour match"
        out["cmap_table"] = {"values": scale.values.tolist(), "rgb": scale.colours.round(1).tolist()}
    else:
        dark = ~pane & area & ((sat >= 0.2) | (g < 150))
        sel2 = dark[yy, xx]
        for a, b in zip(xx[sel2][::2], yy[sel2][::2]):
            mid = _ray_midpoint(P, (a, b), lim)
            if mid is not None:
                (x, y, z), _, _ = mid
                xs.append(x); ys.append(y); zs.append(z)
        out["flag"] = ("depth not recoverable: no colour bar - the surface's z cannot be read from one view; "
                       "points are guesses along each pixel's line of sight; kept out of the accuracy scores")
        out["depth"] = "UNKNOWN"
    out["series"]["surface"] = {"x": np.round(xs, 6).tolist(), "y": np.round(ys, 6).tolist(), "z": np.round(zs, 6).tolist(), "kind": "surface3d"}
    out["table"] = [{"x": a, "y": b, "z": c} for a, b, c in zip(*[out["series"]["surface"][k] for k in "xyz"])]
    out["checks"].append(("surface samples", len(xs), ">= 50", len(xs) >= 50))
    return out
