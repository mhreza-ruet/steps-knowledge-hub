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


def _camera(img, spec, out, excl=None, judge=None):
    """the 3-D camera. With a judge (data that must land inside the axis box, e.g. coloured
    surface pixels whose colour gives z), the axis-direction hypotheses that fit the ticks about
    equally well are told apart by how much of the data lands inside the box: a mirror-image
    camera (an axis read in the wrong direction) puts the data outside."""
    if judge is None:
        cam, inf = calibrate_3d(img, spec, excl)
    else:
        cands = calibrate_3d(img, spec, excl, candidates=True)
        if not cands:
            cam, inf = calibrate_3d(img, spec, excl)
        else:
            # tick fit first (the 0.15 px penalty per flipped axis keeps the VLM's order on a tie);
            # hypotheses within 0.25 px of the best are told apart by the data-inside-box share;
            # a remaining tie between flipping x or y goes to y (the receding floor axis, the one
            # whose label order a reader most easily gets backwards)
            r0 = cands[0][0]
            close = [c for c in cands if c[0] <= r0 + 0.25]
            def _key(t):
                sh, c = t
                fl = c[2].get("axis_order_flipped") or {}
                return (round(sh / 0.02), -(fl.get("x", False) + fl.get("y", False)), fl.get("y", False), -c[0])
            scored = [(judge(c[1]), c) for c in close]
            share, (r, cam, inf) = max(scored, key=_key)
            scored = [(sh, -c[0], c) for sh, c in scored]
            inf = dict(inf, data_inside_box=round(share, 3),
                       hypotheses_compared=[{"flip": c[2].get("axis_order_flipped"), "tick_rms": round(c[0], 2),
                                             "data_inside_box": round(sh, 3)} for sh, _, c in scored])
            if any((inf.get("axis_order_flipped") or {}).values()):
                fl = [k for k, v in inf["axis_order_flipped"].items() if v]
                out["warnings"].append(f"{'/'.join(fl)}-axis tick labels used in the opposite direction to the VLM's list: "
                                       f"the ticks fit a real (not mirror-image) view clearly better that way; {share:.0%} of the data lands inside the axis box")
    out["camera_info"] = {k: v for k, v in inf.items() if k != "hypotheses"} if inf else {}
    if cam is None:
        out["reject"] = f"3-D axes: {inf.get('reason')}"
        return None
    out["camera"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cam.items()}
    rms = inf.get("reprojection_rms_px", 99)
    pg = inf.get("pane_grid") or {}
    if pg.get("used"):
        # the camera was fitted to the box itself: the hundreds of pane grid pixels are the test;
        # the tick points sit 1-2 px off the box edge (tick marks drawn outward, line width)
        gm = pg.get("grid_med_after", 99)
        out["checks"].append(("3-D camera vs the pane grid lines (median px)", gm, "<= 1.5", gm <= 1.5))
        out["checks"].append(("3-D camera reprojection of the ticks on the box edges (px)", round(rms, 2), "<= 3.5", rms <= 3.5))
    else:
        out["checks"].append(("3-D camera reprojection of the ticks (px)", round(rms, 2), "<= 2.5", rms <= 2.5))
    return cam


def _inside_share(img, scale, area_fn, faded=False, n=600):
    """judge for _camera: share of colour-coded pixels that land inside the axis box when their
    colour gives z (coloured surfaces, markers coloured by z)"""
    def judge(cam):
        P = cam["P"]
        lim = cam.get("limits") or {}
        if not lim:
            return 0.0
        area = area_fn(cam)
        v, d = scale.value(img[area].astype(float), shaded=not faded, faded=faded)
        ys, xs = np.where(area)
        ok = d <= 25
        if ok.sum() < 20:
            return 0.0
        k = np.random.default_rng(0).choice(np.where(ok)[0], min(n, int(ok.sum())), replace=False)
        lx, ly = sorted(lim["x"]), sorted(lim["y"])
        px_, py_ = 0.05 * (lx[1] - lx[0]), 0.05 * (ly[1] - ly[0])
        inside = 0
        for i in k:
            try:
                x, y = solve_xy_given_z(P, (xs[i], ys[i]), float(v[i]))
            except np.linalg.LinAlgError:
                continue
            inside += (lx[0] - px_ <= x <= lx[1] + px_) and (ly[0] - py_ <= y <= ly[1] + py_)
        return inside / len(k)
    return judge


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
    # ---- refine: analysis by synthesis (bar3d.py) - draw all bars with the camera, compare with the
    # picture, adjust every height (partly hidden bars are fitted by what is left of them)
    from .bar3d import fit_bars
    nx, ny = len(xs), len(ys)
    H0 = np.full((nx, ny), np.nan)
    for (i, j), (z, _) in got.items():
        H0[int(round(xs[i])), int(round(ys[j]))] = z
    zb = 0.0 if min(cam["z_values"]) <= 0 <= max(cam["z_values"]) else cam["z_floor"]
    z_top = max(cam["limits"]["z"][1], max(cam["z_values"]))
    H, w_fit, colours, finfo = fit_bars(img, cam, bars, zb, z_top, init=H0)
    out["bar_width_grid_units"] = w_fit
    out["fit"] = {k: v for k, v in finfo.items() if k != "crop"}
    pos_to_x = {int(round(v)): k for k, v in enumerate(xs)}
    pos_to_y = {int(round(v)): k for k, v in enumerate(ys)}
    rows = []
    for a in range(nx):
        for b in range(ny):
            if H[a, b] != H[a, b]:
                continue
            i, j = pos_to_x[a], pos_to_y[b]
            rows.append({"x_category": cam["x_labels"][i], "y_category": cam["y_labels"][j], "height": round(float(H[a, b]), 6),
                         "x_pos": a, "y_pos": b, "rgb": [round(c, 1) for c in colours.get((a, b), [31, 119, 180])]})
    out["hidden_cells"] = [[cam["x_labels"][pos_to_x[a]], cam["y_labels"][pos_to_y[b]]] for a, b in finfo["hidden_cells"]]
    for (a, b) in finfo["hidden_cells"]:
        rows.append({"x_category": cam["x_labels"][pos_to_x[a]], "y_category": cam["y_labels"][pos_to_y[b]], "height": None,
                     "x_pos": a, "y_pos": b, "rgb": None,
                     "note": f"hidden behind taller bars: height at most {finfo['hidden_upper_bound'][(a, b)]:.3g}"})
    out["table"] = rows
    vis = [r_ for r_ in rows if r_["height"] is not None]
    out["series"]["height"] = {"cells": [[r_["x_category"], r_["y_category"]] for r_ in vis], "values": [r_["height"] for r_ in vis], "kind": "bar3d"}
    n_cells = len(cells)
    out["checks"] += [("bars read / grid cells", f"{len(vis)}/{n_cells}", ">= 50 %", len(vis) >= 0.5 * n_cells),
                      ("share of the bar pixels the fitted bars explain", finfo["fit_share_explained"], ">= 0.8",
                       finfo["fit_share_explained"] >= 0.8)]
    if len(vis) < n_cells:
        out["warnings"].append(f"{n_cells - len(vis)} bar(s) hidden behind taller bars: not read (an upper bound is given in the CSV)")
    return out


# ------------------------------------------------------------------ 3-D scatter
def pipeline_scatter3d(img, spec):
    out = _base(spec, img, "scatter3d")
    cb = spec.get("colorbar")
    scale, cinfo = (read_colorbar(img, cb) if cb else (None, {}))
    out["colorbar"] = cinfo
    if scale is not None:
        out["cmap_table"] = {"values": scale.values.tolist(), "rgb": scale.colours.round(1).tolist()}
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
    series = list(spec.get("series") or [])
    if not series:
        # the VLM listed no series: the pixels say what marker colours there are
        name = str((spec.get("z_axis") or {}).get("title") or (cb or {}).get("title") or "points")
        if colour_is_z:
            series = [{"name": name, "color": "colormap", "marker": "circle"}]
        else:
            from .bars import discover_colours
            cols = discover_colours(img, area, sat, min_share=0.05)
            series = [{"name": name if len(cols) == 1 else f"{name} ({k + 1})",
                       "color": "#%02x%02x%02x" % tuple(int(v) for v in c), "marker": "circle"} for k, c in enumerate(cols[:6])]
        out["warnings"].append(f"the VLM listed no series; {len(series)} marker colour(s) found in the pixels")
    for q in series:
        if colour_is_z:
            vals_, dist = scale.value(img[area])
            m = np.zeros(area.shape, bool); m[area] = (dist <= 22) & (sat[area] >= 0.12)
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
        if not colour_is_z:
            # markers partly hidden behind a marker of the same colour (points close in the picture)
            from .extract import find_hidden_markers
            hid = find_hidden_markers(m, pts, float(info.get("marker_px", 6)))
            if hid:
                pts = sorted(list(pts) + hid)
                out.setdefault("partly_hidden_markers", []).extend([[round(a, 1), round(b, 1)] for a, b, _ in hid])
        if stems and not colour_is_z:
            # a 'marker' hanging on another marker's stem (same column, joined by the line) may be a
            # stem piece (two stems drawn over each other look thick) - or a real marker whose stem
            # runs along the same image line (points stacked in depth; touching markers). A real
            # marker head is as wide as a marker across the stem and has a full marker core; a
            # thick stem piece is narrow.
            mpx_ = float(info.get("marker_px", 6)); typ = float(info.get("typical_core_area", 1.0))

            def head_width(cx, cy):
                yi, xi = int(round(cy)), int(round(cx))
                d = _down_dir(P, cx, cy); nrm = np.array([-d[1], d[0]])      # across the stem
                w_ = 0
                for sgn in (1, -1):
                    for k_ in range(1, int(2 * mpx_)):
                        qx, qy = int(round(cx + sgn * k_ * nrm[0])), int(round(cy + sgn * k_ * nrm[1]))
                        if not (0 <= qy < m.shape[0] and 0 <= qx < m.shape[1]) or not m[qy, qx]:
                            break
                        w_ += 1
                return w_ + 1
            keep = []
            for k, (cx, cy, a_) in enumerate(pts):
                above = [p_ for p_ in pts if p_[1] < cy - 3 and abs(p_[0] - cx) <= 3]
                hang = any(m[int(p_[1]):int(cy), max(0, int(round(cx)) - 1):int(round(cx)) + 2].any(1).mean() > 0.9 for p_ in above)
                if hang and head_width(cx, cy) >= 0.7 * mpx_ and a_ >= 0.5 * typ:
                    hang = False
                    out.setdefault("stacked_markers", []).append([round(cx, 1), round(cy, 1)])
                if not hang:
                    keep.append((cx, cy, a_))
            pts = keep
        rec = {"x": [], "y": [], "z": [], "px": [], "rgb_each": [], "kind": "marker3d", "depth": None,
               "marker": q.get("marker") or "circle", "marker_px": float(info.get("marker_px", 6))}
        # the series colour as drawn (median of its marker cores) - used to redraw it in the same colour
        mp_ = int(max(1, 0.3 * info.get("marker_px", 6)))
        n_try = len(pts)
        for (cx, cy, _a) in pts:
            if colour_is_z:
                r_ = int(max(1, 0.3 * info.get("marker_px", 6)))
                core = img[int(cy) - r_:int(cy) + r_ + 1, int(cx) - r_:int(cx) + r_ + 1].reshape(-1, 3)
                # 3-D markers are often depth-faded towards white: match the colour as a faded table colour
                z = float(np.median(scale.value(core)[0]))
                x, y = solve_xy_given_z(P, (cx, cy), z)
                rec["depth"] = "colour = z (colour bar)"
            elif stems:
                # candidate feet, farthest first: the walk may run on into another stem below; the
                # foot is the farthest end the marker sits straight above (within the axis box)
                got = None
                for foot in _stem_foot(m_soft, cx, cy, cam, info.get("marker_px", 6)):
                    x, y = solve_xy_given_z(P, foot, z_base)
                    z, res = solve_z_given_xy(P, (cx, cy), x, y)
                    inside = lim[0][0] - 0.05 * np.ptp(lim[0]) <= x <= lim[0][1] + 0.05 * np.ptp(lim[0]) and \
                        lim[1][0] - 0.05 * np.ptp(lim[1]) <= y <= lim[1][1] + 0.05 * np.ptp(lim[1])
                    if res <= 3.0 and inside and z >= z_base:
                        got = (x, y, z); break
                if got is None:
                    # its stem runs into another stem (both drawn on nearly the same image line): the
                    # stem's own line of sight - straight down from THIS marker, no re-centring onto
                    # the other stem - stays on stem pixels until the two lines part; that is the foot
                    f2 = _straight_foot(m_soft, cx, cy, P, info.get("marker_px", 6))
                    if f2 is not None:
                        x, y = solve_xy_given_z(P, f2, z_base)
                        z, res = solve_z_given_xy(P, (cx, cy), x, y)
                        inside = lim[0][0] - 0.05 * np.ptp(lim[0]) <= x <= lim[0][1] + 0.05 * np.ptp(lim[0]) and \
                            lim[1][0] - 0.05 * np.ptp(lim[1]) <= y <= lim[1][1] + 0.05 * np.ptp(lim[1])
                        if res <= 3.0 and inside and z >= z_base:
                            got = (x, y, z)
                            out.setdefault("feet_on_shared_stem", []).append([round(cx, 1), round(cy, 1)])
                if got is None:
                    continue
                x, y, z = got
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
            core = img[max(0, int(cy) - mp_):int(cy) + mp_ + 1, max(0, int(cx) - mp_):int(cx) + mp_ + 1].reshape(-1, 3).astype(float)
            rec["rgb_each"].append(np.median(core, 0).round(1).tolist())
        if rec["rgb_each"]:
            rec["rgb"] = np.median(np.array(rec["rgb_each"]), 0).round(1).tolist()
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
    out["table"] = [{"series": n, "x": a, "y": b, "z": c, "depth": d["depth"],
                     "rgb": "#%02x%02x%02x" % tuple(int(round(v)) for v in rgb)} for n, d in out["series"].items()
                    for a, b, c, rgb in zip(d["x"], d["y"], d["z"], d["rgb_each"])]
    n = len(out["table"])
    out["checks"].append(("points extracted", n, ">= 1", n >= 1))
    return out


def _down_dir(P, x, y):
    """image direction of -z at pixel (x, y). Under perspective, vertical lines converge, so the
    direction depends on where the point is: d ~ p * c3 - c_xy with c = P's z column."""
    c = np.asarray(P, float)[:, 2]
    d = np.array([x * c[2] - c[0], y * c[2] - c[1]])
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.array([0.0, 1.0])
    d = d / n
    return d if d[1] > 0 else -d               # -z points down the image in a normal 3-D view


def _straight_foot(m, cx, cy, P, mpx, tol=1, max_gap=2):
    """end of the stem pixels met when walking from the marker exactly along the image direction
    of -z (perspective-corrected at every step), accepting only pixels within tol px of that
    line: a neighbouring stem that diverges from it is left where they part"""
    H, W = m.shape
    d = _down_dir(P, cx, cy)
    x, y = cx + 0.6 * mpx * d[0], cy + 0.6 * mpx * d[1]
    last, gap = None, 0
    for _ in range(2 * H):
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < W and 0 <= yi < H):
            break
        nrm = (-d[1], d[0])
        hit = any(0 <= int(round(y + k * nrm[1])) < H and 0 <= int(round(x + k * nrm[0])) < W and
                  m[int(round(y + k * nrm[1])), int(round(x + k * nrm[0]))] for k in range(-tol, tol + 1))
        if hit:
            last, gap = (x, y), 0
        else:
            gap += 1
            if gap > max_gap:
                break
        d = _down_dir(P, x, y)
        x, y = x + d[0], y + d[1]
    if last is None or np.hypot(last[0] - cx, last[1] - cy) < mpx:
        return None
    return last


def _stem_foot(m, cx, cy, cam, mpx):
    """follow a drop line from the marker downwards (image direction of -z, re-computed at every
    step because of perspective) to its end"""
    P = cam["P"]
    H, W = m.shape
    x, y = float(cx), float(cy)
    d = _down_dir(P, x, y)
    x, y = x + 0.6 * mpx * d[0], y + 0.6 * mpx * d[1]
    last, gap, t = None, 0, 0
    ends = []                                   # end of every run of stem pixels (a stem may pass behind things)
    while gap <= 8 and t < 2 * H:
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < W and 0 <= yi < H):
            break
        win = m[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2]
        if win.any():
            # re-centre on the stem across the walking direction (a 1-px drift is corrected)
            ys_, xs_ = np.where(win)
            if gap == 0 and len(xs_):
                x += 0.5 * (np.mean(xs_) + max(0, xi - 1) - xi) * abs(d[1])
            last, gap = (x, y), 0
        else:
            if gap == 0 and last is not None:
                ends.append(last)
            gap += 1
        d = _down_dir(P, x, y)
        x, y = x + d[0], y + d[1]
        t += 1
    if last is not None and (not ends or ends[-1] != last):
        ends.append(last)
    ends = [e for e in ends if np.hypot(e[0] - cx, e[1] - cy) >= mpx]
    return ends[::-1]                           # farthest first


# ------------------------------------------------------------------ 3-D surface
def _trace_lines_3d(mask, P, lim, max_pts=6000):
    """the drawn lines (wireframe) as 3-D segments for the redraw: the line pixels are thinned to
    1 px and every pair of touching skeleton pixels becomes a segment; each pixel is placed on its
    line of sight (depth unknown, so the redraw shows the lines exactly where the input has them)"""
    from skimage.morphology import skeletonize
    sk = skeletonize(mask)
    ys, xs = np.where(sk)
    if len(xs) == 0:
        return []
    if len(xs) > max_pts:
        k = np.random.default_rng(0).choice(len(xs), max_pts, replace=False); keep = np.zeros(len(xs), bool); keep[k] = True
    else:
        keep = np.ones(len(xs), bool)
    idx = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs)) if keep[i]}
    pts = {}
    for (y, x), i in idx.items():
        mid = _ray_midpoint(P, (x, y), lim)
        if mid is not None:
            pts[(y, x)] = [round(float(v), 5) for v in mid[0]]
    segs = []
    for (y, x), p in pts.items():
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            q = pts.get((y + dy, x + dx))
            if q is not None:
                segs.append([p, q])
    return segs


def pipeline_surface3d(img, spec, step=3):
    out = _base(spec, img, "surface3d")
    cb = spec.get("colorbar")
    scale, cinfo = (read_colorbar(img, cb) if cb else (None, {}))
    out["colorbar"] = cinfo
    excl = colorbar_exclusion(cinfo, img.shape, 8) if cinfo.get("found") else None
    judge = None
    if scale is not None:
        g0 = img.mean(-1); s0 = (img.max(-1).astype(float) - img.min(-1)) / np.maximum(img.max(-1), 1)
        judge = _inside_share(img, scale, lambda c: _plot_area(img, c, excl) & ~((s0 < 0.06) & (g0 > 200)))
    cam = _camera(img, spec, out, excl, judge)
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
        # grey axis lines / tick marks can sit close to a dark end of the colour map: a pixel must be
        # at least half as saturated as the colour it is matched to
        C = scale.colours
        csat = (C.max(1) - C.min(1)) / np.maximum(C.max(1), 1)
        jj = np.abs(scale.values[None, :] - v[:, None]).argmin(1)
        ok &= ~((csat[jj] > 0.2) & (sat[py, px] < 0.5 * csat[jj]))
        dpx, dpy, drgb = [], [], []
        for a, b, z in zip(px[ok], py[ok], v[ok]):
            try:
                x, y = solve_xy_given_z(P, (a, b), z)
            except np.linalg.LinAlgError:
                continue
            if lim[0][0] <= x <= lim[0][1] and lim[1][0] <= y <= lim[1][1]:
                xs.append(x); ys.append(y); zs.append(float(z))
                dpx.append(int(a)); dpy.append(int(b)); drgb.append(img[b, a].tolist())
        # for the redraw: where each sample sits in the image and its colour as drawn (shading and
        # mesh lines included), so the surface can be rebuilt as a mesh in the input's own colours
        # also the colour of every other grid node in the plot area (white-ish surface parts look
        # like the panes and do not match the colour scale; the redraw fills them in place)
        out["draw"] = {"step": step, "px": dpx, "py": dpy, "rgb": drgb,
                       "all_px": xx[area[yy, xx]].tolist(), "all_py": yy[area[yy, xx]].tolist(),
                       "all_rgb": img[yy[area[yy, xx]], xx[area[yy, xx]]].tolist()}
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
        lc = img[yy[sel2], xx[sel2]].astype(float)
        out["line_rgb"] = np.median(lc, 0).round(1).tolist() if len(lc) else None
        # the axis lines themselves are not part of the wireframe
        ax_m = np.zeros_like(dark)
        from PIL import Image as _I, ImageDraw as _D
        im_ = _I.new("L", (dark.shape[1], dark.shape[0]), 0); dd_ = _D.Draw(im_)
        for (x0, y0), (x1, y1) in (cam.get("axis_lines") or {}).values():
            dd_.line([(x0, y0), (x1, y1)], fill=255, width=7)
        ax_m = np.array(im_) > 0
        out["draw_lines"] = _trace_lines_3d(dark & ~ax_m, P, lim)
        out["flag"] = ("depth not recoverable: no colour bar - the surface's z cannot be read from one view; "
                       "points are guesses along each pixel's line of sight; kept out of the accuracy scores")
        out["depth"] = "UNKNOWN"
    out["series"]["surface"] = {"x": np.round(xs, 6).tolist(), "y": np.round(ys, 6).tolist(), "z": np.round(zs, 6).tolist(), "kind": "surface3d"}
    out["table"] = [{"x": a, "y": b, "z": c} for a, b, c in zip(*[out["series"]["surface"][k] for k in "xyz"])]
    out["checks"].append(("surface samples", len(xs), ">= 50", len(xs) >= 50))
    return out
