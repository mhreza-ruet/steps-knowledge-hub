"""Heatmaps (a grid of coloured cells with category labels) and 2-D fields
(filled contours / colour maps over numeric x-y axes, i.e. a surface seen from above).

The VLM reads: row and column names (heatmap) or the x / y tick labels (field),
and the colorbar's tick labels. The pixels give everything else:

heatmap
  1. colorbar -> colour-to-value table (colorbar.py)
  2. the matrix = the large coloured blocks outside the colorbar
  3. the grid: the VLM's row / column counts, checked against the colour edges
     in the pixels (a count that does not line up with the edges is replaced
     by the count the edges show)
  4. every cell: median colour of its inner half, after dropping pixels far
     from that median (printed numbers) -> value
field
  1. colorbar; the colorbar is painted out, then frame + x / y calibration
     exactly as for line plots
  2. a regular grid of sample points inside the frame; at each point the
     pixel in its 3x3 neighbourhood that is closest to the colour scale is used
     (this steps around contour lines and labels)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import detect as D
from .colorbar import read_colorbar, colorbar_exclusion, _not_white


def _edge_profile(img, region, axis):
    """mean colour change across the region at every position along one axis"""
    x0, y0, x1, y1 = region
    sub = img[y0:y1, x0:x1].astype(float)
    if axis == "x":
        g = np.abs(np.diff(sub, axis=1)).max(-1).mean(0)
    else:
        g = np.abs(np.diff(sub, axis=0)).max(-1).mean(1)
    return g


def _grid_score(prof, n):
    """how strongly the colour changes at ALL n-1 expected cell boundaries
    (lower quartile, so a count that misses some boundaries scores low)"""
    if n < 2:
        return 0.0
    L = len(prof) + 1
    pos = [round(k * L / n) - 1 for k in range(1, n)]
    vals = [prof[max(0, p - 2):p + 3].max() for p in pos if 0 <= p < len(prof)]
    return float(np.percentile(vals, 25)) / (float(np.median(prof)) + 1.0)


def choose_count(prof, n_vlm, n_max=80):
    """number of cells along one axis: the VLM's count unless the colour edges
    disagree. Counts whose every boundary shows an edge are 'good'; a divisor of
    the true count is good too, so the largest good count is the pixels' answer."""
    scores = {n: _grid_score(prof, n) for n in range(2, min(n_max, len(prof) // 4) + 1)}
    if not scores:
        return max(1, n_vlm), {}
    best = max(scores.values())
    if best < 3:                          # no visible cell edges: trust the VLM
        return max(1, n_vlm), scores
    good = [n for n, s_ in scores.items() if s_ >= 0.5 * best]
    bigger = [n for n in good if n_vlm >= 1 and n > n_vlm and n % n_vlm == 0]
    if n_vlm in scores and scores[n_vlm] >= 0.3 * best and not bigger:
        return n_vlm, scores                   # the VLM's count lines up with the edges
    return max(good), scores


def matrix_region(img, excl):
    """bounding box of the large coloured blocks (cells) outside the colorbar"""
    m = _not_white(img) & ~excl
    lab, n = ndi.label(m)
    boxes = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        area = int((lab[sl] == i).sum())
        if area >= 150 and min(h, w) >= 6 and area >= 0.5 * h * w:
            boxes.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop, area))
    if not boxes:
        return None
    # the cells form one block: keep boxes overlapping / adjacent to the biggest one, grow
    boxes.sort(key=lambda b: -b[4])
    x0, y0, x1, y1, _ = boxes[0]
    changed = True
    while changed:
        changed = False
        for b in boxes:
            if b[0] <= x1 + 6 and b[2] >= x0 - 6 and b[1] <= y1 + 6 and b[3] >= y0 - 6:
                nx0, ny0, nx1, ny1 = min(x0, b[0]), min(y0, b[1]), max(x1, b[2]), max(y1, b[3])
                if (nx0, ny0, nx1, ny1) != (x0, y0, x1, y1):
                    x0, y0, x1, y1 = nx0, ny0, nx1, ny1; changed = True
    return int(x0), int(y0), int(x1), int(y1)


def _strip_frame(img, region, max_px=6):
    """an axes frame (dark line) or a light anti-aliased edge drawn around the
    cells is not part of them: peel rows / columns that are mostly dark-grey or
    mostly near-white off each side"""
    x0, y0, x1, y1 = region
    g = img.mean(-1)
    sat = img.max(-1).astype(float) - img.min(-1)

    def junk(v, s_):
        return (np.mean((v < 90) & (s_ < 40)) > 0.6) or (np.mean(v > 238) > 0.6)
    for _ in range(max_px):
        moved = False
        if junk(g[y0:y1, x0], sat[y0:y1, x0]): x0 += 1; moved = True
        if junk(g[y0:y1, x1 - 1], sat[y0:y1, x1 - 1]): x1 -= 1; moved = True
        if junk(g[y0, x0:x1], sat[y0, x0:x1]): y0 += 1; moved = True
        if junk(g[y1 - 1, x0:x1], sat[y1 - 1, x0:x1]): y1 -= 1; moved = True
        if not moved:
            break
    return x0, y0, x1, y1


def cell_colour(img, x0, y0, x1, y1):
    """robust colour of one cell: inner half, pixels near the median"""
    w, h = x1 - x0, y1 - y0
    sub = img[int(y0 + 0.25 * h):int(np.ceil(y1 - 0.25 * h)), int(x0 + 0.25 * w):int(np.ceil(x1 - 0.25 * w))].reshape(-1, 3).astype(float)
    if len(sub) == 0:
        sub = img[int((y0 + y1) / 2), int((x0 + x1) / 2)].reshape(1, 3).astype(float)
    med = np.median(sub, 0)
    keep = np.abs(sub - med).max(1) <= 30
    return np.median(sub[keep], 0) if keep.sum() >= 3 else med


def cell_gaps(img, reg, nr, nc):
    """lines drawn between the cells (white gaps or black borders): colour and width"""
    x0, y0, x1, y1 = reg
    ch, cw = (y1 - y0) / nr, (x1 - x0) / nc
    found = []
    for j in range(1, nc):
        p = int(round(x0 + j * cw))
        strip = img[int(y0 + 2):int(y1 - 2), max(0, p - 4):p + 5].astype(float)
        if strip.size == 0:
            continue
        g = strip.mean(-1); sat = strip.max(-1) - strip.min(-1)
        for kind, m in (("white", (g > 232) & (sat < 20)), ("black", (g < 60) & (sat < 30))):
            col = m.mean(0) >= 0.85
            if col.any():
                found.append((kind, int(col.sum())))
    if len(found) < 0.6 * max(1, nc - 1):
        return None
    kinds = [k for k, _ in found]
    kind = max(set(kinds), key=kinds.count)
    w = float(np.median([n for k, n in found if k == kind]))
    return {"colour": kind, "rgb": [255, 255, 255] if kind == "white" else [0, 0, 0], "width_px": w}


def text_colour(img, x0, y0, x1, y1, cell_rgb, gap=None):
    """colour of the number printed in a cell (pixels far from the cell colour), or None"""
    pad = (gap or {}).get("width_px", 0) + 2
    sub = img[int(y0 + pad):int(np.ceil(y1 - pad)), int(x0 + pad):int(np.ceil(x1 - pad))].reshape(-1, 3).astype(float)
    if len(sub) < 20:
        return None
    far = np.abs(sub - cell_rgb).max(1) > 70
    if far.sum() < max(6, 0.01 * len(sub)) or far.mean() > 0.45:
        return None
    core = sub[far]
    d = np.abs(core - cell_rgb).max(1)
    return np.median(core[d >= np.percentile(d, 50)], 0).round(1).tolist()


def label_rotation(img, reg, axis):
    """tick labels under the grid: horizontal (0), slanted (45) or vertical (90).
    One label is a cluster of glyphs; its height relative to one text line tells."""
    x0, y0, x1, y1 = reg
    H, W = img.shape[:2]
    band = img[min(H - 1, y1 + 3):min(H, y1 + 120), max(0, x0 - 40):min(W, x1 + 40)]
    if band.size == 0:
        return 0
    ink = band.mean(-1) < 140
    if ink.sum() < 20:
        return 0
    rows = np.where(ink.any(1))[0]
    # the labels are the first block of ink rows below the grid (a title / colour bar may follow)
    stop = rows[0]
    for r in rows[1:]:
        if r - stop > 6:
            break
        stop = r
    blk = ink[rows[0]:stop + 1]
    lab, n = ndi.label(ndi.binary_dilation(blk, np.ones((3, 3)), iterations=2))
    if n == 0:
        return 0
    hs, ws = [], []
    for sl in ndi.find_objects(lab):
        h_, w_ = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if h_ > 7 or w_ > 7:                          # skip tick marks
            hs.append(h_); ws.append(w_)
    if not hs:
        return 0
    h, w = float(np.median(hs)), float(np.median(ws))
    # one horizontal text line (after the 2-px dilation) is ~12-16 px high at 8-10 pt
    if h > 2.2 * w and h > 17:
        return 90
    if h > 17 and 0.55 <= h / w <= 1.8:
        return 45
    return 0


def decimals_from_width(img, reg, nr, nc, vals, rgb, gap):
    """how many decimals the printed numbers have, from the width of the printed text: the text
    width in character units is compared with each candidate format of the cell's value"""
    x0, y0, x1, y1 = reg
    ch, cw = (y1 - y0) / nr, (x1 - x0) / nc
    pad = (gap or {}).get("width_px", 0) + 2
    widths, heights, vv = [], [], []
    for i in range(nr):
        for j in range(nc):
            a, b = int(x0 + j * cw + pad), int(np.ceil(x0 + (j + 1) * cw - pad))
            c, d = int(y0 + i * ch + pad), int(np.ceil(y0 + (i + 1) * ch - pad))
            far = np.abs(img[c:d, a:b].astype(float) - rgb[i, j]).max(-1) > 70
            if far.sum() < 6:
                continue
            ys, xs = np.where(far)
            widths.append(np.ptp(xs) + 1); heights.append(np.ptp(ys) + 1); vv.append(vals[i, j])
    if len(widths) < 3:
        return None
    cw_px = 0.85 * float(np.median(heights))         # digit advance ~ 0.85 x digit ink height (DejaVu / Arial)
    n = np.array(widths) / cw_px

    def units(t):
        return sum(0.45 if ch_ == "." else 1.0 for ch_ in t)
    err = {d: float(np.median([abs(units(f"{v:.{d}f}") - k) for v, k in zip(vv, n)])) for d in range(4)}
    return min(err, key=err.get)


def _decimals(s):
    s = str(s).strip().replace("\u2212", "-")
    return len(s.split(".")[1]) if "." in s else 0


def verify_cell_text(spec, px_vals, scale, nr, nc):
    """compare the VLM's printed cell numbers with the value of each cell's colour.
    Agreeing cells take the printed number (exact); disagreeing ones keep the colour value
    and are listed. If most cells disagree the VLM's table is not used at all."""
    from .calibrate import parse_number
    info = {"n_read": 0}
    cv = spec.get("cell_values") or []
    if not cv:
        return None, info
    grid = [[x for x in r] for r in cv]
    shape_ok = len(grid) == nr and all(len(r) == nc for r in grid)
    if not shape_ok and len(grid) == nc and all(len(r) == nr for r in grid):
        grid = [list(r) for r in zip(*grid)]; shape_ok = True; info["transposed"] = True
    if not shape_ok:
        info["note"] = (f"VLM printed-number table is {len(grid)} x {max((len(r) for r in grid), default=0)}, "
                        f"the grid is {nr} x {nc} - not used")
        return None, info
    rng = float(np.ptp(scale.values)) or 1.0
    tol = max(0.08 * rng, 1.5 * rng / max(len(scale.values), 1))
    res, agree, dis, decs = {}, 0, [], []
    for i in range(nr):
        for j in range(nc):
            v = parse_number(grid[i][j])
            if v is None:
                continue
            info["n_read"] += 1
            decs.append(_decimals(grid[i][j]))
            ok = abs(v - px_vals[i, j]) <= tol
            agree += ok
            if not ok:
                dis.append((i, j, str(grid[i][j]), float(px_vals[i, j])))
            res[(i, j)] = (float(v), str(grid[i][j]).strip(), ok)
    if not info["n_read"]:
        return None, info
    info["agree_share"] = agree / info["n_read"]
    info["decimals"] = int(max(set(decs), key=decs.count)) if decs else None
    info["tolerance"] = tol
    if info["agree_share"] < 0.5:
        info["note"] = (f"only {agree}/{info['n_read']} printed cell numbers agree with the cell colours - "
                        "the VLM's numbers are not used (colour values kept)")
        return None, info
    info["disagree"] = dis
    return res, info


def pipeline_heatmap(img, spec):
    H, W = img.shape[:2]
    rows = list(spec.get("rows") or [])
    cols = list(spec.get("columns") or [])
    out = {"chart_type": "heatmap", "size": (W, H), "spec": spec, "warnings": [], "checks": [], "series": {}}
    scale, cinfo = read_colorbar(img, spec.get("colorbar"))
    out["colorbar"] = cinfo
    if scale is None:
        out["reject"] = f"heatmap: {cinfo.get('reason')} - cell colours cannot be turned into values"
        return out
    excl = colorbar_exclusion(cinfo, img.shape)
    reg = matrix_region(img, excl)
    if reg is None:
        out["reject"] = "heatmap: no block of coloured cells found"; return out
    reg = _strip_frame(img, reg)
    x0, y0, x1, y1 = reg
    out["region"] = reg
    nr, sr = choose_count(_edge_profile(img, reg, "y"), len(rows))
    nc, sc = choose_count(_edge_profile(img, reg, "x"), len(cols))
    for what, n_vlm, n_px, names in (("rows", len(rows), nr, rows), ("columns", len(cols), nc, cols)):
        if n_vlm != n_px:
            out["warnings"].append(f"VLM listed {n_vlm} {what}, the colour edges show {n_px} - pixels used")
    rows = (rows + [f"row {i + 1}" for i in range(len(rows), nr)])[:nr]
    cols = (cols + [f"col {j + 1}" for j in range(len(cols), nc)])[:nc]
    ch, cw = (y1 - y0) / nr, (x1 - x0) / nc
    gap = cell_gaps(img, reg, nr, nc)
    out["cell_gap"] = gap
    vals = np.zeros((nr, nc)); dist = np.zeros((nr, nc)); rgb = np.zeros((nr, nc, 3)); txt = {}
    for i in range(nr):
        for j in range(nc):
            c = cell_colour(img, x0 + j * cw, y0 + i * ch, x0 + (j + 1) * cw, y0 + (i + 1) * ch)
            v, d = scale.value(c)
            vals[i, j], dist[i, j], rgb[i, j] = float(v), float(d), c
            t = text_colour(img, x0 + j * cw, y0 + i * ch, x0 + (j + 1) * cw, y0 + (i + 1) * ch, c, gap)
            if t is not None:
                txt[(i, j)] = t
    px_vals = vals.copy()
    # second layer: the numbers printed in the cells, read by the VLM, checked against the colours
    vlm_vals, vinfo = verify_cell_text(spec, px_vals, scale, nr, nc)
    out["cell_text_check"] = vinfo
    shown = [[None] * nc for _ in range(nr)]
    if vlm_vals is not None:
        for (i, j), (v_, s_, ok) in vlm_vals.items():
            shown[i][j] = s_
            if ok:
                vals[i, j] = v_
        if vinfo.get("disagree"):
            out["warnings"].append(f"{len(vinfo['disagree'])} printed cell numbers disagree with the cell colour - colour value kept there: "
                                   + ", ".join(f"({a + 1},{b + 1}) printed {t} / colour {c_:.3g}" for a, b, t, c_ in vinfo["disagree"][:5]))
    if vinfo.get("note"):
        out["warnings"].append(vinfo["note"])
    dec = vinfo.get("decimals")
    if dec is None and txt:
        dec = decimals_from_width(img, reg, nr, nc, px_vals, rgb, gap)
    out["annotation"] = {"present": bool(txt) and (spec.get("annotated") or len(txt) >= 0.5 * nr * nc),
                         "text": shown, "decimals": dec,
                         "rgb": {f"{i},{j}": t for (i, j), t in txt.items()}}
    out["tick_rotation"] = {"x": label_rotation(img, reg, "x"), "y": 0}
    out["pixel_values"] = px_vals.round(6).tolist()
    out.update(rows=rows, columns=cols, values=vals.round(6).tolist(), colour_distance=dist.round(1).tolist(),
               cell_rgb=rgb.round(1).tolist(), cmap_table={"values": scale.values.tolist(), "rgb": scale.colours.round(1).tolist()})
    out["table"] = [{"row": r, "column": c, "value": float(vals[i, j]), "colour_distance": float(dist[i, j])}
                    for i, r in enumerate(rows) for j, c in enumerate(cols)]
    med = float(np.median(dist))
    if vinfo.get("n_read") and vinfo.get("agree_share", 0) >= 0.5:      # the VLM's numbers are used: they must mostly agree
        out["checks"].append(("printed cell numbers that agree with the cell colour (%)", round(100 * vinfo["agree_share"], 1), ">= 70",
                              vinfo["agree_share"] >= 0.7))
    out["checks"] += [("colorbar label fit (px)", round(cinfo["rms_px"], 2), "<= 3", cinfo["rms_px"] <= 3),
                      ("median cell colour distance to the colour scale", round(med, 1), "<= 20", med <= 20),
                      ("cells whose colour is not on the scale (%)", round(100 * float((dist > 40).mean()), 1), "<= 20",
                       float((dist > 40).mean()) <= 0.2)]
    return out


def nice_levels(edges, tick_values):
    """contour level boundaries measured on a stepped colour bar (e.g. 2.004, 3.988, ...) snapped
    to the regular grid the printed tick labels imply (2, 4, ...): step = tick spacing / k"""
    e = np.sort(np.asarray(edges, float))
    if len(e) < 3:
        return e.tolist()
    step = float(np.median(np.diff(e)))
    t = np.sort(np.asarray([v for v in tick_values if v is not None], float))
    if len(t) >= 2:
        ts = float(np.median(np.diff(t)))
        cands = [ts * m / k for k in (1, 2, 3, 4, 5, 10) for m in (1, 2, 3, 5)]
        nice = min(cands, key=lambda c: abs(c - step) / step)
        if abs(nice - step) <= 0.12 * step:
            anchor = t[np.argmin(np.abs(t - e[0]))]
            k0 = np.round((e - anchor) / nice)
            snapped = anchor + k0 * nice
            if np.max(np.abs(snapped - e)) <= 0.25 * nice:
                return [float(round(v, 10)) for v in snapped]
    return e.tolist()


def contour_lines(img, region, scale):
    """thin lines drawn over a filled map (contour lines): pixels far from every colour of the
    scale.  Returns colour and share of the area, or None"""
    L, T, R, B = region
    sub = img[T:B, L:R].astype(float)
    v, d = scale.value(sub)
    far = d > 60
    g = sub.mean(-1)
    dark = far & (g < 110)
    share = float(dark.mean())
    if share < 0.003:
        return None
    # lines are thin: most far pixels have a near-scale pixel within 2 px
    thin = float((ndi.binary_erosion(dark, np.ones((3, 3))).sum()) / max(dark.sum(), 1))
    if thin > 0.35:
        return None
    return {"rgb": np.median(sub[dark], 0).round(1).tolist(), "share": round(share, 4), "width_px": 1.0}


def _snap_limits(lim, labels, frac=0.02):
    """a map usually ends exactly on a tick value: snap a measured end that lies within 2 % of
    the span of a printed tick value onto it"""
    from .calibrate import parse_number
    t = [v for v in (parse_number(x) for x in labels or []) if v is not None]
    span = abs(lim[1] - lim[0]) or 1.0
    out = []
    for e in lim:
        near = min(t, key=lambda v: abs(v - e)) if t else None
        out.append(float(near) if near is not None and abs(near - e) <= frac * span else float(e))
    return out


def pipeline_field(img, spec, step_px=3):
    """filled contour / colour map over numeric axes"""
    from .extract import calibrate_all
    H, W = img.shape[:2]
    out = {"chart_type": "contour", "size": (W, H), "spec": spec, "warnings": [], "checks": [], "series": {}}
    scale, cinfo = read_colorbar(img, spec.get("colorbar"))
    out["colorbar"] = cinfo
    if scale is None:
        out["reject"] = f"contour/colour map: {cinfo.get('reason')}"; return out
    excl = colorbar_exclusion(cinfo, img.shape, pad=6)
    clean = img.copy(); clean[excl] = 255
    frame = D.detect_frame(clean)
    out["frame"] = frame
    cal, cal_info = calibrate_all(clean, frame, spec)
    out["calibration"] = {k: [m.as_dict() for m in v] for k, v in cal.items()}
    out["calibration_info"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("ticks", "labels")} for k, v in cal_info.items()}
    if any(cal_info.get(k, {}).get("relative") or k not in cal for k in ("x_axis", "y_axis")):
        out["reject"] = "contour/colour map: x or y axis has no numeric calibration"; return out
    X, Y = cal["x_axis"][0], cal["y_axis"][0]
    lw = int(np.ceil(frame.get("line_width", 1)))
    L, R, T, B = (int(round(v)) for v in (frame["left"] + lw, frame["right"] - lw, frame["top"] + lw, frame["bottom"] - lw))
    # every pixel of the map -> value (vectorised); a sample takes the pixel of its 3x3 neighbourhood
    # closest to the colour scale (steps around contour lines and labels)
    sub = img[T - 1:B + 2, L - 1:R + 2].astype(float)
    Vp, Dp = scale.value(sub)
    nx = max(20, int((R - L) / step_px) + 1); ny = max(20, int((B - T) / step_px) + 1)
    px = np.linspace(L, R, nx); py = np.linspace(B, T, ny)
    ix = np.round(px - (L - 1)).astype(int); iy = np.round(py - (T - 1)).astype(int)
    best_d = np.full((ny, nx), np.inf); Z = np.full((ny, nx), np.nan)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy = np.clip(iy + dy, 0, Vp.shape[0] - 1)[:, None]; xx = np.clip(ix + dx, 0, Vp.shape[1] - 1)[None, :]
            d_, v_ = Dp[yy, xx], Vp[yy, xx]
            better = d_ < best_d
            best_d = np.where(better, d_, best_d); Z = np.where(better, v_, Z)
    Z = np.where(best_d <= 35, Z, np.nan)
    xs, ys = X.to_value(px), Y.to_value(py)
    from .calibrate import parse_number
    ticks = [parse_number(t) for t in cinfo.get("tick_labels", [])]
    levels = nice_levels(cinfo["levels"], ticks) if cinfo.get("levels") else None
    out.update(x=xs.round(6).tolist(), y=ys.round(6).tolist(), z=np.where(np.isnan(Z), None, Z.round(6)).tolist(),
               levels=levels, levels_measured=cinfo.get("levels"),
               cmap_table={"values": scale.values.tolist(), "rgb": scale.colours.round(1).tolist()})
    out["contour_lines"] = contour_lines(img, (L, T, R, B), scale)
    raw = {"x": [float(X.to_value(frame["left"])), float(X.to_value(frame["right"]))],
           "y": [float(Y.to_value(frame["bottom"])), float(Y.to_value(frame["top"]))]}
    out["axis_limits_measured"] = raw
    out["axis_limits"] = {k: _snap_limits(v, spec.get(f"{k}_axis", {}).get("tick_labels")) for k, v in raw.items()}
    out["table"] = [{"x": float(xs[j]), "y": float(ys[i]), "z": float(Z[i, j])}
                    for i in range(ny) for j in range(nx) if not np.isnan(Z[i, j])]
    miss = float(np.isnan(Z).mean())
    rms = max(cal_info["x_axis"].get("rms_px", 0), cal_info["y_axis"].get("rms_px", 0))
    out["checks"] += [("axis label fit (px)", round(rms, 2), "<= 3", rms <= 3),
                      ("colorbar label fit (px)", round(cinfo["rms_px"], 2), "<= 3", cinfo["rms_px"] <= 3),
                      ("sample points whose colour is not on the scale (%)", round(100 * miss, 1), "<= 30", miss <= 0.3)]
    return out
