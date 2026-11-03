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
    vals = np.zeros((nr, nc)); dist = np.zeros((nr, nc)); rgb = np.zeros((nr, nc, 3))
    for i in range(nr):
        for j in range(nc):
            c = cell_colour(img, x0 + j * cw, y0 + i * ch, x0 + (j + 1) * cw, y0 + (i + 1) * ch)
            v, d = scale.value(c)
            vals[i, j], dist[i, j], rgb[i, j] = float(v), float(d), c
    out.update(rows=rows, columns=cols, values=vals.round(6).tolist(), colour_distance=dist.round(1).tolist(),
               cell_rgb=rgb.round(1).tolist(), cmap_table={"values": scale.values.tolist(), "rgb": scale.colours.round(1).tolist()})
    out["table"] = [{"row": r, "column": c, "value": float(vals[i, j]), "colour_distance": float(dist[i, j])}
                    for i, r in enumerate(rows) for j, c in enumerate(cols)]
    med = float(np.median(dist))
    out["checks"] += [("colorbar label fit (px)", round(cinfo["rms_px"], 2), "<= 3", cinfo["rms_px"] <= 3),
                      ("median cell colour distance to the colour scale", round(med, 1), "<= 20", med <= 20),
                      ("cells whose colour is not on the scale (%)", round(100 * float((dist > 40).mean()), 1), "<= 20",
                       float((dist > 40).mean()) <= 0.2)]
    return out


def pipeline_field(img, spec, samples=60):
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
    L, R, T, B = frame["left"] + lw, frame["right"] - lw, frame["top"] + lw, frame["bottom"] - lw
    px = np.linspace(L, R, samples); py = np.linspace(B, T, samples)
    Z = np.full((samples, samples), np.nan); Dd = np.full((samples, samples), np.nan)
    for i, yy in enumerate(py):
        for j, xx in enumerate(px):
            a, b = int(round(xx)), int(round(yy))
            nb = img[b - 1:b + 2, a - 1:a + 2].reshape(-1, 3)
            v, d = scale.value(nb)
            k = int(np.argmin(d))
            if d[k] <= 35:
                Z[i, j], Dd[i, j] = v[k], d[k]
    xs, ys = X.to_value(px), Y.to_value(py)
    out.update(x=xs.round(6).tolist(), y=ys.round(6).tolist(), z=np.where(np.isnan(Z), None, Z.round(6)).tolist(),
               levels=cinfo.get("levels"), cmap_table={"values": scale.values.tolist(), "rgb": scale.colours.round(1).tolist()})
    out["table"] = [{"x": float(xs[j]), "y": float(ys[i]), "z": float(Z[i, j])}
                    for i in range(samples) for j in range(samples) if not np.isnan(Z[i, j])]
    miss = float(np.isnan(Z).mean())
    rms = max(cal_info["x_axis"].get("rms_px", 0), cal_info["y_axis"].get("rms_px", 0))
    out["checks"] += [("axis label fit (px)", round(rms, 2), "<= 3", rms <= 3),
                      ("colorbar label fit (px)", round(cinfo["rms_px"], 2), "<= 3", cinfo["rms_px"] <= 3),
                      ("sample points whose colour is not on the scale (%)", round(100 * miss, 1), "<= 30", miss <= 0.3)]
    return out
