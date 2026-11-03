"""Pie charts (flat or 3D / tilted): VLM spec (slice names, colours, printed %) + pixels -> shares.

No per-figure settings:
  1. bright, saturated pixels = the top face of the pie; their hue peaks are the
     slice colours, paired with the VLM's colour names (Hungarian on hue)
  2. the apex = where the most slice colours meet
  3. the top face's extent from the apex (left, right, up, down) gives the
     perspective ellipse; every pixel angle is measured in that normalised frame
  4. walking around the apex on 25 rings, each slice's angular share = its percentage
  5. slices ordered clockwise from 12 o'clock (= the benchmark's order)
The percentages the VLM read from the printed labels are used when they agree
with the measured angle (pixel check); otherwise the measurement is used.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .extract import colour_target, hsv_arrays, hue_dist


def pie_point(cal, theta):
    cx, cy = cal["apex"]
    dx, dy = np.cos(theta), np.sin(theta)
    return (cx + np.where(dx >= 0, cal["right"], cal["left"]) * dx,
            cy - np.where(dy >= 0, cal["up"], cal["down"]) * dy)


def hue_peaks(hue, mask, k, min_sep=18):
    hist, edges = np.histogram(hue[mask], bins=180, range=(0, 360))
    hist = ndi.uniform_filter1d(hist.astype(float), 3, mode="wrap")
    cent = (edges[:-1] + edges[1:]) / 2
    order = np.argsort(hist)[::-1]
    peaks = []
    for i in order:
        if hist[i] < 0.0005 * mask.sum() or len(peaks) >= k:
            break
        if all(hue_dist(cent[i], p) > min_sep for p in peaks):
            peaks.append(float(cent[i]))
    return peaks


def refine_apex(cls, ids, cx, cy, radius=80):
    """Slice boundaries are straight lines through the apex: fit a line to every
    boundary between two slices (near the first guess) and take the point
    closest to all of them (least squares)."""
    H, W = cls.shape
    y0, y1 = max(0, int(cy - radius)), min(H, int(cy + radius))
    x0, x1 = max(0, int(cx - 2 * radius)), min(W, int(cx + 2 * radius))
    sub = cls[y0:y1, x0:x1]
    A, b = [], []
    for ii, i in enumerate(ids):
        mi = ndi.binary_dilation(sub == i, iterations=2)
        for j in ids[ii + 1:]:
            mj = ndi.binary_dilation(sub == j, iterations=2)
            ys, xs = np.where(mi & mj & (sub >= 0))
            if len(xs) < 25:
                continue
            P = np.c_[xs + x0, ys + y0].astype(float)
            c = P.mean(0)
            u, sv, vt = np.linalg.svd(P - c, full_matrices=False)
            if sv[0] < 3 * max(sv[1], 1e-6):     # not a straight segment
                continue
            nrm = vt[1]                           # line normal
            A.append(nrm); b.append(nrm @ c)
    if len(A) < 2:
        return cx, cy
    sol, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    if np.hypot(sol[0] - cx, sol[1] - cy) > radius:
        return cx, cy
    return float(sol[0]), float(sol[1])


def pipeline_pie(img, spec, agree_pts=3.0, n_angles=3600):
    H, W = img.shape[:2]
    hue, sat, val = hsv_arrays(img)
    slices = spec.get("slices") or []
    out = {"chart_type": "pie", "size": (W, H), "spec": spec, "series": {}, "styles": {}, "warnings": [],
           "categories": [], "calibration": {}, "calibration_info": {}, "legend": [], "legend_boxes": [],
           "text_boxes": []}
    top = (sat >= 0.35) & (val >= 0.62)
    if not slices:   # the VLM named no slices: one slice per distinct colour of the top face
        pk = hue_peaks(hue, top, 12)
        from .extract import HUES
        slices = [{"name": f"slice {i + 1}", "color": min(HUES, key=lambda n: float(hue_dist(HUES[n], h))),
                   "printed_value": None} for i, h in enumerate(pk)]
        out["warnings"].append(f"VLM gave no slices - {len(slices)} slice colour(s) found in the pixels")
        if not slices:
            return out
    peaks = hue_peaks(hue, top, len(slices))
    # pair colour names with hue peaks
    from scipy.optimize import linear_sum_assignment
    C = np.full((len(slices), len(peaks)), 1e6)
    for i, s in enumerate(slices):
        t = colour_target(s["color"])
        for j, p in enumerate(peaks):
            C[i, j] = float(hue_dist(t.get("hue", 0), p)) if t["kind"] == "hue" else 300
    r, c = linear_sum_assignment(C) if len(peaks) else ([], [])
    slice_hue = {int(i): peaks[int(j)] for i, j in zip(r, c)}
    if len(slice_hue) < len(slices):
        out["warnings"].append(f"only {len(slice_hue)} of {len(slices)} slice colours found in the image")
    # classify the top face
    cls = np.full((H, W), -1, int)
    best = np.full((H, W), 1e9)
    for i, h in slice_hue.items():
        d = np.where(top, hue_dist(hue, h), 1e9)
        sel = (d < best) & (d <= 20)
        cls[sel] = i; best[sel] = d[sel]
    # anti-aliased slice boundaries mix two colours into a third hue: drop thin specks
    for i in slice_hue:
        m = cls == i
        cls[m & ~ndi.binary_opening(m, np.ones((3, 3)))] = -1
    face = cls >= 0
    # apex: where the most classes meet (ties -> nearest the face centre)
    pres = [ndi.maximum_filter((cls == i).astype(np.uint8), size=11) > 0 for i in slice_hue]
    cnt = np.sum(pres, axis=0)
    mx = cnt.max()
    if mx < 2:
        out["warnings"].append("pie apex not found"); return out
    ys, xs = np.where((cnt == mx) & ndi.binary_dilation(face, iterations=6))
    fy, fx = np.where(face)
    fcx, fcy = 0.5 * (fx.min() + fx.max()), 0.5 * (fy.min() + fy.max())
    lab, n = ndi.label((cnt == mx) & ndi.binary_dilation(face, iterations=6))
    comps = []
    for k in range(1, n + 1):
        yy, xx = np.where(lab == k)
        comps.append((np.hypot(xx.mean() - fcx, yy.mean() - fcy), xx.mean(), yy.mean()))
    _, ax_, ay_ = min(comps)
    cx, cy = refine_apex(cls, list(slice_hue), float(ax_), float(ay_))
    # extents of the top face from the apex: the bounding box of the (hole-filled) face.
    # In the perspective model the extreme points lie on the lines through the apex.
    filled = ndi.binary_fill_holes(ndi.binary_closing(face, np.ones((5, 5))))
    lab2, n2 = ndi.label(filled)
    if n2 > 1:
        sizes = ndi.sum(filled, lab2, range(1, n2 + 1)); filled = lab2 == (int(np.argmax(sizes)) + 1)
    fy, fx = np.where(filled)
    left, right = cx - fx.min(), fx.max() - cx
    up, down = cy - fy.min(), fy.max() - cy
    # 3D pie: a darker side wall (same hues, lower brightness) hangs below the front of the face
    side = (sat >= 0.3) & (val >= 0.12) & (val < 0.62) & ~filled
    depths, shade = [], []
    for x in np.linspace(cx - 0.6 * left, cx + 0.6 * right, 9).astype(int):
        col = np.where(filled[:, x])[0]
        if not len(col):
            continue
        y = col.max() + 1
        n = 0
        while y + n < H and side[y + n, x]:
            n += 1
        if n >= 3:
            depths.append(n)
            shade.append(float(val[y:y + n, x].mean()) / max(float(val[col.max() - 3:col.max(), x].mean()), 1e-3))
    depth = float(np.median(depths)) if len(depths) >= 3 else 0.0
    cal = {"apex": (cx, cy), "left": left, "right": right, "up": up, "down": down,
           "depth": depth, "side_shade": float(np.median(shade)) if depth else 1.0}
    out["pie_calibration"] = cal
    # angular shares
    th = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    votes = []
    for rr in np.linspace(0.35, 0.92, 25):
        x, y = pie_point(cal, th)
        x = cx + (x - cx) * rr; y = cy + (y - cy) * rr
        votes.append(cls[np.clip(np.round(y).astype(int), 0, H - 1), np.clip(np.round(x).astype(int), 0, W - 1)])
    V = np.array(votes)
    labs = np.array([np.bincount(v[v >= 0], minlength=len(slices)).argmax() if (v >= 0).any() else -1 for v in V.T])
    good = labs >= 0
    share = np.array([100.0 * (labs[good] == i).sum() / max(good.sum(), 1) for i in range(len(slices))])
    # clockwise order from 12 o'clock: where each slice begins when walking clockwise
    deg = np.degrees(th)
    cw = (90 - deg) % 360                      # clockwise distance from 12 o'clock
    o = np.argsort(cw); seq, cws = labs[o], cw[o]
    start, runlen = {}, {}
    n_ = len(seq)
    for j in range(n_):
        i = seq[j]
        if i < 0 or seq[j - 1] == i:
            continue
        k = j
        while seq[(k + 1) % n_] == i and k - j < n_:
            k += 1
        if k - j + 1 > runlen.get(i, -1):
            start[i], runlen[i] = float(cws[j]), k - j + 1
    order = sorted(start, key=lambda i: start[i]) + [i for i in range(len(slices)) if i not in start]
    xs, ys, flags, meas = [], [], [], []
    for k, i in enumerate(order):
        pv = slices[i].get("printed_value")
        m = float(share[i])
        if pv is not None and abs(pv - m) <= agree_pts:
            v, fl = float(pv), "printed label (agrees with the measured angle)"
        else:
            v, fl = m, ("measured angle" if pv is None else f"measured angle (printed {pv} disagrees)")
        xs.append(k); ys.append(round(v, 4)); flags.append(fl); meas.append(round(m, 3))
    names = [slices[i]["name"] for i in order]
    out["categories"] = names
    out["series"]["share (%)"] = {"x": xs, "y": ys, "kind": "pie", "y_axis": "y_axis", "flags": flags,
                                  "measured": meas, "categories": names}
    out["styles"]["share (%)"] = {"slice_hue": {slices[i]["name"]: slice_hue.get(i) for i in order},
                                  "slice_rgb": {slices[i]["name"]: (np.median(img[cls == i], 0).tolist() if (cls == i).any() else [128, 128, 128]) for i in order}}
    return out
