"""Camera of a 3-D plot (matplotlib-style 3-D axes) from its three axis lines.

The VLM reads the three axes' tick labels (x and y: in the order met left ->
right across the image; z: bottom -> top). The pixels give:

  1. the axis lines: long dark straight lines (Hough), merged when collinear,
     extended along the dark pixels; the ones with a row of tick labels beside
     them are axes; the most vertical one is z
  2. x vs y: label counts / label widths; for numeric axes also the
     right-handed rule (seen from above, turning from +x to +y is counter-clockwise)
  3. tick positions on every axis line: the short tick marks, paired with the
     labels (labels sit at a constant offset from their ticks)
  4. an affine camera first: pixel per unit along each axis from the tick
     spacing; the x- and y-lines meet at the floor corner; the z-line's foot
     on the floor gives the floor level
  5. then a full projective camera (3x4, DLT) from every tick's known 3-D
     position -> perspective is handled; its reprojection error is a quality gate

project(P, X) maps 3-D data -> pixels; the extractors invert it with whatever
extra information the plot holds (bar on the floor grid, colour = z, stems...).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

from . import detect as D
from .calibrate import parse_number


def _dark(img):
    g = img.mean(-1)
    sat = (img.max(-1).astype(float) - img.min(-1)) / np.maximum(img.max(-1), 1)
    return (g < 110) & (sat < 0.25)


def axis_lines(img, excl=None):
    """long straight dark lines -> [{p0, p1, u, n, length, angle}] (merged, extended)"""
    from skimage.transform import probabilistic_hough_line
    H, W = img.shape[:2]
    d = _dark(img)
    if excl is not None:
        d &= ~excl
    segs = probabilistic_hough_line(d, threshold=10, line_length=int(0.1 * min(H, W)), line_gap=3, rng=0)
    lines = []
    for (x0, y0), (x1, y1) in segs:
        p0, p1 = np.array([x0, y0], float), np.array([x1, y1], float)
        u = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
        if u[0] < 0 or (u[0] == 0 and u[1] < 0):
            u = -u; p0, p1 = p1, p0
        nrm = np.array([-u[1], u[0]])
        for L in lines:
            if abs(L["u"] @ u) > math.cos(math.radians(3)) and abs(L["n"] @ (p0 - L["p0"])) <= 3:
                ts = [L["u"] @ (q - L["p0"]) for q in (p0, p1)]
                L["t0"], L["t1"] = min(L["t0"], *ts), max(L["t1"], *ts)
                break
        else:
            lines.append({"p0": p0, "u": u, "n": nrm, "t0": 0.0, "t1": float(np.linalg.norm(p1 - p0))})
    out = []
    for L in lines:
        # extend along the dark pixels (tolerate gaps up to 4 px)
        for end, step in (("t1", 1), ("t0", -1)):
            t, gap = L[end], 0
            while gap <= 4:
                q = L["p0"] + (t + step) * L["u"]
                xi, yi = int(round(q[0])), int(round(q[1]))
                if not (0 <= xi < W and 0 <= yi < H):
                    break
                if d[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2].any():
                    gap = 0; L[end] = t + step
                else:
                    gap += 1
                t += step
        a, b = L["p0"] + L["t0"] * L["u"], L["p0"] + L["t1"] * L["u"]
        out.append({"a": a, "b": b, "u": L["u"], "n": L["n"], "length": float(np.linalg.norm(b - a)),
                    "angle": math.degrees(math.atan2(-(b[1] - a[1]), b[0] - a[0])) % 180})
    return [L for L in out if L["length"] >= 0.12 * min(H, W)]


def words(img, max_h=26):
    """text words: character components merged only with neighbours on the same
    text row (rotated axis titles do not swallow nearby tick labels)"""
    d = D.dark_mask(img, 150, 0.4)
    allc = D.components(d, min_area=3)
    cs = [c for c in allc if 4 <= (c["y1"] - c["y0"]) <= max_h and (c["x1"] - c["x0"]) <= 30]
    dashes = [c for c in allc if (c["y1"] - c["y0"]) <= 3 and 3 <= (c["x1"] - c["x0"]) <= 12]     # minus signs
    cs.sort(key=lambda c: c["x0"])
    ws = []
    for c in cs:
        h = c["y1"] - c["y0"]
        for w in ws:
            wh = w["y1"] - w["y0"]
            ov = min(w["y1"], c["y1"]) - max(w["y0"], c["y0"])
            gap = c["x0"] - w["x1"]
            if ov >= 0.6 * min(h, wh) and -2 <= gap <= 0.5 * max(h, wh) and abs(h - wh) <= 0.5 * max(h, wh):
                w.update(x0=min(w["x0"], c["x0"]), x1=max(w["x1"], c["x1"]), y0=min(w["y0"], c["y0"]), y1=max(w["y1"], c["y1"]))
                w["n"] += 1
                break
        else:
            ws.append(dict(c, n=1))
    # a character broken into an upper and a lower piece (thin strokes over a grid line)
    merged = True
    while merged:
        merged = False
        for i in range(len(ws)):
            for j in range(len(ws)):
                a, b = ws[i], ws[j]
                if i == j:
                    continue
                hov = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
                gap = b["y0"] - a["y1"]
                if hov >= 0.5 * min(a["x1"] - a["x0"], b["x1"] - b["x0"]) and -1 <= gap <= 2 and \
                        (a["y1"] - a["y0"]) + (b["y1"] - b["y0"]) <= max_h:
                    a.update(x0=min(a["x0"], b["x0"]), x1=max(a["x1"], b["x1"]), y0=min(a["y0"], b["y0"]), y1=max(a["y1"], b["y1"]))
                    ws.pop(j); merged = True
                    break
            if merged:
                break
    for dsh in dashes:                      # a minus sign just left of a word belongs to it
        cy = (dsh["y0"] + dsh["y1"]) / 2
        for w in ws:
            if w["y0"] <= cy <= w["y1"] and 0 <= w["x0"] - dsh["x1"] <= 4:
                w["x0"] = dsh["x0"]; break
    for w in ws:
        w["c"] = np.array([(w["x0"] + w["x1"] - 1) / 2, (w["y0"] + w["y1"] - 1) / 2])
    return ws


def labels_beside(L, ws, max_d=45):
    """words in a row beside the line, on its more populated side: (words, side sign)"""
    best = ([], 1)
    for sgn in (1, -1):
        got = []
        for w in ws:
            v = w["c"] - L["a"]
            t, dd = v @ L["u"], sgn * (v @ L["n"])
            if -12 <= t <= L["length"] + 12 and 3 <= dd <= max_d:
                got.append(w)
        if got:
            # the tick labels form one row parallel to the line (titles lie further out)
            dd = np.array([sgn * ((w["c"] - L["a"]) @ L["n"]) for w in got])
            rows = [[w for w, e in zip(got, dd) if abs(e - a) <= 6] for a in dd]
            row = max(rows, key=lambda r: (len(r), -np.mean([sgn * ((w["c"] - L["a"]) @ L["n"]) for w in r])))
            got = row
        if len(got) > len(best[0]):
            best = (got, sgn)
    return best


def tick_marks(img, L, side=None, band=(2, 8)):
    """positions t along the line where short tick marks stick out (either side,
    any slant): dark pixels in a band beside the line, counted per position"""
    # dark tick stubs, and the grey grid lines that meet the axis line at every tick
    g = img.mean(-1)
    sat = (img.max(-1).astype(float) - img.min(-1)) / np.maximum(img.max(-1), 1)
    d = (g < 215) & (sat < 0.2)
    ys, xs = np.where(d)
    v = np.c_[xs, ys].astype(float) - L["a"]
    t, o = v @ L["u"], np.abs(v @ L["n"])
    sel = (o >= band[0]) & (o <= band[1]) & (t >= -3) & (t <= L["length"] + 3)
    if not sel.any():
        return np.array([])
    hist = np.bincount(np.clip(np.round(t[sel]).astype(int) + 3, 0, None), minlength=int(L["length"]) + 8).astype(float)
    hist = np.convolve(hist, [0.5, 1, 0.5], mode="same")
    peaks = []
    for k in range(1, len(hist) - 1):
        if hist[k] >= 3 and hist[k] >= hist[k - 1] and hist[k] > hist[k + 1]:
            if peaks and k - peaks[-1] <= 4:
                if hist[k] > hist[peaks[-1]]:
                    peaks[-1] = k
                continue
            peaks.append(k)
    return np.array(peaks, float) - 3


def _fit_1d(vals, ts):
    """t = (a v + b) / (c v + 1): projective when >= 4 pairs and it helps, else linear"""
    v, t = np.asarray(vals, float), np.asarray(ts, float)
    A = np.c_[v, np.ones(len(v))]
    lin, *_ = np.linalg.lstsq(A, t, rcond=None)
    res_l = float(np.sqrt(np.mean((A @ lin - t) ** 2)))
    best = ("linear", lin, res_l)
    if len(v) >= 4:
        M = np.c_[v, np.ones(len(v)), -v * t]
        pr, *_ = np.linalg.lstsq(M, t, rcond=None)
        pred = (pr[0] * v + pr[1]) / (pr[2] * v + 1)
        res_p = float(np.sqrt(np.mean((pred - t) ** 2)))
        mild = float(np.max(np.abs(pr[2] * v))) < 0.5            # perspective only bends a line a little
        if len(v) >= 5 and mild and res_p < 0.7 * res_l and res_l - res_p > 0.3:
            best = ("projective", pr, res_p)
    return best


def _t_of(fit, v):
    kind, c, _ = fit
    v = np.asarray(v, float)
    return c[0] * v + c[1] if kind == "linear" else (c[0] * v + c[1]) / (c[2] * v + 1)


def _v_of(fit, t):
    kind, c, _ = fit
    t = np.asarray(t, float)
    return (t - c[1]) / c[0] if kind == "linear" else (t - c[1]) / (c[0] - c[2] * t)


def axis_values(spec_axis):
    """numbers (numeric axis) or positions 0..n-1 (category names, in the listed order)"""
    labs = list((spec_axis or {}).get("tick_labels") or [])
    nums = [parse_number(x) for x in labs]
    if labs and all(v is not None for v in nums) and len(set(nums)) == len(nums):
        return np.array(nums, float), labs, False
    return np.arange(len(labs), dtype=float), labs, True


def calibrate_axis_line(img, L, side_words, side, vals, is_z=False, strings=None):
    """pair the tick values with positions along the line -> 1-D fit value -> t.
    Ticks are used when their count equals the label count; otherwise the label
    words are paired with the ticks at one constant offset (labels sit at a
    fixed offset from their ticks); missing labels are allowed."""
    # t must grow in the VLM's reading order: x / y left -> right (u points right), z bottom -> top
    sgn = -1.0 if (is_z and L["u"][1] > 0) else 1.0
    vals = np.asarray(vals, float)
    orig_vals = vals.copy()
    nv = len(vals)
    ticks = tick_marks(img, L)
    ws_sorted = sorted(side_words, key=lambda w: sgn * ((w["c"] - L["a"]) @ L["u"]))
    tw = np.array([(w["c"] - L["a"]) @ L["u"] for w in ws_sorted])
    wid = np.array([w["x1"] - w["x0"] for w in ws_sorted], float)
    info = {"n_labels": int(len(tw)), "n_ticks_found": int(len(ticks)), "n_values": nv}
    ordered_ticks = np.sort(ticks) if sgn > 0 else np.sort(ticks)[::-1]
    pairs = None
    tick_only = None
    if len(ticks) == nv and nv >= 2:
        sp = np.abs(np.diff(ordered_ticks))
        if sp.min() > 0.3 * np.median(sp):
            tick_only = ordered_ticks
    if len(tw) >= 2:
        # pair label words with values: every (2 words, 2 values) hypothesis predicts all
        # label positions; keep the one most words agree with (extra / missing words anywhere)
        best = None
        tol = 3.0
        for i1 in range(len(tw)):
            for i2 in range(i1 + 1, len(tw)):
                for a1 in range(nv):
                    for a2 in range(a1 + 1, nv):
                        k = (tw[i2] - tw[i1]) / (vals[a2] - vals[a1])
                        pred = tw[i1] + k * (vals - vals[a1])
                        dd = np.abs(tw[:, None] - pred[None, :])
                        jw = dd.argmin(0)
                        ok = dd[jw, np.arange(nv)] <= max(tol, 0.08 * abs(k) * np.median(np.abs(np.diff(vals))))
                        if ok.sum() < 2 or len(set(jw[ok])) < ok.sum():
                            continue
                        idx = [(a, int(jw[a])) for a in range(nv) if ok[a]]
                        fit = _fit_1d(vals[[a for a, _ in idx]], tw[[j for _, j in idx]])
                        key = -len(idx) + 0.1 * fit[2]
                        if strings is not None:
                            from .calibrate import _width_mismatch
                            key += 0.5 * _width_mismatch([strings[a] for a, _ in idx], [wid[j] for _, j in idx])
                        if best is None or key < best[0]:
                            best = (key, idx)
        if best is not None:
            idx = best[1]
            vals = vals[[i for i, _ in idx]]
            pairs = tw[[j for _, j in idx]]
            info["paired"] = f"{len(idx)} label words"
            if len(ticks) >= 2:
                sh = np.median([pairs[i] - ticks[np.abs(ticks - pairs[i]).argmin()] for i in range(len(pairs))])
                tt = np.array([ticks[np.abs(ticks - (p - sh)).argmin()] for p in pairs])
                ok = np.abs(tt - (pairs - sh)) <= 3
                if ok.sum() >= max(2, int(0.6 * len(pairs))):
                    pairs = np.where(ok, tt, pairs - sh)
                    info.update(paired=info["paired"] + f" moved onto {int(ok.sum())} tick marks", label_offset_px=float(sh))
    # ticks forming a regular run of exactly nv marks that the label words agree with
    run = _tick_run(np.sort(ticks), nv, tw)
    if run is not None and (pairs is None or len(pairs) < nv or info.get("rms_words", 0) > 1.5):
        seq = run if sgn > 0 else run[::-1]
        pairs, vals = seq, np.asarray(orig_vals, float)
        info["paired"] = f"regular run of {nv} tick marks, confirmed by the labels"
    elif tick_only is not None:
        f_t = _fit_1d(np.arange(nv, dtype=float), tick_only)
        if pairs is None or (len(pairs) < nv and f_t[2] <= 1.5):
            pairs, vals = tick_only, np.asarray(orig_vals, float)
            info["paired"] = "ticks in order"
    if pairs is None:
        return None, {**info, "reason": f"{len(tw)} labels and {len(ticks)} ticks for {nv} values"}
    fit = _fit_1d(vals, pairs)
    # a label snapped onto the wrong mark (grid line, line end) breaks the smooth spacing:
    # drop pairs far off the fit, as long as >= 4 pairs remain
    vals, pairs = np.asarray(vals, float), np.asarray(pairs, float)
    for _ in range(2):
        res = np.abs(_t_of(fit, vals) - pairs)
        bad = res > max(3.5, 3.0 * np.median(res))
        if bad.any() and (~bad).sum() >= 4:
            vals, pairs = vals[~bad], pairs[~bad]
            fit = _fit_1d(vals, pairs)
            info["dropped_outlier_ticks"] = info.get("dropped_outlier_ticks", 0) + int(bad.sum())
        else:
            break
    info.update(model=fit[0], rms_px=fit[2])
    return {"fit": fit, "vals": vals, "t": np.asarray(pairs, float), "line": L}, info


def _tick_run(ticks, nv, tw, tol=0.2):
    """the run of nv tick marks with (slowly varying) regular spacing that the label
    positions agree with best (labels at one constant offset from their ticks)"""
    if len(ticks) < nv or nv < 2:
        return None
    best = None
    for a in range(len(ticks)):
        for b in range(a + 1, len(ticks)):
            seq = [ticks[a], ticks[b]]
            sp = ticks[b] - ticks[a]
            if sp < 6:
                continue
            while len(seq) < nv:
                exp = seq[-1] + sp
                k = np.argmin(np.abs(ticks - exp))
                if abs(ticks[k] - exp) > tol * sp or ticks[k] <= seq[-1]:
                    break
                sp = ticks[k] - seq[-1]
                seq.append(ticks[k])
            if len(seq) != nv:
                continue
            seq = np.array(seq)
            if len(tw) >= 2:
                sh = np.median([w - seq[np.abs(seq - w).argmin()] for w in tw])
                agree = int(sum(np.abs(seq - (w - sh)).min() <= 4 for w in tw))
            else:
                agree = 0
            sps = np.diff(seq)
            key = (agree, -float(np.std(np.diff(sps))) if len(sps) > 1 else 0.0)
            if best is None or key > best[0]:
                best = (key, seq)
    if best is None or best[0][0] < max(2, int(0.6 * min(nv, len(tw)))):
        return None
    return best[1]


def _point(ax, v):
    return ax["line"]["a"] + float(_t_of(ax["fit"], v)) * ax["line"]["u"]


def _intersect(L1, L2):
    A = np.c_[L1["u"], -L2["u"]]
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    s, _ = np.linalg.solve(A, L2["a"] - L1["a"])
    return L1["a"] + s * L1["u"]


def dlt(Xw, xi, lines=None):
    """3x4 camera from 3-D <-> pixel points, plus optional line constraints:
    lines = [(l, X1, X2)]: the image line l = (a, b, c) must pass through the projections of
    the 3-D points X1 and X2 (a grid line on a pane: its 3-D position is known, its pixels too)"""
    Xw, xi = np.asarray(Xw, float), np.asarray(xi, float)
    mu, sd = Xw.mean(0), Xw.std(0) + 1e-9
    m2, s2 = xi.mean(0), xi.std(0).mean() + 1e-9
    T3 = np.diag(np.r_[1 / sd, 1]); T3[:3, 3] = -mu / sd
    T2 = np.diag([1 / s2, 1 / s2, 1]); T2[:2, 2] = -m2 / s2
    Xn = (T3 @ np.c_[Xw, np.ones(len(Xw))].T).T
    xn = (T2 @ np.c_[xi, np.ones(len(xi))].T).T
    rows = []
    for X, x in zip(Xn, xn):
        rows.append(np.r_[np.zeros(4), -x[2] * X, x[1] * X])
        rows.append(np.r_[x[2] * X, np.zeros(4), -x[0] * X])
    T2i = np.linalg.inv(T2)
    for l, X1, X2 in lines or []:
        ln = T2i.T @ np.asarray(l, float)
        ln = ln / np.linalg.norm(ln[:2])
        for X in (X1, X2):
            Xh = T3 @ np.r_[np.asarray(X, float), 1.0]
            rows.append(np.kron(ln, Xh))
    _, _, Vt = np.linalg.svd(np.array(rows))
    Pn = Vt[-1].reshape(3, 4)
    return np.linalg.inv(T2) @ Pn @ T3


def viewer(P, limits, near_xy):
    """where the viewer is: ("point", camera centre) or ("dir", unit direction towards the viewer).
    The camera matrix alone can be mirrored (front / back look alike in one picture); the axis
    lines settle it - they are drawn on the box edges nearest the viewer (near_xy = the floor
    corner where the x- and y-axis lines meet), the grid panes on the far sides."""
    _, _, Vt = np.linalg.svd(P)
    C = Vt[-1]
    centre = np.array([np.mean(limits["x"]), np.mean(limits["y"]), np.mean(limits["z"])])
    near = np.array([near_xy[0], near_xy[1], limits["z"][1]])
    if abs(C[3]) > 1e-9 * np.abs(C[:3]).max():
        c = C[:3] / C[3]
        if (c - centre) @ (near - centre) < 0:
            c = 2 * centre - c
        return "point", c
    v = C[:3] / np.linalg.norm(C[:3])
    return "dir", (v if v @ (near - centre) > 0 else -v)


def _depth_of(P, X):
    _, _, Vt = np.linalg.svd(P)
    C = Vt[-1]
    X = np.asarray(X, float)
    if abs(C[3]) > 1e-9 * np.abs(C[:3]).max():
        return float(np.linalg.norm(X - C[:3] / C[3]))
    v = C[:3] / np.linalg.norm(C[:3]); v = v if v[2] > 0 else -v
    return -float(X @ v)


def measure_grid_line(gmask, p1, p2, search=8, free=None):
    """the grey grid line near the predicted segment p1-p2: best perpendicular offset on each half
    -> (a, b, c) of the measured line, or None"""
    d = p2 - p1
    L = float(np.linalg.norm(d))
    if L < 30:
        return None
    u = d / L; n = np.array([-u[1], u[0]])
    H, W = gmask.shape
    pts = []
    for lo, hi in ((0.1, 0.45), (0.55, 0.9)):
        ts = np.arange(lo * L, hi * L, 1.5)
        offs = np.arange(-search, search + 0.01, 0.5)
        prof = np.full(len(offs), -1.0)
        for k, o in enumerate(offs):
            q = p1 + ts[:, None] * u + o * n
            xs, ys = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
            ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
            if free is not None:                      # only where the pane is not hidden by data
                ok[ok] = free[ys[ok], xs[ok]]
            if ok.sum() >= 8:
                prof[k] = gmask[ys[ok], xs[ok]].mean()
        # the grid line nearest the prediction (not the strongest one: that may be its neighbour)
        peaks = [k for k in range(len(offs)) if prof[k] >= 0.35 and prof[k] >= prof[max(0, k - 1)] and prof[k] >= prof[min(len(offs) - 1, k + 1)]]
        if not peaks:
            return None
        k = min(peaks, key=lambda k: abs(offs[k]))
        # centre of the peak (lines are ~1 px wide, sampled every 0.5 px)
        sel = [j for j in range(max(0, k - 2), min(len(offs), k + 3)) if prof[j] >= 0.5 * prof[k]]
        o = float(np.average(offs[sel], weights=prof[sel]))
        pts.append(p1 + 0.5 * (lo + hi) * L * u + o * n)
    a, b = pts
    l = np.cross(np.r_[a, 1.0], np.r_[b, 1.0])
    return l / np.linalg.norm(l[:2])


def measure_ray(gmask, free, anchor, p_end, max_deg=15.0):
    """a grid line that starts at a known tick point: the direction (within +-max_deg of the
    predicted one) along which the grey grid line runs. -> image line (a, b, c) or None"""
    d = p_end - anchor
    L = float(np.linalg.norm(d))
    if L < 30:
        return None
    base = np.arctan2(d[1], d[0])
    H, W = gmask.shape
    ts = np.arange(6.0, 0.95 * L, 1.5)
    best = None
    for dth in np.radians(np.arange(-max_deg, max_deg + 0.01, 0.25)):
        u = np.array([np.cos(base + dth), np.sin(base + dth)])
        q = anchor + ts[:, None] * u
        xs, ys = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        ok[ok] = free[ys[ok], xs[ok]]
        if ok.sum() < 12:
            continue
        cov = gmask[ys[ok], xs[ok]].mean()
        if best is None or cov > best[0]:
            best = (cov, u)
    if best is None or best[0] < 0.5:
        return None
    l = np.cross(np.r_[anchor, 1.0], np.r_[anchor + best[1], 1.0])
    return l / np.linalg.norm(l[:2])


def _fit_with_lines(Xw, xi, lines):
    """DLT with line constraints; lines that disagree with the rest are dropped"""
    lr = np.array([0.0])
    P_new = None
    for _ in range(4):
        P_new = dlt(Xw, xi, lines)
        lr = []
        for l, X1, X2 in lines:
            q = project(P_new, [X1, X2])
            lr.append(max(abs(l @ np.r_[q[0], 1]), abs(l @ np.r_[q[1], 1])))
        lr = np.array(lr)
        keep = lr <= max(1.5, 2.5 * np.median(lr))
        if keep.all() or keep.sum() < 4:
            break
        lines = [ln for ln, k in zip(lines, keep) if k]
    res = float(np.sqrt(np.mean(np.sum((project(P_new, Xw) - xi) ** 2, 1))))
    return P_new, lines, res, float(np.median(lr))


def refine_with_grid(img, P, Xw, xi, limits, axis_pts, z_floor, near_xy, z_corner, excl=None):
    """perspective needs constraints far from the three axis lines. The grid lines on the floor
    start at the x / y tick marks (known pixels) and run to the far edge; the vertical grid lines
    on the two back walls start where those floor lines end; the horizontal ones on the walls are
    at the z tick values. Each is measured in the pixels (grey line) and used as a line
    constraint in the camera fit - floor first, then the walls with the improved camera."""
    g = img.mean(-1)
    sat = (img.max(-1).astype(float) - img.min(-1)) / np.maximum(img.max(-1), 1)
    gmask = (g > 140) & (g < 237) & (sat < 0.1)
    free = (sat < 0.1) & (g > 140)                     # pane or grid, not data / text
    if excl is not None:
        gmask &= ~excl; free &= ~excl
    (x0, x1), (y0, y1), (z0, z1) = limits["x"], limits["y"], limits["z"]
    xe, ye = near_xy
    xf = x0 if abs(x0 - xe) > abs(x1 - xe) else x1            # far sides: away from the axis lines
    yf = y0 if abs(y0 - ye) > abs(y1 - ye) else y1
    # 1. floor lines, anchored at the tick points
    lines = []
    for v, px in axis_pts["x"]:
        X1, X2 = (v, ye, z_floor), (v, yf, z_floor)
        l = measure_ray(gmask, free, np.asarray(px, float), project(P, [X2])[0])
        if l is not None:
            lines.append((l, X1, X2))
    for w, px in axis_pts["y"]:
        X1, X2 = (xe, w, z_floor), (xf, w, z_floor)
        l = measure_ray(gmask, free, np.asarray(px, float), project(P, [X2])[0])
        if l is not None:
            lines.append((l, X1, X2))
    if len(lines) < 4:
        return P, 0
    P1, lines, res, mlr = _fit_with_lines(Xw, xi, lines)
    if res > 2.5 or mlr > 1.5:
        return P, 0
    # 2. back walls: vertical lines from the far ends of the floor lines, horizontal lines at z ticks
    wall = []
    for v, _ in axis_pts["x"]:
        A_, B_ = (v, yf, z_floor), (v, yf, z1)
        l = measure_ray(gmask, free, project(P1, [A_])[0], project(P1, [B_])[0], max_deg=6)
        if l is not None:
            wall.append((l, A_, B_))
    for w, _ in axis_pts["y"]:
        A_, B_ = (xf, w, z_floor), (xf, w, z1)
        l = measure_ray(gmask, free, project(P1, [A_])[0], project(P1, [B_])[0], max_deg=6)
        if l is not None:
            wall.append((l, A_, B_))
    for u, _ in axis_pts["z"]:
        for A_, B_ in (((xf, ye, u), (xf, yf, u)), ((xe, yf, u), (xf, yf, u))):
            p1, p2 = project(P1, [A_, B_])
            l = measure_grid_line(gmask, p1, p2, search=6, free=free)
            if l is not None:
                wall.append((l, A_, B_))
    if len(wall) >= 3:
        P2, kept, res2, mlr2 = _fit_with_lines(Xw, xi, lines + wall)
        if res2 <= 2.5 and mlr2 <= 1.5:
            return P2, len(kept)
    return P1, len(lines)


def project(P, X):
    X = np.atleast_2d(np.asarray(X, float))
    h = (P @ np.c_[X, np.ones(len(X))].T).T
    return h[:, :2] / h[:, 2:3]


def solve_xy_given_z(P, px, z):
    """pixel + known z -> (x, y)"""
    u, v = px
    A = np.array([[P[0, 0] - u * P[2, 0], P[0, 1] - u * P[2, 1]],
                  [P[1, 0] - v * P[2, 0], P[1, 1] - v * P[2, 1]]])
    b = -np.array([P[0, 2] * z + P[0, 3] - u * (P[2, 2] * z + P[2, 3]),
                   P[1, 2] * z + P[1, 3] - v * (P[2, 2] * z + P[2, 3])])
    return np.linalg.solve(A, b)


def solve_z_given_xy(P, px, x, y):
    """pixel + known (x, y) -> z (least squares over the two image equations), residual px"""
    u, v = px
    a = np.array([P[0, 2] - u * P[2, 2], P[1, 2] - v * P[2, 2]])
    b = -np.array([P[0, 0] * x + P[0, 1] * y + P[0, 3] - u * (P[2, 0] * x + P[2, 1] * y + P[2, 3]),
                   P[1, 0] * x + P[1, 1] * y + P[1, 3] - v * (P[2, 0] * x + P[2, 1] * y + P[2, 3])])
    z = float(a @ b / (a @ a))
    res = float(np.linalg.norm(project(P, [x, y, z])[0] - np.asarray(px, float)))
    return z, res


def calibrate_3d(img, spec, excl=None, candidates=False):
    """-> (camera dict or None, info). The VLM may list a floor axis's labels in the wrong
    direction (a category axis has no direction of its own; a numeric one is sometimes read
    right-to-left as left-to-right), so both directions of x and y are tried: a wrong direction
    makes a mirror-image (left-handed) box that no real camera produces, so the view camera fits
    the ticks worse. The listed order is kept unless a flip fits clearly better."""
    best = None
    allc = []
    for fx in (False, True):
        for fy in (False, True):
            cam, info = _calibrate_3d(img, spec, excl, flip=(fx, fy))
            if cam is not None:
                allc.append(((cam.get("view") or {}).get("tick_rms_px", info.get("reprojection_rms_px", np.inf)) + 0.15 * (fx + fy),
                             cam, dict(info, axis_order_flipped={"x": fx, "y": fy})))
            if cam is None:
                if best is None:
                    best = (np.inf, cam, info)
                continue
            # flipping both category axes is the same picture seen from the opposite azimuth:
            # prefer the listed order unless a flip fits clearly better
            r = (cam.get("view") or {}).get("tick_rms_px", np.inf) + 0.15 * (fx + fy)
            if best is None or r < best[0]:
                best = (r, cam, dict(info, axis_order_flipped={"x": fx, "y": fy}))
    if candidates:
        return sorted(allc, key=lambda c: c[0])
    if best[1] is not None and any(best[2].get("axis_order_flipped", {}).values()):
        fl = [k for k, v in best[2]["axis_order_flipped"].items() if v]
        best[2]["note"] = f"{'/'.join(fl)} tick labels read in the opposite direction to the VLM's list (the pixels fit better)"
    return best[1], best[2]


def _calibrate_3d(img, spec, excl=None, flip=(False, False)):
    """-> (camera dict or None, info)"""
    info = {}
    lines = axis_lines(img, excl)
    ws = words(img)
    cands = []
    for L in lines:
        got, sgn = labels_beside(L, ws)
        if len(got) >= 2:
            cands.append((L, got, sgn))
    info["axis_line_candidates"] = len(cands)
    if len(cands) < 3:
        return None, {**info, "reason": f"only {len(cands)} axis lines with tick labels found (need 3)"}
    vx, lx, catx = axis_values(spec.get("x_axis"))
    vy, ly, caty = axis_values(spec.get("y_axis"))
    if str(spec.get("chart_type", "")).lower() == "bar3d":
        # 3-D bars stand on a grid of categories: the floor labels are names (even '0.5', '1', '2', '5'),
        # placed one grid step apart
        vx, catx = np.arange(len(lx), dtype=float), True
        vy, caty = np.arange(len(ly), dtype=float), True
    if flip[0]:
        vx = np.asarray(vx, float)[::-1].copy()
    if flip[1]:
        vy = np.asarray(vy, float)[::-1].copy()
    vz, lz, catz = axis_values(spec.get("z_axis"))
    if min(len(vx), len(vy), len(vz)) < 2:
        return None, {**info, "reason": "the VLM read fewer than 2 tick labels on an axis"}
    vert = [c for c in cands if abs(c[0]["angle"] - 90) <= 15]
    if not vert:
        return None, {**info, "reason": "no vertical (z) axis line with labels"}
    zc = max(vert, key=lambda c: (len(c[1]) == len(vz), -abs(len(c[1]) - len(vz)), c[0]["length"]))
    rest = [c for c in cands if c is not zc and abs(c[0]["angle"] - 90) > 15]
    best = None
    for i in range(len(rest)):
        for j in range(len(rest)):
            if i == j or abs(((rest[i][0]["angle"] - rest[j][0]["angle"]) + 90) % 180 - 90) < 15:
                continue
            ca, cb = rest[i], rest[j]                       # hypothesis: ca = x, cb = y
            xa, xi_ = calibrate_axis_line(img, ca[0], ca[1], ca[2], vx, strings=lx)
            ya, yi_ = calibrate_axis_line(img, cb[0], cb[1], cb[2], vy, strings=ly)
            if xa is None or ya is None:
                continue
            cost = abs(len(ca[1]) - len(vx)) + abs(len(cb[1]) - len(vy)) + 0.2 * (xi_["rms_px"] + yi_["rms_px"])
            # label widths: long names belong to the axis whose printed labels are long
            wa = np.median([w["x1"] - w["x0"] for w in ca[1]]); wb = np.median([w["x1"] - w["x0"] for w in cb[1]])
            la = np.median([len(s) for s in lx]); lb = np.median([len(s) for s in ly])
            if la != lb:
                cost += 0 if (wa - wb) * (la - lb) > 0 else 3
            if not catx and not caty:
                # right-handed axes seen from above: +x to +y turns counter-clockwise (y up)
                dx = (_point(xa, vx.max()) - _point(xa, vx.min())) * np.array([1, -1])     # towards larger values
                dy = (_point(ya, vy.max()) - _point(ya, vy.min())) * np.array([1, -1])
                cost += 0 if dx[0] * dy[1] - dx[1] * dy[0] > 0 else 5
            info.setdefault("hypotheses", []).append({"x_line_angle": round(ca[0]["angle"]), "y_line_angle": round(cb[0]["angle"]), "cost": round(float(cost), 2)})
            if best is None or cost < best[0]:
                best = (cost, xa, ya, xi_, yi_)
    if best is None:
        return None, {**info, "reason": "no pair of floor axis lines could be calibrated"}
    _, xa, ya, xinf, yinf = best
    za, zinf = calibrate_axis_line(img, zc[0], zc[1], zc[2], vz, is_z=True, strings=lz)
    if za is None:
        return None, {**info, "reason": f"z-axis: {zinf.get('reason')}"}
    info.update(x_axis=xinf, y_axis=yinf, z_axis=zinf)
    # ---- affine camera
    I = _intersect(xa["line"], ya["line"])
    if I is None:
        return None, {**info, "reason": "x and y axis lines are parallel"}
    x_e = float(_v_of(xa["fit"], (I - xa["line"]["a"]) @ xa["line"]["u"]))
    y_e = float(_v_of(ya["fit"], (I - ya["line"]["a"]) @ ya["line"]["u"]))
    zl = za["line"]
    foot = zl["a"] if zl["a"][1] > zl["b"][1] else zl["b"]          # lower end of the z-line
    # the z-axis stands on a floor corner: where its line meets the end of the x- or y-line
    # (a gap at the bottom of the z-line, e.g. under a marker, then does not matter)
    for ax_ in (xa, ya):
        q = _intersect(zl, ax_["line"])
        if q is None:
            continue
        ends = (ax_["line"]["a"], ax_["line"]["b"])
        if min(np.linalg.norm(q - e) for e in ends) <= 15 and np.linalg.norm(q - foot) <= 40:
            foot = q
            info["z_foot"] = "z-line meets the end of a floor axis line"
            break
    z_lo = float(_v_of(za["fit"], (foot - zl["a"]) @ zl["u"]))

    def unit(ax, vals):
        return (_point(ax, vals[-1]) - _point(ax, vals[0])) / (vals[-1] - vals[0])
    dX, dY, dZ = unit(xa, vx), unit(ya, vy), unit(za, vz)
    t0 = I - x_e * dX - y_e * dY - z_lo * dZ
    # z-line foot on the floor: (x_z, y_z) from the affine floor
    A = np.c_[dX, dY]
    xz, yz = np.linalg.solve(A, foot - t0 - z_lo * dZ)
    # ---- 3-D positions of every tick -> projective camera (DLT)
    Xw, xi = [], []
    for v, t in zip(xa["vals"], xa["t"]):
        Xw.append([v, y_e, z_lo]); xi.append(xa["line"]["a"] + t * xa["line"]["u"])
    for v, t in zip(ya["vals"], ya["t"]):
        Xw.append([x_e, v, z_lo]); xi.append(ya["line"]["a"] + t * ya["line"]["u"])
    for v, t in zip(za["vals"], za["t"]):
        Xw.append([xz, yz, v]); xi.append(zl["a"] + t * zl["u"])
    Xw, xi = np.array(Xw), np.array(xi)
    Paff = np.zeros((3, 4)); Paff[:2, :3] = np.c_[dX, dY, dZ]; Paff[:2, 3] = t0; Paff[2, 3] = 1
    res_aff = float(np.sqrt(np.mean(np.sum((project(Paff, Xw) - xi) ** 2, 1))))
    P, res = Paff, res_aff
    if len(Xw) >= 8:
        try:
            Pd = dlt(Xw, xi)
            res_d = float(np.sqrt(np.mean(np.sum((project(Pd, Xw) - xi) ** 2, 1))))
            if res_d < res_aff:
                P, res = Pd, res_d
        except np.linalg.LinAlgError:
            pass
    def _lim(ax):
        L = ax["line"]
        v = sorted(float(_v_of(ax["fit"], t)) for t in (0.0, L["length"]))
        return v
    cam_limits = {"x": _lim(xa), "y": _lim(ya), "z": _lim(za)}
    view = None
    if len(Xw) >= 8:
        try:
            Pv, rms_v, view = fit_view_camera(Xw, xi, cam_limits)
            view["tick_rms_px"] = rms_v
            info["view_camera"] = view
            if rms_v <= 1.5:
                # the view model is rigid: right everywhere in the box when it fits the ticks
                P, res = Pv, rms_v
                info["camera_model"] = "view (elevation / azimuth / focal length)"
                # the ticks sit on three edges only; the grid lines on the three back panes cover
                # the whole box: add them as correspondences and fit the view again
                tk = {"x": (xa["vals"], [xa["line"]["a"] + t * xa["line"]["u"] for t in xa["t"]]),
                      "y": (ya["vals"], [ya["line"]["a"] + t * ya["line"]["u"] for t in ya["t"]]),
                      "z": (za["vals"], [zl["a"] + t * zl["u"] for t in za["t"]])}
                en = {"x": (xa["line"]["a"], xa["line"]["b"]), "y": (ya["line"]["a"], ya["line"]["b"]), "z": (zl["a"], zl["b"])}
                ref = refine_with_pane_grid(img, Pv, view, Xw, xi, x_e, y_e, z_lo, vx, vy, vz, excl, ticks=tk, ends=en)
                info["pane_grid"] = ref["info"]
                if ref.get("P") is not None:
                    P, view = ref["P"], ref["view"]
                    res = ref["tick_rms"]
                    view["tick_rms_px"] = rms_v              # the hypothesis ranking compares the same measure for all
                    view["tick_rms_box_px"] = ref["tick_rms"]
                    info["view_camera"] = view
                    info["camera_model"] = "view (elevation / azimuth / focal length), refined on the pane grid lines"
        except Exception as e:                       # never let the optional model break calibration
            info["view_camera_error"] = str(e)
    cam = {"P": P, "view": view, "limits": cam_limits, "x_edge": x_e, "y_edge": y_e, "z_floor": z_lo, "z_line_foot_xy": (float(xz), float(yz)),
           "x_values": vx.tolist(), "y_values": vy.tolist(), "z_values": vz.tolist(),
           "x_labels": lx, "y_labels": ly, "z_labels": lz, "x_categorical": catx, "y_categorical": caty,
           "axis_lines": {k: (a["line"]["a"].tolist(), a["line"]["b"].tolist()) for k, a in (("x", xa), ("y", ya), ("z", za))}}
    info.update(reprojection_rms_px=res, affine_rms_px=res_aff, camera="projective (DLT)" if P is not Paff else "affine")
    return cam, info


# ------------------------------------------------------------------ pane grid refinement
def _pane_grid_lines(lim, x_e, y_e, z_lo, vx, vy, vz, keyed=False, all_ticks=False):
    """3-D segments of the grid lines a 3-D plot draws on its three back panes: the floor
    (z = bottom) and the two walls on the far side from the axis lines (the y axis runs along
    x = x_e, so the far x wall is the other x limit; same for y). One line per tick value."""
    (x0, x1), (y0, y1), (z0, z1) = lim["x"], lim["y"], lim["z"]
    xf = x1 if abs(x_e - x0) < abs(x_e - x1) else x0
    yf = y1 if abs(y_e - y0) < abs(y_e - y1) else y0
    zf = z0 if abs(z_lo - z0) < abs(z_lo - z1) else z1
    inside = (lambda v, a, b: True) if all_ticks else (lambda v, a, b: min(a, b) + 1e-9 < v < max(a, b) - 1e-9)
    segs = []
    for i, v in enumerate(vx):
        if inside(v, x0, x1):
            segs.append((("x", i, 0), (v, y0, zf), (v, y1, zf)))          # floor
            segs.append((("x", i, 1), (v, yf, z0), (v, yf, z1)))          # far y wall
    for i, v in enumerate(vy):
        if inside(v, y0, y1):
            segs.append((("y", i, 0), (x0, v, zf), (x1, v, zf)))          # floor
            segs.append((("y", i, 1), (xf, v, z0), (xf, v, z1)))          # far x wall
    for i, v in enumerate(vz):
        if inside(v, z0, z1):
            segs.append((("z", i, 0), (x0, yf, v), (x1, yf, v)))          # far y wall
            segs.append((("z", i, 1), (xf, y0, v), (xf, y1, v)))          # far x wall
    # keyed: the same line keeps its key when a limit moves past a tick value (the fit must be
    # free to move a limit across a tick)
    return segs if keyed else [(a, b) for _, a, b in segs]


def measure_grid_points(img, P, segs, excl=None, n=24, search=6, want_seg=False):
    """for sample points along every predicted grid line, the grid line found across it in the
    pixels: a thin grey line, darker than the pane on both sides. Returns (3-D points, pixels)."""
    H, W = img.shape[:2]
    g = img.mean(-1).astype(float)
    sat = (img.max(-1).astype(float) - img.min(-1)) / np.maximum(img.max(-1), 1)
    Xs, ps, si = [], [], []
    for k_seg, (A, B) in enumerate(segs):
        A, B = np.array(A, float), np.array(B, float)
        pa, pb = project(P, np.array([A, B]))
        d = pb - pa
        L = np.linalg.norm(d)
        if L < 20:
            continue
        u = d / L; nrm = np.array([-u[1], u[0]])
        for t in np.linspace(0.08, 0.92, n):
            X = A + t * (B - A)
            p = project(P, X[None])[0]
            offs = np.arange(-search, search + 1)
            q = p[None, :] + offs[:, None] * nrm[None, :]
            xi_, yi_ = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
            if xi_.min() < 1 or yi_.min() < 1 or xi_.max() >= W - 1 or yi_.max() >= H - 1:
                continue
            if excl is not None and excl[yi_, xi_].any():
                continue
            prof = g[yi_, xi_]; sp = sat[yi_, xi_]
            if sp.max() > 0.12:                      # data drawn over the grid here
                continue
            k = int(np.argmin(prof))
            bg = np.median(np.r_[prof[:3], prof[-3:]])
            if bg - prof[k] < 8 or k in (0, len(offs) - 1):
                continue
            # sub-pixel centre: darkness-weighted around the minimum
            w = np.clip(bg - prof[max(0, k - 2):k + 3], 0, None)
            o = offs[max(0, k - 2):k + 3]
            c = float((w * o).sum() / w.sum()) if w.sum() > 0 else float(offs[k])
            Xs.append(X); ps.append(p + c * nrm); si.append(k_seg)
    if want_seg:
        return np.array(Xs), np.array(ps), np.array(si, int)
    return np.array(Xs), np.array(ps)


def refine_with_pane_grid(img, P, view, Xw, xi, x_e, y_e, z_lo, vx, vy, vz, excl=None, ticks=None, ends=None):
    """Refit the view camera on everything that is drawn at a known place of the axes box:
      * the tick marks - and the axis lines they sit on are box EDGES: the x axis runs along
        y = one y limit, z = the floor limit (not at the values read off where the drawn lines
        cross, which carry the reading error of two other axes). The tick 3-D points are
        therefore tied to the box limits being fitted;
      * the ends of the three axis lines = box corners (robust weight; an end may be hidden);
      * the grid lines on the three back panes (floor at the floor limit, the two far walls at
        the far limits): hundreds of measurements across the whole box (distance to the
        predicted line - sliding along it costs nothing), measured again with every improved
        camera.
    The box limits (axes are usually padded beyond their first / last tick by an unknown
    amount) move freely; the image placement (scale + shift) is fitted with the view numbers."""
    import matplotlib
    import sys as _sys
    if "ipykernel" not in _sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import least_squares
    out = {"info": {}}
    fig = plt.figure(figsize=(4, 4))
    try:
        ax = fig.add_subplot(projection="3d")
        lim0 = {k: list(v) for k, v in view["limits"].items()}
        span = {k: lim0[k][1] - lim0[k][0] for k in "xyz"}
        persp = view.get("proj_type") != "ortho" and view.get("focal_length") is not None
        a0 = view.get("box_aspect", (4, 4, 3))
        # which limit each axis line lies on
        ys = 0 if abs(y_e - lim0["y"][0]) < abs(y_e - lim0["y"][1]) else 1     # x line at y = limit[ys]
        xs = 0 if abs(x_e - lim0["x"][0]) < abs(x_e - lim0["x"][1]) else 1     # y line at x = limit[xs]
        zs = 0 if abs(z_lo - lim0["z"][0]) < abs(z_lo - lim0["z"][1]) else 1   # floor
        if ticks is None:
            ticks = {}
        # z line: at the (x, y) corner nearest its foot as read on the floor
        zx = int(np.argmin([abs(Xw[-1][0] - v) for v in lim0["x"]])) if len(Xw) else 1 - xs
        zy = int(np.argmin([abs(Xw[-1][1] - v) for v in lim0["y"]])) if len(Xw) else 1 - ys

        def unpack(q):
            lim = {k: (lim0[k][0] + q[5 + 2 * i] * span[k], lim0[k][1] + q[6 + 2 * i] * span[k]) for i, k in enumerate("xyz")}
            Pn = _view_matrix(ax, lim, q[0], q[1], float(np.exp(q[2])) if persp else None,
                              (4, 4 * np.exp(q[4]), 3 * np.exp(q[3])))
            S = np.array([[q[11], 0, q[12]], [0, -q[11], q[13]], [0, 0, 1.0]])
            return lim, S @ Pn

        def tick_pts(lim):
            X3, x2 = [], []
            for k, (vals, pts) in ticks.items():
                for v, pp in zip(vals, pts):
                    if k == "x":
                        X3.append([v, lim["y"][ys], lim["z"][zs]])
                    elif k == "y":
                        X3.append([lim["x"][xs], v, lim["z"][zs]])
                    else:
                        X3.append([lim["x"][zx], lim["y"][zy], v])
                    x2.append(pp)
            return np.array(X3, float).reshape(-1, 3), np.array(x2, float).reshape(-1, 2)

        def corner_pairs(lim):
            if not ends:
                return []
            c = {"x": ([lim["x"][0], lim["y"][ys], lim["z"][zs]], [lim["x"][1], lim["y"][ys], lim["z"][zs]]),
                 "y": ([lim["x"][xs], lim["y"][0], lim["z"][zs]], [lim["x"][xs], lim["y"][1], lim["z"][zs]]),
                 "z": ([lim["x"][zx], lim["y"][zy], lim["z"][0]], [lim["x"][zx], lim["y"][zy], lim["z"][1]])}
            return [(k, c[k]) for k in "xyz" if k in ends]

        drop = set()

        def corner_resid(Pp, lim, raw=False):
            r = []
            for k, (A3, B3) in corner_pairs(lim):
                pa, pb = project(Pp, np.array([A3, B3]))
                e0, e1 = np.asarray(ends[k][0], float), np.asarray(ends[k][1], float)
                # match measured ends to predicted corners
                if np.linalg.norm(pa - e0) + np.linalg.norm(pb - e1) > np.linalg.norm(pa - e1) + np.linalg.norm(pb - e0):
                    e0, e1 = e1, e0
                for nm, d_ in (((k, 0), pa - e0), ((k, 1), pb - e1)):
                    r += [0.0, 0.0] if (nm in drop and not raw) else list(d_)
            return np.array(r, float)

        def line_resid(Pp, lim, pg, si, segs_n):
            sg = _pane_grid_lines(lim, x_e, y_e, z_lo, vx, vy, vz, keyed=True, all_ticks=True)
            A = np.array([sg[k][1] for k in si], float); B = np.array([sg[k][2] for k in si], float)
            pa, pb = project(Pp, A), project(Pp, B)
            d = pb - pa; d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
            v = pg - pa
            return v[:, 0] * d[:, 1] - v[:, 1] * d[:, 0]

        # initial image placement from the current camera
        Pn0 = _view_matrix(ax, lim0, view["elev"], view["azim"], view.get("focal_length") if persp else None, a0)
        X3, x2 = tick_pts(lim0)
        Ps, _ = _similarity_fit(Pn0, X3 if len(X3) >= 3 else Xw, x2 if len(X3) >= 3 else xi)
        S0 = Ps[:2, :] @ np.linalg.pinv(Pn0)                    # recover s, tx, ty
        s0 = float(np.mean([Ps[0, :3] @ np.linalg.pinv(Pn0[0:1, :3]).ravel()])) if False else None
        h = Pn0 @ np.c_[X3, np.ones(len(X3))].T
        u, v = h[0] / h[2], h[1] / h[2]
        Am = np.zeros((2 * len(u), 3)); bm = np.zeros(2 * len(u))
        Am[0::2, 0], Am[0::2, 1], bm[0::2] = u, 1, x2[:, 0]
        Am[1::2, 0], Am[1::2, 2], bm[1::2] = -v, 1, x2[:, 1]
        (sc, tx, ty), *_ = np.linalg.lstsq(Am, bm, rcond=None)
        q = np.array([view["elev"], view["azim"], np.log(view.get("focal_length") or 1.0),
                      np.log(a0[2] / 3.0), np.log(a0[1] / 4.0), 0, 0, 0, 0, 0, 0, sc, tx, ty], float)
        # 1) the box limits from the edge-tied ticks + axis-line ends alone (the limits read off
        #    the axis-line ends can be far off; the far panes sit at those limits, so the grid
        #    lines cannot be searched for before they are right)
        def F0(qq):
            lim, Pp = unpack(qq)
            T3, t2 = tick_pts(lim)
            return np.r_[(project(Pp, T3) - t2).ravel(), corner_resid(Pp, lim) * 0.5, qq[:5] * 0.0]
        for _k in range(2):
            q = least_squares(F0, q, loss="soft_l1", f_scale=2.0,
                              x_scale=np.r_[2, 2, 0.1, 0.05, 0.05, [0.01] * 6, abs(q[11]) * 0.01, 2, 2], max_nfev=400).x
            cr = np.abs(corner_resid(unpack(q)[1], unpack(q)[0], raw=True)).reshape(-1, 2, 2).max(-1)
            ks = [k for k, _ in corner_pairs(unpack(q)[0])]
            drop.clear(); drop.update((ks[i], j) for i in range(len(ks)) for j in range(2) if cr[i, j] > 4.0)
        lim_c, P_c = unpack(q)
        all_keys = [k_ for k_, _, _ in _pane_grid_lines(lim0, x_e, y_e, z_lo, vx, vy, vz, keyed=True, all_ticks=True)]
        kidx = {k_: i_ for i_, k_ in enumerate(all_keys)}

        def meas(Pp, lim, search=6):
            sk = _pane_grid_lines(lim, x_e, y_e, z_lo, vx, vy, vz, keyed=True)
            Xa, pa_, sa = measure_grid_points(img, Pp, [(a_, b_) for _, a_, b_ in sk], excl, search=search, want_seg=True)
            return Xa, pa_, np.array([kidx[sk[j][0]] for j in sa], int)
        segs = all_keys
        Xg, pg, si = meas(P_c, lim_c, search=8)
        out["info"]["grid_points"] = int(len(Xg))
        if len(Xg) < 30 or len(X3) < 6:
            out["info"]["note"] = "too few pane grid points / ticks"
            return out

        def stats(Pp, lim, pg, si, nseg):
            T3, t2 = tick_pts(lim)
            rt = float(np.sqrt(np.mean(np.sum((project(Pp, T3) - t2) ** 2, 1))))
            rg = np.abs(line_resid(Pp, lim, pg, si, nseg))
            return rt, float(np.median(rg))
        # the old camera judged on the same grid measurements (its own limits)
        Xq, pq, sq = meas(np.asarray(P, float), lim0)
        rt0, rg0 = stats(np.asarray(P, float), lim0, pq, sq, len(segs)) if len(pq) else (np.inf, np.inf)
        for _round in range(4):
            nseg = len(segs)
            r_ = np.abs(line_resid(unpack(q)[1], unpack(q)[0], pg, si, nseg))
            keep = r_ <= max(3.0, 3.0 * np.median(r_))
            pg_, si_ = pg[keep], si[keep]
            if _round > 0:          # a hidden / shortened axis-line end is not a box corner
                cr = np.abs(corner_resid(unpack(q)[1], unpack(q)[0], raw=True)).reshape(-1, 2, 2).max(-1)
                ks = [k for k, _ in corner_pairs(unpack(q)[0])]
                drop.clear(); drop.update((ks[i], j) for i in range(len(ks)) for j in range(2) if cr[i, j] > 4.0)
            wg = 1.0 / np.sqrt(max(len(pg_), 1) / 120.0)        # the grid as a whole ~ 120 points' worth

            def F(qq):
                lim, Pp = unpack(qq)
                T3, t2 = tick_pts(lim)
                rt = (project(Pp, T3) - t2).ravel() * 0.5      # tick points sit ~1-2 px off the edge (tick marks / line width)
                rg = line_resid(Pp, lim, pg_, si_, nseg) * wg
                rc = corner_resid(Pp, lim) * 0.5
                # weak prior: the plotting program's defaults (focal length 1, box 4:4:3) - focal
                # length and box limits trade off against each other on a small box
                rp = np.r_[qq[2], qq[3], qq[4]] * 15.0
                return np.r_[rt, rg, rc, rp]
            sol = least_squares(F, q, loss="soft_l1", f_scale=2.0,
                                x_scale=np.r_[2, 2, 0.1, 0.05, 0.05, [0.01] * 6, abs(q[11]) * 0.01, 2, 2], max_nfev=600)
            q = sol.x
            lim_c, P_c = unpack(q)
            Xg2, pg2, si2 = meas(P_c, lim_c)
            if len(Xg2) >= 30:
                Xg, pg, si = Xg2, pg2, si2
        rt1, rg1 = stats(P_c, lim_c, pg, si, len(segs))
        rt_old = float(np.sqrt(np.mean(np.sum((project(np.asarray(P, float), Xw) - xi) ** 2, 1))))
        out["info"].update(grid_points=int(len(Xg)), grid_med_before=round(rg0, 2), grid_med_after=round(rg1, 2),
                           tick_rms_before=round(rt0, 2), tick_rms_after=round(rt1, 2), used=False,
                           corner_resid=np.round(np.abs(corner_resid(P_c, lim_c, raw=True)), 1).tolist(), corners_dropped=sorted(map(str, drop)))
        if rg1 <= max(1.0, rg0) and rt1 <= max(3.0, rt_old + 1.5):
            v1 = dict(view, elev=float(q[0]), azim=float(q[1]), focal_length=float(np.exp(q[2])) if persp else None,
                      box_aspect=(4.0, float(4 * np.exp(q[4])), float(3 * np.exp(q[3]))),
                      limits={k: [float(a_), float(b_)] for k, (a_, b_) in lim_c.items()})
            out.update(P=P_c, view=v1, tick_rms=rt1)
            out["info"]["used"] = True
    except Exception as e:
        import traceback
        out["info"]["error"] = traceback.format_exc()[-600:]
    finally:
        plt.close(fig)
    return out


def _view_matrix(ax, limits, elev, azim, focal, aspect):
    ax.set_xlim(*limits["x"]); ax.set_ylim(*limits["y"]); ax.set_zlim(*limits["z"])
    ax.set_box_aspect(aspect)
    if focal is None:
        ax.set_proj_type("ortho")
    else:
        ax.set_proj_type("persp", focal_length=focal)
    ax.view_init(elev, azim)
    M = ax.get_proj()
    return M[[0, 1, 3], :]                          # (x, y, w) rows: a 3x4 camera in normalised units


def _similarity_fit(Pn, Xw, xi):
    """pixels = s * normalised (x, -y) + t  (closed-form least squares) -> 3x4 pixel camera, rms"""
    h = (Pn @ np.c_[Xw, np.ones(len(Xw))].T).T
    u, v = h[:, 0] / h[:, 2], h[:, 1] / h[:, 2]
    A = np.zeros((2 * len(u), 3)); b = np.zeros(2 * len(u))
    A[0::2, 0], A[0::2, 1], b[0::2] = u, 1, xi[:, 0]
    A[1::2, 0], A[1::2, 2], b[1::2] = -v, 1, xi[:, 1]
    (s_, tx, ty), *_ = np.linalg.lstsq(A, b, rcond=None)
    S = np.array([[s_, 0, tx], [0, -s_, ty], [0, 0, 1.0]])
    P = S @ Pn
    rms = float(np.sqrt(np.mean(np.sum((project(P, Xw) - xi) ** 2, 1))))
    return P, rms


def fit_view_camera(Xw, xi, limits):
    """A camera described like a plotting program's 3-D view: elevation, azimuth, focal length
    (or orthographic), box aspect, plus where the drawing sits in the image. Only ~6 numbers
    instead of the 11 of a free camera, so three axis lines of ticks pin it down everywhere in
    the box (a free camera is poorly constrained far from the axis lines). Used when it explains
    the ticks as well as the free camera; it also gives the view angles to redraw the plot."""
    import sys as _sys
    import matplotlib
    if "ipykernel" not in _sys.modules:  # keep a notebook's inline backend (plt.show() must keep working there)
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection="3d")
    Xw, xi = np.asarray(Xw, float), np.asarray(xi, float)

    def evaluate(p, focal_free=True):
        elev, azim, lf, ar_z, ar_y = p
        focal = None if not focal_free else float(np.exp(lf))
        try:
            Pn = _view_matrix(ax, limits, elev, azim, focal, (4, 4 * np.exp(ar_y), 3 * np.exp(ar_z)))
            return _similarity_fit(Pn, Xw, xi)
        except Exception:
            return None, 1e9
    best = None
    for persp in (True, False):
        for elev in range(0, 91, 10):
            for azim in range(-180, 180, 10):
                P, r = evaluate((elev, azim, 0.0, 0.0, 0.0), persp)
                if best is None or r < best[0]:
                    best = (r, persp, (elev, azim, 0.0, 0.0, 0.0))
    r0, persp, p0 = best
    res = minimize(lambda p: evaluate(p, persp)[1], np.array(p0, float), method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 1500})
    # the other projection type, refined from the same start, may fit better
    res2 = minimize(lambda p: evaluate(p, not persp)[1], res.x, method="Nelder-Mead",
                    options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 800})
    if res2.fun < res.fun:
        res, persp = res2, not persp
    # the box limits (from the axis-line ends) may be a little off: let them move too
    span = {k: limits[k][1] - limits[k][0] for k in "xyz"}

    def lim_of(d):
        return {k: (limits[k][0] + d[2 * i] * span[k], limits[k][1] + d[2 * i + 1] * span[k]) for i, k in enumerate("xyz")}

    def evaluate2(q):
        nonlocal_l = lim_of(q[5:])
        elev, azim, lf, ar_z, ar_y = q[:5]
        focal = float(np.exp(lf)) if persp else None
        try:
            Pn = _view_matrix(ax, nonlocal_l, elev, azim, focal, (4, 4 * np.exp(ar_y), 3 * np.exp(ar_z)))
            return _similarity_fit(Pn, Xw, xi)
        except Exception:
            return None, 1e9
    res3 = minimize(lambda q: evaluate2(q)[1] + 2.0 * float(np.abs(q[5:]).sum()), np.r_[res.x, np.zeros(6)],
                    method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 4000})
    P3, rms3 = evaluate2(res3.x)
    if rms3 < evaluate(res.x, persp)[1] - 0.05:
        res_x, limits_fit = res3.x[:5], lim_of(res3.x[5:])
    else:
        res_x, limits_fit = res.x, limits
    Pn = _view_matrix(ax, limits_fit, res_x[0], res_x[1], float(np.exp(res_x[2])) if persp else None,
                      (4, 4 * np.exp(res_x[4]), 3 * np.exp(res_x[3])))
    P, rms = _similarity_fit(Pn, Xw, xi)
    elev, azim, lf, ar_z, ar_y = res_x
    plt.close(fig)
    return P, rms, {"elev": float(elev), "azim": float(azim), "proj_type": "persp" if persp else "ortho",
                    "focal_length": float(np.exp(lf)) if persp else None,
                    "box_aspect": (4.0, float(4 * np.exp(ar_y)), float(3 * np.exp(ar_z))),
                    "limits": {k: [float(a), float(b)] for k, (a, b) in limits_fit.items()}}

