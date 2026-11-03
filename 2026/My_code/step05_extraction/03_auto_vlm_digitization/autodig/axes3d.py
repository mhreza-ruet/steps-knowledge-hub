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


def dlt(Xw, xi):
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
    _, _, Vt = np.linalg.svd(np.array(rows))
    Pn = Vt[-1].reshape(3, 4)
    return np.linalg.inv(T2) @ Pn @ T3


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


def calibrate_3d(img, spec, excl=None):
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
    cam = {"P": P, "x_edge": x_e, "y_edge": y_e, "z_floor": z_lo, "z_line_foot_xy": (float(xz), float(yz)),
           "x_values": vx.tolist(), "y_values": vy.tolist(), "z_values": vz.tolist(),
           "x_labels": lx, "y_labels": ly, "z_labels": lz, "x_categorical": catx, "y_categorical": caty,
           "axis_lines": {k: (a["line"]["a"].tolist(), a["line"]["b"].tolist()) for k, a in (("x", xa), ("y", ya), ("z", za))}}
    info.update(reprojection_rms_px=res, affine_rms_px=res_aff, camera="projective (DLT)" if P is not Paff else "affine")
    return cam, info
