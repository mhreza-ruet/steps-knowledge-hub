"""Scoring of the synthetic gallery (gallery/truth/*.json) - one row per ground-truth item.

error % = |extracted - truth| / span x 100 (capped at 100), hit = error <= threshold.
span per type: heatmap - range of the true cell values; field / 3-D surface - true z range;
histogram, box, stacked / horizontal bars, 3-D bars - the value-axis span from the truth's
own maximum (0 .. max); polar - radial span of the truth; 3-D scatter - per coordinate span.
A truth item that was not extracted counts as missed (error 100 %).
Filled contours: the image only says which colour BAND a point is in, so the truth
there is the middle of the band that holds the true value (what the picture encodes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAP = 100.0


def _row(item, truth, got, span, thr):
    if got is None or (isinstance(got, float) and np.isnan(got)):
        return {"item": item, "truth": truth, "extracted": None, "error %": CAP, "hit": False}
    e = min(CAP, 100.0 * abs(float(got) - float(truth)) / max(span, 1e-12))
    return {"item": item, "truth": float(truth), "extracted": float(got), "error %": round(e, 2), "hit": e <= thr}


def _norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def score_heatmap(t, r, thr):
    tv = np.array(t["values"], float)
    span = float(tv.max() - tv.min())
    got = {(_norm(a), _norm(b)): row["value"] for row in r.get("table", []) for a, b in [(row["row"], row["column"])]}
    rows = []
    for i, rn in enumerate(t["rows"]):
        for j, cn in enumerate(t["columns"]):
            rows.append(_row(f"{rn} / {cn}", tv[i, j], got.get((_norm(rn), _norm(cn))), span, thr))
    return rows


def _nearest_valid_interp(xs, ys, Z, qx, qy):
    xs, ys, Z = np.asarray(xs, float), np.asarray(ys, float), np.array(Z, dtype=float)
    ok = ~np.isnan(Z)
    if ok.sum() == 0:
        return np.full(len(qx), np.nan)
    from scipy.interpolate import griddata
    GX, GY = np.meshgrid(xs, ys)
    pts = np.c_[GX[ok], GY[ok]]
    v = griddata(pts, Z[ok], np.c_[qx, qy], method="linear")
    sx, sy = np.ptp(xs) / max(1, len(xs) - 1), np.ptp(ys) / max(1, len(ys) - 1)    # one sample step of slack
    inside = (qx >= xs.min() - sx) & (qx <= xs.max() + sx) & (qy >= ys.min() - sy) & (qy <= ys.max() + sy)
    nn = griddata(pts, Z[ok], np.c_[qx, qy], method="nearest")
    v = np.where(np.isnan(v) & inside, nn, v)
    return np.where(inside, v, np.nan)


def score_field(t, r, thr):
    gx, gy, gz = np.array(t["x"]), np.array(t["y"]), np.array(t["z"], float)
    span = float(t["z_range"][1] - t["z_range"][0])
    GX, GY = np.meshgrid(gx, gy)
    qx, qy, tz = GX.ravel(), GY.ravel(), gz.ravel()
    if t.get("levels"):
        lv = np.array(t["levels"])
        k = np.clip(np.searchsorted(lv, tz) - 1, 0, len(lv) - 2)
        tz = 0.5 * (lv[k] + lv[k + 1])
    if "z" not in r:
        got = np.full(len(qx), np.nan)
    else:
        Z = np.array([[np.nan if v is None else v for v in row] for row in r["z"]], float)
        got = _nearest_valid_interp(r["x"], r["y"], Z, qx, qy)
    return [_row(f"({x:.3g}, {y:.3g})", z, g, span, thr) for x, y, z, g in zip(qx, qy, tz, got)]


def _match_series(truth_names, got_names):
    from .evaluate import match_names
    return match_names(list(got_names), list(truth_names))


def _bar_values(series):
    # The original bar extractor stores values in y; stacked bars use values.
    return series.get("values", series.get("y", []))


def score_hist(t, r, thr):
    rows = []
    got = r.get("series", {})
    mp = _match_series(t["series"], got)
    for name, d in t["series"].items():
        e, h = np.array(d["edges"]), np.array(d["heights"], float)
        span = float(h.max())
        g = got.get(mp.get(name)) if mp.get(name) else None
        for k in range(len(h)):
            c = 0.5 * (e[k] + e[k + 1])
            if g is None or not all(k in g for k in ("left", "right", "height")):
                rows.append(_row(f"{name} [{e[k]:.3g}, {e[k + 1]:.3g})", h[k], None, span, thr)); continue
            v = 0.0
            for lo, hi, hv in zip(g["left"], g["right"], g["height"]):
                if lo <= c < hi:
                    v = hv; break
            rows.append(_row(f"{name} [{e[k]:.3g}, {e[k + 1]:.3g})", h[k], v, span, thr))
    return rows


def score_box(t, r, thr):
    rows = []
    got = r.get("series", {})
    mp = _match_series(t["series"], got)
    for name, d in t["series"].items():
        allv = [v for st in d["stats"] for v in (st["whislo"], st["whishi"], *st["fliers"])]
        span = float(max(allv) - min(0.0, min(allv)))
        g = got.get(mp.get(name)) if mp.get(name) else None
        gst = dict(zip([_norm(c) for c in g["categories"]], g["stats"])) if g else {}
        for c, st in zip(d["categories"], d["stats"]):
            e = gst.get(_norm(c))
            for q in ("whislo", "q1", "median", "q3", "whishi"):
                rows.append(_row(f"{name} / {c} / {q}", st[q], None if e is None else e.get(q), span, thr))
            # fliers: each true flier matched to the nearest extracted one
            ef = list(e["fliers"]) if e else []
            for f in st["fliers"]:
                if ef:
                    k = int(np.argmin([abs(x - f) for x in ef])); v = ef.pop(k)
                else:
                    v = None
                rows.append(_row(f"{name} / {c} / flier", f, v, span, thr))
    return rows


def score_stacked(t, r, thr):
    rows = []
    got = r.get("series", {})
    mp = _match_series(t["series"], got)
    if t.get("stacked", True):
        tot = np.sum([np.array(v, float) for v in t["series"].values()], 0)
        span = float(tot.max())
    else:
        span = float(max(max(v) for v in t["series"].values()))
    for name, vals in t["series"].items():
        g = got.get(mp.get(name)) if mp.get(name) else None
        gv = dict(zip([_norm(c) for c in g.get("categories", [])], _bar_values(g))) if g else {}
        for c, v in zip(t["categories"], vals):
            rows.append(_row(f"{name} / {c}", v, gv.get(_norm(c)), span, thr))
    return rows


def score_polar(t, r, thr):
    rows = []
    got = r.get("series", {})
    mp = _match_series(t["series"], got)
    for name, d in t["series"].items():
        th, rr = np.radians(np.array(d["theta_deg"], float)), np.array(d["r"], float)
        span = float(rr.max())
        g = got.get(mp.get(name)) if mp.get(name) else None
        step = 1 if len(th) <= 60 else int(np.ceil(len(th) / 60))      # long curves: 60 points
        for k in range(0, len(th), step):
            item = f"{name} @ {np.degrees(th[k]):.1f} deg"
            if g is None or not g["r"]:
                rows.append(_row(item, rr[k], None, span, thr)); continue
            gt, gr = np.radians(np.array(g["theta_deg"])), np.array(g["r"])
            if d["kind"] == "bar":
                dd = np.abs((np.degrees(gt - th[k]) + 180) % 360 - 180)
                j = int(np.argmin(dd))
                rows.append(_row(item, rr[k], gr[j] if dd[j] <= 6 else None, span, thr)); continue
            # nearest extracted point in the plane; its radius is the comparison
            X, Y = rr[k] * np.cos(th[k]), rr[k] * np.sin(th[k])
            dist = np.hypot(gr * np.cos(gt) - X, gr * np.sin(gt) - Y)
            j = int(np.argmin(dist))
            if dist[j] > 0.15 * span:
                rows.append(_row(item, rr[k], None, span, thr)); continue
            rows.append({"item": item, "truth": float(rr[k]), "extracted": float(gr[j]),
                         "error %": round(min(CAP, 100 * dist[j] / span), 2), "hit": 100 * dist[j] / span <= thr})
    return rows


def score_bar3d(t, r, thr):
    H = np.array(t["heights"], float)
    span = float(H.max())
    got = {}
    for row in r.get("table", []):
        got[(_norm(row["x_category"]), _norm(row["y_category"]))] = row["height"]
    rows = []
    for i, xc in enumerate(t["x_categories"]):
        for j, yc in enumerate(t["y_categories"]):
            rows.append(_row(f"{xc} / {yc}", H[i, j], got.get((_norm(xc), _norm(yc))), span, thr))
    return rows


def score_scatter3d(t, r, thr):
    rows = []
    got = r.get("series", {})
    allp = np.array([[a, b, c] for d in t["series"].values() for a, b, c in zip(d["x"], d["y"], d["z"])])
    span = np.ptp(allp, 0) + 1e-9
    mp = _match_series(t["series"], got)
    for name, d in t["series"].items():
        g = got.get(mp.get(name)) if mp.get(name) else None
        G = np.c_[g["x"], g["y"], g["z"]] if g and g["x"] else np.zeros((0, 3))
        used = set()
        for a, b, c in zip(d["x"], d["y"], d["z"]):
            item = f"{name} ({a:.3g}, {b:.3g}, {c:.3g})"
            if len(G) == 0:
                rows.append(_row(item, c, None, span[2], thr)); continue
            dist = np.sqrt((((G - [a, b, c]) / span) ** 2).sum(1)) * 100 / np.sqrt(3)
            for k in np.argsort(dist):
                if k not in used:
                    break
            used.add(int(k))
            e = float(dist[k])
            rows.append({"item": item, "truth": float(c), "extracted": float(G[k, 2]), "error %": round(min(CAP, e), 2), "hit": e <= thr})
    return rows


def score_surface3d(t, r, thr):
    gx, gy, gz = np.array(t["x"]), np.array(t["y"]), np.array(t["z"], float)
    span = float(t["z_range"][1] - t["z_range"][0])
    s_ = r.get("series", {}).get("surface")
    GX, GY = np.meshgrid(gx, gy)
    rows = []
    if not s_ or not s_["x"]:
        return [_row(f"({x:.3g}, {y:.3g})", z, None, span, thr) for x, y, z in zip(GX.ravel(), GY.ravel(), gz.ravel())]
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((gy, gx), gz, bounds_error=False, fill_value=np.nan)
    ex, ey, ez = np.array(s_["x"]), np.array(s_["y"]), np.array(s_["z"])
    dx, dy = np.ptp(gx) / (len(gx) - 1), np.ptp(gy) / (len(gy) - 1)
    # each truth grid point: the extracted samples within half a grid step; their z vs the true
    # surface at THEIR (x, y). No sample nearby = a hidden (or missed) part of the surface
    for x, y, z in zip(GX.ravel(), GY.ravel(), gz.ravel()):
        near = (np.abs(ex - x) <= dx / 2) & (np.abs(ey - y) <= dy / 2)
        if not near.any():
            rows.append(_row(f"({x:.3g}, {y:.3g})", z, None, span, thr)); continue
        tz = f(np.c_[ey[near], ex[near]])
        e = float(np.nanmedian(np.abs(ez[near] - tz))) * 100 / span
        rows.append({"item": f"({x:.3g}, {y:.3g})", "truth": float(z), "extracted": float(np.median(ez[near])),
                     "error %": round(min(CAP, e), 2), "hit": e <= thr})
    return rows


SCORERS = {"bar3d": score_bar3d, "scatter3d": score_scatter3d, "surface3d": score_surface3d, "polar": score_polar, "stacked": score_stacked, "box": score_box, "histogram": score_hist, "heatmap": score_heatmap, "field": score_field}


def score(truth, res, thr=5.0):
    """-> DataFrame (one row per truth item); empty for unsupported truth types"""
    f = SCORERS.get(truth.get("type"))
    if f is None:
        return pd.DataFrame(columns=["item", "truth", "extracted", "error %", "hit"])
    return pd.DataFrame(f(truth, res, thr))


def summary(name, df, status, truth=None, res=None, thr=5.0):
    n = len(df)
    if n == 0:
        return {"figure": name, "status": status, "truth items": 0}
    found = df["extracted"].notna()
    out = {"figure": name, "status": status, "truth items": n, "hit rate %": round(100 * df["hit"].mean(), 1),
           "accuracy %": round(100 - df["error %"].mean(), 1), "coverage %": round(100 * found.mean(), 1),
           "mean error % (found)": round(df.loc[found, "error %"].mean(), 2) if found.any() else None}
    if truth is not None and res is not None:
        p = prf_gallery(truth, res, df, thr)
        out.update({k: p.get(k) for k in ("precision %", "recall %", "F1 %", "extracted items")})
    return out


# ===================================================================== precision side
# For precision every EXTRACTED item is checked against the truth (same tolerance as the hit
# rate); an extracted item with no true counterpart is wrong. Recall = hit rate of the truth
# items above (histograms: only bins with a non-zero true height count as items).

def _ok(v, tv, span, thr):
    return tv is not None and tv == tv and v is not None and v == v and 100 * abs(v - tv) / max(span, 1e-12) <= thr


def ext_heatmap(t, r, thr):
    tv = np.array(t["values"], float); span = float(tv.max() - tv.min())
    T = {(_norm(a), _norm(b)): tv[i, j] for i, a in enumerate(t["rows"]) for j, b in enumerate(t["columns"])}
    rows = r.get("table", [])
    return sum(_ok(x["value"], T.get((_norm(x["row"]), _norm(x["column"]))), span, thr) for x in rows), len(rows)


def ext_field(t, r, thr):
    from scipy.interpolate import RegularGridInterpolator
    gx, gy, gz = np.array(t["x"]), np.array(t["y"]), np.array(t["z"], float)
    span = float(t["z_range"][1] - t["z_range"][0])
    f = RegularGridInterpolator((gy, gx), gz, bounds_error=False, fill_value=np.nan)
    rows = r.get("table", [])
    if not rows:
        return 0, 0
    # A misclassified field can produce heatmap rows (row/column/value).
    # Those are false positives, not spatial samples; keep them in the denominator.
    samples = [x for x in rows if all(k in x for k in ("x", "y", "z"))]
    if not samples:
        return 0, len(rows)
    ex = np.array([[x["x"], x["y"], x["z"]] for x in samples], float)
    tz = f(np.c_[ex[:, 1], ex[:, 0]])
    if t.get("levels"):
        lv = np.array(t["levels"]); k = np.clip(np.searchsorted(lv, tz) - 1, 0, len(lv) - 2)
        tz = np.where(np.isnan(tz), np.nan, 0.5 * (lv[k] + lv[k + 1]))
    return int(sum(_ok(a, b, span, thr) for a, b in zip(ex[:, 2], tz))), len(rows)


def ext_hist(t, r, thr):
    got = r.get("series", {}); mp = _match_series(t["series"], got)
    inv = {v: k for k, v in mp.items() if v}
    good = n = 0
    for gname, g in got.items():
        if not all(k in g for k in ("left", "right", "height")):
            # A line mistaken for a histogram has points, not extracted bins.
            n += len(g.get("x", []))
            continue
        n += len(g["height"])
        d = t["series"].get(inv.get(gname))
        if d is None:
            continue
        e, h = np.array(d["edges"]), np.array(d["heights"], float); span = float(h.max())
        for lo, hi, hv in zip(g["left"], g["right"], g["height"]):
            c = 0.5 * (lo + hi)
            k = np.searchsorted(e, c, side="right") - 1
            tv = float(h[k]) if 0 <= k < len(h) else 0.0
            good += _ok(hv, tv, span, thr)
    return good, n


def ext_box(t, r, thr):
    got = r.get("series", {}); mp = _match_series(t["series"], got)
    inv = {v: k for k, v in mp.items() if v}
    good = n = 0
    for gname, g in got.items():
        d = t["series"].get(inv.get(gname))
        tst = dict(zip([_norm(c) for c in d["categories"]], d["stats"])) if d else {}
        if d:
            allv = [v for st in d["stats"] for v in (st["whislo"], st["whishi"], *st["fliers"])]
            span = float(max(allv) - min(0.0, min(allv)))
        for c, st in zip(g["categories"], g["stats"]):
            ts = tst.get(_norm(c))
            for q in ("whislo", "q1", "median", "q3", "whishi"):
                if st.get(q) is not None:
                    n += 1
                    good += bool(ts) and _ok(st[q], ts[q], span, thr)
            tf = list(ts["fliers"]) if ts else []
            for f in st["fliers"]:
                n += 1
                if tf:
                    k = int(np.argmin([abs(x - f) for x in tf]))
                    if _ok(f, tf[k], span, thr):
                        good += 1; tf.pop(k)
    return good, n


def ext_stacked(t, r, thr):
    got = r.get("series", {}); mp = _match_series(t["series"], got)
    inv = {v: k for k, v in mp.items() if v}
    span = float(np.sum([np.array(v, float) for v in t["series"].values()], 0).max()) if t.get("stacked", True) \
        else float(max(max(v) for v in t["series"].values()))
    good = n = 0
    for gname, g in got.items():
        tv = t["series"].get(inv.get(gname))
        T = dict(zip([_norm(c) for c in t["categories"]], tv)) if tv else {}
        for c, v in zip(g.get("categories", []), _bar_values(g)):
            n += 1; good += _ok(v, T.get(_norm(c)), span, thr)
    return good, n


def ext_polar(t, r, thr):
    from .prf import match_points
    got = r.get("series", {}); mp = _match_series(t["series"], got)
    inv = {v: k for k, v in mp.items() if v}
    good = n = 0
    for gname, g in got.items():
        n += len(g["r"])
        d = t["series"].get(inv.get(gname))
        if d is None or not g["r"]:
            continue
        th, rr = np.radians(np.array(d["theta_deg"], float)), np.array(d["r"], float); span = float(rr.max())
        gt_, gr = np.radians(np.array(g["theta_deg"], float)), np.array(g["r"], float)
        if d["kind"] == "line":                      # each extracted point against the true curve
            TX, TY = rr * np.cos(th), rr * np.sin(th)
            for a, b in zip(gt_, gr):
                good += 100 * np.hypot(TX - b * np.cos(a), TY - b * np.sin(a)).min() / span <= thr
        elif d["kind"] == "bar":
            def ok(i, j):
                da = abs((np.degrees(th[i] - gt_[j]) + 180) % 360 - 180)
                return da <= 6 and 100 * abs(rr[i] - gr[j]) / span <= thr, da
            good += match_points(np.c_[th, rr], np.c_[gt_, gr], ok)
        else:
            def ok(i, j):
                e = 100 * np.hypot(rr[i] * np.cos(th[i]) - gr[j] * np.cos(gt_[j]), rr[i] * np.sin(th[i]) - gr[j] * np.sin(gt_[j])) / span
                return e <= thr, e
            good += match_points(np.c_[th, rr], np.c_[gt_, gr], ok)
    return good, n


def ext_bar3d(t, r, thr):
    H = np.array(t["heights"], float); span = float(H.max())
    T = {(_norm(a), _norm(b)): H[i, j] for i, a in enumerate(t["x_categories"]) for j, b in enumerate(t["y_categories"])}
    rows = r.get("table", [])
    return sum(_ok(x["height"], T.get((_norm(x["x_category"]), _norm(x["y_category"]))), span, thr) for x in rows), len(rows)


def ext_scatter3d(t, r, thr):
    from .prf import match_points
    allp = np.array([[a, b, c] for d in t["series"].values() for a, b, c in zip(d["x"], d["y"], d["z"])])
    span = np.ptp(allp, 0) + 1e-9
    got = r.get("series", {}); mp = _match_series(t["series"], got)
    inv = {v: k for k, v in mp.items() if v}
    good = n = 0
    for gname, g in got.items():
        n += len(g["x"])
        d = t["series"].get(inv.get(gname))
        if d is None or not g["x"]:
            continue
        T, G = np.c_[d["x"], d["y"], d["z"]], np.c_[g["x"], g["y"], g["z"]]

        def ok(i, j):
            e = float(np.sqrt((((G[j] - T[i]) / span) ** 2).sum())) * 100 / np.sqrt(3)
            return e <= thr, e
        good += match_points(T, G, ok)
    return good, n


def ext_surface3d(t, r, thr):
    from scipy.interpolate import RegularGridInterpolator
    s_ = r.get("series", {}).get("surface")
    if not s_ or not s_["x"]:
        return 0, 0
    gx, gy, gz = np.array(t["x"]), np.array(t["y"]), np.array(t["z"], float)
    span = float(t["z_range"][1] - t["z_range"][0])
    f = RegularGridInterpolator((gy, gx), gz, bounds_error=False, fill_value=np.nan)
    tz = f(np.c_[s_["y"], s_["x"]])
    return int(sum(_ok(a, b, span, thr) for a, b in zip(s_["z"], tz))), len(s_["z"])


EXTRACTED = {"heatmap": ext_heatmap, "field": ext_field, "histogram": ext_hist, "box": ext_box, "stacked": ext_stacked,
             "polar": ext_polar, "bar3d": ext_bar3d, "scatter3d": ext_scatter3d, "surface3d": ext_surface3d}


def prf_gallery(truth, res, df, thr=5.0):
    """precision / recall / F1 for one gallery image (df = score() rows)"""
    from .prf import prf
    f = EXTRACTED.get(truth.get("type"))
    if f is None or df is None or not len(df):
        return {}
    d = df[df["truth"] > 0] if truth.get("type") == "histogram" else df
    good, n = f(truth, res, thr) if res.get("status") != "rejected" else (0, 0)
    return prf(int(d["hit"].sum()), len(d), good, n)
