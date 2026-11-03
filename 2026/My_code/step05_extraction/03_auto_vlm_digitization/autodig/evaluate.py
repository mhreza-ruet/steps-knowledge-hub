"""CSV export and scoring.

Scoring rule (same as the multi-model benchmark in 02_real_plots_multimodel):
  * reference points: every marker; a traced curve is compared at `line_points`
    evenly spaced x.
  * the candidate series is interpolated at each reference x; a reference x
    outside the candidate's x-range (beyond x_tol x reference spacing) or a
    series not found -> missing point (error 100 %).
  * point error = |y_cand - y_ref| / y-axis span x 100 (capped at 100)
  * hit rate = % points with error <= 5 %;  accuracy = 100 - mean error.
"""
from __future__ import annotations

import csv
import difflib
import re

import numpy as np

from .calibrate import AxisMap

SCORING = {"hit_threshold": 5.0, "line_points": 25, "x_tolerance": 0.5, "error_cap": 100.0}


# ------------------------------------------------------------------- CSV
def to_rows(res):
    rows = []
    cat_axis = res.get("category_axis")
    for name, d in res["series"].items():
        cats = d.get("categories")
        errs = d.get("yerr")
        for k, (x, y) in enumerate(zip(d["x"], d["y"])):
            ax = "right" if d["y_axis"] == "y_axis_right" else "left"
            if cat_axis is not None and cats is not None:
                ax = cat_axis[int(x)] if int(x) < len(cat_axis) else ax
            rows.append({"series": name, "x": x, "category": cats[k] if cats else "", "y": y,
                         "yerr": errs[k] if errs else "", "y_axis": ax, "kind": d["kind"]})
    return rows


def write_csv(res, path):
    if res.get("status") == "rejected":
        return None
    if "table" in res:                     # chart types with their own columns (heatmap, box, polar, 3-D ...)
        rows = res["table"]
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        if res.get("status") == "flagged":
            keys.append("flag")
            rows = [dict(r, flag=res.get("status_reasons", [""])[0]) for r in rows]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys or ["empty"])
            w.writeheader(); w.writerows(rows)
        return path
    rows = to_rows(res)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["series", "x", "category", "y", "yerr", "y_axis", "kind"])
        w.writeheader(); w.writerows(rows)
    return path


def read_csv(path):
    """CSV -> {'series': {name: {'x': [...], 'y': [...], 'y_axis':..., 'kind':...}}}"""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            s = out.setdefault(r["series"], {"x": [], "y": [], "y_axis": r.get("y_axis", "left"), "kind": r.get("kind", "marker")})
            s["x"].append(float(r["x"])); s["y"].append(float(r["y"]))
    return {"series": out}


# ------------------------------------------------------------------- matching
def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def match_names(cands, refs):
    """Map each reference series name to a candidate name (exact, containment, fuzzy, then order)."""
    out, used = {}, set()
    for r in refs:
        nr = _norm(r)
        hit = next((c for c in cands if c not in used and _norm(c) == nr), None)
        hit = hit or next((c for c in cands if c not in used and nr and (_norm(c) in nr or nr in _norm(c)) and _norm(c)), None)
        if hit is None:
            close = difflib.get_close_matches(nr, [_norm(c) for c in cands if c not in used], n=1, cutoff=0.5)
            if close:
                hit = next(c for c in cands if c not in used and _norm(c) == close[0])
        if hit is not None:
            out[r] = hit; used.add(hit)
    # whatever is left: pair by order
    rest_c = [c for c in cands if c not in used]
    for r in refs:
        if r not in out and rest_c:
            out[r] = rest_c.pop(0)
    return out


# ------------------------------------------------------------------- scoring
def axis_spans(res):
    """y-axis span (value units) of each axis, from the calibration at the frame edges.
    Log axes are scored in decades."""
    fr = res["frame"]
    spans = {}
    for key in ("y_axis", "y_axis_right"):
        maps = [AxisMap(**m) for m in res["calibration"].get(key, [])]
        if not maps:
            continue
        lo, hi = fr["bottom"], fr["top"]
        vals = []
        for m in maps:
            for p in (lo, hi, m.lo_px, m.hi_px):
                if p is not None and m.contains(p) and fr["top"] <= p <= fr["bottom"]:
                    v = float(m.to_value(p))
                    vals.append(np.log10(v) if m.log and v > 0 else v)
        spans[key] = (max(vals) - min(vals)) if vals else 1.0
        spans[key + "_log"] = maps[0].log
    return spans


def _interp(px, py, qx, tol):
    px = np.asarray(px, float); py = np.asarray(py, float)
    if len(px) == 0:
        return np.full(len(qx), np.nan)
    o = np.argsort(px); px, py = px[o], py[o]
    ux = np.unique(px)
    uy = np.array([py[px == v].mean() for v in ux])
    out = np.interp(qx, ux, uy)
    inside = (qx >= ux.min() - tol) & (qx <= ux.max() + tol) if len(ux) > 1 else np.abs(qx - ux[0]) <= tol
    out[~inside] = np.nan
    return out


def score(ref, cand, spans, kinds=None, S=SCORING):
    """ref, cand: {'series': {name: {'x','y',('y_axis')}}}. spans: from axis_spans(reference result).
    Returns (summary dict, per-series list, point rows)."""
    names = list(ref["series"])
    mapping = match_names(list((cand or {}).get("series", {})), names)
    rows = []
    for name in names:
        s = ref["series"][name]
        rx = np.asarray(s["x"], float); ry = np.asarray(s["y"], float)
        if len(rx) == 0:
            continue
        kind = (kinds or {}).get(name, s.get("kind", "marker"))
        if kind == "line" and len(rx) > S["line_points"]:
            o = np.argsort(rx); rx, ry = rx[o], ry[o]
            gx = np.linspace(rx.min(), rx.max(), S["line_points"])
            rx, ry = gx, np.interp(gx, rx, ry)
        key = "y_axis_right" if s.get("y_axis") in ("right", "y_axis_right") else "y_axis"
        span = spans.get(key, spans.get("y_axis", 1.0)) or 1.0
        log = spans.get(key + "_log", False)
        tol = S["x_tolerance"] * float(np.median(np.diff(np.sort(rx)))) if len(rx) > 1 else 0.0
        c = (cand or {}).get("series", {}).get(mapping.get(name, ""), None)
        cy = _interp(c["x"], c["y"], rx, tol) if c and c.get("x") else np.full(len(rx), np.nan)
        for xv, yt, ym in zip(rx, ry, cy):
            miss = bool(np.isnan(ym))
            if miss:
                err = 100.0
            else:
                d = abs(np.log10(max(ym, 1e-12)) - np.log10(max(yt, 1e-12))) if log else abs(ym - yt)
                err = float(min(d / span * 100, S["error_cap"]))
            rows.append({"series": name, "matched_to": mapping.get(name), "x": float(xv), "y_ref": float(yt),
                         "y_extracted": None if miss else float(ym), "error_pct": err, "missing": miss,
                         "hit": (not miss) and err <= S["hit_threshold"]})

    def summ(r):
        if not r:
            return {"accuracy": np.nan, "hit_rate": np.nan, "coverage": np.nan, "n_points": 0}
        e = np.array([x["error_pct"] for x in r]); m = np.array([x["missing"] for x in r]); h = np.array([x["hit"] for x in r])
        return {"accuracy": round(float(100 - e.mean()), 1), "hit_rate": round(float(100 * h.mean()), 1),
                "coverage": round(float(100 * (~m).mean()), 1), "n_points": int(len(r))}
    per = [{"series": n, "matched_to": mapping.get(n), **summ([r for r in rows if r["series"] == n])} for n in names]
    return summ(rows), per, rows
