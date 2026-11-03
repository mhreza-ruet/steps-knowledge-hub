"""Score extracted chart data against a reference dataset.

All settings come from ``config.CONFIG["scoring"]``.

* Reference points: every marker of a marker series; for traced curves (FTIR,
  fitted kinetic curves) ``line_points`` evenly spaced points along the curve.
* The model's points for the same series are sorted by x and linearly
  interpolated at each reference x. A reference x outside the x-range the model
  covered (by more than ``x_tolerance`` x the reference spacing), or a series
  the model did not return, is a **missing** point.
* point error = |y_model - y_truth| / (span of that series' y-axis) x 100,
  capped at ``error_cap``. Normalising by the axis span keeps near-zero values
  from exploding and makes FTIR (arbitrary units), dual- and broken-axis plots
  comparable.

Per figure:
  accuracy  = 100 - mean point error, with missing points counted as 100 %.
              LENIENT: a flat line through the middle of the plot already
              scores ~70 (see reporting.midline_baseline).
  hit_rate  = % of reference points read within ``hit_threshold`` (5 %) of the
              axis span - the strict number, comparable to ChartQA's "relaxed
              accuracy" (5 % tolerance).
  coverage  = % of reference points the model returned at all.
"""
from __future__ import annotations

import numpy as np

from . import config as cfgmod
from .figures import FIGURES, axis_span, is_categorical


def _S():
    return cfgmod.CONFIG["scoring"]


def _ref_points(fig_id, name, s):
    x = np.asarray(s["x"], float); y = np.asarray(s["y"], float)
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    n = _S()["line_points"]
    if FIGURES[fig_id]["series"].get(name, {}).get("kind") == "line" and len(x) > n:
        o = np.argsort(x); x, y = x[o], y[o]
        # compare inside the labelled tick range (drops the few px a traced curve
        # was extended to reach the frame, e.g. kinetics x < 0)
        p0, v0, p1, v1 = FIGURES[fig_id]["calib"]["x"]
        lo, hi = max(x.min(), min(v0, v1)), min(x.max(), max(v0, v1))
        gx = np.linspace(lo, hi, n)
        return gx, np.interp(gx, x, y)
    return x, y


def _interp(px, py, qx, tol):
    px = np.asarray(px, float); py = np.asarray(py, float)
    ok = ~(np.isnan(px) | np.isnan(py)); px, py = px[ok], py[ok]
    if len(px) == 0:
        return np.full(len(qx), np.nan)
    o = np.argsort(px); px, py = px[o], py[o]
    ux = np.unique(px)
    uy = np.array([py[px == v].mean() for v in ux])
    out = np.interp(qx, ux, uy)
    inside = (qx >= ux.min() - tol) & (qx <= ux.max() + tol)
    if len(ux) == 1:
        inside = np.abs(qx - ux[0]) <= tol
    out[~inside] = np.nan
    return out


def point_rows(fig_id, ref, pred):
    """One row per reference point: truth, model value, error, hit."""
    S = _S()
    rows = []
    for name, s in ref["series"].items():
        rx, ry = _ref_points(fig_id, name, s)
        if len(rx) == 0:
            continue
        tol = S["x_tolerance"] * float(np.median(np.diff(np.sort(rx)))) if len(rx) > 1 else 0.0
        p = (pred or {}).get("series", {}).get(name)
        if is_categorical(fig_id):
            # bars / pie slices: a category is matched only by its own value (no interpolation)
            py = np.full(len(rx), np.nan)
            if p and p.get("x"):
                px_, py_ = np.asarray(p["x"], float), np.asarray(p["y"], float)
                for i, xv in enumerate(rx):
                    k = np.where(np.abs(px_ - xv) < 0.3)[0]
                    if len(k):
                        py[i] = float(np.nanmean(py_[k]))
        else:
            py = _interp(p["x"], p["y"], rx, tol) if p and p.get("x") else np.full(len(rx), np.nan)
        for xv, yt, ym in zip(rx, ry, py):
            span = axis_span(fig_id, name, xv, yt)
            missing = bool(np.isnan(ym))
            err = S["missing_point_error"] if missing else float(min(abs(ym - yt) / span * 100, S["error_cap"]))
            rows.append({"series": name, "x": float(xv), "y_true": float(yt),
                         "y_model": None if missing else float(ym), "error_pct_of_axis": err,
                         "missing": missing, "hit": (not missing) and err <= S["hit_threshold"]})
    return rows


def summarize(rows):
    if not rows:
        return {"accuracy": np.nan, "hit_rate": np.nan, "coverage": np.nan, "mean_error_covered": np.nan,
                "n_points": 0}
    err = np.array([r["error_pct_of_axis"] for r in rows])
    miss = np.array([r["missing"] for r in rows])
    hit = np.array([r["hit"] for r in rows])
    return {"accuracy": float(100 - err.mean()), "hit_rate": float(100 * hit.mean()),
            "coverage": float(100 * (~miss).mean()),
            "mean_error_covered": float(err[~miss].mean()) if (~miss).any() else np.nan,
            "n_points": int(len(rows))}


def score_figure(fig_id, ref, pred):
    rows = point_rows(fig_id, ref, pred)
    out = summarize(rows)
    out["figure"] = fig_id
    out["series"] = []
    for name in ref["series"]:
        r = [x for x in rows if x["series"] == name]
        if r:
            out["series"].append({"series": name, **summarize(r)})
    # legacy names used by older notebook cells
    out["mean_error"] = out["mean_error_covered"]
    out["coverage_frac"] = out["coverage"] / 100 if out["coverage"] == out["coverage"] else 0
    return out
