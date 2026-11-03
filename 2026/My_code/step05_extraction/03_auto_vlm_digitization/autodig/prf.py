"""Precision, recall and F1 of an extraction.

Hit rate (the older metric) only asks "was each TRUE point recovered?" - it never
punishes points the digitizer invented. Precision does:

  recall     = correct true items / all true items            (what was found)
  precision  = correct extracted items / all extracted items  (how much of the output is right)
  F1         = 2 P R / (P + R)

"Correct" uses the same tolerance as the hit rate: value error <= threshold (5 %) of the
axis span. How items are paired depends on what they are:

  discrete items (markers, bars, pie slices, heatmap cells, box statistics, histogram bins,
  polar markers / sectors, 3-D bars and points): one-to-one pairing (Hungarian, or by
  category / name); an extracted item paired with nothing is a false positive, a true
  item paired with nothing is a false negative.
  markers: a pair must also agree in x (within half the typical spacing of the true markers).
  continuous items (traced curves, colour maps, 3-D surfaces, polar curves): recall = share of
  true sample points whose curve value is right; precision = share of EXTRACTED samples lying
  within the tolerance of the true curve / surface (samples outside the true x-range are wrong).
  Series the digitizer made up (paired with no true series) count fully as false positives.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def prf(tp_recall, n_true, tp_precision, n_extracted):
    r = tp_recall / n_true if n_true else np.nan
    p = tp_precision / n_extracted if n_extracted else (0.0 if n_true else np.nan)
    f = (2 * p * r / (p + r) if p + r > 0 else 0.0) if (p == p and r == r) else np.nan
    pct = lambda v: round(100 * float(v), 1) if v == v else None
    return {"precision %": pct(p), "recall %": pct(r), "F1 %": pct(f),
            "TP (true side)": int(tp_recall), "true items": int(n_true),
            "TP (extracted side)": int(tp_precision), "extracted items": int(n_extracted)}


def match_points(T, E, ok):
    """one-to-one pairing of true items T (n x d) and extracted items E (m x d);
    ok(i, j) -> (feasible, cost). Returns the number of feasible pairs."""
    if len(T) == 0 or len(E) == 0:
        return 0
    C = np.full((len(T), len(E)), 1e6)
    for i in range(len(T)):
        for j in range(len(E)):
            f, c = ok(i, j)
            if f:
                C[i, j] = c
    r, c = linear_sum_assignment(C)
    return int(sum(1 for a, b in zip(r, c) if C[a, b] < 1e6))


# ------------------------------------------------------------------ rebuilt plots (notebook 02 truth)
def prf_rebuilt(fid, gt, res, mapping, thr=5.0):
    """gt: notebook-02 ground truth; res: pipeline() output; mapping: truth name -> extracted name"""
    from chartbench.figures import FIGURES, axis_span, is_categorical
    from chartbench.scoring import _ref_points
    F = FIGURES[fid]
    tp_r = n_t = tp_p = n_e = 0
    used = set()
    for name, s in gt["series"].items():
        ex = res["series"].get(mapping.get(name)) if mapping.get(name) else None
        if ex is not None:
            used.add(mapping[name])
        ex_x = np.asarray(ex["x"], float) if ex and ex.get("x") else np.array([])
        ex_y = np.asarray(ex["y"], float) if ex and ex.get("y") else np.array([])
        kind = F["series"].get(name, {}).get("kind")
        if is_categorical(fid):
            tx, ty = np.asarray(s["x"], float), np.asarray(s["y"], float)
            n_t += len(tx); n_e += len(ex_x)
            good = 0
            for xv, yv in zip(tx, ty):
                k = np.where(np.abs(ex_x - xv) < 0.3)[0]
                if len(k) and abs(ex_y[k[0]] - yv) / axis_span(fid, name, xv, yv) * 100 <= thr:
                    good += 1
            tp_r += good; tp_p += good
        elif kind == "line":
            rx, ry = _ref_points(fid, name, s)          # 25 evenly spaced true samples
            n_t += len(rx)
            o = np.argsort(np.asarray(s["x"], float))
            fx, fy = np.asarray(s["x"], float)[o], np.asarray(s["y"], float)[o]
            if len(ex_x):
                oe = np.argsort(ex_x)
                vy = np.interp(rx, ex_x[oe], ex_y[oe], left=np.nan, right=np.nan)
                tp_r += int(sum(1 for xv, a, b in zip(rx, ry, vy) if b == b and abs(a - b) / axis_span(fid, name, xv, a) * 100 <= thr))
                # precision: every extracted sample against the true curve
                n_e += len(ex_x)
                ty = np.interp(ex_x, fx, fy, left=np.nan, right=np.nan)
                tp_p += int(sum(1 for xv, a, b in zip(ex_x, ex_y, ty) if b == b and abs(a - b) / axis_span(fid, name, xv, b) * 100 <= thr))
        else:                                           # markers
            tx, ty = np.asarray(s["x"], float), np.asarray(s["y"], float)
            n_t += len(tx); n_e += len(ex_x)
            xtol = 0.5 * float(np.median(np.diff(np.sort(tx)))) if len(tx) > 1 else 1e9

            def ok(i, j):
                dy = abs(ex_y[j] - ty[i]) / axis_span(fid, name, tx[i], ty[i]) * 100
                dx = abs(ex_x[j] - tx[i])
                return (dy <= thr and dx <= xtol), dy / thr + dx / max(xtol, 1e-12)
            tp = match_points(np.c_[tx, ty], np.c_[ex_x, ex_y], ok)
            tp_r += tp; tp_p += tp
    for n, d in res["series"].items():                  # series that match no true series: all wrong
        if n not in used:
            n_e += len(d.get("x", []))
    return prf(tp_r, n_t, tp_p, n_e)
