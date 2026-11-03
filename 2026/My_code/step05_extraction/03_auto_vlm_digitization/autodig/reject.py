"""Rejection of plots the pipeline cannot read reliably (unknown types, failed self-checks).

Every image ends with a status:
  ok        extracted, all self-checks passed       -> CSV, recreated plot, counted in the scores
  flagged   extracted, but part of the data is a guess (3-D depth without a cue)
            -> CSV (with the reason), recreated plot, NOT counted in the scores
  rejected  not extracted (unsupported type, or a self-check failed)
            -> no CSV; the reason is listed; NOT counted in the scores

Gates, in order:
  1. chart type: the VLM's answer mapped to a supported type; 'other' (violin, radar,
     network, Venn, ternary, table, photo, schematic ...) is rejected straight away
  2. several plot panels in one image (more than one axes frame found) -> rejected
  3. the extractor itself refused (no colour bar for a heatmap, a glyph that is not a
     box, no circular axis, no 3-D axis lines, ...) -> rejected with its reason
  4. self-checks of the extraction (tick-label fit, counts found vs expected, colour
     distances ...); any failed check -> rejected, with the measured value
For the original line / scatter / bar / pie readers the checks are built here from
their calibration and output. Thresholds are in GATES (one place to tune them).
"""
from __future__ import annotations

import numpy as np

from . import detect as D
from .spec_types import SUPPORTED, canonical_type

GATES = {"label_fit_px": 3.0,            # tick labels must line up with a straight (or 2-piece) scale
         "min_points": 1,                # at least this many data points / bars / slices
         "panel_min_frac": 0.2,          # a frame side must span >= 20 % of the image to count as a panel
         "pie_fill": 0.80}               # share of the pie's ellipse covered by slice colours


def count_panels(img):
    """axes frames in the image: bottom-left corners where a long vertical dark line
    meets a long horizontal one"""
    H, W = img.shape[:2]
    d = D.dark_mask(img, 120, 0.3)
    fr = GATES["panel_min_frac"]
    hs, vs = [], []
    for y in range(H):
        for a, b in D._runs(d[y]):
            if b - a >= fr * W:
                hs.append((y, a, b))
    for x in range(W):
        for a, b in D._runs(d[:, x]):
            if b - a >= fr * H:
                vs.append((x, a, b))
    corners = []
    for x, va, vb in vs:
        for y, ha, hb in hs:
            if abs(vb - 1 - y) <= 3 and abs(ha - x) <= 3:
                if not any(abs(cx - x) <= 6 and abs(cy - y) <= 6 for cx, cy in corners):
                    corners.append((x, y))
    return len(corners), corners


def _legacy_checks(res):
    """self-checks for the line / scatter / bar / pie readers"""
    ct = res.get("chart_type") or res.get("spec", {}).get("chart_type")
    checks = []
    ci = res.get("calibration_info", {})
    npts = sum(len(d.get("y", d.get("values", []))) for d in res.get("series", {}).values())
    if ct == "pie":
        sl = res.get("series", {}).get("share (%)", {})
        n = len(sl.get("y", []))
        checks.append(("slices found", n, ">= 2", n >= 2))
        return checks
    for k in ("x_axis", "y_axis", "y_axis_right"):
        if k not in ci or (k == "x_axis" and ct == "bar"):
            continue
        inf = ci[k]
        if inf.get("relative"):
            if k == "x_axis":
                checks.append(("x-axis has numeric tick labels", "no", "yes", False))
            continue
        rms = inf.get("rms_px")
        nf, npair = inf.get("n_labels_found", 0), len(inf.get("pairs", []))
        nv = inf.get("n_values", 0)
        if nv >= 3 and npair < 3:
            # two points always fit a straight line: with fewer than 3 labels paired nothing was checked
            checks.append((f"{k.replace('_', ' ')} labels paired with a value", npair, ">= 3", False))
        if nf >= 4 and npair < 0.6 * nf:
            # e.g. a broken axis read as one piece: half the printed labels have no value
            checks.append((f"{k.replace('_', ' ')} labels with a value / labels found", f"{npair}/{nf}", ">= 60 %", False))
        if rms is None:
            checks.append((f"{k.replace('_', ' ')} calibrated", "no", "yes", False))
        else:
            checks.append((f"{k.replace('_', ' ')} label fit (px)", round(float(rms), 2), f"<= {GATES['label_fit_px']}",
                           float(rms) <= GATES["label_fit_px"]))
    checks.append(("data points extracted", npts, f">= {GATES['min_points']}", npts >= GATES["min_points"]))
    if ct == "bar":
        nc, ns = len(res.get("categories", [])), max(1, len(res.get("series", {})))
        checks.append(("bars found / expected", f"{npts}/{nc * ns}", ">= 50 %", npts >= 0.5 * nc * ns))
    return checks


def pie_fill(img, res):
    pc = res.get("pie_calibration") or {}
    if "apex" not in pc:
        return None
    H, W = img.shape[:2]
    cx, cy = pc["apex"]
    yy, xx = np.mgrid[:H, :W]
    dx, dy = xx - cx, yy - cy
    rx = np.where(dx >= 0, pc["right"], pc["left"]); ry = np.where(dy >= 0, pc["down"], pc["up"])
    inside = (dx / np.maximum(rx, 1)) ** 2 + (dy / np.maximum(ry, 1)) ** 2 <= 0.85 ** 2
    if inside.sum() < 50:
        return 0.0
    colour = (255 - img.astype(np.int16)).max(-1) > 30
    return float(colour[inside].mean())


def assess(img, spec, res):
    """-> (status, reasons, checks)"""
    ct = canonical_type(spec.get("chart_type"))
    reasons = []
    if ct not in SUPPORTED:
        return "rejected", [f"unsupported chart type ({spec.get('vlm_chart_type') or spec.get('chart_type')})"], []
    if res.get("reject"):
        return "rejected", [res["reject"]], res.get("checks", [])
    if ct in ("scatter", "line", "scatter_line", "bar", "stacked_bar", "histogram", "box", "contour"):
        n, _ = count_panels(img)
        if n >= 2:
            return "rejected", [f"{n} plot panels (axes frames) in one image - split the figure into single plots first"], []
    checks = list(res.get("checks") or [])
    if ct in ("scatter", "line", "scatter_line", "pie") or (ct == "bar" and res.get("chart_type") == "bar" and "stacked" not in res):
        checks += _legacy_checks(res)
    if res.get("chart_type") == "pie":
        f = pie_fill(img, res)
        if f is not None:
            checks.append(("pie ellipse filled with slice colours", round(f, 2), f">= {GATES['pie_fill']}", f >= GATES["pie_fill"]))
        else:
            checks.append(("pie centre found", "no", "yes", False))
    failed = [c for c in checks if not c[3]]
    if failed:
        return "rejected", [f"self-check failed: {c[0]} = {c[1]} (needs {c[2]})" for c in failed], checks
    if res.get("flag"):
        return "flagged", [res["flag"]], checks
    return "ok", reasons, checks
