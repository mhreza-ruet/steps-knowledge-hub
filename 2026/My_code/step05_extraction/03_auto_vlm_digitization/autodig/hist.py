"""Histograms: bins on a numeric x-axis, one height per bin.

The VLM reads the axes' tick labels and every series' name, colour and style
('filled' bars or a 'step' outline). The pixels give:

  1. frame + x / y calibration (same code as line plots)
  2. per series a colour mask: legend sample colour (or the colour name)
     - filled: the fill colour, plus the colours made where a semi-transparent
       series is drawn over another (an overlap colour c = top + k (under - white)
       with one blending factor k is recognised and counted for both series)
     - step: the outline colour
  3. per pixel column: the highest pixel of the series -> a height profile
  4. the profile is cut into bins where it jumps or where a gap / edge line
     separates bars; long flat stretches are split by the common bin width
     (two neighbouring bins of equal height look like one wide bar)
  5. bin edges and heights -> data units
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import detect as D
from .extract import (calibrate_all, colour_target, hsv_arrays, hue_dist, plot_region, exclusion_mask,
                      expand_legend_boxes, find_legend, refine_hue)


def _quant_colours(img, area, min_share=0.004):
    q = (img[area] // 6).astype(np.int32)
    keys, cnt = np.unique(q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2], return_counts=True)
    keep = cnt >= max(30, min_share * area.sum())
    out = []
    for k, c in zip(keys[keep], cnt[keep]):
        rgb = np.array([k // 10000, (k // 100) % 100, k % 100]) * 6 + 3
        out.append((rgb.astype(float), int(c)))
    return out


def _blend_of(c, a, b):
    """is colour c = b drawn semi-transparently over a?  c = b + k (a - 255)"""
    d = a - 255.0
    if np.abs(d).max() < 8:
        return None
    k = float(np.dot(c - b, d) / np.dot(d, d))
    if not 0.15 <= k <= 0.95:
        return None
    res = np.abs(c - (b + k * d)).max()
    return k if res <= 14 else None


def series_colours(img, spec, frame, hue, sat, val, area, legend):
    """reference colour (RGB) of every series: legend sample, else the most
    common plot colour near the colour name"""
    refs = []
    from .bars import match_patches, colour_from_name
    match = match_patches(spec["series"], legend) if legend else {}
    taken = []
    for i, s in enumerate(spec["series"]):
        if i in match:
            refs.append((np.array(legend[match[i]]["rgb"], float), "legend sample"))
        else:
            c = colour_from_name(s.get("color", ""), img, hue, sat, val, area, taken)
            refs.append((np.array(c, float) if c is not None else None, "colour name"))
            if c is not None:
                from .bars import _rgb_hue
                taken.append(_rgb_hue(c)[0])
    return refs


def _segments(top, gap_px, jump_px=1.5):
    """cut a height profile (np.nan = no bar) into flat pieces"""
    segs, start = [], None
    for x in range(len(top) + 1):
        v = top[x] if x < len(top) else np.nan
        if start is None:
            if not np.isnan(v):
                start = x
            continue
        if np.isnan(v) or abs(v - np.nanmedian(top[start:x])) > jump_px:
            segs.append((start, x))
            start = x if not np.isnan(v) else None
    return [(a, b) for a, b in segs if b - a >= 2]


def _edge_colour(img, top, segs, fill, base_px):
    """the line drawn between neighbouring bars (black, white, ...) or None"""
    cols = []
    for (a, b), (c, d) in zip(segs, segs[1:]):
        if abs(c - b) > 4:
            continue
        x = int(round(0.5 * (b + c)))
        t = np.nanmax([top[a:b].mean() if b > a else np.nan, top[c:d].mean() if d > c else np.nan])
        r0, r1 = int(t) + 3, int(base_px) - 3
        if r1 - r0 < 4:
            continue
        win = img[r0:r1, max(0, x - 2):x + 3].astype(float)        # rows x 5 columns
        dev = np.abs(win - fill).max(-1).mean(0)
        k = int(np.argmax(dev))
        if dev[k] > 40:
            cols.append(np.median(win[:, k], 0))
    if len(cols) < 2:
        return None
    return np.median(cols, 0).round(1).tolist()


def _outline_width(img, top, ref):
    """thickness of a step outline (pixels of the series colour below the plateau top)"""
    d = np.sqrt(((img.astype(float) - ref) ** 2).sum(-1)) <= 60
    th = []
    for x in np.where(~np.isnan(top))[0][::3]:
        t = int(top[x]); n = 0
        while t + n < img.shape[0] and d[t + n, x]:
            n += 1
        if 0 < n <= 8:
            th.append(n)
    return float(np.median(th)) if th else None


def _overlay_curves(img, area, refs, blends, X, Y, frame):
    """thin curves drawn over the bars (fitted density, cumulative line): a saturated or dark
    colour that is neither a series colour nor an overlap colour, forming a thin line"""
    from .detect import _runs
    pix = img[area].astype(float)
    known = [np.asarray(r, float) for r in refs] + [np.asarray(b["rgb"], float) for b in blends]
    sat = pix.max(-1) - pix.min(-1)
    cand = (sat > 60) | (pix.mean(-1) < 90)
    for k in known:
        cand &= np.abs(pix - k).max(-1) > 45
    if cand.sum() < 0.25 * (frame["right"] - frame["left"]):
        return []
    q = (pix[cand] // 24).astype(int)
    keys, cnt = np.unique(q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2], return_counts=True)
    out = []
    for key in keys[np.argsort(-cnt)][:3]:
        c = np.array([key // 10000, (key // 100) % 100, key % 100]) * 24 + 12.0
        m = area & (np.abs(img.astype(float) - c).max(-1) <= 40)
        for kc in known:
            m &= np.abs(img.astype(float) - kc).max(-1) > 45
        if any(np.abs(c - o["rgb_q"]).max() < 30 for o in out):
            continue
        colsum = m.sum(0)
        xs = np.where((colsum > 0) & (colsum <= 8))[0]          # thin: few pixels per column
        if len(xs) < 0.3 * (frame["right"] - frame["left"]):
            continue
        ys = np.array([np.where(m[:, x])[0].mean() for x in xs])
        runs = _runs(np.isin(np.arange(img.shape[1]), xs))
        if max((b - a for a, b in runs), default=0) < 0.2 * (frame["right"] - frame["left"]):
            continue
        rgb = np.median(img[m].astype(float), 0)
        out.append({"rgb": rgb.round(1).tolist(), "rgb_q": c, "x": X.to_value(xs.astype(float)).round(6).tolist(),
                    "y": Y.to_value(ys).round(6).tolist(), "width_px": float(np.median(colsum[xs]))})
    for o in out:
        o.pop("rgb_q")
    return out


def _legend_box(boxes, frame):
    """where the legend sits (for the redraw): a matplotlib 'loc' word"""
    if not boxes:
        return None
    b = max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
    cx, cy = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
    fx = (cx - frame["left"]) / max(1, frame["right"] - frame["left"])
    fy = (cy - frame["top"]) / max(1, frame["bottom"] - frame["top"])
    v = "upper" if fy < 0.4 else ("lower" if fy > 0.6 else "center")
    h = "left" if fx < 0.4 else ("right" if fx > 0.6 else "center")
    return {"loc": "center" if v == h == "center" else (f"{v} {h}" if not (v == "center" and h == "center") else "center"), "box": list(map(float, b))}


def pipeline_hist(img, spec):
    H, W = img.shape[:2]
    frame = D.detect_frame(img)
    cal, cal_info = calibrate_all(img, frame, spec)
    out = {"chart_type": "histogram", "size": (W, H), "frame": frame, "spec": spec, "warnings": [], "checks": [], "series": {},
           "calibration": {k: [m.as_dict() for m in v] for k, v in cal.items()},
           "calibration_info": {k: {kk: vv for kk, vv in v.items() if kk not in ("ticks", "labels")} for k, v in cal_info.items()}}
    for k in ("x_axis", "y_axis"):
        if k not in cal or cal_info.get(k, {}).get("relative"):
            out["reject"] = f"histogram: the {k.replace('_', ' ')} has no numeric calibration"; return out
    X, Y = cal["x_axis"][0], cal["y_axis"][0]
    hue, sat, val = hsv_arrays(img)
    region = plot_region(frame, (H, W))
    lines = D.text_lines_inside(img, frame)
    from .bars import find_patch_legend
    legend, boxes = find_patch_legend(img, frame, lines, hue, sat, val)
    if not legend:                                    # step histograms: line samples in the legend
        lg, bx = find_legend(img, frame, lines, hue, sat, val)
        legend = [{"rgb": e["rgb"], "text": e["text"]} for e in lg]
        boxes = bx
    boxes = expand_legend_boxes(img, boxes, lines, [{"text": e["text"]} for e in legend])
    area = region & ~exclusion_mask((H, W), lines, boxes)
    series = spec.get("series") or []
    if not series:
        out["reject"] = "histogram: the VLM listed no series"; return out
    refs = series_colours(img, spec, frame, hue, sat, val, area, legend)
    cols = _quant_colours(img, area)
    base_px = float(Y.to_pixel(0.0)) if Y.to_value(frame["bottom"]) <= 0 <= Y.to_value(frame["top"]) else float(frame["bottom"])
    tops = {}
    blends = []
    for i, ((ref, src), s) in enumerate(zip(refs, series)):
        if ref is None:
            out["warnings"].append(f"colour '{s.get('color')}' of '{s['name']}' not found"); continue
        style = str(s.get("style", "filled")).lower()
        d = np.sqrt(((img.astype(float) - ref) ** 2).sum(-1))
        m = (d <= 30) & area
        if style != "step":
            # overlap colours: this series drawn over another one, or another one over it
            for c, n in cols:
                if np.abs(c - ref).max() <= 20:
                    continue
                for j, (r2, _) in enumerate(refs):
                    if j == i or r2 is None:
                        continue
                    k_top = _blend_of(c, r2, ref)          # this series drawn over series j
                    k_und = _blend_of(c, ref, r2)          # series j drawn over this one
                    if k_top is not None or k_und is not None:
                        m |= (np.abs(img.astype(float) - c).max(-1) <= 9) & area
                        blends.append({"top": i if k_top is not None else j, "under": j if k_top is not None else i,
                                       "k": k_top if k_top is not None else k_und, "rgb": c.tolist(), "pixels": n})
        m = ndi.binary_opening(m, np.ones((2, 2))) if style != "step" else m
        lab, n = ndi.label(m)
        if n == 0:
            out["warnings"].append(f"no pixels of '{s['name']}' found"); continue
        top = np.full(W, np.nan)
        cols_any = m.any(0)
        first = np.argmax(m, 0)
        top[cols_any] = first[cols_any]
        if style != "step":
            # a filled bar reaches down to the baseline: ignore columns that only hold specks
            depth = m.sum(0)
            top[depth < 0.5 * np.maximum(0, base_px - np.nan_to_num(top, nan=base_px))] = np.nan
        tops[s["name"]] = (top, style, ref, src)

    # common bin width: the most frequent flat-piece width over all series
    widths = []
    segs_all = {}
    for name, (top, style, ref, src) in tops.items():
        runs = [(a, b) for a, b in D._runs(~np.isnan(top)) if b - a >= 2]
        rw = np.array([b - a for a, b in runs], float)
        if len(runs) >= 3 and np.std(rw) <= 0.25 * np.median(rw):
            # every bar is separated from the next by an edge line: one run = one bin
            # (a line drawn across the bars, e.g. a fitted density curve, does not cut bins)
            segs = runs
        else:
            segs = _segments(top, 2)
        segs_all[name] = segs
        widths += [b - a for a, b in segs]
    if not widths:
        out["reject"] = "histogram: no bars found"; return out
    w = np.array(widths, float)
    w0 = float(np.median(w[w <= np.percentile(w, 60)])) if len(w) > 3 else float(np.median(w))
    # the bars fill the column between edges: an edge line (dark / white) eats ~1 px; snap to the axis grid
    for name, (top, style, ref, src) in tops.items():
        edges, heights, pxs = [], [], []
        for a, b in segs_all[name]:
            k = max(1, int(round((b - a) / w0)))
            for t in range(k):
                aa, bb = a + (b - a) * t / k, a + (b - a) * (t + 1) / k
                seg_top = float(np.nanmedian(top[int(round(aa)):max(int(round(aa)) + 1, int(round(bb)))]))
                # an outline drawn above the fill
                c0 = int(round((aa + bb) / 2))
                tt = int(round(seg_top))
                if style != "step" and tt >= 2 and img[tt - 1, c0].mean() < 110:
                    seg_top -= 1.0
                edges.append((aa, bb)); heights.append(seg_top); pxs.append((aa, bb, seg_top))
        # widen each bar by half the gap to its neighbours (edge lines between bars)
        xe = []
        for k, (aa, bb) in enumerate(edges):
            lo = aa - 0.5 if k == 0 or abs(edges[k - 1][1] - aa) > 3 else 0.5 * (edges[k - 1][1] + aa)
            hi = bb + 0.5 if k == len(edges) - 1 or abs(edges[k + 1][0] - bb) > 3 else 0.5 * (bb + edges[k + 1][0])
            if k > 0 and abs(edges[k - 1][1] - aa) <= 3:
                lo = 0.5 * (edges[k - 1][1] + aa)
            xe.append((lo, hi))
        left = X.to_value(np.array([e[0] for e in xe])); right = X.to_value(np.array([e[1] for e in xe]))
        hv = Y.to_value(np.array(heights)) - (Y.to_value(base_px) if base_px != frame["bottom"] else 0.0)
        hv = np.maximum(hv, 0)
        out["series"][name] = {"left": left.round(6).tolist(), "right": right.round(6).tolist(), "height": hv.round(6).tolist(),
                               "kind": "histogram", "style": style, "rgb": ref.tolist(), "from": src}
    out["bin_width_px"] = w0
    # drawing style, measured: transparency (from overlap colours), bar edge colour, outline width
    names = [s["name"] for s in series]
    if blends:
        kk = float(np.median([b["k"] for b in blends]))
        alpha = float(np.clip(1 - kk, 0.1, 1.0))
        order = sorted({b["under"] for b in blends}) + sorted({b["top"] for b in blends} - {b["under"] for b in blends})
        out["transparency"] = {"alpha": round(alpha, 3), "draw_order": [names[i] for i in order if i < len(names)],
                               "overlap_rgb": blends[0]["rgb"]}
    for name, (top, style, ref, src) in tops.items():
        if name in out["series"]:
            e = _edge_colour(img, top, segs_all[name], ref, base_px) if style != "step" else None
            known = [r for r, _ in refs if r is not None] + [np.asarray(b["rgb"]) for b in blends]
            if e is not None and any(np.abs(np.asarray(e) - k).max() <= 35 for k in known):
                e = None                                   # that is another series / an overlap, not an edge line
            out["series"][name]["edge_rgb"] = e
            out["series"][name]["line_px"] = _outline_width(img, top, ref) if style == "step" else None
    out["overlay_lines"] = _overlay_curves(img, area, [r for r, _ in refs if r is not None], blends, X, Y, frame)
    out["legend_box"] = _legend_box(boxes, frame)
    out["table"] = [{"series": n, "bin_left": l, "bin_right": r, "height": h}
                    for n, d in out["series"].items() for l, r, h in zip(d["left"], d["right"], d["height"])]
    rms = max(cal_info["x_axis"].get("rms_px", 0), cal_info["y_axis"].get("rms_px", 0))
    nb = sum(len(d["height"]) for d in out["series"].values())
    out["checks"] += [("axis label fit (px)", round(rms, 2), "<= 3", rms <= 3),
                      ("bins found", nb, ">= 2", nb >= 2),
                      ("series found", f"{len(out['series'])}/{len(series)}", "all", len(out["series"]) == len(series))]
    return out
