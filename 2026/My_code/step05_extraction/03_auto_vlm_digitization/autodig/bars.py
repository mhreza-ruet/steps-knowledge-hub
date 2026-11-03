"""Bar charts: VLM spec (categories, series names / colours / hatching) + pixels -> bar heights.

No per-figure settings. Steps:
  1. frame + y-axis calibration from the VLM's tick labels (same code as line plots;
     broken and right axes included)
  2. legend patches inside the plot (if any) give the exact fill colour of each
     series; otherwise the colour name is snapped to the nearest colour present
  3. per series: fill-colour mask -> runs of columns = bars; bar top = highest
     filled row (robust to hatching), minus the outline
  4. bars of all series are grouped into the VLM's number of categories by the
     largest horizontal gaps (left to right = category order)
  5. error bar = dark run in the bar's centre column that crosses the bar top
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import detect as D
from .calibrate import map_for_pixel
from .extract import (calibrate_all, colour_target, hsv_arrays, hue_dist, plot_region, exclusion_mask,
                      expand_legend_boxes)


# ------------------------------------------------------------------ legend patches
def find_patch_legend(img, frame, lines, hue, sat, val):
    """Legend entries of a bar chart: a text line with a filled colour patch
    just to its left (patches can be pale or hatched, so the median colour of
    the non-white, non-text pixels is used)."""
    H, W = img.shape[:2]
    g = img.mean(-1)
    ents = []
    for l in lines:
        h = l["y1"] - l["y0"]
        y0, y1 = max(0, l["y0"] - 2), min(H, l["y1"] + 2)
        x1 = l["x0"] - 2
        x0 = max(0, x1 - int(max(40, 3.5 * h)))
        if x1 - x0 < 6:
            continue
        sub = img[y0:y1, x0:x1].astype(float)
        dw = np.sqrt(((255.0 - sub) ** 2).sum(-1))
        gsub = sub.mean(-1)
        ssub = sat[y0:y1, x0:x1]
        # patch pixels: clearly not white, and not black text / outline
        ok = (dw > 40) & ~((gsub < 60) & (ssub < 0.15))
        if ok.sum() < 30:
            continue
        # the patch is solid: its colour is the most common one (hatching, tinted
        # see-through legend backgrounds and anti-aliasing are minorities)
        q = (sub[ok] // 8).astype(int)
        keys, cnt = np.unique(q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2], return_counts=True)
        best = keys[np.argmax(cnt)]
        selq = (q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]) == best
        rgb = np.median(sub[ok][selq], 0)
        near = np.zeros(ok.shape, bool)
        near[ok] = np.abs(sub[ok] - rgb).max(1) <= 24
        if near.sum() < 25:
            continue
        lab, n = ndi.label(near)
        sizes = ndi.sum(near, lab, range(1, n + 1))
        mk = lab == (int(np.argmax(sizes)) + 1)
        ys, xs = np.where(mk)
        ents.append({"text": l, "rgb": rgb.tolist(),
                     "sample": (int(xs.min()) + x0, int(ys.min()) + y0, int(xs.max()) + 1 + x0, int(ys.max()) + 1 + y0)})
    # a legend = >= 2 entries on one row or in one column
    if len(ents) < 2:
        return [], []
    rows = []                      # entries on one row: same text top (descenders change the bottom)
    for e in sorted(ents, key=lambda e: e["text"]["y0"]):
        if rows and abs(e["text"]["y0"] - rows[-1][-1]["text"]["y0"]) <= 4:
            rows[-1].append(e)
        else:
            rows.append([e])
    groups = [g_ for g_ in rows if len(g_) >= 2]
    if not groups:   # vertical legend
        groups = [ents]
    out, boxes = [], []
    for g_ in groups:
        g_.sort(key=lambda e: (e["text"]["y0"] // 8, e["sample"][0]))
        out += g_
        boxes.append((min(e["sample"][0] for e in g_), min(min(e["sample"][1], e["text"]["y0"]) for e in g_),
                      max(e["text"]["x1"] for e in g_), max(max(e["sample"][3], e["text"]["y1"]) for e in g_)))
    return out, boxes


def legend_patches(img, frame, hue, sat, val):
    """legend patches inside the plot frame or just outside it (above / right of the frame).
    Returns (entries, legend boxes, text lines inside the frame)."""
    H, W = img.shape[:2]
    lines_in = D.text_lines_inside(img, frame)
    outs = []
    T, R = int(frame["top"]), int(frame["right"])
    if T > 12:
        outs += D.text_lines_inside(img, {"left": 0, "right": W - 1, "top": 0, "bottom": T - 2}, inset=0)
    if W - R > 40:
        outs += D.text_lines_inside(img, {"left": R + 2, "right": W - 1, "top": 0, "bottom": H - 1}, inset=0)
    legend, boxes = find_patch_legend(img, frame, lines_in, hue, sat, val)
    if not legend and outs:
        legend, boxes = find_patch_legend(img, frame, outs, hue, sat, val)
    return legend, boxes, lines_in


def _rgb_hue(rgb):
    import colorsys
    r, g, b = [v / 255 for v in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return 360 * h, s, v


def match_patches(series, legend):
    """Pair VLM series with legend patches by colour (Hungarian); order breaks ties."""
    if not legend:
        return {}
    from scipy.optimize import linear_sum_assignment
    C = np.full((len(series), len(legend)), 1e6)
    for i, s in enumerate(series):
        t = colour_target(s["color"])
        for j, e in enumerate(legend):
            h, sa, v = _rgb_hue(e["rgb"])
            order = 2.0 * abs(i - j)
            if t["kind"] == "dark":
                C[i, j] = (0 if sa < 0.15 and v < 0.6 else 120) + order
            elif sa >= 0.05:
                d = float(hue_dist(t["hue"], h))
                C[i, j] = (d if d <= 60 else 500) + order
    r, c = linear_sum_assignment(C)
    return {int(i): int(j) for i, j in zip(r, c) if C[i, j] < 1e5}


def discover_series(img, area, sat, legend, min_share=0.03):
    """Distinct fill colours in the plot area (not white / grey) -> unnamed series."""
    cols = [e["rgb"] for e in legend] if legend else discover_colours(img, area, sat, min_share)
    from .extract import HUES
    out = []
    for i, c in enumerate(cols):
        h, sa, v = _rgb_hue(c)
        name = min(HUES, key=lambda n: float(hue_dist(HUES[n], h))) if sa > 0.15 else "gray"
        out.append({"name": f"series {i + 1}", "kind": "bar", "color": name, "hatched": False, "error_bars": True,
                    "marker": "none", "marker_fill": "filled", "line": "none", "line_color": None, "y_axis": "left",
                    "position": None, "n_points": None})
    return out


def discover_colours(img, area, sat, min_share=0.03):
    """Distinct fill colours (RGB) in the plot area, most common first; a colour
    gradient inside a bar counts as one colour."""
    if True:
        m = area & (sat >= 0.08) & (img.mean(-1) < 245)
        q = (img[m] // 16).astype(int)
        keys, cnt = np.unique(q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2], return_counts=True)
        cols = []
        for kk in keys[np.argsort(cnt)[::-1]]:
            if cnt[keys == kk][0] < min_share * m.sum():
                break
            rgb = np.array([kk // 256, (kk // 16) % 16, kk % 16]) * 16 + 8
            h, sa, _ = _rgb_hue(rgb)
            same = lambda c: (np.linalg.norm(rgb - np.array(c)) <= 60 or
                              (sa >= 0.3 and _rgb_hue(c)[1] >= 0.3 and float(hue_dist(h, _rgb_hue(c)[0])) <= 15))
            if not any(same(c) for c in cols):   # a colour gradient inside a bar is one series
                cols.append(rgb.tolist())
    return cols


# ------------------------------------------------------------------ masks
def fill_mask(img, ref, hue, sat, val):
    """Pixels of one bar fill. Saturated fills (possibly with a colour gradient)
    are matched by hue; pale fills by RGB distance (with a small saturation
    floor that keeps grey grid lines and anti-aliased text out)."""
    h, s, v = _rgb_hue(ref)
    if s >= 0.35:
        return (hue_dist(hue, h) <= 15) & (sat >= 0.2) & (val >= 0.3)
    if s < 0.05:                   # grey fill
        d = np.sqrt(((img.astype(np.int32) - np.asarray(ref, np.int32)) ** 2).sum(-1))
        return (d <= 30) & (sat < 0.1)
    d = np.sqrt(((img.astype(np.int32) - np.asarray(ref, np.int32)) ** 2).sum(-1))
    return (d <= 38) & (sat >= 0.05)


def colour_from_name(name, img, hue, sat, val, region, taken=()):
    """Reference fill colour for a series without a legend patch: the most
    common colour in the plot whose hue is near the colour name."""
    t = colour_target(name)
    n_ = (name or "").lower()
    if t["kind"] == "dark" and ("gray" in n_ or "grey" in n_) and not any(w in n_ for w in ("dark", "black", "charcoal")):
        g_ = img.mean(-1)
        m = region & (g_ >= 80) & (g_ <= 215) & (sat < 0.12)
    elif t["kind"] == "dark":
        m = region & (img.mean(-1) < 110) & (sat < 0.2)
    else:
        m = region & (hue_dist(hue, t["hue"]) <= 30) & (sat >= 0.06) & (val >= 0.3)
        for th in taken:
            m &= hue_dist(hue, th) > 12
    if m.sum() < 30:
        return None
    q = (img[m] // 8).astype(int)
    keys, cnt = np.unique(q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2], return_counts=True)
    best = keys[np.argmax(cnt)]
    sel = (q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]) == best
    return np.median(img[m][sel], 0).tolist()


# ------------------------------------------------------------------ bars
def find_bars(mask, min_w=3, min_h=4):
    """Runs of columns holding fill pixels -> [(x0, x1, top_row)] (x1 exclusive)."""
    colcnt = mask.sum(0)
    on = colcnt >= min_h
    runs = D._runs(on)
    merged = []
    for s_, e in runs:   # hatching can cut a bar into slivers: merge runs 1-2 px apart
        if merged and s_ - merged[-1][1] <= 2:
            merged[-1][1] = e
        else:
            merged.append([s_, e])
    out = []
    for s_, e in merged:
        if e - s_ < min_w:
            continue
        a, b = s_ + int(0.2 * (e - s_)), e - int(0.2 * (e - s_))
        tops = [np.where(mask[:, x])[0].min() for x in range(a, max(b, a + 1)) if mask[:, x].any()]
        out.append((s_, e, float(np.percentile(tops, 20))))
    return out


def group_bars(centres, n_groups):
    """Split sorted bar centres into n_groups at the largest gaps."""
    c = np.sort(np.asarray(centres, float))
    if len(c) == 0 or n_groups <= 1:
        return [list(c)]
    gaps = np.diff(c)
    k = min(n_groups - 1, len(gaps))
    cut = np.sort(np.argsort(gaps)[::-1][:k])
    groups, start = [], 0
    for i in cut:
        groups.append(list(c[start:i + 1])); start = i + 1
    groups.append(list(c[start:]))
    return groups


def error_bar(gray, cx, top, T, dark=100):
    """Upper half-length (px) of the error bar crossing the bar top at column cx."""
    H, W = gray.shape
    c0, c1 = max(int(round(cx)) - 1, 0), min(int(round(cx)) + 2, W)
    col = (gray[:, c0:c1] < dark).any(1)
    r0 = int(round(top))
    win = col[max(r0 - 3, 0):r0 + 4]
    if not win.any():
        return 0.0
    y = max(r0 - 3, 0) + int(np.argmax(win))
    while y > T and col[y - 1]:
        y -= 1
    return max(0.0, top - y - 1.0)   # minus half the cap line


def pipeline_bars(img, spec, frame=None):
    H, W = img.shape[:2]
    frame = frame or D.detect_frame(img)
    cal, cal_info = calibrate_all(img, frame, spec)
    hue, sat, val = hsv_arrays(img)
    region = plot_region(frame, (H, W))
    lines = D.text_lines_inside(img, frame)
    legend, boxes = find_patch_legend(img, frame, lines, hue, sat, val)
    boxes = expand_legend_boxes(img, boxes, lines, [{"text": e["text"]} for e in legend])
    excl = exclusion_mask((H, W), lines, boxes)
    area = region & ~excl
    gray = img.mean(-1)
    series = spec["series"]
    cats = spec.get("categories") or []
    out = {"chart_type": "bar", "size": (W, H), "frame": frame, "spec": {**spec, "series": [dict(q) for q in series]},
           "calibration": {k: [m.as_dict() for m in v] for k, v in cal.items()},
           "calibration_info": {k: {kk: vv for kk, vv in v.items() if kk not in ("ticks", "labels")} for k, v in cal_info.items()},
           "legend": [{"rgb": e["rgb"], "text_box": (e["text"]["x0"], e["text"]["y0"], e["text"]["x1"], e["text"]["y1"])} for e in legend],
           "legend_boxes": boxes, "text_boxes": [(l["x0"], l["y0"], l["x1"], l["y1"]) for l in lines],
           "categories": [c["name"] for c in cats], "series": {}, "styles": {}, "warnings": []}
    if not cats:
        out["warnings"].append("VLM gave no categories - nothing extracted"); return out
    if "y_axis" not in cal:
        out["warnings"].append("y-axis calibration failed - no data extracted"); return out

    if not series:   # the VLM named no series: one series per distinct fill colour in the plot
        series = discover_series(img, area, sat, legend)
        out["spec"]["series"] = series
        out["warnings"].append(f"VLM gave no series - {len(series)} fill colour(s) found in the pixels")
    # ---- reference colour of every series
    match = match_patches(series, legend)
    refs, taken = [], []
    for i, s in enumerate(series):
        if i in match:
            ref, src = legend[match[i]]["rgb"], "legend patch"
        else:
            ref, src = colour_from_name(s["color"], img, hue, sat, val, area, taken), "colour name"
        refs.append((ref, src))
        if ref is not None:
            taken.append(_rgb_hue(ref)[0])
    if any(r is None for r, _ in refs):
        # a colour name that matches nothing in the image (e.g. 'light coral' for a red
        # gradient): give that series the fill colour no other series uses
        free = [c for c in discover_colours(img, area, sat)
                if all(np.linalg.norm(np.array(c) - np.array(r)) > 60 for r, _ in refs if r is not None)]
        for k, (r, src) in enumerate(refs):
            if r is None and free:
                refs[k] = (free.pop(0), "unclaimed fill colour (name not found)")

    # ---- bars of every series
    found, extents = {}, []
    for (ref, src), s in zip(refs, series):
        if ref is None:
            out["warnings"].append(f"colour '{s['color']}' of series '{s['name']}' not found"); continue
        m = fill_mask(img, ref, hue, sat, val) & area
        bars = find_bars(m)
        found[s["name"]] = [(0.5 * (a + b), b - a, top) for a, b, top in bars]
        extents += [(a, b) for a, b, _ in bars]
        out["styles"][s["name"]] = {"rgb": list(ref), "from": src, "hatched": s.get("hatched", False)}

    # ---- categories: all bar centres split at the largest gaps
    allc = sorted(c for v in found.values() for c, _, _ in v)
    if not allc:
        out["warnings"].append("no bars found"); return out
    # pixel check of the category count: bars of one group touch (or nearly), groups are
    # separated by a clear gap -> the number of groups the pixels show
    ext = sorted(extents)
    medw = float(np.median([b - a for a, b in ext]))
    natural = 1 + sum(1 for (a0, b0), (a1, b1) in zip(ext, ext[1:]) if a1 - b0 > 0.6 * medw)
    if natural != len(cats):
        out["warnings"].append(f"VLM listed {len(cats)} categories, the pixels show {natural} groups of bars - "
                               f"pixels used")
        cats = (cats + [{"name": f"category {i + 1}", "y_axis": "left"} for i in range(len(cats), natural)])[:natural]
        out["categories"] = [c["name"] for c in cats]
    groups = group_bars(allc, len(cats))
    centres = [float(np.mean(g)) for g in groups]
    bounds = [(-np.inf if i == 0 else 0.5 * (groups[i - 1][-1] + groups[i][0]),
               np.inf if i == len(groups) - 1 else 0.5 * (groups[i][-1] + groups[i + 1][0])) for i in range(len(groups))]
    out["category_centres_px"] = centres

    for s in series:
        bars = found.get(s["name"], [])
        xs, ys, es, pxs, pys, offs, wids, outl = [], [], [], [], [], [], [], []
        for ci, (lo, hi) in enumerate(bounds):
            inside = [b for b in bars if lo <= b[0] < hi]
            if not inside:
                continue
            if len(inside) > 1:   # keep the widest (a sliver is debris)
                inside = [max(inside, key=lambda b: b[1])]
            cx, w, top = inside[0]
            # outline: a dark row right above the fill
            a, b = int(cx - 0.3 * w), int(cx + 0.3 * w) + 1
            if top >= 1 and gray[int(top) - 1, a:b].mean() < 120:
                top -= 1.0
                outl.append(1)
            else:
                outl.append(0)
            cat_ax = cats[ci]["y_axis"] if ci < len(cats) else "left"
            key = "y_axis_right" if cat_ax == "right" and "y_axis_right" in cal else "y_axis"
            maps = cal[key]
            mp = map_for_pixel(maps, top)
            v = float(mp.to_value(top))
            e_px = error_bar(gray, cx, top, frame["top"]) if s.get("error_bars") else 0.0
            e = abs(float(mp.to_value(top - e_px)) - v) if e_px > 1.2 else 0.0
            xs.append(ci); ys.append(round(v, 6)); es.append(round(e, 6)); pxs.append(round(cx, 1)); pys.append(round(top, 1))
            offs.append(cx - centres[ci]); wids.append(w)
        rec = {"x": xs, "y": ys, "yerr": es, "kind": "bar", "y_axis": "y_axis", "px": pxs, "py": pys,
               "categories": [cats[i]["name"] for i in xs]}
        out["series"][s["name"]] = rec
        if s["name"] in out["styles"]:
            out["styles"][s["name"]].update(outline=bool(outl) and float(np.mean(outl)) >= 0.5,
                                            offset_px=float(np.median(offs)) if offs else 0.0,
                                            width_px=float(np.median(wids)) if wids else 10.0)
        if len(xs) < len(cats):
            miss = [cats[i]["name"] for i in range(len(cats)) if i not in xs]
            out["warnings"].append(f"'{s['name']}': no bar found for {miss}")
    out["category_axis"] = [c["y_axis"] for c in cats]
    return out
