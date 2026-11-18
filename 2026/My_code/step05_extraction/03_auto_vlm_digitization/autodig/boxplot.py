"""Box plots (vertical or horizontal; plain, filled, notched, grouped; with fliers).

The VLM reads the value-axis tick labels, the category names, the orientation
and the series (legend) names / colours. The pixels give, per box glyph:

  box      the longest run of rows whose drawn width is >= 40 % of the glyph's
           widest row, with straight side edges (a violin has curved sides and
           is rejected here)          -> Q1, Q3
  whiskers the glyph's first and last row (cap or whisker end) -> lower / upper whisker
  median   the row inside the box that differs from the box's own fill across
           (nearly) its whole width   -> median
  fliers   small separate marks above / below the glyph -> outlier values

Horizontal boxes are handled by transposing the image (rows <-> columns), so the
same code reads both orientations.
"""
from __future__ import annotations

import re

import numpy as np
from scipy import ndimage as ndi

from . import detect as D
from .extract import calibrate_all, hsv_arrays, plot_region, exclusion_mask, expand_legend_boxes


def _ink(img, area):
    g = img.mean(-1)
    sat = (img.max(-1).astype(float) - img.min(-1)) / np.maximum(img.max(-1), 1)
    return area & ((g < 200) | ((sat > 0.12) & (g < 250)))


def _drop_gridlines(m, img):
    """grid lines: rows / columns inked across (nearly) the whole plot, or across
    most of it in a light colour (a long dark whisker is not a grid line)"""
    m = m.copy()
    g = img.mean(-1)
    for ax in (1, 0):
        frac = m.mean(ax)
        light = np.array([np.median(g[k][m[k]]) if ax == 1 and m[k].any() else
                          (np.median(g[:, k][m[:, k]]) if ax == 0 and m[:, k].any() else 255)
                          for k in range(m.shape[1 - ax])])
        kill = (frac > 0.9) | ((frac > 0.6) & (light > 150))
        if ax == 1:
            m[kill] = False
        else:
            m[:, kill] = False
    return m


def analyse_glyph(gm, img_t, vmap):
    """gm: bool mask of one glyph (in 'vertical' orientation, full image size).
    Returns stats dict (pixels and values) or None with a reason."""
    ys, xs = np.where(gm)
    y0, y1 = ys.min(), ys.max()
    rows = np.arange(y0, y1 + 1)
    left = np.array([xs[ys == y].min() if (ys == y).any() else -1 for y in rows])
    right = np.array([xs[ys == y].max() if (ys == y).any() else -1 for y in rows])
    w = np.where(left >= 0, right - left + 1, 0)
    maxw = w.max()
    wide = w >= 0.4 * maxw
    runs = D._runs(wide)
    if not runs:
        return None, "no box"
    a, b = max(runs, key=lambda r: r[1] - r[0])
    if b - a < 3:
        return None, "box too short"
    L = left[a:b]
    mode = np.bincount(L - L.min()).argmax() + L.min()
    straight = float(np.mean(np.abs(L - mode) <= 1))
    if straight < 0.35:
        return None, f"box sides are not straight ({straight:.0%} of rows on one edge) - not a box plot glyph"
    top_r, bot_r = rows[a], rows[b - 1]
    # median: the row inside the box that differs from the box's fill across its width
    inner = []
    for k in range(a + 2, b - 2):
        l, r = left[k] + 2, right[k] - 2
        if r - l >= 3:
            inner.append((rows[k], img_t[rows[k], l:r + 1].astype(float)))
    med_r = None
    med_rows = []
    if inner:
        F = np.median(np.concatenate([p for _, p in inner]), 0)
        dev = np.array([np.mean(np.abs(p - F).max(1) > 40) for _, p in inner])
        if dev.max() >= 0.6:
            best = np.where(dev >= 0.6)[0]
            # the longest run of deviating rows is the median line
            grp = np.split(best, np.where(np.diff(best) > 1)[0] + 1)
            g = max(grp, key=len)
            med_r = float(np.mean([inner[i][0] for i in g]))
            med_rows = [inner[i][0] for i in g]
        fill = F
    else:
        fill = None
    st = {"px": {"top": float(top_r), "q_hi": float(top_r), "q_lo": float(bot_r), "bottom": float(y1), "whisk_top": float(y0),
                 "median": med_r, "x_centre": float(0.5 * (L.min() + right[a:b].max())), "width": float(maxw)},
          "fill_rgb": None if fill is None else fill.round(1).tolist(), "straight_edges": straight}
    st["style"] = _glyph_style(gm, img_t, rows, left, right, a, b, mode, fill, inner, med_r)
    v = lambda p: float(vmap(p))
    vals = sorted([v(top_r), v(bot_r)])
    wh = sorted([v(y0), v(y1)])
    st.update(q1=vals[0], q3=vals[1], whislo=wh[0], whishi=wh[1], median=(v(med_r) if med_r is not None else None))
    st["style"]["median_lw_px"] = len(med_rows) if med_rows else None
    if st["style"].get("mean_px") is not None:
        st["mean"] = v(st["style"]["mean_px"])
    return st, None


def _glyph_style(gm, img_t, rows, left, right, a, b, mode, fill, inner, med_r):
    """how the box is drawn - measured, so the recreated plot can look the same:
    edge colour, median-line colour, notch, a mean marker inside the box"""
    sty = {}
    # edge colour: the box's left side column over the box rows
    ed = np.array([img_t[rows[k], max(0, left[k] + d)] for k in range(a, b) if left[k] >= 0 for d in (0, 1)], float)
    if len(ed):                                   # the stroke's core: darkest half (anti-aliasing greys the rest)
        ed = ed[ed.mean(1) <= np.median(ed.mean(1))]
    sty["edge_rgb"] = np.median(ed, 0).round(1).tolist() if len(ed) else [0, 0, 0]
    # notch: the left side moves inwards around the median and back out
    L = left[a:b].astype(float)
    ind = L - mode
    sty["notch"] = bool(ind.max() >= max(3, 0.12 * (right[a:b] - left[a:b]).max()) and np.mean(ind <= 1) >= 0.25)
    if med_r is not None and fill is not None:
        k = int(round(med_r))
        if 0 <= k < img_t.shape[0]:
            kk = int(np.argmin(np.abs(rows - k)))
            l, r = left[kk] + 2, right[kk] - 2
            if r > l:
                px = img_t[k, l:r + 1].astype(float)
                dev = np.abs(px - fill).max(1) > 40
                if dev.any():
                    sty["median_rgb"] = np.median(px[dev], 0).round(1).tolist()
    # mean marker: a compact blob inside the box that is neither fill, edge nor median line
    if fill is not None and len(inner):
        H_, W_ = img_t.shape[:2]
        blob = np.zeros((H_, W_), bool)
        others = [np.asarray(c, float) for c in (sty.get("median_rgb"), sty.get("edge_rgb")) if c is not None]
        for rr, p in inner:
            kk = int(np.argmin(np.abs(rows - rr)))
            l = left[kk] + 2
            d = np.abs(p - fill).max(1) > 60                 # not the fill ...
            for c in others:                                  # ... nor the median line / edge colour
                d &= np.abs(p - c).max(1) > 60
            if med_r is not None and abs(rr - med_r) <= 3 and not others:
                d[:] = False
            blob[rr, l:l + len(p)] = d
        lab, n = ndi.label(blob)
        if n:
            sizes = ndi.sum(blob, lab, range(1, n + 1))
            j = int(np.argmax(sizes)) + 1
            if sizes[j - 1] >= 8:
                ys, xs = np.where(lab == j)
                hh, ww = np.ptp(ys) + 1, np.ptp(xs) + 1
                if 0.4 <= hh / ww <= 2.5 and ww <= 0.6 * (right[a:b] - left[a:b]).max():
                    sty["mean_px"] = float(ys.mean())
                    sty["mean_rgb"] = np.median(img_t[ys, xs].astype(float), 0).round(1).tolist()
                    # triangle: width grows from top to bottom; else a circle / square
                    top_w = (ys <= ys.min() + hh / 3).sum(); bot_w = (ys >= ys.max() - hh / 3).sum()
                    sty["mean_marker"] = "^" if bot_w > 1.6 * top_w else ("v" if top_w > 1.6 * bot_w else "o")
    return sty


def _flier_style(mask, img_t):
    """marker of an outlier: hollow circle / filled circle / cross"""
    ys, xs = np.where(mask)
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    hh, ww = np.ptp(ys) + 1, np.ptp(xs) + 1
    fillr = len(ys) / float(hh * ww)
    centre = mask[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2].any()
    px = img_t[ys, xs].astype(float)
    dark = px.mean(1) <= np.percentile(px.mean(1), 35)      # thin strokes: the core, not the anti-aliased rim
    rgb = np.median(px[dark], 0).round(1).tolist()
    if not centre:
        mk, filled = "o", False
    elif fillr >= 0.6:
        mk, filled = ("s" if fillr >= 0.85 else "o"), True
    else:
        # a cross has its ink on the diagonals, a plus on the centre row / column
        diag = sum(mask[min(max(ys.min() + d, 0), mask.shape[0] - 1), min(xs.min() + d, mask.shape[1] - 1)] for d in range(min(hh, ww)))
        mk, filled = ("x" if diag >= 0.5 * min(hh, ww) else "+"), True
    return {"marker": mk, "filled": filled, "rgb": rgb, "size_px": float(max(hh, ww))}


def _value_axis(spec):
    from .spec_types import value_axis
    return value_axis(spec)


def _has_value_grid(img, area, vkey):
    """light lines across the whole plot, perpendicular to the value axis"""
    g = img.mean(-1)
    light = area & (g < 245) & (g > 150)
    frac = light.mean(1) if vkey == "y_axis" else light.mean(0)
    span = area.mean(1) if vkey == "y_axis" else area.mean(0)
    k = (span > 0) & (frac >= 0.6 * np.maximum(span, 1e-9))
    return int(D._runs(k).__len__()) >= 3 if hasattr(D, "_runs") else k.sum() >= 3


def pipeline_box(img, spec):
    H, W = img.shape[:2]
    frame = D.detect_frame(img)
    cal, cal_info = calibrate_all(img, frame, spec)
    vkey = _value_axis(spec)
    out = {"chart_type": "box", "orientation": "horizontal" if vkey == "x_axis" else "vertical", "size": (W, H), "frame": frame,
           "spec": spec, "warnings": [], "checks": [], "series": {},
           "calibration": {k: [m.as_dict() for m in v] for k, v in cal.items()},
           "calibration_info": {k: {kk: vv for kk, vv in v.items() if kk not in ("ticks", "labels")} for k, v in cal_info.items()}}
    if vkey not in cal or cal_info.get(vkey, {}).get("relative"):
        out["reject"] = "box plot: the value axis has no numeric calibration"; return out
    VM = cal[vkey][0]
    hue, sat, val = hsv_arrays(img)
    lines = D.text_lines_inside(img, frame)
    from .bars import find_patch_legend, match_patches
    legend, boxes = find_patch_legend(img, frame, lines, hue, sat, val)
    boxes = expand_legend_boxes(img, boxes, lines, [{"text": e["text"]} for e in legend])
    area = plot_region(frame, (H, W), inset=2) & ~exclusion_mask((H, W), lines, boxes)
    m = _drop_gridlines(_ink(img, area), img)
    if vkey == "x_axis":
        m, img_t = m.T, img.transpose(1, 0, 2)
    else:
        img_t = img
    lab, n = ndi.label(m, np.ones((3, 3)))
    objs = ndi.find_objects(lab)
    span = (frame["right"] - frame["left"]) if vkey == "x_axis" else (frame["bottom"] - frame["top"])
    glyphs, small = [], []
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        area_i = int((lab[sl] == i).sum())
        if h >= 0.04 * span and w >= 4 and area_i >= 30:
            glyphs.append(i)
        elif area_i >= 4 and max(h, w) <= 0.06 * span:
            small.append(i)
    cats = list(spec.get("categories") or [])
    series = list(spec.get("series") or []) or [{"name": "values", "color": ""}]
    if not glyphs:
        out["reject"] = "box plot: no box glyphs found"; return out
    # narrow 'glyphs' are stacks of touching flier markers, not boxes
    gw = {i: objs[i - 1][1].stop - objs[i - 1][1].start for i in glyphs}
    wmax = max(gw.values())
    small += [i for i in glyphs if gw[i] < 0.35 * wmax]
    glyphs = [i for i in glyphs if gw[i] >= 0.35 * wmax]
    stats, reasons = [], []
    for i in glyphs:
        st, why = analyse_glyph(lab == i, img_t, VM.to_value)
        if st is None:
            reasons.append(why)
        else:
            st["label"] = i
            stats.append(st)
    if reasons and len(reasons) >= max(1, len(glyphs) // 2):
        out["reject"] = f"box plot: {len(reasons)}/{len(glyphs)} glyphs are not boxes ({reasons[0]})"; return out
    stats.sort(key=lambda s: s["px"]["x_centre"])
    # fliers: small marks in a glyph's column, outside its whiskers
    for i in small:
        ys, xs = np.where(lab == i)
        cx = xs.mean()
        hh, ww = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        k = max(1, int(round(hh / max(ww, 1))))            # touching markers stacked along the value axis
        best = min(stats, key=lambda s: abs(s["px"]["x_centre"] - cx))
        if abs(best["px"]["x_centre"] - cx) > 0.6 * best["px"]["width"]:
            continue
        if k == 1 or True:
            fs_ = _flier_style(lab == i, img_t) if k == 1 else None
            if fs_:
                best.setdefault("flier_styles", []).append(fs_)
        for j in range(k):
            cy = ys.min() + (j + 0.5) * hh / k - 0.5
            if not (best["px"]["whisk_top"] - 2 <= cy <= best["px"]["bottom"] + 2):
                best.setdefault("fliers", []).append(float(VM.to_value(cy)))
    ns = len(series)
    if ns > 1 and not legend and cats and len(stats) == len(cats):
        # no legend and one box per category: the VLM listed each box colour as a 'series';
        # it is one series whose boxes are coloured individually
        out["warnings"].append(f"VLM listed {ns} series but there is no legend and one box per category - read as one series with per-box colours")
        base = re.sub(r"\s*#\d+$", "", series[0]["name"]).strip() or series[0]["name"]
        series = [dict(series[0], name=base, color="per box")]
        ns = 1
    expect = len(cats) * ns if cats else len(stats)
    if cats and len(stats) != expect:
        out["warnings"].append(f"VLM: {len(cats)} categories x {ns} series = {expect} boxes, pixels found {len(stats)}")
    # categories: group boxes by position (series sit side by side inside a category)
    from .bars import group_bars
    ncat = len(cats) if cats and len(stats) % max(1, len(cats)) == 0 else max(1, len(stats) // ns)
    groups = group_bars([s["px"]["x_centre"] for s in stats], ncat) if ncat > 1 else [[s["px"]["x_centre"] for s in stats]]
    cats = (cats + [f"category {k + 1}" for k in range(len(cats), ncat)])[:ncat]
    # series inside a category: by legend colour if there is a legend, else left to right
    match = match_patches(series, legend) if legend and ns > 1 else {}
    for s_ in series:
        out["series"][s_["name"]] = {"categories": [], "stats": [], "kind": "box"}
    for ci, g in enumerate(groups):
        members = [s for s in stats if s["px"]["x_centre"] in g]
        members.sort(key=lambda s: s["px"]["x_centre"])
        if match and len(match) == ns and all(s["fill_rgb"] for s in members):
            order = []
            for s in members:
                d = [np.abs(np.array(s["fill_rgb"]) - np.array(legend[match[k]]["rgb"])).max() for k in range(ns)]
                order.append(int(np.argmin(d)))
        else:
            order = list(range(len(members)))
        for s, k in zip(members, order):
            if k >= ns:
                continue
            rec = out["series"][series[k]["name"]]
            rec["categories"].append(cats[ci])
            rec["stats"].append({q: s.get(q) for q in ("q1", "median", "q3", "whislo", "whishi")} | {"fliers": sorted(s.get("fliers", []))}
                                | ({"mean": s["mean"]} if s.get("mean") is not None else {}))
            rec.setdefault("_glyph", []).append(s)
    out["glyphs"] = stats
    # category positions in pixels -> category index (for box positions and widths)
    gc = [float(np.mean(g)) for g in groups]
    if len(gc) >= 2:
        sl, ic = np.polyfit(gc, np.arange(len(gc)), 1)
    else:
        sl, ic = 1.0 / max(stats[0]["px"]["width"] * 2, 1), -gc[0] / max(stats[0]["px"]["width"] * 2, 1)
    for s in stats:
        s["pos"] = float(sl * s["px"]["x_centre"] + ic)
        s["width_units"] = float(abs(sl) * s["px"]["width"])
    ext = (frame["left"], frame["right"]) if vkey == "x_axis" else (frame["bottom"], frame["top"])
    out["value_limits"] = sorted([float(VM.to_value(ext[0])), float(VM.to_value(ext[1]))])
    cext = (frame["top"], frame["bottom"]) if vkey == "x_axis" else (frame["left"], frame["right"])
    out["category_limits"] = sorted([float(sl * cext[0] + ic), float(sl * cext[1] + ic)])
    # image rows grow downwards, a matplotlib y axis upwards
    out["category_axis_reversed"] = bool(sl > 0) if vkey == "x_axis" else bool(sl < 0)
    out["grid"] = bool(_has_value_grid(img, area, vkey))
    # outlier marker: one style per series (stacked, touching markers cannot be read alone)
    for rec in out["series"].values():
        fl = [f for g_ in rec.get("_glyph", []) for f in g_.get("flier_styles", [])]
        if fl:
            from collections import Counter
            mk = Counter((f["marker"], f["filled"]) for f in fl).most_common(1)[0][0]
            pick = [f for f in fl if (f["marker"], f["filled"]) == mk]
            common = {"marker": mk[0], "filled": mk[1], "rgb": np.median([f["rgb"] for f in pick], 0).round(1).tolist()}
            for g_ in rec.get("_glyph", []):
                g_["flier_styles"] = [common]
    for rec in out["series"].values():
        gl = rec.pop("_glyph", [])
        rec["draw"] = [{"pos": g_["pos"], "width": g_["width_units"], "fill_rgb": g_["fill_rgb"], **g_["style"],
                        "flier": (g_.get("flier_styles") or [None])[0]} for g_ in gl]
    out["table"] = [{"series": n, "category": c, **{k: v for k, v in st.items() if k != "fliers"},
                     "fliers": ";".join(f"{f:.6g}" for f in st["fliers"])}
                    for n, d in out["series"].items() for c, st in zip(d["categories"], d["stats"])]
    rms = cal_info[vkey].get("rms_px", 0)
    nomed = sum(1 for s in stats if s.get("median") is None)
    out["checks"] += [("value-axis label fit (px)", round(rms, 2), "<= 3", rms <= 3),
                      ("boxes found / expected", f"{len(stats)}/{expect}", "equal", len(stats) == expect),
                      ("boxes without a median line", nomed, "0", nomed == 0)]
    return out
