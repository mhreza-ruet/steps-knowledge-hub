"""Stacked bar charts (vertical or horizontal, also 100 % stacks) and horizontal
grouped bar charts. (Vertical grouped bars are read by bars.py.)

The VLM reads: orientation, the category names (vertical: left -> right;
horizontal: top -> bottom), the series (stack order, legend names, fill
colours) and the value-axis tick labels. The pixels give:

  1. frame + value-axis calibration
  2. per series the fill colour (legend patch, else colour name) -> fill mask
  3. horizontal charts are transposed, so one code path reads both
  4. per series: runs of columns = bar segments; a segment's two ends are the
     first / last filled row in its middle columns, moved half-way into an edge
     line drawn between segments (an edge line is centred on the boundary)
  5. stacked: value = segment length in data units; not stacked: value = the
     bar's far end read on the axis
  6. all segments are grouped into the VLM's number of categories by position
"""
from __future__ import annotations

import numpy as np

from . import detect as D
from .extract import calibrate_all, hsv_arrays, plot_region, exclusion_mask, expand_legend_boxes


def _dark_run(col, start, step, limit=6):
    """length of the dark run starting at index start going in direction step"""
    n, i = 0, start
    while 0 <= i < len(col) and n < limit and col[i] < 110:
        n += 1; i += step
    return n


def pipeline_stacked(img, spec):
    from .bars import find_patch_legend, match_patches, colour_from_name, fill_mask, _rgb_hue, discover_colours, group_bars
    H, W = img.shape[:2]
    frame = D.detect_frame(img)
    cal, cal_info = calibrate_all(img, frame, spec)
    horiz = str(spec.get("orientation") or "vertical").lower().startswith("h")
    stacked = str(spec.get("chart_type", "")).lower().startswith("stacked")
    vkey = "x_axis" if horiz else "y_axis"
    out = {"chart_type": "stacked_bar" if stacked else "bar", "orientation": "horizontal" if horiz else "vertical",
           "stacked": stacked, "size": (W, H), "frame": frame, "spec": spec, "warnings": [], "checks": [], "series": {},
           "calibration": {k: [m.as_dict() for m in v] for k, v in cal.items()},
           "calibration_info": {k: {kk: vv for kk, vv in v.items() if kk not in ("ticks", "labels")} for k, v in cal_info.items()}}
    if vkey not in cal or cal_info.get(vkey, {}).get("relative"):
        out["reject"] = "bar chart: the value axis has no numeric calibration"; return out
    VM = cal[vkey][0]
    hue, sat, val = hsv_arrays(img)
    from .bars import legend_patches
    legend, boxes, lines = legend_patches(img, frame, hue, sat, val)
    boxes = expand_legend_boxes(img, boxes, lines, [{"text": e["text"]} for e in legend])
    area = plot_region(frame, (H, W), inset=1) & ~exclusion_mask((H, W), lines, boxes)
    series = list(spec.get("series") or [])
    cats = list(spec.get("categories") or [])
    if not series or not cats:
        out["reject"] = "bar chart: the VLM gave no series or no categories"; return out
    match = match_patches(series, legend) if legend else {}
    refs, taken = [], []
    for i, s in enumerate(series):
        if i in match:
            ref, src = legend[match[i]]["rgb"], "legend patch"
        else:
            ref, src = colour_from_name(s.get("color", ""), img, hue, sat, val, area, taken), "colour name"
        refs.append((ref, src))
        if ref is not None:
            taken.append(_rgb_hue(ref)[0])
    if any(r is None for r, _ in refs):
        free = [c for c in discover_colours(img, area, sat)
                if all(np.linalg.norm(np.array(c) - np.array(r)) > 60 for r, _ in refs if r is not None)]
        refs = [(free.pop(0), "unclaimed fill colour") if r is None and free else (r, src) for r, src in refs]
    gray = img.mean(-1)
    if horiz:
        gray_t = gray.T
    else:
        gray_t = gray
    segs = {}
    for (ref, src), s in zip(refs, series):
        if ref is None:
            out["warnings"].append(f"colour of '{s['name']}' not found"); continue
        m = fill_mask(img, ref, hue, sat, val) & area
        # stacked series often share a hue (light / mid / dark blue): a pixel belongs to
        # the series whose reference colour is nearest
        others = [np.array(r, float) for r, _ in refs if r is not None and r is not ref]
        if others:
            d_own = np.sqrt(((img.astype(float) - np.array(ref, float)) ** 2).sum(-1))
            d_oth = np.min([np.sqrt(((img.astype(float) - o) ** 2).sum(-1)) for o in others], 0)
            m &= d_own <= d_oth
        if horiz:
            m = m.T
        colcnt = m.sum(0)
        found = []
        for a, b in D._runs(colcnt >= 3):
            if b - a < 3:
                continue
            c0, c1 = a + int(0.3 * (b - a)), max(a + int(0.3 * (b - a)) + 1, b - int(0.3 * (b - a)))
            tops, bots = [], []
            for c in range(c0, c1):
                rr = np.where(m[:, c])[0]
                if len(rr):
                    tops.append(rr.min()); bots.append(rr.max())
            if not tops:
                continue
            t, bt = float(np.median(tops)), float(np.median(bots))
            cc = int(0.5 * (a + b))
            du = _dark_run(gray_t[:, cc], int(t) - 1, -1)
            dd = _dark_run(gray_t[:, cc], int(bt) + 1, +1)
            e_lo, e_hi = t - 0.5 - du / 2.0, bt + 0.5 + dd / 2.0       # pixel ends of the segment
            found.append({"centre": 0.5 * (a + b - 1), "width": b - a, "ends_px": (e_lo, e_hi)})
        segs[s["name"]] = found
        out.setdefault("styles", {})[s["name"]] = {"rgb": list(map(float, ref)), "from": src}
    allc = sorted(f["centre"] for v in segs.values() for f in v)
    if not allc:
        out["reject"] = "bar chart: no bars found"; return out
    if len(allc) < len(cats):
        out["warnings"].append(f"VLM listed {len(cats)} categories, only {len(allc)} bar segments found")
    groups = group_bars(allc, min(len(cats), len(allc)))
    bounds = [(-np.inf if i == 0 else 0.5 * (groups[i - 1][-1] + groups[i][0]),
               np.inf if i == len(groups) - 1 else 0.5 * (groups[i][-1] + groups[i + 1][0])) for i in range(len(groups))]
    base = None
    lo_v, hi_v = sorted([float(VM.to_value(frame["left"] if horiz else frame["bottom"])), float(VM.to_value(frame["right"] if horiz else frame["top"]))])
    if lo_v <= 0 <= hi_v:
        base = float(VM.to_pixel(0.0))
    for s in series:
        cs, vs, px = [], [], []
        for ci, (lo, hi) in enumerate(bounds):
            inside = [f for f in segs.get(s["name"], []) if lo <= f["centre"] < hi]
            if not inside:
                continue
            f = max(inside, key=lambda q: q["width"])
            a, b = f["ends_px"]
            if stacked:
                v = abs(float(VM.to_value(b)) - float(VM.to_value(a)))
            else:
                # far end from the baseline (0 if visible, else the axis start)
                ref_px = base if base is not None else (frame["left"] if horiz else frame["bottom"])
                far = a if abs(a - ref_px) > abs(b - ref_px) else b
                v = float(VM.to_value(far))
            cs.append(cats[ci] if ci < len(cats) else f"category {ci + 1}"); vs.append(round(v, 6)); px.append([round(a, 1), round(b, 1), round(f["centre"], 1)])
        out["series"][s["name"]] = {"categories": cs, "values": vs, "px": px, "kind": "bar"}
        miss = [c for c in cats if c not in cs]
        if miss:
            out["warnings"].append(f"'{s['name']}': no bar segment found for {miss}")
    out["categories"] = cats
    out["category_centres_px"] = [float(np.mean(g)) for g in groups]
    out["table"] = [{"series": n, "category": c, "value": v} for n, d in out["series"].items() for c, v in zip(d["categories"], d["values"])]
    rms = cal_info[vkey].get("rms_px", 0)
    found = sum(len(d["values"]) for d in out["series"].values())
    out["checks"] += [("value-axis label fit (px)", round(rms, 2), "<= 3", rms <= 3),
                      ("bar segments found / expected", f"{found}/{len(cats) * len(series)}", ">= 75 %",
                       found >= 0.75 * len(cats) * len(series))]
    return out
