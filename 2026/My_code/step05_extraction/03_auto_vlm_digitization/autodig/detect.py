"""Pixel-level layout detection for a chart image - no per-figure settings.

Everything here works from the image alone:
  * plot frame (x-axis line, left / right y-axis lines, top line if boxed)
  * tick marks along each axis (major vs minor by length)
  * tick-label text blobs outside the frame (positions only - their *values*
    come from the VLM, see spec.py)
  * text lines inside the frame (legend text, annotations) to be excluded
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.int16)


def gray(img):
    return img.mean(-1)


def saturation(img):
    mx, mn = img.max(-1), img.min(-1)
    return np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0.0)


def dark_mask(img, max_gray=140, max_sat=0.35):
    return (gray(img) <= max_gray) & (saturation(img) <= max_sat)


def _runs(v):
    """(start, end) of every run of True in a 1-D bool array (end exclusive)."""
    v = np.r_[False, v, False].astype(np.int8)
    d = np.diff(v)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _longest_run_gap(v, gap=15):
    """Longest run allowing interruptions of up to `gap` px (markers sitting on an axis)."""
    rs = _runs(v)
    if not rs:
        return 0, 0, 0
    merged = [list(rs[0])]
    for s, e in rs[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    s, e = max(merged, key=lambda r: r[1] - r[0])
    return e - s, s, e


def _longest_run(v):
    rs = _runs(v)
    if not rs:
        return 0, 0, 0
    s, e = max(rs, key=lambda r: r[1] - r[0])
    return e - s, s, e


# --------------------------------------------------------------------- frame
def detect_frame(img):
    try:
        return _detect_frame(img)
    except RuntimeError:
        return _frame_from_x_axis(img)


def _detect_frame(img):
    """Return dict(left, right, top, bottom, has_right, has_top, line_width).

    left/right/top/bottom are the pixel coordinates of the axis lines. The
    y-axis is the strongest long vertical dark line in the left half, the
    x-axis the lowest long horizontal line between the y-axis lines.
    """
    H, W = img.shape[:2]
    d = dark_mask(img, max_gray=150, max_sat=0.3)
    # axis lines may be coloured (e.g. a magenta left axis and a blue right axis):
    # strongly saturated pixels count too, but only where they form a THIN line -
    # filled areas (bars) and markers are wider than a few pixels
    sat, gr = saturation(img), gray(img)
    col_line = (sat > 0.6) & (gr < 200)
    thin = col_line & ~(np.roll(col_line, 4, 1) & np.roll(col_line, -4, 1))
    d_v = d | thin
    col_cnt = d_v.sum(0).astype(float)
    # discount filled regions: a column only counts where its neighbours 4 px away are not ink too
    iso = d_v & ~np.roll(d_v, 4, 1) & ~np.roll(d_v, -4, 1)
    col_cnt = iso.sum(0).astype(float)

    def strongest(lo, hi, leftmost):
        seg = col_cnt[lo:hi]
        c = int(np.argmax(seg)) + lo
        near = [x for x in range(max(lo, c - 6), min(hi, c + 7)) if col_cnt[x] >= 0.6 * col_cnt[c]]
        return (near[0] if leftmost else near[-1]), col_cnt[c]
    lc, ln = strongest(0, W // 2, leftmost=True)
    rc, rn = strongest(W // 2, W, leftmost=False)
    if ln < 0.3 * H:
        return _frame_from_x_axis(img)
    # x-axis: lowest row that is mostly dark between the y-axis and the right end
    x_hi = rc if rn >= 0.3 * H else W - 1
    frac = d[:, lc:x_hi + 1].mean(1)
    rows = np.where(frac >= 0.5)[0]
    groups = _group_consecutive(list(rows))
    groups = [g for g in groups if g[0] > 0.3 * H]
    if not groups:
        raise RuntimeError("no x-axis line found")
    bottom_group = groups[-1]
    bottom = float(np.mean(bottom_group))
    lw = len(bottom_group)
    yb = int(bottom_group[0])
    # top of the y-axis line (allow a gap < 25 px, e.g. a broken axis)
    col = d_v[:yb, max(lc - 1, 0):lc + 2].any(1)
    top_left = yb
    for s_, e in reversed(_runs(col)):
        if top_left - e <= 25:
            top_left = s_
    has_right = rn >= 0.5 * (yb - top_left)
    top_rows = [g for g in _group_consecutive(list(rows)) if g[-1] < yb - 20 and abs(g[0] - top_left) <= 6]
    has_top = bool(top_rows)
    top = float(np.mean(top_rows[0])) if has_top else float(top_left)
    left = _line_centre(d_v, lc, axis=1, lo=int(top), hi=yb)
    right = _line_centre(d_v, rc, axis=1, lo=int(top), hi=yb) if has_right else float(x_hi)
    return {"left": left, "right": right, "top": top, "bottom": bottom,
            "has_right": bool(has_right), "has_top": has_top, "line_width": int(lw), "size": (W, H)}


def _frame_from_x_axis(img):
    """Charts without a y-axis line (spreadsheet style: an x-axis line and
    horizontal grid lines only). The frame is the extent of the lowest long
    horizontal line; its top is the highest long horizontal (grid) line."""
    H, W = img.shape[:2]
    g = gray(img)
    lines = (g < 205) & (saturation(img) < 0.25)
    best = []
    for y in range(int(0.3 * H), H):
        n, s_, e = _longest_run(lines[y])
        if n >= 0.4 * W:
            best.append((y, s_, e))
    if not best:
        raise RuntimeError("no axis line found")
    # the x-axis: lowest long line that is not the figure's outer border
    cand = [b for b in best if b[0] < H - 0.04 * H] or best
    yb, xl, xr = max(cand, key=lambda b: b[0])
    rows = [y for y in range(0, yb - 10) if lines[y, xl:xr].mean() > 0.6 and y > 0.02 * H]
    top = float(min(rows)) if rows else float(0.05 * H)
    # skip an outer figure border at the very top
    grp = _group_consecutive(rows)
    if len(grp) > 1 and grp[0][0] < 0.03 * H:
        top = float(grp[1][0])
    return {"left": float(xl), "right": float(xr - 1), "top": top, "bottom": float(yb),
            "has_right": False, "has_top": False, "line_width": 1, "size": (W, H), "no_y_axis_line": True}


def _group_consecutive(idx, gap=1):
    out, cur = [], []
    for i in idx:
        if cur and i - cur[-1] > gap:
            out.append(cur); cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def _line_centre(d, c, axis, lo, hi):
    """Refine a vertical line position: mean of the columns around c that are
    mostly dark between rows lo..hi."""
    if hi <= lo:
        return float(c)
    cols = [x for x in range(c - 3, c + 4) if 0 <= x < d.shape[1] and d[lo:hi, x].mean() > 0.5]
    return float(np.mean(cols)) if cols else float(c)


# --------------------------------------------------------------------- ticks
def detect_ticks(img, frame, side, max_len=14):
    """Tick marks on one axis. side in {"bottom", "left", "right"}.
    Returns list of dicts (pos, length, direction) - pos is the pixel coordinate
    along the axis (x for bottom, y for left/right)."""
    d = dark_mask(img, max_gray=150, max_sat=0.35) | ((saturation(img) > 0.5) & (gray(img) < 200))
    hw = max(1, frame["line_width"] // 2 + 1)
    L, R, T, B = frame["left"], frame["right"], frame["top"], frame["bottom"]
    out = []
    if side == "bottom":
        y = int(round(B))
        span = range(int(L) + 3, int(R) - 2)
        for direction, rng in (("out", range(y + hw + 1, y + hw + 1 + max_len)), ("in", range(y - hw - 1, y - hw - 1 - max_len, -1))):
            for x in span:
                n = 0
                for yy in rng:
                    if 0 <= yy < d.shape[0] and d[yy, x]:
                        n += 1
                    else:
                        break
                if n >= 3:
                    out.append((x, n, direction))
    else:
        x = int(round(L if side == "left" else R))
        span = range(int(T) + 3, int(B) - 2)
        outward = range(x - hw - 1, x - hw - 1 - max_len, -1) if side == "left" else range(x + hw + 1, x + hw + 1 + max_len)
        inward = range(x + hw + 1, x + hw + 1 + max_len) if side == "left" else range(x - hw - 1, x - hw - 1 - max_len, -1)
        for direction, rng in (("out", outward), ("in", inward)):
            for y in span:
                n = 0
                for xx in rng:
                    if 0 <= xx < d.shape[1] and d[y, xx]:
                        n += 1
                    else:
                        break
                if n >= 3:
                    out.append((y, n, direction))
    # merge adjacent positions (thick ticks)
    ticks = []
    for direction in ("out", "in"):
        pts = sorted((p, n) for p, n, dr in out if dr == direction)
        for g in _group_consecutive([p for p, _ in pts]):
            ns = [n for p, n in pts if p in g]
            if len(g) > 4:          # too wide to be a tick (a marker / text touching the axis)
                continue
            ticks.append({"pos": float(np.mean(g)), "length": int(max(ns)), "direction": direction})
    # drop "in" ticks that are really data touching the axis: keep the direction with more ticks
    if ticks:
        n_out = sum(t["direction"] == "out" for t in ticks)
        n_in = len(ticks) - n_out
        keep = "out" if n_out >= n_in else "in"
        ticks = [t for t in ticks if t["direction"] == keep]
        if ticks:
            mx = max(t["length"] for t in ticks)
            for t in ticks:
                t["major"] = t["length"] >= 0.75 * mx
    return sorted(ticks, key=lambda t: t["pos"])


# --------------------------------------------------------------------- text
def components(mask, min_area=3):
    lab, n = ndi.label(mask)
    objs = ndi.find_objects(lab)
    out = []
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        area = int((lab[sl] == i).sum())
        if area < min_area:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        out.append({"x0": x0, "x1": x1, "y0": y0, "y1": y1, "area": area})
    return out


def _merge_words(comps, horizontal=True, gap_factor=0.7):
    """Merge character components into words / labels."""
    comps = sorted(comps, key=lambda c: (c["x0"] if horizontal else c["y0"]))
    words = []
    for c in comps:
        h = c["y1"] - c["y0"]
        merged = False
        for w in words:
            wh = w["y1"] - w["y0"]
            if horizontal:
                vov = min(w["y1"], c["y1"]) - max(w["y0"], c["y0"])
                gap = c["x0"] - w["x1"]
                if vov > -2 and gap <= gap_factor * max(h, wh, 6) and gap > -max(h, wh):
                    merged = True
            else:
                hov = min(w["x1"], c["x1"]) - max(w["x0"], c["x0"])
                gap = c["y0"] - w["y1"]
                ww = w["x1"] - w["x0"]
                if hov > -2 and gap <= gap_factor * max(c["x1"] - c["x0"], ww, 6) and gap > -ww:
                    merged = True
            if merged:
                w["x0"], w["x1"] = min(w["x0"], c["x0"]), max(w["x1"], c["x1"])
                w["y0"], w["y1"] = min(w["y0"], c["y0"]), max(w["y1"], c["y1"])
                w["n"] += 1
                break
        if not merged:
            words.append(dict(c, n=1))
    return words


def _first_band(profile, start, step, max_gap=2, max_len=120, min_size=5):
    """From `start`, walk outward to the first occupied stretch at least
    `min_size` px thick (thinner stretches are tick ends / specks), allowing
    gaps <= max_gap inside it. Returns (lo, hi)."""
    pos = start
    for _ in range(6):
        band = _band_once(profile, pos, step, max_gap, max_len - abs(pos - start))
        if band is None:
            return None
        if band[1] - band[0] + 1 >= min_size:
            return band
        pos = (band[1] + 1) if step > 0 else (band[0] - 1)
    return None


def _band_once(profile, start, step, max_gap, max_len):
    n = len(profile)
    i, k = start, 0
    while 0 <= i < n and not profile[i] and k < max_len:
        i += step; k += 1
    if not (0 <= i < n) or not profile[i]:
        return None
    first = i
    last = i
    gap = 0
    while 0 <= i < n and k < max_len:
        if profile[i]:
            last = i; gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
        i += step; k += 1
    return (min(first, last), max(first, last))


def tick_label_blobs(img, frame, side, ticks=None):
    """Bounding boxes of the tick labels next to one axis: the first band of
    text outside the frame (the axis title further out is excluded by the
    whitespace gap between them). Returns words with centre 'c' along the axis."""
    d = dark_mask(img, max_gray=160, max_sat=0.4) | ((saturation(img) > 0.5) & (gray(img) < 200))
    H, W = d.shape
    L, R, T, B = frame["left"], frame["right"], frame["top"], frame["bottom"]
    tick_len = max([t["length"] for t in (ticks or []) if t["direction"] == "out"] + [0])
    lw = frame["line_width"]
    if side == "bottom":
        x_lo, x_hi = int(max(0, L - 40)), int(min(W, R + 40))
        start = int(B + lw / 2 + tick_len + 2)
        prof = d[:, x_lo:x_hi].any(1)
        band = _first_band(prof, start, +1)
        if band is None:
            return []
        sub = d[band[0]:band[1] + 1, x_lo:x_hi]
        cs = [dict(c, x0=c["x0"] + x_lo, x1=c["x1"] + x_lo, y0=c["y0"] + band[0], y1=c["y1"] + band[0])
              for c in components(sub, min_area=2)]
        words = _group_words(cs, horizontal=True)
        # a y-axis label sitting low at the corner (the left/right '0') is not an x label
        words = [w for w in words if w["x1"] >= L - 1 and w["x0"] <= R + 1]
        for w in words:
            w["c"] = (w["x0"] + w["x1"] - 1) / 2
        return sorted(words, key=lambda w: w["c"])
    y_lo, y_hi = int(max(0, T - 15)), int(min(H, B + 15))
    prof = d[y_lo:y_hi, :].any(0)
    if side == "left":
        band = _first_band(prof, int(L - lw / 2 - tick_len - 2), -1)
    else:
        band = _first_band(prof, int(R + lw / 2 + tick_len + 2), +1)
    if band is None:
        return []
    sub = d[y_lo:y_hi, band[0]:band[1] + 1]
    cs = [dict(c, x0=c["x0"] + band[0], x1=c["x1"] + band[0], y0=c["y0"] + y_lo, y1=c["y1"] + y_lo)
          for c in components(sub, min_area=2)]
    words = _group_words(cs, horizontal=False)
    # a rotated axis title right next to the axis (no tick labels) shows up as
    # tall, narrow pieces - reject bands whose 'words' are taller than wide
    # (narrow digits such as "1" are tall-and-thin too, so compare with the other words' height)
    if words:
        med_h = float(np.median([w["y1"] - w["y0"] for w in words]))
        words = [w for w in words if (w["y1"] - w["y0"]) <= 1.6 * max(w["x1"] - w["x0"], 6) + 4
                 or (w["y1"] - w["y0"]) <= 1.25 * med_h + 2]
    # tick labels hug the axis (right-aligned on the left axis, left-aligned on
    # the right one); pieces of a rotated title sitting in the same band don't
    if words and side == "left":
        edge = max(w["x1"] for w in words)
        words = [w for w in words if w["x1"] >= edge - 8]
    elif words:
        edge = min(w["x0"] for w in words)
        words = [w for w in words if w["x0"] <= edge + 8]
    if len(words) < 2:
        return []
    for w in words:
        w["c"] = (w["y0"] + w["y1"] - 1) / 2
    # letters of a rotated axis title sit in a dense column (gaps ~ letter height);
    # tick labels are spread out (gaps >= ~2 label heights)
    cs = np.sort([w["c"] for w in words])
    med_h = float(np.median([w["y1"] - w["y0"] for w in words]))
    if len(cs) >= 4 and float(np.median(np.diff(cs))) < 1.5 * med_h:
        return []
    return sorted(words, key=lambda w: w["c"])


def _group_words(cs, horizontal):
    """Group the components of a label band into labels. Horizontal band (x-axis):
    a new label starts after a gap wider than ~0.8 x text height. Vertical band
    (y-axis): one label per text row (components overlapping vertically)."""
    if not cs:
        return []
    hs = np.median([c["y1"] - c["y0"] for c in cs])
    if horizontal:
        cs = sorted(cs, key=lambda c: c["x0"])
        words = [dict(cs[0], n=1)]
        for c in cs[1:]:
            w = words[-1]
            if c["x0"] - w["x1"] <= 0.8 * hs:
                w.update(x0=min(w["x0"], c["x0"]), x1=max(w["x1"], c["x1"]),
                         y0=min(w["y0"], c["y0"]), y1=max(w["y1"], c["y1"]), n=w["n"] + 1)
            else:
                words.append(dict(c, n=1))
    else:
        cs = sorted(cs, key=lambda c: c["y0"])
        words = [dict(cs[0], n=1)]
        for c in cs[1:]:
            w = words[-1]
            if c["y0"] <= w["y1"] - 0.3 * min(c["y1"] - c["y0"], w["y1"] - w["y0"]):
                w.update(x0=min(w["x0"], c["x0"]), x1=max(w["x1"], c["x1"]),
                         y0=min(w["y0"], c["y0"]), y1=max(w["y1"], c["y1"]), n=w["n"] + 1)
            else:
                words.append(dict(c, n=1))
    # drop specks (e.g. a stray dot) much smaller than a typical label
    big = np.median([(w["x1"] - w["x0"]) * (w["y1"] - w["y0"]) for w in words])
    return [w for w in words if (w["x1"] - w["x0"]) * (w["y1"] - w["y0"]) >= 0.15 * big]


def text_lines_inside(img, frame, inset=3):
    """Text lines inside the plot frame (legend text, annotations). A text line
    is >= 2 character-sized dark components sitting side by side."""
    d = dark_mask(img, max_gray=150, max_sat=0.4)
    L, R, T, B = [int(round(v)) for v in (frame["left"], frame["right"], frame["top"], frame["bottom"])]
    sub = d[T + inset:B - inset, L + inset:R - inset]
    cs = [dict(c, x0=c["x0"] + L + inset, x1=c["x1"] + L + inset, y0=c["y0"] + T + inset, y1=c["y1"] + T + inset)
          for c in components(sub, min_area=4)]
    # character-sized components
    ch = [c for c in cs if 5 <= (c["y1"] - c["y0"]) <= 30 and (c["x1"] - c["x0"]) <= 30]
    words = _merge_words(ch, horizontal=True, gap_factor=0.8)
    # >= 2 glyphs side by side; a single wide component is a legend line sample, not text
    lines = [w for w in words if (w["x1"] - w["x0"]) >= 1.4 * (w["y1"] - w["y0"]) and w["n"] >= 2]
    return lines
