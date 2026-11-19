"""Panel-aware classical digitizer for the extended figure set (``profiles.py``).

Same idea as ``digitize.py`` (hand-checked pixel calibration per axis + the colour of
every series -> mechanical extraction), generalised so that one image can hold

* several plot PANELS (a 2x2 grid of isotherms, five stacked XRD patterns, ...),
* an INSET plot inside the main plot (a bar chart inside a line plot),
* several y-axes per panel (left / right, a broken left axis),

and the series kinds met in the new figures: lines, markers (with error bars), bars,
stacked bars, reference sticks (XRD/PDF cards) and - through the autodig package of
notebook 03 - 3-D surfaces and filled contour maps.

Every panel is digitized on its own (its frame, its axes, its series); the result keeps
the panels apart:

    {"figure": id, "layout": "single" | "panels" | "main+inset",
     "panels": {panel_id: {"role": "main"|"panel"|"inset", "series": {name: {x, y, yerr?, kind}}, ...}}}

Nothing is cropped: masks are restricted to a panel's frame inside the full image.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .digitize import Axis, dark_mask, hue_mask, load_rgb


def color_mask(img, rgb, tol=60, min_sat=None):
    d = np.sqrt(((img.astype(np.int32) - np.asarray(rgb, np.int32)) ** 2).sum(-1))
    m = d <= tol
    if min_sat is not None:
        m &= (img.max(-1) - img.min(-1)) >= min_sat
    return m


# ----------------------------------------------------------------------------- axes
class Piecewise:
    """pixel <-> value for an axis broken into segments: [{"px": (lo, hi), "map": (p0, v0, p1, v1)}]"""

    def __init__(self, segs):
        self.segs = [(float(min(g["px"])), float(max(g["px"])), Axis(*g["map"])) for g in segs]

    def to_value(self, p):
        p = np.asarray(p, float)
        out = np.full(p.shape, np.nan)
        for lo, hi, A in self.segs:
            out = np.where((p >= lo - 0.5) & (p <= hi + 0.5) & np.isnan(out), A.to_value(p), out)
        return out if out.shape else float(out)

    def to_pixel(self, v):
        v = np.asarray(v, float)
        out = np.full(v.shape, np.nan)
        for lo, hi, A in self.segs:
            a, b = sorted((float(A.to_value(lo)), float(A.to_value(hi))))
            out = np.where((v >= a - 1e-9) & (v <= b + 1e-9) & np.isnan(out), A.to_pixel(v), out)
        return out if out.shape else float(out)


def axis_of(P, key):
    if key == "x" and P.get("x_segments"):
        return Piecewise(P["x_segments"])
    if key == "y" and P.get("y_segments") and P.get("pixel_space"):
        return Piecewise([{"px": g["rows"], "map": g["map"]} for g in P["y_segments"]])
    c = P.get(key)
    if c is None:
        return None
    return Axis(c[0], c[1], c[2], c[3], log=bool(P.get(key + "_log", False)))


def y_axis_for(P, s, py=None):
    """the y Axis a series (or one pixel row of it) is read against"""
    ax = s.get("axis", "left")
    if ax == "right":
        return axis_of(P, "y_right")
    if "split_px" in P:                                  # broken left axis: the row decides
        if py is None:
            return axis_of(P, "y_lower")
        return axis_of(P, "y_upper") if py < P["split_px"] else axis_of(P, "y_lower")
    return axis_of(P, "y")


def y_value(P, s, py):
    py = np.asarray(py, float)
    if P.get("y_segments") and s.get("axis", "left") != "right":
        # several stacked segments (axis broken more than once): the pixel row decides the segment
        out = np.full(py.shape, np.nan)
        for seg in P["y_segments"]:
            A = Axis(*seg["map"])
            sel = (py >= seg["rows"][0]) & (py <= seg["rows"][1])
            out = np.where(sel, A.to_value(py), out)
        return out
    if "split_px" in P and s.get("axis", "left") != "right":
        up, lo = axis_of(P, "y_upper"), axis_of(P, "y_lower")
        return np.where(py < P["split_px"], up.to_value(py), lo.to_value(py))
    return y_axis_for(P, s).to_value(py)


# ----------------------------------------------------------------------------- masks
def series_mask(img, s):
    m = s.get("match", {})
    if m.get("dark"):
        mask = dark_mask(img, max_gray=m.get("max_gray", 110), max_sat=m.get("max_sat", 40))
    elif "hue" in m:
        mask = hue_mask(img, m["hue"], m.get("hue_tol", 20), m.get("min_sat", 0.3), m.get("min_val", 0.3))
        if m.get("close"):
            mask = ndi.binary_closing(mask, structure=np.ones((m["close"], m["close"])))
        if "max_sat_frac" in m:                            # muted colours (brown next to grey): an upper saturation bound
            mx = img.max(-1).astype(float)
            mask &= (mx - img.min(-1)) / np.maximum(mx, 1) <= m["max_sat_frac"]
    elif m.get("outline"):
        mask = dark_mask(img, max_gray=110, max_sat=60)
    elif "ink" in m:                                       # hatched / dotted fills: any ink, closed over the pattern
        mask = img.astype(float).mean(-1) < m["ink"]
        if m.get("drop_solid"):                            # solid black bars next to the patterned ones are not part of them
            solid = ndi.binary_opening(img.astype(float).mean(-1) < 70, structure=np.ones((5, 5)))
            mask &= ~ndi.binary_dilation(solid, iterations=1)
        if m.get("close"):
            k = m["close"]
            mask = ndi.binary_closing(mask, structure=np.ones((k, k)))
        if m.get("close_h"):                               # outlined bars: fill between the two side edges
            mask = ndi.binary_closing(mask, structure=np.ones((1, m["close_h"])))
    elif "nonwhite" in m:                                  # gradient / shaded fills: anything not background
        mask = img.min(-1) < m["nonwhite"]
    elif m.get("gray"):                                   # grey fills with a light-to-dark gradient
        g = img.astype(float).mean(-1); sat = img.max(-1) - img.min(-1)
        lo, hi = m.get("gray_range", (55, 215))
        mask = (g >= lo) & (g <= hi) & (sat <= m.get("max_sat", 8))
    elif "rgb" in m:
        mask = color_mask(img, m["rgb"], m.get("tol", 50), m.get("min_sat"))
    else:
        raise ValueError(f"series has no colour match: {s}")
    if "gray_range" in m and not m.get("gray"):           # grey shade window (hatched / grey bars)
        g = img.mean(-1); sat = img.max(-1) - img.min(-1)
        lo, hi = m["gray_range"]
        mask = (g >= lo) & (g <= hi) & (sat <= m.get("max_sat", 30))
    return mask


def inside(mask, P, extra_exclude=(), pad=None):
    L, R, T, B = [int(round(v)) for v in P["frame"]]
    pad = P.get("frame_pad", 2) if pad is None else pad
    out = np.zeros_like(mask)
    out[T + pad:B - pad + 1, L + pad:R - pad + 1] = mask[T + pad:B - pad + 1, L + pad:R - pad + 1]
    for box in list(P.get("exclude", [])) + list(extra_exclude):
        x0, x1, y0, y1 = [int(round(v)) for v in box]
        out[y0:y1, x0:x1] = False
    return out


# ----------------------------------------------------------------------------- lines
def trace_curve(mask, x_range=None, max_jump=25, gap_fill=30, prefer="continuity", restart_gap=25, vert_run=5):
    """one curve: per pixel column the run of mask pixels that continues the curve best.
    Returns pixel xs, ys (gaps up to gap_fill px are bridged linearly, longer gaps break)."""
    H, W = mask.shape
    cols = np.where(mask.any(0))[0]
    if x_range is not None:
        cols = cols[(cols >= x_range[0]) & (cols <= x_range[1])]
    xs, ys = [], []
    prev = None
    for x in cols:
        rows = np.where(mask[:, x])[0]
        runs = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
        cents = np.array([r.mean() for r in runs])
        if prev is None:
            if prefer == "top":
                k = int(np.argmin(cents))
            elif prefer == "bottom":
                k = int(np.argmax(cents))
            else:
                k = int(np.argmax([len(r) for r in runs]))
        else:
            dist = np.array([0.0 if r_.min() - 2 <= prev <= r_.max() + 2 else min(abs(r_.min() - prev), abs(r_.max() - prev)) for r_ in runs])
            k = int(np.argmin(dist))
            if abs(cents[k] - prev) > max_jump and xs and x - xs[-1] <= 2:
                # a steep curve spans the jump inside one column: take the run touching the previous row
                touch = [i for i, r in enumerate(runs) if r.min() - 2 <= prev <= r.max() + 2]
                if touch:
                    k = touch[0]
            if dist[k] > max_jump:
                # far away from the curve (a tick mark, text, another curve's piece): skip the column,
                # unless the curve has been lost for long (then restart from here)
                if x - xs[-1] <= restart_gap:
                    continue
        r = runs[k]
        if len(r) > vert_run and prev is not None:
            # a steep stretch of the curve inside one column (sharp peak / dip): keep both ends, the
            # one next to the previous point first, so the peak height survives
            a, b = (float(r.min()), float(r.max())) if abs(r.min() - prev) <= abs(r.max() - prev) else (float(r.max()), float(r.min()))
            xs.append(float(x)); ys.append(a)
            xs.append(float(x) + 0.01); ys.append(b); prev = b
            continue
        y = float(r.mean())
        xs.append(float(x)); ys.append(y); prev = y
    return np.array(xs), np.array(ys)


def trace_spectrum(mask, x_range=None):
    """dense spectra with sharp peaks (XRD, noisy XPS): per column the full vertical extent
    of the trace is kept (top and bottom of the run), so a one-pixel-wide peak keeps its
    height; returns xs and the run TOP (peak side) and BOTTOM per column."""
    cols = np.where(mask.any(0))[0]
    if x_range is not None:
        cols = cols[(cols >= x_range[0]) & (cols <= x_range[1])]
    xs, tops, bots = [], [], []
    for x in cols:
        rows = np.where(mask[:, x])[0]
        xs.append(float(x)); tops.append(float(rows.min())); bots.append(float(rows.max()))
    return np.array(xs), np.array(tops), np.array(bots)


def drop_slivers(x, y, min_len, max_jump, gap=12):
    """pieces of a traced curve shorter than min_len columns, cut off from the rest by a gap or a
    jump (crossing points of two other curves, anti-aliased text), are not the curve"""
    if len(x) == 0:
        return x, y
    br = np.where((np.diff(x) > gap) | (np.abs(np.diff(y)) > max_jump))[0] + 1
    keep = np.zeros(len(x), bool)
    for seg in np.split(np.arange(len(x)), br):
        if len(seg) >= min_len:
            keep[seg] = True
    if not keep.any():
        return x, y
    return x[keep], y[keep]


def bridge_gaps(x, y, max_gap):
    """a curve hidden for a short stretch (under another curve, a label, a marker) is continued
    straight across the gap (gaps up to max_gap px); longer gaps stay open"""
    if len(x) < 2 or max_gap <= 0:
        return x, y
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(x)):
        g = x[i] - x[i - 1]
        if 2 < g <= max_gap:
            for t in np.arange(x[i - 1] + 2, x[i], 2.0):
                xs.append(t); ys.append(y[i - 1] + (y[i] - y[i - 1]) * (t - x[i - 1]) / g)
        xs.append(x[i]); ys.append(y[i])
    return np.array(xs), np.array(ys)


def downsample(x, y, n):
    """thin a traced curve to about n points, keeping the extremes of every stretch (sharp peaks
    and dips of spectra survive)"""
    if len(x) <= n:
        return x, y
    edges = np.unique(np.round(np.linspace(0, len(x), n // 2 + 1)).astype(int))
    keep = set()
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = np.arange(a, b)
        keep.update((seg[0], seg[int(np.argmin(y[a:b]))], seg[int(np.argmax(y[a:b]))]))
    keep.add(len(x) - 1)
    idx = np.array(sorted(keep))
    return x[idx], y[idx]


# ----------------------------------------------------------------------------- markers
def marker_centres(mask, ms_px, hollow=False, min_area_frac=0.25, split=True, shape="o"):
    m = mask.copy()
    if hollow:
        m = ndi.binary_closing(m, structure=np.ones((2, 2)))
        f = ndi.binary_fill_holes(m)
        # an outline broken by anti-aliasing does not enclose its hole: close it a little more
        f2 = ndi.binary_fill_holes(ndi.binary_dilation(m, iterations=1))
        m = f | ndi.binary_erosion(f2, iterations=1)
    k = max(2, int(round(ms_px * 0.45)))
    m = ndi.binary_opening(m, structure=np.ones((k, k)))
    lab, n = ndi.label(m)
    out = []
    a1 = max(4.0, {"s": 0.95, "o": 0.78, "p": 0.7, "D": 0.5, "^": 0.45, "v": 0.45}.get(shape, 0.6) * ms_px * ms_px)
    for i in range(1, n + 1):
        yy, xx = np.where(lab == i)
        a = len(xx)
        if a < min_area_frac * a1:
            continue
        kk = int(round(a / a1)) if split else 1
        if kk >= 2 and kk <= 60:
            # fused markers: split along the longer extent
            ext_x, ext_y = np.ptp(xx), np.ptp(yy)
            v = xx if ext_x >= ext_y else yy
            qs = np.quantile(v, (np.arange(kk) + 0.5) / kk)
            lbl = np.argmin(np.abs(v[:, None] - qs[None, :]), 1)
            for q in range(kk):
                sel = lbl == q
                if sel.sum():
                    out.append((float(xx[sel].mean()), float(yy[sel].mean())))
        else:
            out.append((float((xx.min() + xx.max()) / 2), float((yy.min() + yy.max()) / 2)))
    return sorted(out)


def _outline_template(marker, ms):
    """1-px outline of a hollow marker of size ms (px) as a boolean array"""
    from PIL import Image, ImageDraw
    n = int(np.ceil(ms)) + 4
    im = Image.new("L", (n, n), 0); d = ImageDraw.Draw(im)
    c = (n - 1) / 2; r = ms / 2
    if marker in ("o", "circle"):
        d.ellipse([c - r, c - r, c + r, c + r], outline=255, width=1)
    elif marker in ("s", "square"):
        d.rectangle([c - r, c - r, c + r, c + r], outline=255, width=1)
    elif marker in ("^", "triangle"):
        d.polygon([(c, c - r), (c - r, c + r * 0.75), (c + r, c + r * 0.75)], outline=255)
    elif marker == "v":
        d.polygon([(c, c + r), (c - r, c - r * 0.75), (c + r, c - r * 0.75)], outline=255)
    elif marker in ("d", "D", "diamond"):
        d.polygon([(c, c - r), (c + r, c), (c, c + r), (c - r, c)], outline=255)
    elif marker == "<":
        d.polygon([(c - r, c), (c + r * 0.75, c - r), (c + r * 0.75, c + r)], outline=255)
    else:
        d.ellipse([c - r, c - r, c + r, c + r], outline=255, width=1)
    return np.asarray(im) > 0


def markers_by_outline(mask, X, xs, ms_px, marker, P=None, dx=3, min_score=0.45, filled=False):
    """hollow markers at known x: the position (within +-dx px of x, any row) where the mask covers
    the marker's outline best. Lines passing through, other series' markers on top and touching
    markers do not move it - only a whole outline scores high."""
    t = _outline_template(marker, ms_px)
    if filled:
        t = ndi.binary_fill_holes(t)
    th, tw = t.shape
    md = ndi.binary_dilation(mask, iterations=1)
    L, R, T, B = P["frame"] if P is not None else (0, mask.shape[1], 0, mask.shape[0])
    out = []
    for xv in xs:
        px = float(cat_px(P, xv)) if P is not None and P.get("cat_pos") else float(X.to_pixel(xv))
        if px != px:
            continue
        best = None
        for cx in range(int(round(px)) - dx, int(round(px)) + dx + 1):
            x0 = cx - tw // 2
            if x0 < 0 or x0 + tw > mask.shape[1]:
                continue
            strip = md[:, x0:x0 + tw].astype(np.float32)
            sc = ndi.correlate(strip, t.astype(np.float32), mode="constant")[:, tw // 2] / t.sum()
            sc[:int(T) + 1] = 0; sc[int(B):] = 0
            y = int(np.argmax(sc))
            if best is None or sc[y] > best[0]:
                best = (float(sc[y]), cx, y)
        if best and best[0] >= min_score:
            out.append((float(best[1]), float(best[2] - th // 2 + th // 2)))
    return out


def markers_at(mask, X, xs, ms_px, hollow=False, P=None, on_line=False, gap_rows=4):
    """markers known to sit at given x values (one per x, e.g. one per bar / sampling time): in a
    narrow column around each x the marker is the run of mask rows about one marker tall; its
    centre row is the y. Robust for chains of overlapping markers that blob detection cannot split."""
    m = mask
    if hollow:
        c = ndi.binary_closing(m, structure=np.ones((2, 2)))
        m = ndi.binary_fill_holes(c) | ndi.binary_erosion(ndi.binary_fill_holes(ndi.binary_dilation(c, iterations=1)), iterations=1) \
            | ndi.binary_erosion(ndi.binary_fill_holes(ndi.binary_dilation(c, iterations=2)), iterations=2)
    out = []
    for xv in xs:
        px = float(cat_px(P, xv)) if P is not None and P.get("cat_pos") else float(X.to_pixel(xv))
        if on_line:
            # light / hollow markers on their connecting line: the line runs through the marker centre
            rows = np.where(mask[:, int(round(px)) - 1:int(round(px)) + 2].any(1))[0]
            if len(rows):
                runs = np.split(rows, np.where(np.diff(rows) > 2)[0] + 1)

                # where the line enters / leaves the marker: the line rows just left and right of it,
                # each continued towards the marker centre; the marker's run is the one nearest that
                ests = []
                for a0, a1 in ((px - 1.6 * ms_px, px - 0.8 * ms_px), (px + 0.8 * ms_px, px + 1.6 * ms_px)):
                    sub = mask[:, max(0, int(a0)):max(0, int(a1)) + 1]
                    if sub.shape[1] == 0 or not sub.any():
                        continue
                    cy_ = [np.median(np.where(sub[:, j])[0]) for j in range(sub.shape[1]) if sub[:, j].any()]
                    jj = [j for j in range(sub.shape[1]) if sub[:, j].any()]
                    if len(cy_) >= 2:
                        k_, b_ = np.polyfit(np.array(jj) + max(0, int(a0)), cy_, 1)
                        ests.append(k_ * px + b_)
                    else:
                        ests.append(cy_[0])
                if ests:
                    ye = float(np.mean(ests))
                    r = min(runs, key=lambda r: abs(r.mean() - ye))
                else:
                    r = min(runs, key=lambda r: abs(len(r) - ms_px))
                out.append((px, float(np.median(r))))
            continue
        x0, x1 = int(round(px - ms_px * 0.3)), int(round(px + ms_px * 0.3))
        col = m[:, max(0, x0):x1 + 1].mean(1) >= 0.3
        rows = np.where(col)[0]
        if not len(rows):
            continue
        # a curve / line of another colour drawn across the marker splits it: bridge gaps of a line width
        runs = np.split(rows, np.where(np.diff(rows) > gap_rows)[0] + 1)
        runs = [r for r in runs if (r.max() - r.min() + 1) >= 0.35 * ms_px and len(r) >= 0.3 * ms_px]
        if not runs:
            continue

        def line_score(r):
            # a marker on a connecting line has line pixels left / right of it at about its rows
            y0, y1 = max(0, int(r.min()) - 3), int(r.max()) + 4
            a = m[y0:y1, max(0, int(px - 1.6 * ms_px)):max(0, int(px - 0.8 * ms_px))].any()
            b = m[y0:y1, int(px + 0.8 * ms_px):int(px + 1.6 * ms_px)].any()
            return int(a) + int(b)
        r = min(runs, key=lambda r: (-line_score(r), abs(len(r) - ms_px)))
        # centre of the marker: if the run is a chain of two stacked markers take the one ~ms tall
        ext = r.max() - r.min() + 1
        out.append((px, float((r.min() + r.max()) / 2) if ext <= 1.5 * ms_px else float(r.min() + ms_px / 2)))
    return out


def error_bar(img, cx, cy, r, frame, white=215, max_len=200, exclude=(), ink=None):
    """half-length (px) of a vertical error bar through a marker / bar top: walk up from the
    marker edge along the centre column while there is ink; the cap is where it stops."""
    L, R, T, B = frame
    x = int(round(cx))
    g = img[:, max(0, x - 1):x + 2].astype(float).mean(-1).min(1)
    if ink is not None:                                    # only the error bar's own colour counts (bars behind it do not)
        g = np.where(ink[:, max(0, x - 1):x + 2].any(1), 0.0, 255.0)
    for (x0, x1, y0, y1) in exclude:                       # letters / labels above a bar are not the error bar
        if x0 <= x <= x1:
            g[int(y0):int(y1)] = 255
    y = int(round(cy - r - 1)); top = None; gap = 0
    while y > T and cy - y < max_len:
        if g[y] < white:
            top = y; gap = 0
        else:
            gap += 1
            if gap > 2:
                break
        y -= 1
    if top is None:
        return 0.0
    # the cap is the wide ink at the top: its centre row is the error-bar end
    cap = []
    for yy in range(top, min(top + 4, int(cy))):
        row = img[yy, max(0, x - 8):x + 9].astype(float).mean(-1) < white
        if row.sum() >= 5:
            cap.append(yy)
    end = float(np.mean(cap)) if cap else float(top)
    return max(0.0, cy - end)


# ----------------------------------------------------------------------------- bars
def cat_px(P, i):
    """pixel centre of category i (fractional i interpolates). Uneven spacing (bars in pairs):
    the profile lists every centre in "cat_pos"."""
    if P.get("cat_pos"):
        cp = P["cat_pos"]
        return float(np.interp(i, np.arange(len(cp)), cp))
    return axis_of(P, "x").to_pixel(i)


def column_runs(mask, x0, x1):
    """rows covered by the mask in the central part of a bar column (>= half the columns)"""
    cols = mask[:, int(round(x0)):int(round(x1)) + 1]
    if cols.shape[1] == 0:
        return np.zeros(0, int)
    frac = cols.mean(1)
    return np.where(frac >= 0.5)[0]


def bar_value(P, s, mask, i, img):
    if "category_axis" in P:                             # dual-axis bar charts: the category decides the axis
        s = dict(s, axis=P["category_axis"].get(i, s.get("axis", "left")))
    cx = cat_px(P, i) + s.get("offset_px", 0.0)
    w = s.get("width_px", 20)
    if s.get("match", {}).get("outline"):
        # white / patterned bars drawn as an outline: the top edge is the highest dark row in two
        # columns left and right of the centre (the error bar runs through the centre)
        g = img.astype(float).mean(-1)
        L, R, T, B = P["frame"]
        tops = []
        for xx in (cx - 0.3 * w, cx + 0.3 * w):
            col = g[int(T) + 3:int(B) - 1, int(round(xx))]
            dk = np.where(col < 110)[0]
            if len(dk):
                tops.append(dk.min() + int(T) + 3)
        if not tops:
            return None, 0.0
        top = float(np.median(tops))
        val = float(y_value(P, s, top))
        e = error_bar(img, cx, top + 1, 0, P["frame"], exclude=P.get("exclude", ())) if s.get("err") else 0.0
        if e:
            e = abs(float(y_value(P, s, top - e)) - val)
        return val, e
    rows = column_runs(mask, cx - 0.3 * w, cx + 0.3 * w)
    if len(rows) == 0:
        return None, 0.0
    runs = np.split(rows, np.where(np.diff(rows) > s.get("max_row_gap", 6))[0] + 1)
    if P.get("y_segments"):
        seg0 = [g for g in P["y_segments"] if min(g["map"][1], g["map"][3]) <= 0 <= max(g["map"][1], g["map"][3])][0]
        base_px = float(Axis(*seg0["map"]).to_pixel(0.0))
    else:
        Y = y_axis_for(P, s)
        base_px = float(s.get("base_px", Y.to_pixel(s.get("base", 0.0)) if "split_px" not in P else axis_of(P, "y_lower").to_pixel(0.0)))
    # the run touching / nearest the baseline is the bar
    best = min(runs, key=lambda r: min(abs(r.min() - base_px), abs(r.max() - base_px)))
    if s.get("touch_base") and min(abs(best.min() - base_px), abs(best.max() - base_px)) > 4:
        return None, 0.0                                     # nothing standing on the baseline here (no bar)
    if P.get("y_segments"):                                 # bars crossing a break: the whole column of runs
        allr = np.concatenate(runs)
        up = abs(best.max() - base_px) <= abs(best.min() - base_px)
        conn = [r for r in runs if (r.max() <= base_px + 3 if up else r.min() >= base_px - 3)]
        if conn:
            lo, hi = min(r.min() for r in conn), max(r.max() for r in conn)
            # keep only runs chained to the base with gaps no bigger than a break gap
            chain = sorted(conn, key=lambda r: -r.max() if up else r.min())
            edge = chain[0]
            bounds = [g["rows"][1] for g in P["y_segments"][:-1]]
            for r in chain[1:]:
                lo_, hi_ = (r.max(), edge.min()) if up else (edge.max(), r.min())
                # only a gap that holds a segment boundary (the drawn break) is bridged
                if hi_ - lo_ <= s.get("break_gap", 14) and any(lo_ - 3 <= b <= hi_ + 3 for b in bounds):
                    edge = np.r_[edge, r]
                else:
                    break
            tip = float(edge.min()) if up else float(edge.max())
            if abs(tip - base_px) < s.get("min_px", 3):
                return None, 0.0
            return float(y_value(P, s, tip)), 0.0
    if abs(best.max() - base_px) <= abs(best.min() - base_px):
        top = float(best.min())                          # bar grows upwards
        if s.get("edge"):                                # the outline above the fill is the bar's real top
            g = img[:, int(round(cx - 0.3 * w)):int(round(cx + 0.3 * w)) + 1].astype(float).mean(-1).mean(1)
            dk = [yy for yy in range(int(top) - 3, int(top)) if yy >= 0 and g[yy] < 120]
            if dk:
                top = float(np.mean(dk))
        if "split_px" in P:                              # bar crossing a broken axis: the top segment holds its end
            allrows = np.concatenate(runs)
            top = float(allrows.min())
        val = float(y_value(P, s, top))
        e = error_bar(img, cx, top + 1, 0, P["frame"], exclude=P.get("exclude", ())) if s.get("err") else 0.0
        if e:
            e = abs(float(y_value(P, s, top - e)) - val)
    else:
        bot = float(best.max())                          # negative bar
        val = float(y_value(P, s, bot))
        e = 0.0
    return val, e


def stacked_values(P, names, masks, i, S):
    """segment heights of one stacked column: every series' run in the column, measured
    as top-bottom extent in data units (segments are read independently)"""
    out = {}
    for n in names:
        s = S[n]
        cx = cat_px(P, i) + s.get("offset_px", 0.0)
        w = s.get("width_px", 20)
        rows = column_runs(masks[n], cx - 0.3 * w, cx + 0.3 * w)
        if len(rows) == 0:
            out[n] = (0.0, None, None); continue
        # one segment of a colour per column; labels written on the bar split it (gaps up to ~a text
        # line are bridged), markers of the same colour above the bar stay separate
        runs = np.split(rows, np.where(np.diff(rows) > s.get("max_row_gap", 16))[0] + 1)
        r = max(runs, key=lambda q: q.max() - q.min())
        # a thin sliver above / below a gap (marker edge, error-bar cap of the same colour) is not the bar
        sub = np.split(r, np.where(np.diff(r) > 1)[0] + 1)
        while len(sub) > 1 and len(sub[0]) < 4:
            sub = sub[1:]
        while len(sub) > 1 and len(sub[-1]) < 4:
            sub = sub[:-1]
        r = np.concatenate(sub)
        top, bot = float(r.min()) - 0.5, float(r.max()) + 0.5
        yt, yb = float(y_value(P, s, top)), float(y_value(P, s, bot))
        out[n] = (yt - yb, yb, yt)
    return out


# ----------------------------------------------------------------------------- sticks
def sticks(mask, P, s, min_h=3):
    """reference sticks (PDF cards): vertical lines standing on the baseline. The stick columns are
    found just above the baseline; each stick is followed straight up in its own column(s) until it
    ends (hkl labels written above the sticks, separated by a small gap, are not part of it)."""
    Y = y_axis_for(P, s)
    base = float(Y.to_pixel(s.get("base", 0.0)))
    yb = int(round(base))
    band = mask[max(0, yb - 4):yb, :].any(0)
    cols = np.where(band)[0]
    tops = {}
    for x in cols:
        sub = mask[:, x]
        y = yb - 1
        while y > 0 and not sub[y] and y > yb - 5:
            y -= 1
        top = None
        while y >= 0 and sub[y]:
            top = y; y -= 1
        if top is not None and base - top >= min_h:
            tops[x] = top
    xs, hs = [], []
    ks = sorted(tops)
    for g in np.split(np.array(ks), np.where(np.diff(ks) > 1)[0] + 1) if ks else []:
        if not len(g) or len(g) > 6:                      # wider than a stick: an axis / text piece
            continue
        j = min(g, key=lambda c: tops[c])                 # the tallest column of the stick
        xs.append(float(j)); hs.append(float(tops[j]))
    return np.array(xs), np.array(hs)


def pattern_stack(img, P, S):
    """stacked bars whose segments are told apart by their FILL PATTERN (solid black, hatched,
    dotted, empty outline) instead of colour, stacked above AND below zero. Per bar column the
    rows are classified by the share of dark pixels across the bar's inner width; outline rows
    (a dark line 1-2 rows thick) separate segments. Returns {series: {x, bottom, top, y}} where
    y = top - bottom is signed (negative below zero)."""
    names = {s["pattern"]: n for n, s in S.items() if s["kind"] == "pattern_stack"}
    g = img.astype(float).mean(-1)
    L, R, T, B = P["frame"]
    Y = axis_of(P, "y")
    zero = float(Y.to_pixel(0.0))
    out = {n: {"x": [], "y": [], "bottom": [], "top": [], "kind": "stacked"} for n in names.values()}
    w = P.get("bar_width_px", 15)
    for i in range(len(P["categories"])):
        cx = cat_px(P, i)
        x0, x1 = int(round(cx - w / 2 + 2)), int(round(cx + w / 2 - 2))
        dark = (g[:, x0:x1 + 1] < 130).mean(1)
        ink = (g[:, x0:x1 + 1] < 200).mean(1)                # pattern density (grey hatch / dot strokes)
        edge = (g[:, int(round(cx - w / 2)) - 1:int(round(cx - w / 2)) + 2] < 130).any(1) | \
               (g[:, int(round(cx + w / 2)) - 1:int(round(cx + w / 2)) + 2] < 130).any(1)
        rows = np.where(edge[int(T) + 2:int(B) - 1])[0] + int(T) + 2      # the bar's side outlines give its extent
        if not len(rows):
            continue
        runs = np.split(rows, np.where(np.diff(rows) > 2)[0] + 1)
        run = [r for r in runs if r.min() - 3 <= zero <= r.max() + 3]
        if not run:
            continue
        top, bot = int(run[0].min()), int(run[0].max())
        # segment boundaries: thin dark lines across the full width
        full = dark >= 0.85
        y = top
        segs = []
        while y <= bot:
            if full[y]:
                y2 = y
                while y2 + 1 <= bot and full[y2 + 1]:
                    y2 += 1
                if y2 - y + 1 >= 4:                       # a solid black segment
                    segs.append((y, y2, "solid"))
                y = y2 + 1
                continue
            y2 = y
            while y2 + 1 <= bot and not full[y2 + 1]:
                y2 += 1
            share = float(ink[y:y2 + 1].mean())
            if y2 - y + 1 >= 2:
                kind = "hatched" if share >= P.get("hatch_share", 0.38) else ("dotted" if share >= P.get("dot_share", 0.08) else "empty")
                segs.append((y - 0.5, y2 + 0.5, kind))
            y = y2 + 1
        # merge neighbours of the same kind (a boundary line drawn inside one fill), split at zero
        merged = []
        for a, b_, k in segs:
            if merged and merged[-1][2] == k and a - merged[-1][1] <= 3:
                merged[-1] = (merged[-1][0], b_, k)
            else:
                merged.append((a, b_, k))
        for a, b_, k in merged:
            n = names.get(k)
            if n is None:
                continue
            for lo_px, hi_px in ((a, min(b_, zero)), (max(a, zero), b_)):
                if hi_px - lo_px < 2:
                    continue
                vt, vb = float(Y.to_value(lo_px)), float(Y.to_value(hi_px))
                above = (lo_px + hi_px) / 2 < zero
                d = out[n]
                d["x"].append(i)
                d["top"].append(round(vt, 6)); d["bottom"].append(round(vb, 6))
                d["y"].append(round((vt - vb) if above else -(vt - vb), 6))
    for d in out.values():                                # 'bottom' = the end nearer zero, as matplotlib draws it
        d["bottom"] = [b if yv >= 0 else t for b, t, yv in zip(d["bottom"], d["top"], d["y"])]
    return out


# ----------------------------------------------------------------------------- panel
def extract_panel(img, P, image_path=None):
    """digitize one panel -> {"series": {...}} (data units of the panel's axes)"""
    if P.get("engine") == "autodig":
        return _autodig_panel(image_path, P)
    X = axis_of(P, "x")
    S = P["series"]
    out = {"series": {}}
    if any(s["kind"] == "pattern_stack" for s in S.values()):
        out["series"].update(pattern_stack(img, P, S))
        return out
    masks = {n: inside(series_mask(img, s), P, s.get("exclude", ()),
                      s.get("frame_pad", 0 if s["kind"] in ("bar", "stacked") else None)) for n, s in S.items()}
    # pixels of one series are removed from the others when the profile says so (e.g. black text vs. black curve)
    for n, s in S.items():
        if s.get("not_on_bars") and P.get("cat_pos"):       # markers: the bar columns are bars, not markers
            for i, cx in enumerate(P["cat_pos"]):
                w = max((t.get("width_px", 0) for t in S.values() if t["kind"] in ("bar", "stacked")), default=0)
                masks[n][:, int(cx - w / 2) - 1:int(cx + w / 2) + 2] = False
        for other in s.get("minus", []):
            masks[n] &= ~ndi.binary_dilation(masks[other], iterations=1)
    stacked = [n for n, s in S.items() if s["kind"] == "stacked"]
    if stacked:
        for i in range(len(P["categories"])):
            vals = stacked_values(P, stacked, masks, i, S)
            for n in stacked:
                d = out["series"].setdefault(n, {"x": [], "y": [], "kind": "stacked", "bottom": []})
                v, b, _ = vals[n]
                d["x"].append(i); d["y"].append(round(v, 6)); d["bottom"].append(None if b is None else round(b, 6))
    for n, s in S.items():
        k = s["kind"]
        m = masks[n]
        if k == "stacked":
            continue
        if k == "bar":
            xs, ys, es = [], [], []
            for i in (s["at"] if "at" in s else range(len(P["categories"]))):
                if i in s.get("skip", []):
                    continue
                v, e = bar_value(P, s, m, i, img)
                if v is None:
                    continue
                xs.append(i); ys.append(round(v, 6)); es.append(round(e, 6))
            d = {"x": xs, "y": ys, "kind": "bar"}
            if s.get("err"):
                d["yerr"] = es
            out["series"][n] = d
        elif k == "line":
            xr = s.get("x_px")
            if s.get("spectrum"):
                px, top, bot = trace_spectrum(m, xr)
                # peaks point up: the run top; a baseline-crossing trace pointing down uses the bottom
                pk = s.get("peaks", "up")
                py = top if pk == "up" else (bot if pk == "down" else (top + bot) / 2)
            else:
                px, py = trace_curve(m, xr, prefer=s.get("prefer", "continuity"), max_jump=s.get("max_jump", 25),
                                     vert_run=s.get("vert_run", 5))
            if len(px) == 0:
                out["series"][n] = {"x": [], "y": [], "kind": "line"}; continue
            if not s.get("spectrum"):          # spectra keep one-column peaks
                px, py = drop_slivers(px, py, s.get("min_seg_px", 6), s.get("max_jump", 25))
            px, py = bridge_gaps(px, py, s.get("gap_fill_px", 40))
            px, py = downsample(px, py, s.get("samples", 300))
            out["series"][n] = {"x": [round(float(v), 6) for v in X.to_value(px)],
                                "y": [round(float(v), 6) for v in y_value(P, s, py)], "kind": "line"}
        elif k == "marker":
            if s.get("expected_x_from") and s["expected_x_from"] in out["series"]:
                s = dict(s, expected_x=out["series"][s["expected_x_from"]]["x"])   # same x as another series' markers
            if s.get("expected_x") is not None and s.get("outline_match"):
                cs = markers_by_outline(m, X, s["expected_x"], s.get("ms_px", 8), s.get("marker", "o"), P,
                                        min_score=s.get("min_score", 0.45), filled=not s.get("hollow", False))
            elif s.get("expected_x") is not None:
                cs = markers_at(m, X, s["expected_x"], s.get("ms_px", 8), s.get("hollow", False), P, s.get("on_line", False))
            else:
              cs = marker_centres(m, s.get("ms_px", 8), s.get("hollow", False), split=s.get("split", True),
                                  min_area_frac=s.get("min_area_frac", 0.25), shape={"circle": "o", "square": "s", "triangle": "^"}.get(s.get("marker", "o"), s.get("marker", "o")))
            if s.get("max_n"):
                cs = cs[:s["max_n"]]
            if P.get("cat_pos"):
                cp = np.asarray(P["cat_pos"], float)
                xs = [float(np.interp(c[0], cp, np.arange(len(cp)))) for c in cs]
            else:
                xs = [float(X.to_value(c[0])) for c in cs]
            ys = [float(y_value(P, s, c[1])) for c in cs]
            d = {"x": [round(v, 6) for v in xs], "y": [round(v, 6) for v in ys], "kind": "marker"}
            if s.get("err"):
                es = []
                for (cx, cy), yv in zip(cs, ys):
                    ink = series_mask(img, s) | dark_mask(img, max_gray=90, max_sat=50) if s.get("err_ink", True) else None
                    e = error_bar(img, cx, cy, s.get("ms_px", 8) / 2, P["frame"], exclude=P.get("exclude", ()), ink=ink)
                    es.append(round(abs(float(y_value(P, s, cy - e)) - yv), 6) if e > 1 else 0.0)
                d["yerr"] = es
            order = np.argsort(d["x"])
            for kk in list(d):
                if isinstance(d[kk], list):
                    d[kk] = [d[kk][j] for j in order]
            out["series"][n] = d
        elif k == "stick":
            px, py = sticks(m, P, s)
            out["series"][n] = {"x": [round(float(v), 6) for v in X.to_value(px)],
                                "y": [round(float(v), 6) for v in y_value(P, s, py)], "kind": "stick"}
        else:
            raise ValueError(f"unknown series kind {k}")
    return out


def _autodig_panel(image_path, P):
    """3-D surfaces / filled contour maps: the autodig package of notebook 03 (camera fit or
    colour bar), driven by the panel's hand-written spec. The calibration it measured is kept."""
    import json
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "03_auto_vlm_digitization"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from autodig.extract import pipeline
    from autodig.spec import renormalize
    r = pipeline(str(image_path), renormalize(json.loads(json.dumps(P["spec"]))))
    out = {"series": {}, "autodig_status": r.get("status"), "autodig_reasons": r.get("status_reasons")}
    for n, d in (r.get("series") or {}).items():
        keep = {k: d[k] for k in ("x", "y", "z", "values", "grid_x", "grid_y", "grid_z", "kind") if k in d}
        out["series"][n] = keep
    out["autodig_result"] = {k: r[k] for k in ("camera", "colorbar", "table") if k in r}
    return out


def extract_figure(F, image_path):
    img = load_rgb(image_path)
    panels = {}
    for P in F["panels"]:
        d = extract_panel(img, P, image_path)
        d["role"] = P.get("role", "main")
        if P.get("printed"):                                 # numbers printed on the figure (fit parameters): exact values
            d["printed_fit_parameters"] = P["printed"]
            d["printed_equations"] = P.get("printed_equations", {})
        if P.get("categories"):
            d["categories"] = list(P["categories"])
        panels[P["id"]] = d
    for pid, vals in (F.get("printed_values") or {}).items():   # numbers printed on the figure (exact)
        panels[pid]["printed_values"] = vals
    roles = {p.get("role", "main") for p in F["panels"]}
    layout = "main+inset" if "inset" in roles else ("panels" if len(F["panels"]) > 1 else "single")
    return {"figure": F["id"], "layout": layout, "panels": panels}
