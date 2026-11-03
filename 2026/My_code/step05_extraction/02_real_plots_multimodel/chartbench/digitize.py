"""Classical (non-ML) chart digitizer: axis calibration + color-based tracing.

This is the "WebPlotDigitizer-style" method from the step 4b plan. It is
*semi-automatic*: each figure in ``figures.py`` carries a hand-checked axis
calibration (two pixel->value reference points per axis, read off the tick
marks) and the RGB color of every series. Given those, extraction is fully
mechanical:

* ``line`` series   -> for every pixel column inside the plot frame, take the
                       median row of pixels matching the series color.
* ``marker`` series -> isolate pixels of the series color, fill hollow markers,
                       erase thin connecting lines with a morphological opening,
                       and take the centroid of every remaining blob.

Because it needs that per-figure calibration, it is not a fair "zero-shot"
competitor to the VLMs - treat it as the precision upper bound / reference
reader, which is exactly why its output on the *original* figures is used as
the first-version data the rebuilt plots are drawn from.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


# ----------------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------------
class Axis:
    """Linear (or log) pixel <-> value map from two reference ticks."""

    def __init__(self, p0, v0, p1, v1, log=False):
        self.p0, self.p1, self.log = float(p0), float(p1), log
        f = np.log10 if log else (lambda v: v)
        self.v0, self.v1 = f(float(v0)), f(float(v1))

    def to_value(self, p):
        v = self.v0 + (np.asarray(p, float) - self.p0) * (self.v1 - self.v0) / (self.p1 - self.p0)
        return 10 ** v if self.log else v

    def to_pixel(self, v):
        v = np.log10(v) if self.log else np.asarray(v, float)
        return self.p0 + (v - self.v0) * (self.p1 - self.p0) / (self.v1 - self.v0)


def load_rgb(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.int16)


def color_mask(img, rgb, tol=60, min_sat=None):
    """Pixels within Euclidean RGB distance ``tol`` of ``rgb``.

    For near-black / gray series pass ``min_sat=None`` and a gray rgb; for
    colored series an extra saturation floor keeps anti-aliased gray text out.
    """
    d = np.sqrt(((img - np.asarray(rgb, np.int16)) ** 2).sum(-1))
    m = d <= tol
    if min_sat is not None:
        sat = img.max(-1) - img.min(-1)
        m &= sat >= min_sat
    return m


def hue_mask(img, hue_deg, hue_tol=25, min_sat=0.25, min_val=0.35):
    """Pixels whose HSV hue is within ``hue_tol`` degrees of ``hue_deg``.

    More robust than RGB distance for anti-aliased marker outlines: a pale pink
    edge pixel of a magenta marker keeps the magenta hue even though its RGB
    is far from (255, 0, 255)."""
    rgb = img.astype(float) / 255.0
    mx, mn = rgb.max(-1), rgb.min(-1)
    delta = mx - mn
    sat = np.where(mx > 0, delta / np.maximum(mx, 1e-9), 0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    d = np.maximum(delta, 1e-9)
    hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    dh = np.abs((hue - hue_deg + 180) % 360 - 180)
    return (dh <= hue_tol) & (sat >= min_sat) & (mx >= min_val)


def dark_mask(img, max_gray=110, max_sat=40):
    """Near-black / dark-gray pixels (black series, axis-colored series)."""
    sat = img.max(-1) - img.min(-1)
    return (img.mean(-1) <= max_gray) & (sat <= max_sat)


def restrict(mask, frame, exclude=()):
    """Keep only pixels inside ``frame`` = (left, right, top, bottom) and
    outside every ``exclude`` box (legends, panel labels, annotation text)."""
    L, R, T, B = frame
    out = np.zeros_like(mask)
    out[T:B, L:R] = mask[T:B, L:R]
    for (x0, x1, y0, y1) in exclude:
        out[y0:y1, x0:x1] = False
    return out


# ----------------------------------------------------------------------------
# tracing
# ----------------------------------------------------------------------------
def trace_line(mask, step=1):
    """Column-wise median row of a thin colored curve. Returns pixel (xs, ys)."""
    xs, ys = [], []
    cols = np.where(mask.any(0))[0]
    for x in cols[::step]:
        rows = np.where(mask[:, x])[0]
        xs.append(x)
        ys.append(np.median(rows))
    return np.array(xs, float), np.array(ys, float)


def find_markers(mask, open_size=3, min_area=6, max_area=400, fill=True, split_overlaps=True):
    """Centroids of marker blobs, with connecting lines removed."""
    m = mask.copy()
    if fill:
        m = ndi.binary_closing(m, structure=np.ones((2, 2)))
        m = ndi.binary_fill_holes(m)
    if open_size > 1:
        m = ndi.binary_opening(m, structure=np.ones((open_size, open_size)))
    lab, n = ndi.label(m)
    blobs = []
    for i in range(1, n + 1):
        yy, xx = np.where(lab == i)
        if min_area <= len(xx) <= max_area:
            blobs.append((xx.astype(float), yy.astype(float)))
    if not blobs:
        return []
    typical = np.median([len(b[0]) for b in blobs])
    pts = []
    for xx, yy in blobs:
        k = int(round(len(xx) / typical)) if split_overlaps else 1
        if k >= 2:  # two (or more) overlapping markers fused into one blob
            pts.extend(_split_blob(xx, yy, k))
        else:
            # bounding-box centre = the plotting anchor of a marker (a triangle's
            # area centroid sits ~1/6 of its height off the anchor)
            pts.append(((xx.min() + xx.max()) / 2, (yy.min() + yy.max()) / 2, len(xx)))
    pts.sort()
    return pts


def _split_blob(xx, yy, k):
    """Split a fused blob into k markers along its principal axis (1-D k-means)."""
    P = np.c_[xx, yy]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    proj = (P - c) @ vt[0]
    centers = np.quantile(proj, (np.arange(k) + 0.5) / k)
    for _ in range(20):
        assign = np.argmin(np.abs(proj[:, None] - centers[None]), 1)
        centers = np.array([proj[assign == j].mean() if (assign == j).any() else centers[j] for j in range(k)])
    return [(xx[assign == j].mean(), yy[assign == j].mean(), int((assign == j).sum())) for j in range(k) if (assign == j).any()]


def error_bar_extent(img, cx, cy, marker_r, frame, white=225, gap=3, half_width=1):
    """Half-length (pixels) of a vertical error bar through marker (cx, cy).

    Scans up and down from the marker edge through non-white pixels in a
    narrow column; the bar ends after ``gap`` consecutive white rows. Error bars
    are symmetric in these figures, so the shorter side is returned (the longer
    side is usually contaminated by a neighbouring curve)."""
    L, R, T, B = frame
    x0, x1 = int(round(cx)) - half_width, int(round(cx)) + half_width + 1
    gray = img[:, x0:x1].mean(-1).min(-1)
    ext = []
    for direction in (-1, 1):
        y = int(round(cy)) + direction * int(marker_r)
        last_dark = y
        whites = 0
        while T < y < B and whites < gap:
            if gray[y] < white:
                last_dark = y
                whites = 0
            else:
                whites += 1
            y += direction
        ext.append(abs(last_dark - cy))
    return min(ext)


def to_data(xs, ys, xaxis: Axis, yaxis: Axis):
    return xaxis.to_value(xs), yaxis.to_value(ys)


def resample_line(x, y, n=200, xmin=None, xmax=None):
    """Sort, de-duplicate and resample a traced curve on an even x grid."""
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    ux, idx = np.unique(x, return_index=True)
    uy = np.array([np.median(y[x == v]) for v in ux])
    lo = ux.min() if xmin is None else max(xmin, ux.min())
    hi = ux.max() if xmax is None else min(xmax, ux.max())
    gx = np.linspace(lo, hi, n)
    return gx, np.interp(gx, ux, uy)


# ----------------------------------------------------------------------------
# figure-level extraction (uses the per-figure config in figures.py)
# ----------------------------------------------------------------------------
def _axis(c, key):
    return Axis(*c[key])


def _series_mask(img, s, cfg):
    if s.get("dark"):
        return dark_mask(img, max_gray=95, max_sat=45)
    return hue_mask(img, s["hue"], 22, cfg.get("sat", 0.35), 0.4)


def extract_figure(fig_id, image_path):
    """Run the classical digitizer on one figure image -> dataset dict.

    dataset = {"figure": id, "series": {name: {"x": [...], "y": [...],
               "yerr": [...] (optional), "flags": [...] (optional)}}}
    """
    from .figures import FIGURES  # local import to avoid a cycle

    F = FIGURES[fig_id]
    c, cfg = F["calib"], F["digitize"]
    img = load_rgb(image_path)
    if F["chart_type"].startswith("bar"):
        out = extract_bars(F, img); out["figure"] = fig_id
        return out
    if F["chart_type"] == "pie":
        out = extract_pie(F, img); out["figure"] = fig_id
        return out
    X = _axis(c, "x")
    excl = cfg.get("exclude", [])
    out = {"figure": fig_id, "source": "classical", "series": {}}
    raw_masks, yaxes = {}, {}

    for name, s in F["series"].items():
        ax = s.get("axis", "left")
        if ax == "right":
            Y = _axis(c, "y_right")
        elif ax == "upper":
            Y = _axis(c, "y_upper")
        elif ax == "lower":
            Y = _axis(c, "y_lower")
        else:
            Y = _axis(c, "y")
        m = restrict(_series_mask(img, s, cfg), cfg["inner"], excl)
        L_, R_, T_, B_ = c["frame"]
        raw_masks[name] = restrict(_series_mask(img, s, cfg), (L_ + 2, R_ - 1, T_ + 2, B_ - 2), excl)
        yaxes[name] = Y

        if s["kind"] == "line":
            xs, ys = trace_line(m)
            if len(xs) < 5:
                continue
            vx, vy = X.to_value(xs), Y.to_value(ys)
            vx, vy = _extend_to_frame(vx, vy, X, c["frame"])
            gx, gy = resample_line(vx, vy, n=cfg.get("line_samples", 200))
            out["series"][name] = {"x": gx.round(4).tolist(), "y": gy.round(4).tolist()}
            continue

        # ---- marker series
        if s.get("dark") and cfg.get("markers_through_curves"):
            # black markers crossed by colored fit curves: re-attach the curve
            # pixels that sit right next to dark pixels before blob finding
            curve = np.zeros_like(m)
            for other in F["series"].values():
                if other.get("hue") is not None:
                    curve |= hue_mask(img, other["hue"], 22, 0.5, 0.5)
            curve = restrict(curve, cfg["inner"], excl)
            m = m | (curve & ndi.binary_dilation(m, np.ones((5, 5))))
        pts = find_markers(m, open_size=cfg.get("open_size", 4), min_area=cfg.get("min_area", 10),
                           max_area=cfg.get("max_area", 400), fill=cfg.get("fill", True))
        if pts:  # drop debris blobs much smaller than a typical marker
            typ = np.median([p[2] for p in pts])
            pts = [p for p in pts if p[2] >= 0.45 * typ]
        if ax in ("upper", "lower"):  # broken axis: keep only blobs in this segment
            sp = c["split_px"]
            pts = [p for p in pts if (p[1] < sp) == (ax == "upper")]
        px = np.array([p[0] for p in pts]); py = np.array([p[1] for p in pts])
        vx, vy = X.to_value(px), Y.to_value(py)
        flags = ["detected"] * len(vx)
        yerr = None
        if cfg.get("error_bars") and len(px):
            e_px = np.array([error_bar_extent(img, a, b, cfg.get("marker_r", 6), cfg["inner"]) for a, b in zip(px, py)])
            e_px = np.where(e_px <= cfg.get("marker_r", 6) + 1.5, 0.0, e_px)
            med = np.median(e_px[e_px > 0]) if (e_px > 0).any() else 0
            e_px = np.where(e_px > 2.2 * med, med, e_px)  # contaminated by a neighbour -> typical
            yerr = (e_px * abs(Y.v1 - Y.v0) / abs(Y.p1 - Y.p0)).round(3)

        if "expected_x" in cfg:  # snap to the known sampling grid
            vx, vy, flags, yerr = _snap(cfg["expected_x"], vx, vy, yerr)
            # markers the blob finder missed (e.g. half-hidden at the axis
            # origin, or cut by a curve): look in a narrow column at that x
            L, R, T, B = c["frame"]
            raw = restrict(_series_mask(img, s, cfg), (L + 2, R - 1, T + 2, B - 2), excl)
            for i in np.where(np.isnan(vy))[0]:
                yy = _local_search(raw, X.to_pixel(vx[i]), cfg.get("marker_r", 5))
                if yy is not None:
                    vy[i] = Y.to_value(yy); flags[i] = "local_search"
        rec = {"x": np.round(vx, 4).tolist(), "y": np.round(vy, 5).tolist(), "flags": flags}
        if yerr is not None:
            rec["yerr"] = np.round(yerr, 3).tolist()
        out["series"][name] = rec

    _fill_occluded(F, out, raw_masks, X, yaxes, cfg.get("marker_r", 5))
    if fig_id == "n2":
        _split_branches(F, out)
    return out


def _extend_to_frame(vx, vy, X, frame):
    """A curve that starts/ends within a few pixels of the frame was only cut by
    the frame mask - extend it linearly to the frame edge."""
    o = np.argsort(vx); vx, vy = vx[o], vy[o]
    lo, hi = sorted([float(X.to_value(frame[0])), float(X.to_value(frame[1]))])
    span = hi - lo
    if len(vx) > 4 and 0 < vx[0] - lo < 0.02 * span:
        k = (vy[3] - vy[0]) / (vx[3] - vx[0])
        vx = np.r_[lo, vx]; vy = np.r_[vy[0] - k * (vx[1] - lo), vy]
    if len(vx) > 4 and 0 < hi - vx[-1] < 0.02 * span:
        k = (vy[-1] - vy[-4]) / (vx[-1] - vx[-4])
        vx = np.r_[vx, hi]; vy = np.r_[vy, vy[-1] + k * (hi - vx[-2])]
    return vx, vy


def _snap(expected, vx, vy, yerr):
    ex = np.asarray(expected, float)
    tol = 0.35 * np.min(np.diff(np.sort(ex))) if len(ex) > 1 else 1.0
    ys, fl, es = [], [], []
    for x in ex:
        if len(vx):
            j = np.argmin(np.abs(vx - x))
            if abs(vx[j] - x) <= tol:
                ys.append(vy[j]); fl.append("detected")
                es.append(None if yerr is None else yerr[j])
                continue
        ys.append(np.nan); fl.append("missing"); es.append(None if yerr is None else 0.0)
    return ex, np.array(ys, float), fl, (None if yerr is None else np.array(es, float))


def _local_search(mask, px, r):
    """Vertical centre of the biggest compact run of series pixels in the
    columns px-r..px+r (None if nothing marker-sized is there)."""
    x0, x1 = int(round(px - r)), int(round(px + r)) + 1
    x0 = max(x0, 0)
    col = mask[:, x0:x1].sum(1)
    rows = np.where(col >= 2)[0]
    if len(rows) < 3:
        return None
    runs, cur = [], [rows[0]]
    for a in rows[1:]:
        if a - cur[-1] <= 2:
            cur.append(a)
        else:
            runs.append(cur); cur = [a]
    runs.append(cur)
    best = max(runs, key=lambda rr: col[rr].sum())
    if len(best) < 3:
        return None
    return float(np.average(best, weights=col[best]))


def _fill_occluded(F, out, raw_masks, X, yaxes, r):
    """A marker hidden under another series' marker -> borrow the value of the
    marker it is hiding behind (flag 'occluded'). The candidate is the other
    series' marker at the same x whose surroundings show the most pixels of
    *this* series' colour peeking out (e.g. corners of a square behind a
    circle). With no candidate, interpolate between the series' own points."""
    names = list(out["series"])
    anchored = {n: np.array(out["series"][n]["y"], float) for n in names}
    for n in names:
        rec = out["series"][n]
        y = anchored[n].copy()
        if not np.isnan(y).any():
            continue
        xs = np.array(rec["x"], float)
        for i in np.where(np.isnan(y))[0]:
            best, best_score = None, -1
            for o in names:
                if o == n or F["series"][o].get("axis", "left") != F["series"][n].get("axis", "left"):
                    continue
                ox, oy = np.array(out["series"][o]["x"], float), anchored[o]
                k = np.argmin(np.abs(ox - xs[i]))
                if abs(ox[k] - xs[i]) > 1e-6 or np.isnan(oy[k]):
                    continue
                px, py = int(round(float(X.to_pixel(xs[i])))), int(round(float(yaxes[n].to_pixel(oy[k]))))
                win = raw_masks[n][max(py - r - 2, 0):py + r + 3, max(px - r - 2, 0):px + r + 3]
                score = int(win.sum())
                if score > best_score:
                    best, best_score = oy[k], score
            if best is not None:
                y[i] = float(best); rec["flags"][i] = "occluded"
        good = ~np.isnan(y)
        for i in np.where(~good)[0]:
            y[i] = float(np.interp(xs[i], xs[good], y[good])); rec["flags"][i] = "interpolated"
        rec["y"] = np.round(y, 5).tolist()


def _split_branches(F, out):
    """N2 isotherm: one blue marker set -> lower (adsorption) / upper (desorption)."""
    rec = out["series"].pop("adsorption")
    out["series"].pop("desorption", None)
    x, y = np.array(rec["x"]), np.array(rec["y"])
    o = np.argsort(x); x, y = x[o], y[o]
    split = F["digitize"]["branch_split_x"]
    lo_x, lo_y, hi_x, hi_y = [], [], [], []
    i = 0
    while i < len(x):
        if x[i] < split:
            lo_x.append(x[i]); lo_y.append(y[i]); i += 1; continue
        # pair markers that share (almost) the same pressure
        if i + 1 < len(x) and abs(x[i + 1] - x[i]) < 0.012:
            a, b = sorted([y[i], y[i + 1]])
            xm = (x[i] + x[i + 1]) / 2
            lo_x.append(x[i]); lo_y.append(a); hi_x.append(x[i + 1]); hi_y.append(b)
            i += 2
        else:  # unpaired: decide by distance to the current lower branch trend
            if lo_y and hi_y and abs(y[i] - hi_y[-1]) < abs(y[i] - lo_y[-1]):
                hi_x.append(x[i]); hi_y.append(y[i])
            else:
                lo_x.append(x[i]); lo_y.append(y[i])
            i += 1
    # loop closes: desorption starts where adsorption ends and rejoins at the split
    if lo_x:
        hi_x = [lo_x[-1]] + hi_x[::-1] if hi_x and hi_x[-1] < lo_x[-1] else hi_x
    out["series"]["adsorption"] = {"x": np.round(lo_x, 4).tolist(), "y": np.round(lo_y, 4).tolist()}
    hx, hy = np.array(hi_x), np.array(hi_y)
    o = np.argsort(hx)
    out["series"]["desorption"] = {"x": np.round(hx[o], 4).tolist(), "y": np.round(hy[o], 4).tolist()}


# ----------------------------------------------------------------------------
# bar charts
# ----------------------------------------------------------------------------
def _bar_axis(F, cat_index, top_px):
    """Which y-axis a bar is read against: the figure's per-category axis
    (dual-axis charts), else the series axis; on a broken axis the bar top
    decides the segment."""
    c = F["calib"]
    ax = F.get("category_axis", {}).get(cat_index, "left")
    if ax == "left" and "split_px" in c and "y_upper" in c:
        ax = "upper" if top_px < c["split_px"] else "lower"
    key = {"left": "y", "right": "y_right", "upper": "y_upper", "lower": "y_lower"}[ax]
    return ax, Axis(*c[key])


def extract_bars(F, img):
    """Bar charts. For every series and every category: the bar sits at
    (category centre + series offset); its fill colour is found in a window of
    the bar's width, the bar top is the highest filled row (robust to hatching:
    20th percentile over the bar's columns, minus the outline), and the error
    bar is the run of dark pixels in the bar's centre column that crosses the
    bar top (half-length above the top = the error)."""
    c, cfg = F["calib"], F["digitize"]
    X = Axis(*c["x"])
    L, R, T, B = c["frame"]
    excl = cfg.get("exclude", [])
    gray = img.mean(-1)
    out = {"figure": None, "source": "classical", "series": {}}
    ncat = len(F["categories"])
    for name, s in F["series"].items():
        if "hue" in s:   # gradient fills: match the hue, not one RGB value
            m = hue_mask(img, s["hue"], s.get("hue_tol", 15), s.get("min_sat", 0.25), s.get("min_val", 0.4))
        else:
            m = color_mask(img.astype(np.int32), s["fill_rgb"], s.get("tol", 40), s.get("min_sat", 15))
        m = restrict(m, (int(L) + 2, int(R) - 1, int(T) + 1, int(B) - 1), excl)
        xs, ys, es, flags = [], [], [], []
        half = s["width_px"] / 2
        for i in range(ncat):
            cx = float(X.to_pixel(i)) + s["offset_px"]
            x0, x1 = int(round(cx - 0.6 * half)), int(round(cx + 0.6 * half)) + 1
            tops = [np.where(m[:, x])[0].min() for x in range(x0, x1) if m[:, x].any()]
            if len(tops) < 0.3 * (x1 - x0):
                xs.append(i); ys.append(0.0); es.append(0.0); flags.append("no bar found (0)")
                continue
            top = float(np.percentile(tops, 20)) - cfg.get("outline_px", 1.0)
            ax, Y = _bar_axis(F, i, top)
            val = float(Y.to_value(top))
            # error bar: dark run in the centre column(s) that reaches the bar top
            e = 0.0
            if cfg.get("error_bars", False):
                ccol = int(round(cx))
                dark = (gray[:, ccol - 1:ccol + 2] < cfg.get("err_gray", 100)).any(1)
                r0 = int(round(top))
                if dark[max(r0 - 3, 0):r0 + 4].any():
                    y = r0 + int(np.argmax(dark[r0 - 3:r0 + 4])) - 3
                    while y > T and dark[y - 1]:
                        y -= 1
                    e_px = top - y
                    if e_px > cfg.get("min_err_px", 1.2):
                        e = abs(float(Y.to_value(y)) - val)
            xs.append(i); ys.append(round(val, 5)); es.append(round(e, 5)); flags.append(f"detected ({ax} axis)")
        rec = {"x": xs, "y": ys, "flags": flags}
        if cfg.get("error_bars", False):
            rec["yerr"] = es
        out["series"][name] = rec
    return out


# ----------------------------------------------------------------------------
# pie charts (3D, tilted)
# ----------------------------------------------------------------------------
def pie_point(cal, theta):
    """Point of the tilted top face at angle theta (rad, counter-clockwise from 3 o'clock).
    The top face is drawn in perspective, so its four half-radii (left, right, up,
    down from the apex) differ; each quadrant is an ellipse quarter."""
    cx, cy = cal["apex"]
    l, r, u, d = cal["left"], cal["right"], cal["up"], cal["down"]
    dx, dy = np.cos(theta), np.sin(theta)
    return cx + np.where(dx >= 0, r, l) * dx, cy - np.where(dy >= 0, u, d) * dy


def extract_pie(F, img, n_angles=3600):
    """Pie charts: classify every pixel of the top face by slice colour, walk
    around the apex on 25 concentric rings (35-92 % of the radius; the slice
    labels printed inside a slice are out-voted) and turn the angular share of
    each colour into a percentage."""
    cal, names = F["calib"]["pie"], list(F["categories"])
    cols = np.array([F["slices"][n]["rgb"] for n in names], float)
    d = np.sqrt(((img[..., None, :].astype(float) - cols[None, None]) ** 2).sum(-1))
    k = d.argmin(-1); k[d.min(-1) > F["digitize"].get("tol", 90)] = -1
    H, W = k.shape
    th = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    votes = []
    for rr in np.linspace(0.35, 0.92, 25):
        cx, cy = cal["apex"]
        x, y = pie_point(cal, th)
        x = cx + (x - cx) * rr; y = cy + (y - cy) * rr
        votes.append(k[np.clip(np.round(y).astype(int), 0, H - 1), np.clip(np.round(x).astype(int), 0, W - 1)])
    V = np.array(votes)
    lab = np.array([np.bincount(v[v >= 0], minlength=len(names)).argmax() if (v >= 0).any() else -1 for v in V.T])
    good = lab >= 0
    share = [100.0 * (lab[good] == i).sum() / good.sum() for i in range(len(names))]
    # slice boundaries (for the QA overlay)
    edges = [float(np.degrees(th[j])) for j in range(n_angles) if lab[j] != lab[j - 1]]
    return {"figure": None, "source": "classical",
            "series": {F["value_series"]: {"x": list(range(len(names))), "y": [round(v, 3) for v in share],
                                           "flags": ["measured slice angle"] * len(names)}},
            "boundaries_deg": edges}
