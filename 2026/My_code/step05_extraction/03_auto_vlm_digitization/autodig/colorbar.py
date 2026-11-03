"""Colour scales (colorbars): find the bar, read its scale, turn colours into values.

Used by heatmaps, 2-D fields (filled contours / colour maps), 3-D scatter and
3-D surfaces. The VLM reads the colorbar's tick labels; the pixels find

  * the bar: a long, narrow rectangle whose colour changes along its length
    and stays the same across it (continuous or stepped scales);
  * the tick-label positions next to it (same code as the plot axes);
  * the colour at every pixel along the bar -> a lookup table colour -> value.

`ColorScale.value(rgb)` returns the value of the nearest colour in the table and
the colour distance (a large distance = this colour is not on the scale).
For shaded 3-D surfaces `shaded=True` compares colours up to a brightness
factor (lighting darkens a colour without changing its hue).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from . import detect as D
from .calibrate import calibrate_axis, parse_number


def _not_white(img, tol=4):
    return (255 - img.astype(np.int16)).max(-1) > tol


def find_colorbars(img, min_len_frac=0.18):
    """Candidate colorbars: [{box: (x0, y0, x1, y1), orient, score}] best first."""
    H, W = img.shape[:2]
    m = _not_white(img)
    lab, n = ndi.label(m)
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        h, w = y1 - y0, x1 - x0
        long_, short = max(h, w), min(h, w)
        if short < 5 or long_ < min_len_frac * min(H, W) or long_ < 4 * short:
            continue
        comp = lab[sl] == i
        # trim protruding tick marks: keep the rows / columns the component mostly fills
        cols_ok = np.where(comp.mean(0) >= 0.5)[0]; rows_ok = np.where(comp.mean(1) >= 0.5)[0]
        if len(cols_ok) < 5 or len(rows_ok) < 5:
            continue
        x0, x1, y0, y1 = x0 + cols_ok[0], x0 + cols_ok[-1] + 1, y0 + rows_ok[0], y0 + rows_ok[-1] + 1
        h, w = y1 - y0, x1 - x0
        if min(h, w) < 5 or max(h, w) < 4 * min(h, w):
            continue
        if (lab[y0:y1, x0:x1] == i).mean() < 0.9:                 # a filled rectangle (frame + fill)
            continue
        orient = "vertical" if h > w else "horizontal"
        inner = img[y0 + 2:y1 - 2, x0 + 2:x1 - 2].astype(float)
        if inner.size == 0 or min(inner.shape[:2]) < 2:
            continue
        if orient == "horizontal":
            inner = inner.transpose(1, 0, 2)
        # across the bar: same colour; along it: the colour changes
        q = inner[:, inner.shape[1] // 5: max(inner.shape[1] // 5 + 1, 4 * inner.shape[1] // 5)]
        across = float(np.median(q.std(1).max(-1)))
        prof = np.median(q, 1)
        along = float(np.abs(np.diff(prof, axis=0)).sum(0).max())
        if across > 12 or along < 120:
            continue
        out.append({"box": (int(x0), int(y0), int(x1), int(y1)), "orient": orient, "across_std": across,
                    "colour_path": along, "score": along / (1 + across)})
    return sorted(out, key=lambda c: -c["score"])


class ColorScale:
    def __init__(self, pos_values, colours, info):
        self.values = np.asarray(pos_values, float)
        self.colours = np.asarray(colours, float)
        self.info = info
        self.tree = cKDTree(self.colours)
        self.vmin, self.vmax = float(self.values.min()), float(self.values.max())

    def value(self, rgb, shaded=False):
        """rgb: (..., 3) -> (values, colour distances)"""
        rgb = np.asarray(rgb, float)
        flat = rgb.reshape(-1, 3)
        if not shaded:
            d, k = self.tree.query(flat)
            return self.values[k].reshape(rgb.shape[:-1]), d.reshape(rgb.shape[:-1])
        # brightness-free match: distance between p and the best multiple of each table colour
        C = self.colours
        cc = (C ** 2).sum(1) + 1e-9
        best_v = np.empty(len(flat)); best_d = np.empty(len(flat))
        for a in range(0, len(flat), 4096):
            P = flat[a:a + 4096]
            k = np.clip((P @ C.T) / cc, 0.35, 1.25)                 # brightness factor per pair
            d = np.sqrt(((P[:, None, :] - k[..., None] * C[None]) ** 2).sum(-1))
            j = d.argmin(1)
            best_v[a:a + 4096] = self.values[j]; best_d[a:a + 4096] = d[np.arange(len(P)), j]
        return best_v.reshape(rgb.shape[:-1]), best_d.reshape(rgb.shape[:-1])


def read_colorbar(img, cb_spec, exclude_box=None):
    """cb_spec: {"tick_labels": [...low -> high...], "orientation": ...}.
    Returns (ColorScale or None, info)."""
    info = {"found": False}
    labels = [t for t in (cb_spec or {}).get("tick_labels") or [] if parse_number(t) is not None]
    cands = find_colorbars(img)
    want = str((cb_spec or {}).get("orientation") or "").lower()
    if want in ("vertical", "horizontal"):
        cands = sorted(cands, key=lambda c: (c["orient"] != want, -c["score"]))
    if exclude_box is not None:
        ex0, ey0, ex1, ey1 = exclude_box
        cands = [c for c in cands if not (c["box"][0] >= ex0 and c["box"][2] <= ex1 and c["box"][1] >= ey0 and c["box"][3] <= ey1)]
    if not cands:
        info["reason"] = "no colorbar found in the image"
        return None, info
    if len(labels) < 2:
        info["reason"] = "the VLM read fewer than 2 colorbar tick labels"
        return None, info
    best = None
    for c in cands[:3]:
        x0, y0, x1, y1 = c["box"]
        fr = {"left": x0, "right": x1 - 1, "top": y0, "bottom": y1 - 1, "line_width": 1}
        sides = ("right", "left") if c["orient"] == "vertical" else ("bottom", "top")
        for side in sides:
            if side == "top":
                # labels above a horizontal bar: flip the image so they are 'below'
                flip = img[::-1]
                fr2 = {"left": x0, "right": x1 - 1, "top": img.shape[0] - y1, "bottom": img.shape[0] - y0 - 1, "line_width": 1}
                words = D.tick_label_blobs(flip, fr2, "bottom")
                for w in words:
                    w["y0"], w["y1"] = img.shape[0] - w["y1"], img.shape[0] - w["y0"]
            else:
                words = D.tick_label_blobs(img, fr, side)
            if len(words) < 2:
                continue
            along = "y" if c["orient"] == "vertical" else "x"
            maps, inf = calibrate_axis(labels, [w["c"] for w in words], None, "linear", False, along,
                                       label_widths=[w["x1"] - w["x0"] for w in words])
            if maps is None:
                continue
            q = inf.get("rms_px", 99) + 3 * abs(len(words) - len(labels))
            if best is None or q < best[0]:
                best = (q, c, side, maps[0], inf, words)
    if best is None:
        info["reason"] = "colorbar tick labels could not be matched to label positions"
        return None, info
    q, c, side, mp, inf, words = best
    x0, y0, x1, y1 = c["box"]
    # colour along the bar (skip the 2-px outline), median across its width
    if c["orient"] == "vertical":
        a0, a1 = x0 + max(2, (x1 - x0) // 5), x1 - max(2, (x1 - x0) // 5)
        pos = np.arange(y0 + 2, y1 - 2)
        cols = np.median(img[y0 + 2:y1 - 2, a0:a1].astype(float), 1)
    else:
        a0, a1 = y0 + max(2, (y1 - y0) // 5), y1 - max(2, (y1 - y0) // 5)
        pos = np.arange(x0 + 2, x1 - 2)
        cols = np.median(img[a0:a1, x0 + 2:x1 - 2].astype(float), 0)
    vals = mp.to_value(pos.astype(float))
    # stepped scales (filled contours): every run of one colour stands for its middle value
    jump = np.abs(np.diff(cols, axis=0)).max(1)
    cuts = [0] + [k + 1 for k in np.where(jump > 10)[0]] + [len(cols)]
    runs = [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]
    long_runs = [r for r in runs if r[1] - r[0] >= 4]
    stepped = (len(cuts) - 2 >= 3 and len(long_runs) >= 3 and sum(r[1] - r[0] for r in long_runs) > 0.85 * len(cols)
               and all(np.abs(cols[b - 1] - cols[a]).max() <= 12 for a, b in long_runs))
    if stepped:
        cols2 = np.array([cols[a:b].mean(0) for a, b in long_runs])
        vals2 = np.array([vals[a:b].mean() for a, b in long_runs])
        edges = sorted([float(vals[a]) for a, b in long_runs] + [float(vals[long_runs[-1][1] - 1])])
        cols, vals = cols2, vals2
    info.update(found=True, box=c["box"], orient=c["orient"], label_side=side, rms_px=float(inf.get("rms_px", 0)),
                n_labels_found=len(words), n_labels_read=len(labels), stepped=bool(stepped),
                levels=(edges if stepped else None), value_range=[float(np.min(vals)), float(np.max(vals))],
                label_boxes=[(w["x0"], w["y0"], w["x1"], w["y1"]) for w in words])
    return ColorScale(vals, cols, info), info


def colorbar_exclusion(info, shape, pad=4):
    """mask of the colorbar and its labels (so they are not taken as data)"""
    m = np.zeros(shape[:2], bool)
    if not info or not info.get("found"):
        return m
    x0, y0, x1, y1 = info["box"]
    boxes = [(x0, y0, x1, y1)] + list(info.get("label_boxes", []))
    for a, b, c, d in boxes:
        m[max(0, b - pad):d + pad, max(0, a - pad):c + pad] = True
    return m
