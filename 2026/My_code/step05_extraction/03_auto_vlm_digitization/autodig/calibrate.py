"""Turn (tick-label values read by the VLM) + (tick / label positions found in
the pixels) into a pixel <-> value calibration for each axis.

Handles: linear and log axes, reversed axes (e.g. FTIR wavenumber 4000 -> 400),
a different number of labels found vs. read (extra / missing labels), labels
split into pieces, and two-segment broken axes.
"""
from __future__ import annotations

import math
import re

import numpy as np

_NUM = re.compile(r"[-+−–]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_number(s):
    """'1,000' -> 1000, '−5' -> -5, '10^3' / '10³' -> 1000, '1e-3' -> 0.001."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    t = str(s).strip().replace(",", "").replace("−", "-").replace("–", "-").replace(" ", "")
    sup = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
    if "^" in t or any(ch in t for ch in "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"):   # power-of-ten labels: 10^3, 10³
        m = re.fullmatch(r"10\^?\{?([-+]?\d+)\}?", t.translate(sup))
        if m:
            return 10.0 ** int(m.group(1))
    m = _NUM.search(t)
    return float(m.group(0).replace("−", "-").replace("–", "-")) if m else None


class AxisMap:
    """value <-> pixel for one axis (or one segment of a broken axis)."""

    def __init__(self, a, b, log=False, lo_px=None, hi_px=None):
        self.a, self.b, self.log = a, b, log    # pixel = a + b * f(value)
        self.lo_px, self.hi_px = lo_px, hi_px   # pixel range this map is valid for (broken axes)

    def to_pixel(self, v):
        v = np.asarray(v, float)
        return self.a + self.b * (np.log10(v) if self.log else v)

    def to_value(self, p):
        f = (np.asarray(p, float) - self.a) / self.b
        return 10 ** f if self.log else f

    def contains(self, p):
        return (self.lo_px is None or p >= self.lo_px) and (self.hi_px is None or p <= self.hi_px)

    def as_dict(self):
        return {"a": self.a, "b": self.b, "log": self.log, "lo_px": self.lo_px, "hi_px": self.hi_px}


def _fit(px, vals, log):
    f = np.log10(vals) if log else np.asarray(vals, float)
    if len(px) < 2 or np.ptp(f) == 0:
        return None, np.inf
    A = np.c_[np.ones(len(f)), f]
    (a, b), *_ = np.linalg.lstsq(A, np.asarray(px, float), rcond=None)
    res = np.asarray(px) - (a + b * f)
    return (a, b), float(np.sqrt(np.mean(res ** 2)))


def _merge_close(centres, widths=None, frac=0.45):
    """Labels split into pieces ('1' '2') sit much closer than the label
    spacing: merge positions closer than frac x the median spacing.
    Returns (centres, widths); a merged label's width covers its pieces."""
    if widths is None:
        widths = [0.0] * len(centres)
    it = sorted(zip(centres, widths))
    if len(it) < 3:
        return [float(c) for c, _ in it], [float(w) for _, w in it]
    sp = np.median(np.diff([c for c, _ in it]))
    out = [[it[0]]]
    for c, w in it[1:]:
        if c - out[-1][-1][0] < frac * sp:
            out[-1].append((c, w))
        else:
            out.append([(c, w)])
    cs, ws = [], []
    for g in out:
        lo = min(c - w / 2 for c, w in g); hi = max(c + w / 2 for c, w in g)
        cs.append(float(np.mean([c for c, _ in g]))); ws.append(float(hi - lo))
    return cs, ws


def _width_mismatch(strings, widths):
    """How badly label lengths (characters) disagree with blob widths (pixels).
    Breaks the tie when evenly spaced labels can be aligned at several offsets
    ('0' is narrower than '30')."""
    n = np.array([max(1, len(re.sub(r"[^0-9.\-]", "", str(s))) or len(str(s))) for s in strings], float)
    w = np.asarray(widths, float)
    if len(n) < 2 or w.min() <= 0 or np.ptp(n) == 0:
        return 0.0
    k = np.median(w / n)
    return float(np.mean(np.abs(w - k * n)) / k)


def _snap(px, ticks, spacing):
    if not ticks:
        return px
    t = np.asarray(ticks, float)
    out = []
    for p in px:
        j = np.argmin(np.abs(t - p))
        out.append(float(t[j]) if abs(t[j] - p) <= 0.3 * spacing else p)
    return out


def calibrate_axis(values, label_pos, tick_pos=None, scale="linear", broken=False, along="x", label_widths=None):
    """values: numbers as the VLM read them, in visual order (x: left->right,
    y: bottom->top). label_pos: pixel centres of the label blobs. Returns
    (list_of_AxisMap, info). For y, pixel grows downward, so values listed
    bottom->top pair with positions sorted descending."""
    strs = [x for x in values if parse_number(x) is not None]
    vals = [parse_number(x) for x in strs]
    info = {"n_values": len(vals), "n_labels_found": len(label_pos), "method": None}
    if len(vals) < 2:
        info["method"] = "no tick values"
        return None, info
    pos, wid = _merge_close(label_pos, label_widths)
    if along == "y":
        pos, wid = pos[::-1], wid[::-1]
    spacing = float(np.median(np.abs(np.diff(pos)))) if len(pos) > 1 else 50.0
    logs = [False, True] if (scale == "log" or (min(vals) > 0 and _geometric(vals))) else [False]
    if scale == "log":
        logs = [True]
    best = None
    # align the shorter list inside the longer one (contiguous), allow up to 3 extra on either side
    nv, nb = len(vals), len(pos)
    for log in logs:
        if log and min(vals) <= 0:
            continue
        cands = []
        if nb >= nv:
            for off in range(0, min(nb - nv, 3) + 1):
                cands.append((vals, strs, pos[off:off + nv], wid[off:off + nv]))
        if nv > nb and nb >= 2:
            for off in range(0, min(nv - nb, 3) + 1):
                cands.append((vals[off:off + nb], strs[off:off + nb], pos, wid))
        if nb >= 2 and nv != nb:
            # a label the pixels missed in the MIDDLE of the axis (or an extra blob):
            # place every blob on the label grid by the typical label spacing
            dif = np.abs(np.diff(pos))
            dd = float(np.median(dif)) if len(dif) else 0
            if dd > 0:
                idx = np.r_[0, np.cumsum(np.maximum(1, np.round(dif / dd)).astype(int))]
                if len(set(idx)) == len(idx):
                    for off in range(0, max(0, nv - idx[-1] - 1) + 1):
                        if off + idx[-1] < nv:
                            cands.append(([vals[off + i] for i in idx], [strs[off + i] for i in idx], pos, wid))
        for v, st, p, w in cands:
            ps = _snap(p, tick_pos, spacing)
            ab, rms = _fit(ps, v, log)
            if ab is None:
                continue
            # evenly spaced labels fit equally well at any offset -> label widths decide
            key = rms + 0.25 * spacing * _width_mismatch(st, w) if label_widths is not None else rms
            if best is None or key < best[5]:
                best = (rms, ab, log, v, ps, key)
    if best is None:
        info["method"] = "alignment failed"
        return None, info
    rms, ab, log, v, ps, _ = best
    info.update(rms_px=rms, log=log, pairs=list(zip(map(float, v), map(float, ps))), method="single fit")
    if (broken or rms > max(2.0, 0.02 * spacing * len(v))) and len(v) >= 5:
        seg = _two_segments(v, ps, log)
        if seg is not None and (broken or seg[0] < 0.5 * rms):
            srms, maps = seg
            info.update(rms_px=srms, method="broken axis (2 segments)")
            return maps, info
    return [AxisMap(ab[0], ab[1], log)], info


def _geometric(vals):
    v = np.asarray(vals, float)
    if len(v) < 3 or np.any(v <= 0):
        return False
    r = v[1:] / v[:-1]
    return np.allclose(r, r[0], rtol=0.05) and abs(r[0] - 1) > 0.5


def _two_segments(v, p, log):
    """Best split of the (value, pixel) pairs into two contiguous groups with
    their own linear fits (a broken axis)."""
    best = None
    for k in range(2, len(v) - 1):
        a1, r1 = _fit(p[:k], v[:k], log)
        a2, r2 = _fit(p[k:], v[k:], log)
        if a1 is None or a2 is None:
            continue
        r = max(r1, r2)
        if best is None or r < best[0]:
            best = (r, k, a1, a2)
    if best is None:
        return None
    r, k, a1, a2 = best
    brk = (p[k - 1] + p[k]) / 2
    # segment 1 is the first part along the visual order (left / bottom)
    lo_first = p[k - 1] < p[k]
    m1 = AxisMap(a1[0], a1[1], log, lo_px=None if lo_first else brk, hi_px=brk if lo_first else None)
    m2 = AxisMap(a2[0], a2[1], log, lo_px=brk if lo_first else None, hi_px=None if lo_first else brk)
    return r, [m1, m2]


def map_for_pixel(maps, p):
    for m in maps:
        if m.contains(p):
            return m
    return maps[0]
