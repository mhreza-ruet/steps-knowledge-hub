"""Pixel <-> VLM cross-check of the spec, before any data is extracted.

The pixels can count things the VLM is bad at counting:
  * how many tick labels sit along each axis      -> if the VLM's list has a
    different length, ask it again with the count ("there are 10 labels on
    the left axis - read all 10"), then keep whichever list calibrates better
    (labels must line up with the label positions as a straight line,
    or two straight pieces for a broken axis);
  * whether there are labels on the right edge at all -> a "right axis" the
    VLM invented without printed numbers is dropped.
Every decision is logged in the returned notes.
"""
from __future__ import annotations

import copy
import re

import numpy as np

from . import detect as D
from .calibrate import _merge_close, calibrate_axis, parse_number

SIDES = {"x_axis": ("bottom", "x"), "y_axis": ("left", "y"), "y_axis_right": ("right", "y")}


def axis_has_break(image, side="left"):
    """A broken y-axis: the axis line is interrupted by a short gap (2-25 px)."""
    img = D.load_rgb(image)
    try:
        fr = D.detect_frame(img)
    except RuntimeError:
        return False
    x = int(round(fr["left"] if side == "left" else fr["right"]))
    ink = (D.dark_mask(img, 150, 0.3) | ((D.saturation(img) > 0.5) & (D.gray(img) < 200)))[:, max(x - 1, 0):x + 2].any(1)
    col = ink[int(fr["top"]) + 3:int(fr["bottom"]) - 3]
    gaps = [e - s_ for s_, e in D._runs(~col)]
    return any(2 <= g <= 25 for g in gaps)


CHECK_VERSION = 2      # bump when the checks change: cached VLM answers are re-checked (not re-asked)


def axis_break_rows(image, side="left"):
    """pixel rows (centres) of short gaps (2-25 px) in a vertical axis line = axis breaks"""
    img = D.load_rgb(image)
    try:
        fr = D.detect_frame(img)
    except RuntimeError:
        return []
    x = int(round(fr["left"] if side == "left" else fr["right"]))
    ink = (D.dark_mask(img, 150, 0.3) | ((D.saturation(img) > 0.5) & (D.gray(img) < 200)))[:, max(x - 1, 0):x + 2].any(1)
    t0 = int(fr["top"]) + 3
    col = ink[t0:int(fr["bottom"]) - 3]
    return [t0 + (s_ + e) / 2 for s_, e in D._runs(~col) if 2 <= e - s_ <= 25]


def _linear_rms(vals, pos):
    """rms (px) of a straight-line fit of label positions against values"""
    v = np.array([parse_number(x) for x in vals], float)
    if len(v) < 2 or np.isnan(v).any() or len(set(v)) < len(v):
        return np.inf
    A = np.c_[v, np.ones_like(v)]
    coef, *_ = np.linalg.lstsq(A, np.asarray(pos, float), rcond=None)
    return float(np.sqrt(np.mean((A @ coef - pos) ** 2)))


def check_broken_axis(image, spec, key, reader, notes):
    """A y-axis whose line has a gap is read piece by piece.

    The pixels find the gap and the tick labels on each side of it. If the
    VLM's labels do not already calibrate BOTH pieces, the VLM is shown the
    whole image with a red rectangle drawn around the labels of one piece
    and asked for just those numbers - once per piece. A piece's answer is
    used only if it has one number per label found there, the numbers
    increase upwards and lie on a straight line against the label positions;
    the pieces must not overlap (upper piece above lower piece)."""
    side = "left" if key == "y_axis" else "right"
    ax = spec.get(key)
    if not ax or ax.get("scale") == "log":
        return False
    rows = axis_break_rows(image, side)
    if not rows:
        return False
    img = D.load_rgb(image)
    fr = D.detect_frame(img)
    blobs = D.tick_label_blobs(img, fr, side, D.detect_ticks(img, fr, side))
    if len(blobs) < 4:
        return False
    cs = np.array([b["c"] for b in blobs], float)
    # the break that splits the labels most evenly
    brk = max(rows, key=lambda r: min((cs < r).sum(), (cs > r).sum()))
    groups = [[b for b in blobs if b["c"] > brk], [b for b in blobs if b["c"] < brk]]   # lower piece, upper piece
    if min(len(g) for g in groups) < 2:
        return False
    # do the current labels already cover both pieces?
    vals = [v for v in ax["tick_labels"] if parse_number(v) is not None]
    allp = sorted(cs, reverse=True)
    if len(vals) == len(blobs) and _linear_rms(vals, allp) <= 3:
        return False          # one straight scale across the gap: the gap is not a scale break
    if len(vals) == len(blobs):
        lo_v, up_v = vals[:len(groups[0])], vals[len(groups[0]):]
        lo_p = sorted((b["c"] for b in groups[0]), reverse=True)
        up_p = sorted((b["c"] for b in groups[1]), reverse=True)
        if _linear_rms(lo_v, lo_p) <= 3 and _linear_rms(up_v, up_p) <= 3 and parse_number(up_v[0]) > parse_number(lo_v[-1]):
            if not ax.get("broken"):
                notes.append(f"{key}: the axis line has a break at pixel row {brk:.0f}; the VLM's labels fit the two "
                             f"pieces ({len(lo_v)} + {len(up_v)}) but it said 'not broken' - set to broken")
                ax["broken"] = True
            return True
    msg = (f"{key}: the axis line has a break at pixel row {brk:.0f} with {len(groups[0])} labels below "
           f"and {len(groups[1])} above; VLM gave {len(vals)} labels (broken={bool(ax.get('broken'))})")
    if reader is None or not hasattr(reader, "reread_ticks_box"):
        notes.append(msg + " - the pieces could not be read separately (no VLM re-ask available)")
        return False
    got = []
    for name, g in zip(("lower", "upper"), groups):
        pad = 4
        box = (min(b["x0"] for b in g) - pad, min(b["y0"] for b in g) - pad,
               max(b["x1"] for b in g) + pad, max(b["y1"] for b in g) + pad)
        raw, new = reader.reread_ticks_box(image, key, box, len(g))
        pos = sorted((b["c"] for b in g), reverse=True)           # bottom -> top
        ok = (new is not None and len(new) == len(g) and all(parse_number(v) is not None for v in new))
        if ok:
            nv = [parse_number(v) for v in new]
            ok = all(b > a for a, b in zip(nv, nv[1:])) and _linear_rms(new, pos) <= 3.0
        notes.append(msg + f"; {name} piece (red box) -> {new}: {'ok' if ok else 'rejected'}")
        if not ok:
            return False
        got.append(new)
    if parse_number(got[1][0]) <= parse_number(got[0][-1]):
        notes.append(msg + "; the pieces overlap - not used")
        return False
    ax["tick_labels"] = got[0] + got[1]
    ax["broken"] = True
    notes.append(msg + f"; pieces read separately -> {ax['tick_labels']}: USED")
    return True


def label_positions(image):
    """{axis: (centres, widths, major tick positions)} found in the pixels."""
    img = D.load_rgb(image)
    frame = D.detect_frame(img)
    out = {}
    for key, (side, _) in SIDES.items():
        if side == "right" and not frame["has_right"]:
            out[key] = ([], [], [])
            continue
        ticks = D.detect_ticks(img, frame, side)
        labels = D.tick_label_blobs(img, frame, side, ticks)
        c, w = _merge_close([l["c"] for l in labels], [l["x1"] - l["x0"] for l in labels])
        out[key] = (c, w, [t["pos"] for t in ticks if t.get("major")])
    return out


def _fit_quality(values, pos, ax, along):
    """lower = better: rms of the label-position fit + a penalty per label left unpaired."""
    c, w, ticks = pos
    if len(c) < 2 or sum(parse_number(v) is not None for v in values) < 2:
        return np.inf, None
    maps, info = calibrate_axis(values, c, ticks, ax.get("scale", "linear"), ax.get("broken", False), along,
                                label_widths=w)
    if maps is None:
        return np.inf, info
    unpaired = abs(len(c) - len(info.get("pairs", [])))
    return float(info.get("rms_px", np.inf)) + 4.0 * unpaired, info


def check_spec(image, spec, reader=None):
    """Returns (checked spec, notes list)."""
    spec = copy.deepcopy(spec)
    notes = []
    if spec.get("chart_type") == "pie":
        return spec, notes            # no axes to check
    try:
        pos = label_positions(image)
    except RuntimeError as e:
        return spec, [f"no axes found in the image ({e})"]
    # a right axis needs printed labels on the right
    if spec.get("y_axis_right") and len(pos["y_axis_right"][0]) < 2:
        notes.append("y_axis_right: VLM reported a right axis but no labels were found there - dropped")
        spec["y_axis_right"] = None
        for s in spec["series"]:
            s["y_axis"] = "left"
    fixed = set()
    for key in ("y_axis", "y_axis_right"):
        if spec.get(key) and check_broken_axis(image, spec, key, reader, notes):
            fixed.add(key)
    for key, (side, along) in SIDES.items():
        ax = spec.get(key)
        if not ax or key in fixed or (key == "x_axis" and spec.get("chart_type") == "bar"):   # bar x labels are category names
            continue
        n_pix, n_vlm = len(pos[key][0]), len(ax["tick_labels"])
        if n_pix < 2 or n_pix == n_vlm or (n_vlm < 2 and n_pix < 3):
            continue                       # nothing to compare, or the counts agree
        q_old, _ = _fit_quality(ax["tick_labels"], pos[key], ax, along)
        msg = f"{key}: VLM listed {n_vlm} tick labels, pixels found {n_pix}"
        if q_old <= 2.0 and abs(n_pix - n_vlm) < 2:
            notes.append(msg + f"; labels still line up (fit {q_old:.1f} px) - kept")
            continue
        if reader is None or not hasattr(reader, "reread_ticks"):
            notes.append(msg + " (no re-ask possible)")
            continue
        brk = ax.get("broken", False) or (key != "x_axis" and axis_has_break(image, "left" if key == "y_axis" else "right"))
        raw, new = reader.reread_ticks(image, key, n_pix, brk)
        if not new:
            notes.append(msg + f"; re-ask gave no list ({raw[:60]!r})")
            continue
        q_new, _ = _fit_quality(new, pos[key], ax, along)
        same = len(set(new) & set(ax["tick_labels"])) / max(1, min(len(new), n_vlm))
        if q_new < q_old and q_new <= 3.0 and (q_old > 2.0 or same >= 0.5):
            notes.append(msg + f"; re-asked -> {new} (fit {q_old:.1f} -> {q_new:.1f}): USED")
            ax["tick_labels"] = new
        else:
            notes.append(msg + f"; re-asked -> {new} (fit {q_new:.1f} not better than {q_old:.1f}): kept first answer")
    # vertical y-axis titles are easy to misread ('H' for 'pH'): read them again with the
    # image turned so the title is horizontal; keep the longer reading if it contains the first
    if reader is not None and hasattr(reader, "reread_title"):
        for key, side in (("y_axis", "left"), ("y_axis_right", "right")):
            ax = spec.get(key)
            if not ax:
                continue
            old = ax.get("title", "") or ""
            raw, new = reader.reread_title(image, side)
            norm = lambda t: re.sub(r"\s+", "", t.lower())
            if new and norm(new) != norm(old) and (not old or norm(old) in norm(new) or len(norm(old)) <= 2):
                notes.append(f"{key} title: first read {old!r}, read again turned 90 deg: {new!r} - USED")
                ax["title"] = new
                if spec.get("chart_type") == "bar":      # a single unnamed series is named after the axis title
                    for q in spec["series"]:
                        if q["name"] == old:
                            q["name"] = new
    return spec, notes
