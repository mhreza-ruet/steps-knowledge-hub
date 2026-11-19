"""Rebuild QA: side-by-side images and a 'too far off' flag per figure.

For every digitized series the pixels of that series are found in the ORIGINAL and in the
REBUILT image with the same colour rule (inside the panel frame). Two distances:
  * miss_px  - 90th percentile distance from an original series pixel to the nearest rebuilt one
               (something in the original the rebuild does not draw);
  * extra_px - the same the other way round (the rebuild draws something the original has not).
A series is OFF when more than 15 % of its pixels (original or rebuilt) are farther than 6 px
(about one marker) from the other image's pixels of that series; a figure is flagged when any
series is off. 3-D / contour panels (autodig) use a colour-difference score.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .digitize import load_rgb
from .engine import inside, series_mask

LIMIT_PX = 6.0
SHARE = 0.15


def _drop_small(m, min_extent=25):
    """text glyphs, arrow heads and tick labels inside the frame are small separate pieces; a
    traced curve is long. Pieces smaller than min_extent px in both directions are dropped (line
    series only)."""
    lab, n = ndi.label(m, np.ones((3, 3)))
    if n == 0:
        return m
    keep = np.zeros(n + 1, bool)
    for i, sl in enumerate(ndi.find_objects(lab)):
        keep[i + 1] = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) >= min_extent
    return keep[lab]


def _mask(img, P, s):
    pad = s.get("frame_pad")
    if s.get("match", {}).get("dark") or s.get("match", {}).get("outline"):
        pad = max(pad or 0, 7)                  # black series: tick marks / frame line widths differ, not data
    m = inside(series_mask(img, s), P, s.get("exclude", ()), pad)
    if s["kind"] == "marker" and not s.get("hollow") and not s.get("connect"):
        k = max(2, int(round(0.3 * s.get("ms_px", 8))))     # thin decorations of the marker colour (dashed arrows) are not data
        m = ndi.binary_opening(m, structure=np.ones((k, k)))
    if s["kind"] in ("bar", "stacked") and (s.get("match", {}).get("dark") or "ink" in s.get("match", {})):
        # black text next to black / hatched bars is not data: only the bar columns count
        from .engine import cat_px
        keep = np.zeros(m.shape[1], bool)
        n = len(P.get("categories") or [])
        for i in (s.get("at") or range(n)):
            cx = cat_px(P, i) + s.get("offset_px", 0.0); w = s.get("width_px", 20)
            keep[max(0, int(cx - 0.35 * w)):int(cx + 0.35 * w) + 1] = True
        return _drop_small(m & keep[None, :], 6)
    if s["kind"] == "stick":                  # hkl labels above the sticks are text, not data
        return _drop_small(m, 30)
    return _drop_small(m) if (s["kind"] == "line" and not s.get("qa_keep_small")) else m


def series_distance(orig, rebuilt, P, s, others=()):
    """others: the panel's other series - a pixel hidden under another series (drawing order,
    line width) is not counted as missing / extra"""
    mo, mr = _mask(orig, P, s), _mask(rebuilt, P, s)
    if mo.sum() < 5 and mr.sum() < 5:
        return 0.0, 0.0, int(mo.sum()), int(mr.sum()), 0.0, 0.0
    if mo.sum() < 5 or mr.sum() < 5:
        return 99.0, 99.0, int(mo.sum()), int(mr.sum()), 1.0, 1.0
    oo = np.zeros_like(mo); orr = np.zeros_like(mr)
    for t in others:
        oo |= _mask(orig, P, t); orr |= _mask(rebuilt, P, t)
    dr = ndi.distance_transform_edt(~(mr | orr))
    # extra: the rebuild draws this series where the original has no ink at all (colours of curves
    # drawn over each other blend, so 'any ink' is the fair reference)
    ink_o = inside(orig.min(-1) < 200, P, s.get("exclude", ()), 2)
    do = ndi.distance_transform_edt(~(mo | oo | ink_o))
    miss = float(np.percentile(dr[mo], 90))
    extra = float(np.percentile(do[mr], 90))
    return miss, extra, int(mo.sum()), int(mr.sum()), float((dr[mo] > LIMIT_PX).mean()), float((do[mr] > LIMIT_PX).mean())


def check_figure(F, orig_path, rebuilt_path, limit=LIMIT_PX):
    o, r = load_rgb(orig_path), load_rgb(rebuilt_path)
    if r.shape != o.shape:
        from PIL import Image
        r = np.asarray(Image.open(rebuilt_path).convert("RGB").resize(o.shape[1::-1])).astype(np.int16)
    rows = []
    for P in F["panels"]:
        if P.get("engine") == "autodig":
            x0, y0, x1, y1 = P.get("region", (0, 0, o.shape[1], o.shape[0]))
            a = ndi.uniform_filter(o[y0:y1, x0:x1].astype(float), (9, 9, 1))
            b = ndi.uniform_filter(r[y0:y1, x0:x1].astype(float), (9, 9, 1))
            d = float(np.mean(np.abs(a - b)))
            rows.append({"panel": P["id"], "series": "(3-D / contour panel)", "miss_px": None, "extra_px": None,
                         "colour_diff": round(d, 1), "off": d > 25})
            continue
        if any(s["kind"] == "pattern_stack" for s in P["series"].values()):
            # patterns are not colours: compare the bar columns' dark share row by row
            from .engine import cat_px, axis_of
            w = P.get("bar_width_px", 15); bad = []
            zero = float(axis_of(P, "y").to_pixel(0.0))
            L_, R_, T_, B_ = P["frame"]

            def extent(img_, cx):
                gg = img_.astype(float).mean(-1)
                e = (gg[:, int(round(cx - w / 2)) - 1:int(round(cx - w / 2)) + 2] < 130).any(1) | \
                    (gg[:, int(round(cx + w / 2)) - 1:int(round(cx + w / 2)) + 2] < 130).any(1)
                rr = np.where(e[int(T_) + 2:int(B_) - 1])[0] + int(T_) + 2
                if not len(rr):
                    return None
                for q in np.split(rr, np.where(np.diff(rr) > 2)[0] + 1):
                    if q.min() - 3 <= zero <= q.max() + 3:
                        return int(q.min()), int(q.max())
                return None
            for i in range(len(P["categories"])):
                cx = cat_px(P, i)
                ea, eb = extent(o, cx), extent(r, cx)
                bad.append(ea is None or eb is None or abs(ea[0] - eb[0]) > LIMIT_PX or abs(ea[1] - eb[1]) > LIMIT_PX)
            share = float(np.mean(bad)) if bad else 1.0
            rows.append({"panel": P["id"], "series": "(pattern-stacked bars: bar extents)", "share_missing": round(share, 3),
                         "share_extra": None, "off": share > SHARE})
            continue
        for n, s in P["series"].items():
            if s.get("qa") is False:
                continue
            others = [t for m, t in P["series"].items() if m != n and t["kind"] in ("line", "marker", "bar", "stacked")]
            miss, extra, no, nr, fm, fe = series_distance(o, r, P, s, others)
            # OFF: more than SHARE of the series' pixels are farther than LIMIT_PX from their counterpart
            # (a few hidden / merged markers are not 'way off'; a misplaced curve or a missing series is)
            rows.append({"panel": P["id"], "series": n, "miss_px": round(miss, 1), "extra_px": round(extra, 1),
                         "share_missing": round(fm, 3), "share_extra": round(fe, 3),
                         "px_orig": no, "px_rebuilt": nr,
                         "off": bool(fm > s.get("qa_share", SHARE) or fe > s.get("qa_share", SHARE))})
    return {"figure": F["id"], "flagged": any(r_["off"] for r_ in rows), "rows": rows}


def side_by_side(orig_path, rebuilt_path, out_path, title="", flag=None):
    from PIL import Image, ImageDraw
    a = Image.open(orig_path).convert("RGB"); b = Image.open(rebuilt_path).convert("RGB").resize(a.size)
    W, H = a.size
    top = 30
    canvas = Image.new("RGB", (2 * W + 24, H + top), "white")
    canvas.paste(a, (0, top)); canvas.paste(b, (W + 24, top))
    d = ImageDraw.Draw(canvas)
    d.text((6, 8), f"original  {title}", fill=(0, 0, 0))
    d.text((W + 30, 8), "rebuilt (ground-truth data)", fill=(0, 0, 0))
    if flag:
        d.rectangle([0, 0, 2 * W + 23, H + top - 1], outline=(220, 0, 0), width=4)
        d.text((W - 60, 8), "FLAG: too far off", fill=(220, 0, 0))
    canvas.save(out_path)
    return out_path
