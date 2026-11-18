"""Polar plots: curves, scatter and polar bars (rose / wind diagrams).

The VLM reads: the angle labels around the circle (e.g. "0°", "45°", or N/E/S/W),
the radial tick labels (inner -> outer) and the series (name, colour, marker,
line). The pixels give:

  1. the outer circle: the large dark ring (circle fit, angle coverage check)
  2. angle calibration: the angle labels sit just outside the ring; their
     positions (as angles seen from the centre) are paired with the VLM's angle
     values, trying both directions (counter-clockwise / clockwise) and every
     rotation -> zero direction + turning direction, as measured, not assumed
  3. radial calibration: the radial labels inside the circle, as distances from
     the centre, paired with the VLM's values (same alignment code as the axes;
     the radial axis may start above 0), snapped to the grid circles
  4. data per series colour: markers (centres), curves (thinned to a 1-px line,
     one point per angle step and branch) or bars (outermost filled radius along
     every ray, cut into sectors)
"""
from __future__ import annotations

import math
import re

import numpy as np
from scipy import ndimage as ndi

from . import detect as D
from .calibrate import calibrate_axis, parse_number
from .extract import colour_target, hsv_arrays, refine_hue, series_mask, find_markers

COMPASS = {"n": 0, "nne": 22.5, "ne": 45, "ene": 67.5, "e": 90, "ese": 112.5, "se": 135, "sse": 157.5, "s": 180,
           "ssw": 202.5, "sw": 225, "wsw": 247.5, "w": 270, "wnw": 292.5, "nw": 315, "nnw": 337.5}


def parse_angle(s):
    """'45°' -> 45, 'NE' -> 45 (compass, clockwise from north), 'π/2' -> 90, '3π/2' -> 270"""
    t = str(s).strip().lower().replace(" ", "")
    if t in COMPASS:
        return float(COMPASS[t]), "compass"
    m = re.fullmatch(r"(\d*\.?\d*)?(π|pi)(/(\d+))?", t)
    if m:
        a = float(m.group(1)) if m.group(1) else 1.0
        b = float(m.group(4)) if m.group(4) else 1.0
        return 180.0 * a / b, "radian"
    v = parse_number(t.replace("°", "").replace("deg", ""))
    return (float(v), "degree") if v is not None else (None, None)


def fit_circle(xs, ys):
    A = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]
    b = xs ** 2 + ys ** 2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0], c[1]
    r = math.sqrt(max(c[2] + cx ** 2 + cy ** 2, 1e-9))
    return cx, cy, r


def find_circle(img):
    """outer ring of the polar axes: (cx, cy, R, info) or None"""
    H, W = img.shape[:2]
    g = img.mean(-1)
    sat = (img.max(-1).astype(float) - img.min(-1)) / np.maximum(img.max(-1), 1)
    d = (g < 120) & (sat < 0.3)
    lab, n = ndi.label(d, np.ones((3, 3)))
    best = None
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if min(h, w) < 0.3 * min(H, W) or abs(h - w) > 0.15 * max(h, w):
            continue
        ys, xs = np.where(lab == i)
        if len(xs) > 6000:
            k = np.random.default_rng(0).choice(len(xs), 6000, replace=False); xs, ys = xs[k], ys[k]
        xs, ys = xs.astype(float), ys.astype(float)
        cx, cy, r = fit_circle(xs, ys)
        for _ in range(3):                       # refit on the ring pixels only
            near = np.abs(np.hypot(xs - cx, ys - cy) - r) <= 3
            if near.sum() < 50:
                break
            cx, cy, r = fit_circle(xs[near], ys[near])
        dist = np.abs(np.hypot(xs - cx, ys - cy) - r)
        near = dist <= 3
        ang = np.degrees(np.arctan2(-(ys[near] - cy), xs[near] - cx)) % 360
        cover = len(np.unique((ang // 5).astype(int))) / 72
        rms = float(np.sqrt(np.mean(dist[near] ** 2))) if near.any() else 99
        if cover >= 0.8 and rms <= 2.5 and (best is None or r > best[2]):
            best = (cx, cy, r, {"coverage": cover, "rms_px": rms})
    return best


def words_everywhere(img, max_h=30):
    d = D.dark_mask(img, 150, 0.4)
    cs = [c for c in D.components(d, min_area=3) if (c["y1"] - c["y0"]) <= max_h and (c["x1"] - c["x0"]) <= 40]
    ws = D._merge_words(cs, horizontal=True, gap_factor=0.6)
    for w in ws:
        w["cx"], w["cy"] = (w["x0"] + w["x1"] - 1) / 2, (w["y0"] + w["y1"] - 1) / 2
    return ws


def _clean_angle_labels(labels):
    """the VLM's angle labels as a SET of angles (its order is not trusted). A stray radial
    number among degree labels ('12' among '0°', '45°', ...) is dropped: when most labels carry a
    degree sign (or are compass words), the ones without it are not angle labels."""
    parsed = [(str(l), *parse_angle(l)) for l in labels]
    parsed = [(t, v, k) for t, v, k in parsed if v is not None]
    deg_sign = [p for p in parsed if "°" in p[0] or "deg" in p[0].lower()]
    if len(deg_sign) >= 0.6 * len(parsed) and len(deg_sign) >= 3:
        parsed = deg_sign
    kinds = [k for _, _, k in parsed]
    if kinds:
        main = max(set(kinds), key=kinds.count)
        parsed = [p for p in parsed if p[2] == main]
    seen, out = set(), []
    for t, v, k in parsed:
        key = round(v % 360, 3)
        if key not in seen:
            seen.add(key); out.append((t, v % 360))
    return sorted(out, key=lambda q: q[1])


def _order_disagreement(listed, phi0, s):
    """share of consecutive labels in the VLM's list that do NOT step counter-clockwise (the
    order the prompt asks for) under this calibration, + 1 if the first is not at 3 o'clock"""
    if len(listed) < 3:
        return 0.0
    img_ang = [(phi0 + s * v) % 360 for v in listed]
    steps = [((b - a) % 360) for a, b in zip(img_ang, img_ang[1:])]
    bad = float(np.mean([not (0 < st < 180) for st in steps]))
    first = 0.0 if abs((img_ang[0] + 180) % 360 - 180) <= 180.0 / len(listed) + 3 else 1.0
    return bad + first


def angle_calibration(words, labels, cx, cy, R):
    """phi_image = phi0 + s * theta  (degrees, image angle counter-clockwise from +x).
    The VLM's angle labels are used as a set of values - their listed order is NOT trusted
    (a small VLM often lists them in reading order).  The label words outside the ring are
    found in the pixels; every turning direction and every rotation that puts some label on
    some word is tried.  Evenly spaced labels fit many rotations equally well, so the label
    widths decide ('0°' is narrower than '135°'); a remaining tie goes to matplotlib's
    defaults (zero at east or north)."""
    from .calibrate import _width_mismatch
    lab = _clean_angle_labels(labels)
    vals = [v for _, v in lab]
    strs = [t for t, _ in lab]
    ring = [w for w in words if R + 1 <= math.hypot(w["cx"] - cx, w["cy"] - cy) <= R + 70]
    if len(vals) < 3 or len(ring) < 3:
        return None, {"reason": f"{len(ring)} angle labels found outside the circle, {len(vals)} read"}
    items = sorted(((math.degrees(math.atan2(-(w["cy"] - cy), w["cx"] - cx)) % 360, w) for w in ring), key=lambda q: q[0])
    merged = []                                       # pieces of one label close together -> one label
    for p, w in items:
        if merged and p - merged[-1][-1][0] < 4:
            merged[-1].append((p, w))
        else:
            merged.append([(p, w)])
    if len(merged) > 1 and (merged[0][0][0] + 360) - merged[-1][-1][0] < 4:
        merged[0] = [(q - 360, w) for q, w in merged[-1]] + merged[0]; merged.pop()
    phis = np.array([np.mean([q for q, _ in m]) % 360 for m in merged])
    # label width = evidence for which label sits where; words merged with a curve or a legend
    # (taller than one text line) or degree-sign specks give no reliable width
    hs = np.array([w["y1"] - w["y0"] for _, w in items], float)
    hline = float(np.median(hs[hs >= 7])) if (hs >= 7).any() else 11.0

    def _width(m):
        clean = [w for _, w in m if 0.7 * hline <= (w["y1"] - w["y0"]) <= 1.4 * hline and (w["x1"] - w["x0"]) <= 4 * hline]
        return (max(w["x1"] for w in clean) - min(w["x0"] for w in clean)) if clean else np.nan
    widths = np.array([_width(m) for m in merged], float)
    # the VLM's listed order is weak evidence (often right, sometimes scrambled)
    listed = [v % 360 for v in (parse_angle(l)[0] for l in labels) if v is not None]
    vv = np.array(vals)
    best = None
    for s in (1, -1):
        for pb in phis:
            for v0 in vv:
                phi0 = (pb - s * v0) % 360
                pred = (phi0 + s * vv) % 360
                dd = np.abs((phis[:, None] - pred[None, :] + 180) % 360 - 180)
                cost = float(np.minimum(dd.min(0), 20).sum() + np.minimum(dd.min(1), 20).sum())
                j = dd.argmin(0)
                ok = (dd[j, np.arange(len(vv))] <= 8) & ~np.isnan(widths[j])
                wm = _width_mismatch([t for t, o in zip(strs, ok) if o], widths[j[ok]]) if ok.sum() >= 3 else 0.0
                prior = 0.0 if min(abs((phi0 - z + 180) % 360 - 180) for z in (0, 90)) < 3 else 0.5
                key = cost + 12.0 * wm + prior + (0.0 if s == 1 else 0.2) + 3.0 * _order_disagreement(listed, phi0, s)
                if best is None or key < best[0] - 1e-9:
                    best = (key, s, phi0, cost, wm)
    if best is None:
        return None, {"reason": "the angle labels could not be placed on the label words"}
    key, s, phi0, cost, wm = best
    pred = (phi0 + s * vv) % 360
    dd = (phis[:, None] - pred[None, :] + 180) % 360 - 180
    j = np.abs(dd).argmin(0)
    r_ = dd[j, np.arange(len(vv))]
    ok = np.abs(r_) <= 8
    if ok.any():
        phi0 = (phi0 + float(np.mean(r_[ok]))) % 360
        rms = float(np.sqrt(np.mean((r_[ok] - np.mean(r_[ok])) ** 2)))
    else:
        rms = 99.0
    listed = [parse_angle(l)[0] for l in labels]
    return (phi0, s), {"phi0": phi0, "direction": "counter-clockwise" if s == 1 else "clockwise", "paired": int(ok.sum()),
                       "n_labels": len(vals), "rms_deg": rms, "zero_at": _zero_name(phi0), "width_mismatch": round(wm, 3),
                       "labels_used": strs, "labels_dropped": [l for l in labels if str(l) not in strs],
                       "label_angles_px": phis.round(1).tolist()}


def _zero_name(phi0):
    return {0: "east (3 o'clock)", 90: "north (12 o'clock)", 180: "west", 270: "south"}.get(int(round(phi0 / 90) * 90) % 360, f"{phi0:.0f} deg")


def ring_radii(img, cx, cy, R, n_ang=720):
    """radii of the grid circles: radii at which (nearly) every direction is inked"""
    g = img.mean(-1)
    sat = (img.max(-1).astype(float) - img.min(-1)) / np.maximum(img.max(-1), 1)
    ink = (g < 240) & (sat < 0.15)
    th = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    cov = []
    for r in np.arange(3, R + 3):
        xs = np.clip(np.round(cx + r * np.cos(th)).astype(int), 0, img.shape[1] - 1)
        ys = np.clip(np.round(cy - r * np.sin(th)).astype(int), 0, img.shape[0] - 1)
        cov.append(ink[ys, xs].mean())
    cov = np.array(cov)
    peaks = [i for i in range(1, len(cov) - 1) if cov[i] >= 0.4 and cov[i] >= cov[i - 1] and cov[i] >= cov[i + 1]]
    radii = []
    for p in peaks:
        r = 3 + p
        if not radii or r - radii[-1] > 3:
            radii.append(float(r))
    return radii


def radial_calibration(words, labels, cx, cy, R, rings):
    vals = [l for l in labels if parse_number(l) is not None]
    inside = [w for w in words if 2 <= math.hypot(w["cx"] - cx, w["cy"] - cy) <= R - 2]
    # a radial label is one short text line; words merged with bar edges or curves are not labels
    hs = np.array([w["y1"] - w["y0"] for w in words], float)
    hline = float(np.median(hs[(hs >= 7) & (hs <= 20)])) if ((hs >= 7) & (hs <= 20)).any() else 11.0
    inside = [w for w in inside if 0.7 * hline <= (w["y1"] - w["y0"]) <= 1.4 * hline and (w["x1"] - w["x0"]) <= 4 * hline]
    if len(vals) < 2:
        return None, {"reason": "fewer than 2 radial tick labels read"}
    if len(inside) < 2:
        return None, {"reason": "no radial tick labels found inside the circle"}
    # radial labels lie along one ray: group words by direction, keep the group that fits best
    best = None
    phis = np.array([math.degrees(math.atan2(-(w["cy"] - cy), w["cx"] - cx)) % 360 for w in inside])
    for p in np.unique(np.round(phis / 5) * 5):
        grp = [w for w, q in zip(inside, phis) if abs((q - p + 180) % 360 - 180) <= 12]
        if len(grp) < 2:
            continue
        dist = sorted(math.hypot(w["cx"] - cx, w["cy"] - cy) for w in grp)
        grp_s = sorted(grp, key=lambda w: math.hypot(w["cx"] - cx, w["cy"] - cy))
        maps, inf = calibrate_axis(vals, dist, None, "linear", False, "x", label_widths=[w["x1"] - w["x0"] for w in grp_s])
        if maps is None:
            continue
        q = inf.get("rms_px", 99) + 3 * abs(len(grp) - len(vals))
        if best is None or q < best[0]:
            best = (q, maps[0], inf, grp)
    from .calibrate import AxisMap
    # grid circles sit at the radial tick values, so the labels are paired with the rings.
    # Equally spaced rings fit equally spaced labels at several offsets: the offset is chosen by
    # (1) agreement with the label words' own distances (when any were found), (2) no label value
    # of 0 on a ring (a ring at radius 0 cannot be drawn: 0 is the centre), (3) fewest labels left
    # unpaired, (4) the outer ring may be an unlabelled frame.
    v_all = sorted(parse_number(x) for x in vals)
    nv, nr_ = len(v_all), len(rings)
    nv_lab = len(vals)
    lab_pairs = (best[2].get("pairs") if best is not None else None) or []
    if len(lab_pairs) < 3 or len(lab_pairs) != nv_lab:   # two words fit any two values; a missed word shifts the rest
        lab_pairs = []
    cand = None
    for k in range(-(nv - 1), nr_):
        pr = [(v_all[i], rings[i + k]) for i in range(nv) if 0 <= i + k < nr_]
        if len(pr) < min(3, nv):
            continue
        v = np.array([a_ for a_, _ in pr]); p = np.array([b_ for _, b_ in pr])
        A = np.c_[np.ones(len(v)), v]
        (a, b), *_ = np.linalg.lstsq(A, p, rcond=None)
        if b <= 0:
            continue
        res = float(np.sqrt(np.mean((A @ np.array([a, b]) - p) ** 2)))
        if res > 3:
            continue
        lab_err = float(np.sqrt(np.mean([((a + b * lv) - lp) ** 2 for lv, lp in lab_pairs]))) if len(lab_pairs) >= 2 else 0.0
        zero_on_ring = any(abs(a_) < 1e-12 for a_, _ in pr)
        unpaired = nv - len(pr)
        # label words sit a few px off their ring (text box centre): only an error of a sizeable
        # share of the ring spacing says the offset is wrong
        spacing = b * float(np.median(np.diff(v_all))) if nv > 1 else 1.0
        lab_pen = 0.0 if lab_err < 0.4 * spacing else 4.0 * lab_err / spacing
        score = res + lab_pen + (8.0 if zero_on_ring else 0.0) + 2.0 * unpaired
        if cand is None or score < cand[0]:
            cand = (score, a, b, res, lab_err, len(pr), unpaired)
    if cand is not None:
        score, a, b, res, lab_err, npair, unpaired = cand
        inf = {"method": "grid circles paired with the radial labels", "rms_px": res, "n_rings": nr_, "pairs_used": npair,
               "labels_without_ring": unpaired, "label_disagreement_px": round(lab_err, 2)}
        if best is not None:
            inf["label_rms_px"] = float(best[2].get("rms_px", 99))
            inf["label_boxes"] = [(w["x0"], w["y0"], w["x1"], w["y1"]) for w in best[3]]
        return AxisMap(a, b), inf
    if best is None:
        return None, {"reason": "radial labels could not be paired with label positions"}
    q, mp, inf, grp = best
    # snap to the grid circles where one lies within 4 px of a label's radius
    pairs = inf.get("pairs", [])
    snapped = []
    for v, p in pairs:
        near = [r for r in rings if abs(r - p) <= 4]
        snapped.append((v, min(near, key=lambda r: abs(r - p)) if near else p))
    if len(snapped) >= 2:
        v = np.array([a for a, _ in snapped]); p = np.array([b for _, b in snapped])
        A = np.c_[np.ones(len(v)), v]
        (a, b), *_ = np.linalg.lstsq(A, p, rcond=None)
        res = float(np.sqrt(np.mean((A @ np.array([a, b]) - p) ** 2)))
        from .calibrate import AxisMap
        mp = AxisMap(a, b)
        inf = dict(inf, rms_px=res, snapped_to_grid_circles=sum(1 for (x, y), (_, z) in zip(pairs, snapped) if y != z))
    inf["label_boxes"] = [(w["x0"], w["y0"], w["x1"], w["y1"]) for w in grp]
    return mp, inf


def equal_bins(th, outer, tol_deg=2.0):
    """polar bars (rose / wind diagrams) almost always use equal angular bins. The bar edges
    are where the outer radius changes for good (a spoke drawn across a bar is only a 1-sample
    dip, so the profile is median-filtered first); the bin width is the largest 360/n whose grid
    (with its best phase) passes through nearly every edge. Each bin's value = the median outer
    radius over its central half. Returns None when no equal-bin grid explains the edges."""
    from scipy.ndimage import median_filter
    step = float(th[1] - th[0])
    f = np.where(np.isnan(outer), -1.0, outer)
    f = median_filter(f, size=5, mode="wrap")
    k = int(round(3.0 / step))
    g = np.r_[f[-k:], f, f[:k]]
    edges = []
    for i in range(len(f)):
        lo, hi = np.median(g[i:i + k]), np.median(g[i + k + 1:i + 2 * k + 1])
        if abs(hi - lo) > 3 and (not edges or th[i] - edges[-1] > 3):
            edges.append(float(th[i]) + step / 2)
    if len(edges) < 3:
        return None
    E = np.array(edges)
    best = None
    for n in range(72, 3, -1):
        w = 360.0 / n
        if w < 3 * tol_deg:
            continue
        ph = np.angle(np.mean(np.exp(1j * 2 * np.pi * (E % w) / w))) / (2 * np.pi) * w % w
        d = np.abs(((E - ph) % w + w / 2) % w - w / 2)
        cover = float(np.mean(d <= tol_deg))
        if cover >= 0.9:
            best = (n, w, ph, cover)
            break                                  # the finest grid that still explains (almost) all edges
    if best is None:
        return None
    n, w, ph, cover = best
    # the finest grid passes every edge trivially only if it is very fine; prefer the coarsest grid
    # whose lines still pass >= 90 % of the edges
    for m in range(4, n):
        w2 = 360.0 / m
        ph2 = np.angle(np.mean(np.exp(1j * 2 * np.pi * (E % w2) / w2))) / (2 * np.pi) * w2 % w2
        d2 = np.abs(((E - ph2) % w2 + w2 / 2) % w2 - w2 / 2)
        if np.mean(d2 <= tol_deg) >= 0.9:
            n, w, ph, cover = m, w2, ph2, float(np.mean(d2 <= tol_deg))
            break
    bins = []
    for i in range(n):
        a0 = (ph + i * w) % 360
        centre = (a0 + w / 2) % 360
        sel = np.abs(((th - centre) + 180) % 360 - 180) <= w / 4
        vals = outer[sel]
        vals = vals[~np.isnan(vals)]
        if len(vals) >= max(2, 0.5 * sel.sum()):
            bins.append((float(centre), float(np.median(vals))))
    if len(bins) < 3:
        return None
    return {"bins": bins, "n_bins": n, "bin_width_deg": round(w, 3), "edge_cover": round(cover, 3), "n_edges": len(E)}


def pipeline_polar(img, spec):
    H, W = img.shape[:2]
    out = {"chart_type": "polar", "size": (W, H), "spec": spec, "warnings": [], "checks": [], "series": {}}
    circ = find_circle(img)
    if circ is None:
        out["reject"] = "polar: no circular axis ring found"; return out
    cx, cy, R, cinf = circ
    out["circle"] = {"cx": cx, "cy": cy, "R": R, **cinf}
    words = words_everywhere(img)
    ac, ainf = angle_calibration(words, spec.get("angle_labels") or [], cx, cy, R)
    out["angle_calibration"] = ainf
    if ac is None:
        out["reject"] = f"polar: angle calibration failed ({ainf.get('reason')})"; return out
    rings = ring_radii(img, cx, cy, R)
    rm, rinf = radial_calibration(words, spec.get("radial_labels") or [], cx, cy, R, rings)
    out["radial_calibration"] = {k: v for k, v in rinf.items() if k != "pairs"}
    if rm is None:
        out["reject"] = f"polar: radial calibration failed ({rinf.get('reason')})"; return out
    phi0, s = ac
    out["calibration"] = {"phi0": phi0, "direction": s, "r_a": rm.a, "r_b": rm.b, "rings_px": rings}
    out["r_limits"] = [float(rm.to_value(0.0)), float(rm.to_value(R))]
    # for the redraw: the ends snapped onto a printed radial label when within 2 % of the span,
    # the direction (data angle) along which the radial labels are printed, the label strings
    rl = [parse_number(x) for x in spec.get("radial_labels") or []]
    rl = [v for v in rl if v is not None]
    span = abs(out["r_limits"][1] - out["r_limits"][0]) or 1.0
    snap_to = sorted(set(rl) | {0.0})            # the centre is usually r = 0 even when '0' is not printed
    out["r_limits_draw"] = [min(snap_to, key=lambda v: abs(v - e)) if min(abs(v - e) for v in snap_to) <= 0.02 * span else e
                            for e in out["r_limits"]]
    if out["r_limits_draw"][0] < 0 and not any(v < 0 for v in rl):
        out["r_limits_draw"][0] = 0.0             # a negative inner limit breaks the polar grid in matplotlib < 3.11
    boxes = rinf.get("label_boxes") or []
    if boxes:
        angs = [math.degrees(math.atan2(-((b[1] + b[3]) / 2 - cy), (b[0] + b[2]) / 2 - cx)) % 360 for b in boxes]
        a_img = math.degrees(math.atan2(np.mean(np.sin(np.radians(angs))), np.mean(np.cos(np.radians(angs))))) % 360
        out["radial_label_angle_deg"] = float(((a_img - phi0) * s) % 360)
    out["angle_labels_used"] = ainf.get("labels_used")

    def to_data(px, py):
        dx, dy = np.asarray(px, float) - cx, -(np.asarray(py, float) - cy)
        phi = np.degrees(np.arctan2(dy, dx)) % 360
        th = ((phi - phi0) * s) % 360
        return th, rm.to_value(np.hypot(dx, dy))

    hue, sat, val = hsv_arrays(img)
    yy, xx = np.mgrid[:H, :W]
    disk = np.hypot(xx - cx, yy - cy) <= R - 1.5
    textm = np.zeros((H, W), bool)
    for (x0, y0, x1, y1) in rinf.get("label_boxes", []):      # the radial labels are the only text inside
        textm[max(0, y0 - 1):y1 + 1, max(0, x0 - 1):x1 + 1] = True
    region = disk & ~textm
    kind = str(spec.get("kind") or "line").lower()
    series = spec.get("series") or []
    targets = [refine_hue(colour_target(q.get("color", "")), hue, sat, val, region) for q in series]
    hues = [t["hue"] for t in targets if t["kind"] == "hue"]
    for q, t in zip(series, targets):
        others = [h for h in hues if t["kind"] != "hue" or abs(h - t["hue"]) > 1e-6]
        m = series_mask(img, t, hue, sat, val, region, others)
        if t["kind"] == "dark":
            m &= ~(np.abs(np.hypot(xx - cx, yy - cy) - R) <= 3)
        if m.sum() < 5:
            out["warnings"].append(f"no pixels of '{q['name']}' ({q.get('color')}) found"); continue
        mk = q.get("marker", "none") not in ("none", None, "")
        pix = img[m].astype(float)
        core = pix[pix.mean(1) <= np.percentile(pix.mean(1), 50)] if len(pix) > 10 else pix   # stroke core, not anti-aliased rim
        look = {"rgb": np.median(core, 0).round(1).tolist(), "marker": q.get("marker") if mk else None,
                "line": str(q.get("line") or ("none" if mk else "solid")).lower()}
        if kind == "bar":
            th = np.arange(0, 360, 0.5)
            rad = np.arange(0, R, 0.5)
            outer = np.full(len(th), np.nan)
            phi = (phi0 + s * th) % 360
            for k, ph in enumerate(np.radians(phi)):
                xs = np.clip(np.round(cx + rad * np.cos(ph)).astype(int), 0, W - 1)
                ys = np.clip(np.round(cy - rad * np.sin(ph)).astype(int), 0, H - 1)
                hit = np.where(m[ys, xs])[0]
                if len(hit) >= 3:
                    outer[k] = rad[hit.max()]
            reg = equal_bins(th, outer)
            if reg is not None:
                thc = [round(c, 3) for c, _ in reg["bins"]]
                rv = [float(rm.to_value(rp + 0.5)) for _, rp in reg["bins"]]
                out["bar_bins"] = {k: v for k, v in reg.items() if k != "bins"}
                out["series"][q["name"]] = {"theta_deg": thc, "r": [round(v, 6) for v in rv], "kind": "bar", **look}
                continue
            # sectors: runs of directions with a bar, cut where the outer radius jumps
            secs, cur = [], []
            for k in range(len(th)):
                if np.isnan(outer[k]) or (cur and abs(outer[k] - np.median(outer[cur])) > 2.5):
                    if len(cur) >= 3:
                        secs.append(cur)
                    cur = [] if np.isnan(outer[k]) else [k]
                else:
                    cur.append(k)
            if len(cur) >= 3:
                if secs and secs[0][0] == 0 and abs(np.median(outer[cur]) - np.median(outer[secs[0]])) <= 2.5:
                    secs[0] = cur + secs[0]                # a sector crossing the 0 direction
                else:
                    secs.append(cur)
            thc, rv = [], []
            for sec in secs:
                a = th[sec]
                c = math.degrees(math.atan2(np.mean(np.sin(np.radians(a))), np.mean(np.cos(np.radians(a))))) % 360
                thc.append(round(c, 3)); rv.append(float(rm.to_value(np.median(outer[sec]) + 0.5)))
            out["series"][q["name"]] = {"theta_deg": thc, "r": [round(v, 6) for v in rv], "kind": "bar", **look}
        elif mk or kind == "scatter":
            pts, info = find_markers(m)
            if not pts:
                out["warnings"].append(f"no markers of '{q['name']}' found"); continue
            th, r = to_data([p[0] for p in pts], [p[1] for p in pts])
            o = np.argsort(th)
            from .extract import joined_by_line
            joined, share = joined_by_line(m, pts)
            if look["line"] not in ("none", "") and not joined:
                out["warnings"].append(f"'{q['name']}': the VLM described a line through the markers, the pixels show none "
                                       f"({share:.0%} of the colour lies off the markers) - drawn as markers only")
                look = dict(look, line="none")
            elif look["line"] in ("none", "") and joined:
                look = dict(look, line="solid")
            out["series"][q["name"]] = {"theta_deg": th[o].round(3).tolist(), "r": r[o].round(6).tolist(), "kind": "marker", **look}
        else:
            from skimage.morphology import skeletonize
            sk = skeletonize(m)
            ys, xs = np.where(sk)
            th, r = to_data(xs, ys)
            # one point per 1-degree step and branch
            pts = []
            for b in range(360):
                sel = (th >= b) & (th < b + 1)
                if not sel.any():
                    continue
                rr = np.sort(r[sel]); tt = th[sel][np.argsort(r[sel])]
                grp = np.split(np.arange(len(rr)), np.where(np.diff(rr) > 0.03 * (out["r_limits"][1] - out["r_limits"][0]))[0] + 1)
                for gi in grp:
                    pts.append((float(np.mean(tt[gi])), float(np.mean(rr[gi]))))
            pts.sort()
            out["series"][q["name"]] = {"theta_deg": [round(a, 3) for a, _ in pts], "r": [round(b, 6) for _, b in pts], "kind": "line", **look}
    out["table"] = [{"series": n, "theta_deg": a, "r": b} for n, d in out["series"].items() for a, b in zip(d["theta_deg"], d["r"])]
    npts = len(out["table"])
    out["checks"] += [("circle fit (px)", round(cinf["rms_px"], 2), "<= 2.5", cinf["rms_px"] <= 2.5),
                      ("angle labels paired", f"{ainf['paired']}/{ainf['n_labels']}", ">= 60 %", ainf["paired"] >= 0.6 * ainf["n_labels"]),
                      ("angle label fit (deg)", round(ainf["rms_deg"], 2), "<= 6", ainf["rms_deg"] <= 6),
                      ("radial label fit (px)", round(rinf.get("rms_px", 99), 2), "<= 3", rinf.get("rms_px", 99) <= 3),
                      ("points extracted", npts, ">= 1", npts >= 1)]
    return out
