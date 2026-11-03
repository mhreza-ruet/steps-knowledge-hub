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


def angle_calibration(words, labels, cx, cy, R):
    """phi_image = phi0 + s * theta  (degrees, image angle counter-clockwise from +x).
    The VLM lists the angle labels as met going counter-clockwise from the
    3 o'clock position. Evenly spaced labels fit any rotation, so the first
    listed label is pinned to the label word nearest 3 o'clock; both turning
    directions are tried and the pixels decide."""
    vals = [parse_angle(l)[0] for l in labels]
    vals = [v for v in vals if v is not None]
    ring = [w for w in words if R + 1 <= math.hypot(w["cx"] - cx, w["cy"] - cy) <= R + 70]
    if len(vals) < 3 or len(ring) < 3:
        return None, {"reason": f"{len(ring)} angle labels found outside the circle, {len(vals)} read"}
    phis = sorted(math.degrees(math.atan2(-(w["cy"] - cy), w["cx"] - cx)) % 360 for w in ring)
    merged = []
    for p in phis:                                   # pieces of one label close together
        if merged and p - merged[-1][-1] < 4:
            merged[-1].append(p)
        else:
            merged.append([p])
    if len(merged) > 1 and (merged[0][0] + 360) - merged[-1][-1] < 4:
        merged[0] = [q - 360 for q in merged[-1]] + merged[0]; merged.pop()
    phis = np.array([np.mean(m) % 360 for m in merged])
    vv = np.array(vals)
    half = 180.0 / len(vv) + 3
    firsts = [p for p in phis if abs((p + 180) % 360 - 180) <= half] or list(phis)
    best = None
    for s in (1, -1):
        # the labels were listed counter-clockwise: each next label must lie a little
        # further counter-clockwise (0 < step < 180 deg) for this turning direction
        steps = (s * np.diff(vv)) % 360
        if len(steps) and np.mean((steps > 0) & (steps < 180)) < 0.75:
            continue
        for p0 in firsts:
            phi0 = (p0 - s * vv[0]) % 360
            pred = (phi0 + s * vv) % 360
            dd = np.abs((phis[:, None] - pred[None, :] + 180) % 360 - 180)
            cost = float(np.minimum(dd.min(0), 20).sum() + np.minimum(dd.min(1), 20).sum())
            if best is None or cost < best[0]:
                best = (cost, s, phi0)
    if best is None:
        return None, {"reason": "the angle labels' order fits neither turning direction"}
    cost, s, phi0 = best
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
    return (phi0, s), {"phi0": phi0, "direction": "counter-clockwise" if s == 1 else "clockwise", "paired": int(ok.sum()),
                       "n_labels": len(vals), "rms_deg": rms, "zero_at": _zero_name(phi0)}


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
    nv = len(vals)
    inner = [r for r in rings if abs(r - R) > 3]
    use = rings if len(rings) == nv else (inner if len(inner) == nv else None)
    if use is not None:
        # one grid circle per radial label (the outer ring counts if it is labelled)
        v = np.array([parse_number(x) for x in vals]); p = np.array(use)
        A = np.c_[np.ones(nv), v]
        (a, b), *_ = np.linalg.lstsq(A, p, rcond=None)
        res = float(np.sqrt(np.mean((A @ np.array([a, b]) - p) ** 2)))
        inf = {"method": "grid circles paired with the radial labels in order", "rms_px": res, "n_rings": len(rings)}
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
            out["series"][q["name"]] = {"theta_deg": thc, "r": [round(v, 6) for v in rv], "kind": "bar"}
        elif mk or kind == "scatter":
            pts, info = find_markers(m)
            if not pts:
                out["warnings"].append(f"no markers of '{q['name']}' found"); continue
            th, r = to_data([p[0] for p in pts], [p[1] for p in pts])
            o = np.argsort(th)
            out["series"][q["name"]] = {"theta_deg": th[o].round(3).tolist(), "r": r[o].round(6).tolist(), "kind": "marker"}
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
            out["series"][q["name"]] = {"theta_deg": [round(a, 3) for a, _ in pts], "r": [round(b, 6) for _, b in pts], "kind": "line"}
    out["table"] = [{"series": n, "theta_deg": a, "r": b} for n, d in out["series"].items() for a, b in zip(d["theta_deg"], d["r"])]
    npts = len(out["table"])
    out["checks"] += [("circle fit (px)", round(cinf["rms_px"], 2), "<= 2.5", cinf["rms_px"] <= 2.5),
                      ("angle labels paired", f"{ainf['paired']}/{ainf['n_labels']}", ">= 60 %", ainf["paired"] >= 0.6 * ainf["n_labels"]),
                      ("angle label fit (deg)", round(ainf["rms_deg"], 2), "<= 6", ainf["rms_deg"] <= 6),
                      ("radial label fit (px)", round(rinf.get("rms_px", 99), 2), "<= 3", rinf.get("rms_px", 99) <= 3),
                      ("points extracted", npts, ">= 1", npts >= 1)]
    return out
