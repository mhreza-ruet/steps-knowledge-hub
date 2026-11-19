"""Figures that hold several plots: a grid / stack of panels, or a main plot with an inset.

1. LAYOUT (VLM): `LAYOUT_PROMPT` asks how many plots the image holds and how they are arranged
   (single / panels / main+inset), with each plot's label and position. The VLM only describes the
   layout - it reads no data.
2. FRAMES (pixels): every axes frame is found as a rectangle of long dark lines (`find_frames`);
   a frame lying inside another one is an inset.
3. One plot at a time: for each frame a FULL-SIZE copy of the image is made in which everything
   outside that plot (its frame plus the strip of tick labels / titles around it) is painted white,
   and an inset is painted white inside its main plot. Nothing is cropped: pixel coordinates stay
   those of the original. The normal pipeline (VLM spec + pixel digitizers) then runs on each copy.
4. The results are kept apart: {"layout", "panels": {label: {"role", "box", "result"}}}, and the
   recreations are pasted back at their places, so the combined redraw looks like the input.

Shared axes: a panel without its own tick labels (stacked panels that print the x labels only
under the bottom one) gets the strip of the panel below / beside it that carries those labels.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from . import detect as D

LAYOUT_PROMPT = """Look at the whole image. Does it contain ONE plot, or SEVERAL plots (panels side by side or
stacked, each with its own axes), or a main plot with a small INSET plot drawn inside it?
Return ONLY this JSON object:
{"layout": "single" or "panels" or "main+inset",
 "plots": [{"label": "<panel label as printed, e.g. (a), B, or empty>",
            "role": "main" or "panel" or "inset",
            "position": "<where: e.g. top-left, top-right, bottom, upper part, inside the main plot top-right>",
            "chart": "<what it is: line, scatter, bar, stacked bar, spectrum, ...>",
            "shares_axis_with": "<label of a plot whose printed x-axis (or y-axis) labels it uses, or empty>"}]}
Rules: an inset is a small plot with its own axes INSIDE another plot. A legend box or a printed table is not a plot.
Equations or fit-parameter tables printed inside a plot are not separate plots. List plots left to right, top to bottom."""


def find_frames(img, min_frac=0.12):
    """axes frames: rectangles whose left and bottom sides are long dark lines; the right / top
    sides may be missing (open frames). Returns [(L, T, R, B)], largest first."""
    H, W = img.shape[:2]
    # axis lines may be coloured (a red x-axis): dark OR strongly saturated pixels
    d = D.dark_mask(img, 140, 0.35) | ((D.saturation(img) > 0.5) & (D.gray(img) < 200))

    def runs(v, gap=4):                                   # runs of True, bridging gaps of a few px (ticks, anti-aliasing)
        r = D._runs(v)
        out = []
        for a, b in r:
            if out and a - out[-1][1] <= gap:
                out[-1] = (out[-1][0], b)
            else:
                out.append((a, b))
        return out
    hs, vs = [], []
    for y in range(H):
        for a, b in runs(d[y]):
            if b - a >= min_frac * W:
                hs.append((y, a, b))
    for x in range(W):
        for a, b in runs(d[:, x]):
            if b - a >= min_frac * H:
                vs.append((x, a, b))
    boxes = []
    for x, va, vb in vs:
        # horizontal lines starting at this vertical line (left end within 3 px) while it runs: every one
        # is the bottom of a frame; stacked panels share one long left line
        hits0 = sorted({int(y) for y, ha, hb in hs if x - 15 <= ha <= x + 3 and va - 3 <= y <= vb + 3})
        # one frame side is a few rows thick: per group of neighbouring rows keep the longest line
        hits = []
        for y in hits0:
            ln = max(hb - x for yy, ha, hb in hs if yy == y and x - 15 <= ha <= x + 3)
            if hits and y - hits[-1][0] <= 6:
                if ln > hits[-1][1]:
                    hits[-1] = (y, ln)
            else:
                hits.append((y, ln))
        hits = [h for h, _ in hits]
        for i, y in enumerate(hits):
            if y - va < min_frac * H * 0.5:
                continue
            hb = max(hb for yy, ha, hb in hs if yy == y and x - 15 <= ha <= x + 3)
            # the right side: a vertical line that spans this frame's height (closed frame); else the line end
            rights = [x2 for x2, a2, b2 in vs if x + 40 < x2 <= hb + 8 and a2 <= y - 0.6 * (y - va) and b2 >= y - 3]
            if rights:
                hb = max(hb, max(rights))
            # a frame side is a THIN line: filled bars standing on it make it thick
            seg = d[max(0, y - 6):max(1, y - 4), int(x):int(hb)]
            if seg.size and seg.mean() > 0.85:
                continue
            top = hits[i - 1] if i > 0 and y - hits[i - 1] >= 20 else int(va)
            box = (int(x), int(top), int(hb), int(y))
            if not any(abs(box[0] - b[0]) <= 6 and abs(box[3] - b[3]) <= 6 for b in boxes):
                boxes.append(box)
    # a printed table or legend box is a rectangle too: an axes frame has tick marks on its left or bottom side
    keep = []
    for (L, T, R, B) in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        fr = {"left": L, "right": R, "top": T, "bottom": B, "line_width": 2}
        try:
            nt = max(len(D.detect_ticks(img, fr, "bottom")), len(D.detect_ticks(img, fr, "left")))
        except Exception:
            nt = 0
        host = any(k[0] <= L and k[1] <= T + 3 and k[2] >= R and k[3] >= B and (k[2] - k[0]) * (k[3] - k[1]) > (R - L) * (B - T)
                   for k in keep)
        if host:
            # a box inside an accepted plot: an inset plot, or a printed table / legend box. A table has
            # several ruled lines across it; an inset has (almost) none.
            inner = D.gray(img)[T + 4:B - 3, L + 4:R - 3] < 235
            ruled = int(((inner.mean(1) > 0.8)).sum()) if inner.size else 0
            if ruled < 2 and (R - L) * (B - T) > 0.04 * max((k[2] - k[0]) * (k[3] - k[1]) for k in keep):
                keep.append((L, T, R, B))
            continue
        if nt >= 3:
            keep.append((L, T, R, B))
    keep.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    return keep


def roles(boxes):
    """'inset' for a frame inside another one, else 'panel' (or 'main' for the host of an inset)"""
    out = []
    for i, b in enumerate(boxes):
        host = next((j for j, c in enumerate(boxes) if j != i and c[0] <= b[0] + 3 and c[1] <= b[1] + 3 and c[2] >= b[2] - 3
                     and c[3] >= b[3] - 3 and (c[2] - c[0]) * (c[3] - c[1]) > (b[2] - b[0]) * (b[3] - b[1])), None)
        out.append(("inset", host) if host is not None else ("panel", None))
    hosts = {h for r, h in out if r == "inset"}
    return [("main", None) if i in hosts else r for i, r in enumerate(out)]


def select_frames(img, layout):
    """the axes frames that are the VLM's plots. Pixel frame finding alone also finds outlined bars,
    tables and surface meshes, so the VLM's layout decides: 'single' -> no split; 'panels' -> the N
    largest frames with tick marks that do not lie inside each other (N = plots the VLM listed);
    'main+inset' -> the largest frame and the largest frame inside it that is not a ruled table."""
    lay = (layout or {}).get("layout", "single")
    plots = (layout or {}).get("plots") or []
    boxes = find_frames(img)
    if lay not in ("panels", "main+inset") or not boxes:
        return []
    def inside(b, c):
        return c[0] <= b[0] + 3 and c[1] <= b[1] + 3 and c[2] >= b[2] - 3 and c[3] >= b[3] - 3 and b != c
    top = [b for b in boxes if not any(inside(b, c) for c in boxes)]
    if lay == "panels":
        n = sum(1 for p in plots if str(p.get("role", "panel")) != "inset") or len(top)
        top = sorted(top, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:n]
        return sorted(top, key=lambda b: (round(b[1] / 40), b[0]))
    main = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    ins = [b for b in boxes if inside(b, main)]
    return [main] + ([max(ins, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))] if ins else [])


def panel_image(img, boxes, k, rl, margin=None):
    """full-size copy showing only plot k (frame + label strip); other plots painted white.
    margin = (left, bottom, right, top) strip kept around the frame for tick labels / titles."""
    H, W = img.shape[:2]
    L, T, R, B = boxes[k]
    if margin is None:     # strip kept around the frame for tick labels / titles (a right axis needs room too)
        margin = (60, 48, 8, 20) if rl[k][0] == "inset" else (95, 70, 75, 30)
    ml, mb, mr, mt = margin
    keep = np.zeros((H, W), bool)
    x0, y0, x1, y1 = max(0, L - ml), max(0, T - mt), min(W, R + mr), min(H, B + mb)
    # do not reach into a neighbouring panel's frame
    for j, (l2, t2, r2, b2) in enumerate(boxes):
        if j == k or rl[j][0] == "inset":
            continue
        if r2 <= L and r2 > x0:
            x0 = r2 + 2
        if l2 >= R and l2 < x1:
            x1 = l2 - 2
        if b2 <= T and b2 > y0:
            y0 = b2 + 2
        if t2 >= B and t2 < y1:
            y1 = t2 - 2
    keep[y0:y1, x0:x1] = True
    if rl[k][0] == "main":                                 # hide the inset inside the main plot
        for j, (l2, t2, r2, b2) in enumerate(boxes):
            if rl[j] == ("inset", k):
                keep[max(0, t2 - 25):b2 + 45, max(0, l2 - 70):r2 + 8] = False
    out = np.full_like(img, 255)
    out[keep] = img[keep]
    # stacked panels that print the x tick labels only under the lowest one: that strip of labels is
    # copied under this panel (same columns) so its x axis can be read too
    below = [b for j, b in enumerate(boxes) if j != k and rl[j][0] != "inset" and abs(b[0] - L) <= 6 and abs(b[2] - R) <= 8 and b[1] >= B - 3]
    if below:
        low = max(below, key=lambda b: b[3])
        h = min(mb, H - low[3] - 1, H - B - 1)
        if h > 5:
            strip = img[low[3] + 1:low[3] + 1 + h, x0:x1]
            out[B + 1:B + 1 + h, x0:x1] = strip
            y1 = max(y1, B + 1 + h)
    return out, (x0, y0, x1, y1)


def process_figure(image, reader, out_root, name=None, use_cache=True, check=True, ask_layout=True):
    """single plot -> run.process_image; several plots -> one run per plot on a masked full-size
    copy, results kept apart and recreations pasted back together."""
    from pathlib import Path
    from .run import process_image, side_by_side
    image = Path(image)
    name = name or image.stem
    od = Path(out_root) / name
    od.mkdir(parents=True, exist_ok=True)
    img = np.asarray(Image.open(image).convert("RGB"))
    layout = {}
    lay_path = od / "vlm_layout.json"
    if ask_layout and hasattr(reader, "ask"):
        if use_cache and lay_path.exists():
            layout = json.loads(lay_path.read_text())
        else:
            from .spec import _json_blob
            raw = reader.ask(str(image), LAYOUT_PROMPT)
            layout = {"raw": raw, "layout": _json_blob(raw) or {}}
            lay_path.write_text(json.dumps(layout, indent=1))
    boxes = select_frames(img, layout.get("layout"))
    if len(boxes) <= 1:
        r = process_image(image, reader, out_root, name, use_cache, check)
        r["layout"] = "single"
        return r
    rl = roles(boxes)
    labels = [p.get("label") for p in (layout.get("layout") or {}).get("plots", [])]
    order = sorted(range(len(boxes)), key=lambda i: (rl[i][0] == "inset", round(boxes[i][1] / 50), boxes[i][0]))
    panels, rec = {}, Image.open(image).convert("RGB").copy()
    rec.paste((255, 255, 255), (0, 0) + rec.size)
    for n, k in enumerate(order):
        pid = (labels[n] if n < len(labels) and labels[n] else f"panel_{n + 1}") if rl[k][0] != "inset" else f"inset_{n + 1}"
        pid = "".join(ch for ch in pid if ch.isalnum() or ch in "_-") or f"panel_{n + 1}"
        pim, keep_box = panel_image(img, boxes, k, rl)
        p_path = od / f"{name}__{pid}.png"
        Image.fromarray(pim).save(p_path)
        r = process_image(p_path, reader, od, f"{pid}", use_cache, check)
        panels[pid] = {"role": rl[k][0], "frame": list(boxes[k]), "region": list(keep_box),
                       "status": r["result"].get("status"), "result": r["result"], "dir": r["dir"],
                       "series": r["result"].get("series", {}), "chart_type": r["result"].get("chart_type"),
                       "spec_info": r.get("spec_info"), "digitize_seconds": r.get("digitize_seconds")}
        pr = Image.open(r["recreated"]).convert("RGB").resize(rec.size)
        rec.paste(pr.crop(keep_box), keep_box[:2])
    lay_name = "main+inset" if any(v["role"] == "inset" for v in panels.values()) else "panels"
    combined = {"figure": name, "layout": lay_name, "vlm_layout": layout.get("layout"),
                "panels": {k: {kk: vv for kk, vv in v.items() if kk != "result"} | {"series": v["result"].get("series", {}),
                                                                                    "chart_type": v["result"].get("chart_type")}
                           for k, v in panels.items()}}
    from .run import _jsonable
    (od / "extraction.json").write_text(json.dumps(_jsonable(combined), indent=1))
    rec_path = od / "recreated.png"
    rec.save(rec_path)
    side_by_side(image, rec_path, od / "side_by_side.png", name)
    statuses = [v["status"] for v in panels.values()]
    status = "rejected" if all(s == "rejected" for s in statuses) else ("ok" if all(s == "ok" for s in statuses) else "flagged")
    return {"name": name, "dir": str(od), "layout": lay_name, "panels": panels, "recreated": rec_path,
            "result": {"status": status, "status_reasons": [f"{k}: {v['status']}" for k, v in panels.items()],
                       "chart_type": lay_name, "series": {f"{k}/{s}": d for k, v in panels.items()
                                                          for s, d in (v["result"].get("series") or {}).items()}}}
