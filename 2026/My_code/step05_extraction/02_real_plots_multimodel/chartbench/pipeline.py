"""Glue for the notebook: folders, reference data, rebuilds, results, tables."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .figures import FIGURES, ORDER


class Paths:
    def __init__(self, work_dir, data_dir):
        self.work = Path(work_dir).resolve()
        self.data = Path(data_dir).resolve()
        self.reference = self.work / "outputs" / "reference_v1"       # classical v1 data from the originals
        self.rebuilt = self.work / "outputs" / "rebuilt_plots"        # rebuilt images (same size as originals)
        self.ground_truth = self.work / "outputs" / "ground_truth"    # exact data behind each rebuilt image
        self.qa = self.work / "outputs" / "qa"                         # overlays / side-by-sides
        self.results = self.work / "outputs" / "model_results"        # raw + parsed answers, per model
        self.model_rebuilds = self.work / "outputs" / "model_rebuilds" # each model's v1 re-plotted
        for p in (self.reference, self.rebuilt, self.ground_truth, self.qa, self.results, self.model_rebuilds):
            p.mkdir(parents=True, exist_ok=True)

    def original(self, fig_id):
        return self.data / FIGURES[fig_id]["file"]

    def rebuilt_img(self, fig_id):
        return self.rebuilt / f"{fig_id}_rebuilt.png"


def check_originals(P):
    missing = [f for f in ORDER if not P.original(f).exists()]
    if missing:
        raise FileNotFoundError(f"original image(s) not found in {P.data}: {missing}")
    return [P.original(f) for f in ORDER]


def build_reference(P, force=False):
    """Stage 1: classical digitizer on the untouched originals -> v1 data."""
    from .digitize import extract_figure
    out = {}
    for f in ORDER:
        path = P.reference / f"{f}.json"
        if path.exists() and not force:
            out[f] = json.loads(path.read_text()); continue
        d = extract_figure(f, P.original(f))
        d["source_image"] = FIGURES[f]["file"]
        if "printed_values" in FIGURES[f]:
            # pie charts print every slice's percentage: those printed numbers ARE the data.
            # The classical digitizer's angle measurement is kept next to them as a check.
            name = FIGURES[f]["value_series"]
            meas = d["series"][name]
            d["series"][name] = {"x": meas["x"], "y": list(FIGURES[f]["printed_values"]),
                                 "flags": ["printed label"] * len(meas["x"]),
                                 "measured_by_digitizer": meas["y"]}
        path.write_text(json.dumps(d, indent=1))
        out[f] = d
    return out


def overlay_reference(P, reference):
    """QA image: v1 points drawn on top of each original."""
    from PIL import Image, ImageDraw
    from .digitize import Axis
    paths = []
    for f in ORDER:
        F = FIGURES[f]; c = F["calib"]
        im = Image.open(P.original(f)).convert("RGB"); d = ImageDraw.Draw(im)
        if F["chart_type"] == "pie":
            _overlay_pie(F, reference[f], d)
            p = P.qa / f"{f}_v1_overlay.png"; im.save(p); paths.append(p)
            continue
        if F["chart_type"].startswith("bar"):
            _overlay_bars(F, reference[f], d)
            p = P.qa / f"{f}_v1_overlay.png"; im.save(p); paths.append(p)
            continue
        X = Axis(*c["x"])
        for name, s in reference[f]["series"].items():
            ax = F["series"][name].get("axis", "left")
            Y = Axis(*c[{"left": "y", "right": "y_right", "upper": "y_upper", "lower": "y_lower"}[ax]])
            px, py = X.to_pixel(np.array(s["x"])), Y.to_pixel(np.array(s["y"]))
            flags = s.get("flags", ["detected"] * len(px))
            step = max(1, len(px) // 40)
            for i in range(0, len(px), step):
                col = {"detected": (0, 200, 0), "local_search": (0, 160, 255)}.get(flags[i], (255, 140, 0))
                d.ellipse([px[i] - 3, py[i] - 3, px[i] + 3, py[i] + 3], outline=col, width=2)
        p = P.qa / f"{f}_v1_overlay.png"; im.save(p); paths.append(p)
    return paths


def _overlay_bars(F, ref, d):
    """QA: a green line across every bar top that was read, and the error-bar ends."""
    from .digitize import Axis
    from .figures import series_axis
    c = F["calib"]; X = Axis(*c["x"])
    key = {"left": "y", "right": "y_right", "upper": "y_upper", "lower": "y_lower"}
    for name, s in ref["series"].items():
        st = F["series"][name]
        for i, (x, y) in enumerate(zip(s["x"], s["y"])):
            ax = series_axis(F_id(F), name, x, y)
            k = key[ax] if key[ax] in c else ("y_lower" if "y_lower" in c else "y")
            Y = Axis(*c[k])
            cx = float(X.to_pixel(x)) + st["offset_px"]; hw = st["width_px"] / 2
            py = float(Y.to_pixel(y))
            d.line([cx - hw, py, cx + hw, py], fill=(0, 200, 0), width=2)
            e = (s.get("yerr") or [0] * len(s["x"]))[i]
            if e:
                pe = float(Y.to_pixel(y + e))
                d.line([cx - 4, pe, cx + 4, pe], fill=(255, 140, 0), width=2)


def F_id(F):
    return next(k for k, v in FIGURES.items() if v is F)


def _overlay_pie(F, ref, d):
    """QA: rays at the slice boundaries the digitizer measured."""
    import math
    from .digitize import pie_point
    cal = F["calib"]["pie"]; cx, cy = cal["apex"]
    for deg in ref.get("boundaries_deg", []):
        x, y = pie_point(cal, math.radians(deg))
        d.line([cx, cy, float(x), float(y)], fill=(0, 200, 0), width=3)


def rebuild_all(P, reference, force=False):
    """Stage 2: draw every figure again from the v1 data; that data becomes the
    exact ground truth of the rebuilt image. Existing rebuilt images are kept
    (force=False) so cached model answers always refer to the same pixels;
    if you re-render, delete outputs/model_results too."""
    from .render import render
    for f in ORDER:
        if force or not P.rebuilt_img(f).exists():
            render(f, reference[f], P.rebuilt_img(f))
        gt = {"figure": f, "image": P.rebuilt_img(f).name, "series": {
            n: {k: v for k, v in s.items() if k in ("x", "y", "yerr")} for n, s in reference[f]["series"].items()}}
        (P.ground_truth / f"{f}.json").write_text(json.dumps(gt, indent=1))
        write_ground_truth_csv(P, f, reference[f])
    return [P.rebuilt_img(f) for f in ORDER]


def write_ground_truth_csv(P, fig_id, ref):
    """Same ground truth as the JSON, as a flat table: one row per point."""
    import csv
    F = FIGURES[fig_id]
    with open(P.ground_truth / f"{fig_id}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "y_axis", "x", "y", "yerr", "how_obtained"])
        for name, s in ref["series"].items():
            n = len(s["x"])
            yerr = s.get("yerr") or [""] * n
            flags = s.get("flags") or (["traced curve"] if F["series"][name]["kind"] == "line" else ["detected"]) * n
            for i in range(n):
                w.writerow([name, F["series"][name].get("axis", "left"), s["x"][i], s["y"][i], yerr[i], flags[i]])


def side_by_side(P, fig_id, right_img, out_name, left_title="original", right_title="rebuilt"):
    from PIL import Image, ImageDraw
    a = Image.open(P.original(fig_id)).convert("RGB"); b = Image.open(right_img).convert("RGB")
    W, H = a.size
    canvas = Image.new("RGB", (2 * W + 20, H + 24), "white")
    canvas.paste(a, (0, 24)); canvas.paste(b.resize(a.size), (W + 20, 24))
    d = ImageDraw.Draw(canvas)
    d.text((6, 4), left_title, fill=(0, 0, 0)); d.text((W + 26, 4), right_title, fill=(0, 0, 0))
    p = P.qa / out_name; canvas.save(p)
    return p


def jobs(P, sets=("original", "rebuilt")):
    out = []
    for s in sets:
        for f in ORDER:
            out.append((s, f, P.original(f) if s == "original" else P.rebuilt_img(f)))
    return out


def load_answer(P, model_key, image_set, fig_id):
    p = P.results / model_key / image_set / f"{fig_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


def ground_truth(P, fig_id):
    return json.loads((P.ground_truth / f"{fig_id}.json").read_text())


def score_model(P, model_key):
    """Per-figure scores of one model on both image sets (see reporting.py for tables)."""
    from .scoring import score_figure
    rows = []
    for f in ORDER:
        gt = ground_truth(P, f)
        for image_set in ("rebuilt", "original"):
            a = load_answer(P, model_key, image_set, f)
            if a is None:
                continue
            s = score_figure(f, gt, a["parsed"])
            rows.append({"model": model_key, "image_set": image_set, "figure": f, "hit_rate": s["hit_rate"],
                         "accuracy": s["accuracy"], "coverage": s["coverage"], "seconds": a.get("seconds"),
                         "error": a.get("error")})
    return rows


def model_rebuild(P, model_key, image_set="original"):
    """Re-plot a model's own extraction in the original's style (visual check
    of 'how close would a rebuild from this model's data have been')."""
    from .render import render
    out = []
    d = P.model_rebuilds / model_key; d.mkdir(parents=True, exist_ok=True)
    for f in ORDER:
        a = load_answer(P, model_key, image_set, f)
        if a is None:
            continue
        p = d / f"{f}_from_{model_key}.png"
        try:
            render(f, a["parsed"], p, clip=True); out.append((f, p))
        except Exception as e:
            print(f"[{model_key}] could not re-plot {f}: {e}")
    return out
