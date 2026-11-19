"""Extended figure set: every image of ``../data`` and ``../data/test_data`` that has a profile in
``profiles.py`` (the prof_*.py files). Same three stages as the original 12 figures:

1. v1 data from the original image (``engine.extract_figure``: hand-checked calibration + colours)
2. rebuild the figure from that data (``render2.render_figure``) - the rebuilt image is the test
   image and the data behind it is its EXACT ground truth
3. QA: original and rebuild side by side (saved, not displayed) + a pixel comparison per series
   that FLAGS rebuilds that are too far off (``qa.check_figure``)

Ground truth keeps every panel / inset apart::

    {"figure": id, "image": "<id>_rebuilt.png", "source_image": "...", "layout": "single|panels|main+inset",
     "panels": {panel_id: {"role": "main|panel|inset", "title": ..., "x_axis": ..., "y_axis": ...,
                           "categories": [...]?, "series": {name: {"kind", "x", "y", "yerr"?, "bottom"?}}}}}

The same module builds the VLM prompt for these figures, parses the answers and scores them, so
Real_Plots_MultiModel_recreation.ipynb can benchmark the models on the extended set too.
"""
from __future__ import annotations

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # silence missing-font notes

import csv
import json
import re
from pathlib import Path

import numpy as np

from . import prof_all  # noqa: F401  (registers every profile)
from .profiles import NOT_PROFILED, PROFILES
from .digitize import Axis


def ids(include_duplicates=False):
    return [k for k, F in PROFILES.items() if include_duplicates or not F.get("duplicate_of")]


def original(P, fid):
    return Path(P.data) / PROFILES[fid]["file"]


def rebuilt_img(P, fid):
    return P.rebuilt / f"{fid}_rebuilt.png"


# ----------------------------------------------------------------------------- ground truth
def _axis_info(Pn):
    """titles / units / ranges of a panel's axes (for the GT file and the VLM prompt)"""
    def rng(key, lo_px, hi_px):
        if Pn.get(key) is None:
            return None
        A = Axis(*Pn[key][:4])
        a, b = float(A.to_value(lo_px)), float(A.to_value(hi_px))
        return [round(min(a, b), 6), round(max(a, b), 6)]
    L, R, T, B = Pn["frame"]
    out = {"x_axis": {"title": Pn.get("x_title"), "unit": Pn.get("x_unit"), "range": rng("x", L, R) if not Pn.get("categories") else None},
           "y_axis": {"title": Pn.get("y_title"), "range": rng("y", B, T)}}
    if Pn.get("y_right"):
        out["y_axis_right"] = {"title": Pn.get("y_right_title"), "range": rng("y_right", B, T)}
    if Pn.get("y_segments"):
        out["y_axis"]["segments"] = [[float(Axis(*g["map"]).to_value(max(g["rows"]))), float(Axis(*g["map"]).to_value(min(g["rows"])))]
                                     for g in Pn["y_segments"]]
    if Pn.get("x_segments"):
        out["x_axis"]["segments"] = [[float(Axis(*g["map"]).to_value(min(g["px"]))), float(Axis(*g["map"]).to_value(max(g["px"])))]
                                     for g in Pn["x_segments"]]
    if "split_px" in Pn:
        out["y_axis"]["broken"] = True
    return out


def ground_truth_of(fid, ref):
    F = PROFILES[fid]
    keep = ("kind", "x", "y", "yerr", "bottom")
    gt = {"figure": fid, "image": f"{fid}_rebuilt.png", "source_image": F["file"], "title": F.get("title"),
          "chart_type": F.get("chart_type"), "layout": ref["layout"], "panels": {}}
    for Pn in F["panels"]:
        d = ref["panels"][Pn["id"]]
        p = {"role": d.get("role", "main"), "title": Pn.get("panel_title") or (Pn.get("label") or {}).get("text"),
             **_axis_info(Pn), "series": {}}
        if Pn.get("categories"):
            p["categories"] = list(Pn["categories"])
        for k in ("printed_fit_parameters", "printed_equations", "printed_values"):
            if d.get(k):
                p[k] = d[k]
        for n, s in d["series"].items():
            p["series"][n] = {k: s[k] for k in keep if k in s and s[k] is not None}
            if Pn["series"].get(n, {}).get("axis") == "right":
                p["series"][n]["y_axis"] = "right"
        gt["panels"][Pn["id"]] = p
    return gt


def write_gt_csv(path, gt):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "role", "series", "kind", "category", "x", "y", "yerr", "bottom", "y_axis"])
        for pid, p in gt["panels"].items():
            cats = p.get("categories")
            for n, s in p["series"].items():
                for i in range(len(s.get("x", []))):
                    x = s["x"][i]
                    cat = cats[int(round(x))] if cats and float(x).is_integer() and 0 <= int(x) < len(cats) else ""
                    w.writerow([pid, p["role"], n, s.get("kind", ""), cat, x, s["y"][i],
                                (s.get("yerr") or [""] * len(s["x"]))[i], (s.get("bottom") or [""] * len(s["x"]))[i],
                                s.get("y_axis", "left")])


def build(P, only=None, force=False, log=print):
    """Stages 1-3 for the extended set. Existing rebuilt images are kept (force=False) so cached
    model answers keep referring to the same pixels. Returns the QA table (list of dicts)."""
    from .engine import extract_figure
    from .render2 import render_figure
    from . import qa
    rows = []
    for fid in (only or ids()):
        F = PROFILES[fid]
        src = original(P, fid)
        ref_path = P.reference / f"{fid}.json"
        if ref_path.exists() and not force:
            ref = json.loads(ref_path.read_text())
        else:
            ref = extract_figure(F, src)
            ref["source_image"] = F["file"]
            ref_path.write_text(json.dumps(ref, indent=1, default=float))
        if force or not rebuilt_img(P, fid).exists():
            render_figure(F, ref, rebuilt_img(P, fid))
        gt = ground_truth_of(fid, ref)
        q = qa.check_figure(F, src, rebuilt_img(P, fid))
        gt["qa"] = {"flagged": q["flagged"], "rule": "a series is OFF when > 15 % of its pixels (original or rebuilt) "
                                                     "are > 6 px from the other image's pixels of that series",
                    "series": q["rows"]}
        (P.ground_truth / f"{fid}.json").write_text(json.dumps(gt, indent=1, default=float))
        write_gt_csv(P.ground_truth / f"{fid}.csv", gt)
        qa.side_by_side(src, rebuilt_img(P, fid), P.qa / f"{fid}_original_vs_rebuilt.png", fid, q["flagged"])
        n_pts = sum(len(s.get("x", [])) for p in gt["panels"].values() for s in p["series"].values())
        off = [f"{r['panel']}/{r['series']}" for r in q["rows"] if r["off"]]
        rows.append({"figure": fid, "source_image": F["file"], "layout": gt["layout"], "panels": len(gt["panels"]),
                     "series": sum(len(p["series"]) for p in gt["panels"].values()), "points": n_pts,
                     "status": "FLAGGED - check the side-by-side" if q["flagged"] else "ok", "series_off": "; ".join(off),
                     "also_in": "; ".join(F.get("duplicates", []))})
        log(f"{fid:22s} {rows[-1]['status']:32s} {n_pts:5d} points  {rows[-1]['series_off']}")
    with open(P.qa / "extended_qa_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return rows


def skipped():
    """images without ground truth in this set, and why"""
    out = [{"image": k, "reason": v} for k, v in NOT_PROFILED.items()]
    out += [{"image": F["file"], "reason": f"same figure as '{F['duplicate_of']}' of the original 12 (ground truth exists there)"}
            for F in PROFILES.values() if F.get("duplicate_of")]
    for F in PROFILES.values():
        for d in F.get("duplicates", []):
            out.append({"image": d, "reason": f"identical file to {F['file']} (ground truth '{F['id']}')"})
    return out


# ----------------------------------------------------------------------------- VLM prompt / parsing / scoring
def build_prompt(fid):
    """Data-extraction prompt for one extended figure: describes every panel / inset (titles, axes,
    series, categories) and asks for ONE JSON object keyed by panel. Never contains data values."""
    F = PROFILES[fid]
    gt_like = ground_truth_of(fid, {"layout": "", "panels": {Pn["id"]: {"series": {n: {} for n in Pn["series"]}, "role": Pn.get("role", "main")}
                                                           for Pn in F["panels"]}})
    lines = [f"This figure: {F.get('title', '')}."]
    if len(F["panels"]) > 1:
        roles = {Pn.get("role", "main") for Pn in F["panels"]}
        kind = "a main plot with a small INSET plot drawn inside it" if "inset" in roles else f"{len(F['panels'])} separate plot panels"
        lines.append(f"The image contains {kind}. Treat every panel separately: each has its own axes, data and scale, "
                     f"even when axis titles are shared or printed once for all panels.")
    for Pn in F["panels"]:
        p = gt_like["panels"][Pn["id"]]
        desc = [f'- panel "{Pn["id"]}" ({p["role"]}' + (f', labelled {p["title"]}' if p.get("title") else "") + "):"]
        xa, ya = p["x_axis"], p["y_axis"]
        if Pn.get("categories"):
            desc.append(f"  x = categories {json.dumps(Pn['categories'])} (answer the category name)")
        else:
            desc.append(f"  x-axis '{xa.get('title') or ''}'" + (f" from {xa['range'][0]:g} to {xa['range'][1]:g}" if xa.get("range") else "") +
                        (" (axis broken into segments)" if xa.get("segments") else ""))
        desc.append(f"  y-axis '{ya.get('title') or ''}'" + (f" from {ya['range'][0]:g} to {ya['range'][1]:g}" if ya.get("range") else "") +
                    (" (axis broken into segments)" if ya.get("segments") or ya.get("broken") else "") +
                    (" - NO numeric ticks: give y as % of the panel height (0 = bottom, 100 = top)" if Pn.get("no_y_ticks") else ""))
        if p.get("y_axis_right"):
            r = p["y_axis_right"]
            desc.append(f"  right y-axis '{r.get('title') or ''}' from {r['range'][0]:g} to {r['range'][1]:g}")
        ser = []
        for n, s in Pn["series"].items():
            k = s["kind"]
            what = {"line": "curve", "marker": "markers", "bar": "bars", "stacked": "stacked bar segments",
                    "pattern_stack": "stacked bar segments (told apart by fill pattern)", "stick": "reference sticks"}.get(k, k)
            ser.append(f"'{n}' ({what}" + (", right axis" if s.get("axis") == "right" else "") + ")")
        desc.append("  series: " + ", ".join(ser))
        lines += desc
    example = {Pn["id"]: {n: ([["<category>", 0.0]] if Pn.get("categories") else [[0.0, 0.0]]) for n in list(Pn["series"])[:1]}
               for Pn in F["panels"][:2]}
    lines.append("Task: extract the underlying data of EVERY series in EVERY panel: every marker / bar / segment, and "
                 "curves sampled densely enough to redraw them. Stacked segments: the segment's own height.")
    lines.append("Respond with ONLY one JSON object: {panel id: {series name: [[x, y], ...]}}, e.g. "
                 f"{json.dumps(example)}. Use exactly the panel ids and series names above. Numbers only, no units, "
                 "no explanation, no markdown.")
    return "\n".join(lines)


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _cat_index(cats, x):
    if isinstance(x, (int, float)):            # a category index (markers between bars may sit at fractions)
        return float(x) if -0.5 <= x <= len(cats) - 0.5 else None
    k = _norm(x)
    for i, c in enumerate(cats):
        if _norm(c) == k:
            return float(i)
    for i, c in enumerate(cats):
        if k and (k in _norm(c) or _norm(c) in k):
            return float(i)
    return None


def parse_answer(text, fid):
    """-> {"panels": {pid: {"series": {name: {"x", "y"}}}}}; a flat {series: points} answer is
    accepted for single-panel figures."""
    from .models import _load_json_blob, match_series
    F = PROFILES[fid]
    obj = _load_json_blob(text or "")
    out = {"panels": {Pn["id"]: {"series": {}} for Pn in F["panels"]}}
    if not isinstance(obj, dict):
        return out
    pids = [Pn["id"] for Pn in F["panels"]]
    if not any(_norm(k) in {_norm(p) for p in pids} for k in obj):
        obj = {pids[0]: obj} if len(pids) == 1 else obj
    for key, val in obj.items():
        pid = next((p for p in pids if _norm(p) == _norm(key)), None)
        if pid is None or not isinstance(val, dict):
            continue
        Pn = next(q for q in F["panels"] if q["id"] == pid)
        names, used = list(Pn["series"]), set()
        for sk, pts in val.items():
            n = match_series(str(sk), names, used)
            if n is None or not isinstance(pts, list):
                continue
            xs, ys = [], []
            for pt in pts:
                if not (isinstance(pt, (list, tuple)) and len(pt) >= 2):
                    continue
                try:
                    x = _cat_index(Pn["categories"], pt[0]) if Pn.get("categories") else float(pt[0])
                    if x is None:
                        continue
                    xs.append(x); ys.append(float(pt[1]))
                except Exception:
                    continue
            if xs:
                out["panels"][pid]["series"][n] = {"x": xs, "y": ys}
                used.add(n)
    return out


def score(fid, gt, pred, hit_pct=5.0, n_line=50):
    """per truth point: the model's value at that x (interpolated for curves, same category for bars),
    error as % of the panel's y-axis span; hit = error <= 5 %."""
    F = PROFILES[fid]
    rows = []
    for Pn in F["panels"]:
        g = gt["panels"][Pn["id"]]
        span = None
        yr = (g.get("y_axis") or {}).get("range")
        for n, s in g["series"].items():
            if not s.get("x"):
                continue
            ys = g["y_axis_right"]["range"] if s.get("y_axis") == "right" and g.get("y_axis_right") else yr
            sp = abs(ys[1] - ys[0]) if ys else (np.ptp(s["y"]) or 1.0)
            x, y = np.asarray(s["x"], float), np.asarray(s["y"], float)
            if s.get("kind") == "line" and len(x) > n_line:
                o = np.argsort(x); x, y = x[o], y[o]
                gx = np.linspace(x.min(), x.max(), n_line); y = np.interp(gx, x, y); x = gx
            p = pred["panels"].get(Pn["id"], {}).get("series", {}).get(n)
            for xv, yv in zip(x, y):
                ym = np.nan
                if p and p.get("x"):
                    px, py = np.asarray(p["x"], float), np.asarray(p["y"], float)
                    if Pn.get("categories") or s.get("kind") in ("bar", "stacked"):
                        k = np.where(np.abs(px - xv) < 0.3)[0]
                        same = [j for j in k if np.sign(py[j]) == np.sign(yv)]
                        k = same if same else k               # positive / negative segments of one stack
                        ym = float(np.mean(py[k])) if len(k) else np.nan
                    else:
                        o = np.argsort(px); px, py = px[o], py[o]
                        tol = 0.05 * (np.ptp(x) or 1.0)
                        if px.min() - tol <= xv <= px.max() + tol:
                            ym = float(np.interp(xv, px, py))
                err = 100.0 if ym != ym else min(abs(ym - yv) / sp * 100, 100.0)
                rows.append({"panel": Pn["id"], "series": n, "x": float(xv), "y_true": float(yv),
                             "y_model": None if ym != ym else ym, "error_pct_of_axis": err, "missing": ym != ym,
                             "hit": (ym == ym) and err <= hit_pct})
    if not rows:
        return {"figure": fid, "hit_rate": np.nan, "accuracy": np.nan, "coverage": np.nan, "n_points": 0, "rows": []}
    err = np.array([r["error_pct_of_axis"] for r in rows]); miss = np.array([r["missing"] for r in rows])
    return {"figure": fid, "hit_rate": 100 * float(np.mean([r["hit"] for r in rows])), "accuracy": float(100 - err.mean()),
            "coverage": 100 * float((~miss).mean()), "n_points": len(rows), "rows": rows}


def run_model(runner, P, image_set="rebuilt", only=None, force=False, log=print):
    """ask a VLM runner (Qwen family: has ._ask) every extended figure; answers cached in
    outputs/model_results/<runner.key>/<image_set>_extended/<id>.json"""
    import time
    if not hasattr(runner, "_ask"):
        log(f"[{runner.key}] has no free-prompt interface - skipped for the extended set")
        return
    todo = []
    for fid in (only or ids()):
        out = P.results / runner.key / f"{image_set}_extended" / f"{fid}.json"
        if force or not out.exists():
            todo.append((fid, out))
    if not todo:
        log(f"[{runner.key}] extended set: all answers cached"); return
    runner.load()
    try:
        for fid, out in todo:
            img = rebuilt_img(P, fid) if image_set == "rebuilt" else original(P, fid)
            prompt = build_prompt(fid)
            t = time.time()
            try:
                path, _, tmp = runner._input_image(img)
                raw = runner._ask(path, prompt); err = None
                if tmp:
                    Path(path).unlink(missing_ok=True)
            except Exception as e:
                raw, err = "", f"{type(e).__name__}: {e}"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"model": runner.key, "figure": fid, "image": str(img), "prompt": prompt,
                                       "seconds": round(time.time() - t, 1), "error": err, "raw": raw,
                                       "parsed": parse_answer(raw, fid)}, indent=1))
            log(f"[{runner.key}] {fid}: {time.time() - t:.0f}s" + (f"  ERROR {err}" if err else ""))
    finally:
        runner.unload()


def leaderboard(P, keys, image_set="rebuilt"):
    import pandas as pd
    rows = []
    for k in keys:
        for fid in ids():
            a = P.results / k / f"{image_set}_extended" / f"{fid}.json"
            g = P.ground_truth / f"{fid}.json"
            if not (a.exists() and g.exists()):
                continue
            s = score(fid, json.loads(g.read_text()), json.loads(a.read_text())["parsed"])
            rows.append({"model": k, "figure": fid, "hit_rate": s["hit_rate"], "accuracy": s["accuracy"], "coverage": s["coverage"]})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby("model")[["hit_rate", "accuracy", "coverage"]].mean().round(1).assign(
        figures=df.groupby("model").size()).sort_values("hit_rate", ascending=False)
