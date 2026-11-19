"""All benchmark images in one run: the 12 rebuilt plots (notebook 02), the synthetic gallery
and the extended set (original data / test_data figures with hand-calibrated ground truth).

collect(cfg)                    -> list of items {set, name, image, truth, ...}
seed_cache(items, out, old)     -> copy VLM answers of earlier runs (no second VLM call)
run_item(item, reader, out,...) -> VLM spec + digitize + recreate + side-by-side for one image
score_item(item, run, thr)      -> one summary row + one table of points for one image

Scoring per set (5 % of the axis span = a hit, as everywhere in the benchmark):
  rebuilt   report.point_table / prf.prf_rebuilt (same as Auto_VLM_Digitization_3D, 'rebuilt')
  gallery   gallery_eval.score / summary (same as the 'gallery' mode); the 'unsupported'
            plots are scored on rejection only
  extended  chartbench.extended.score for the truth side (hit rate, accuracy, coverage = recall
            side) + precision of every extracted item below; panels of a multi-plot figure are
            paired with the truth panels by frame overlap, series by name.
"""
from __future__ import annotations

import difflib
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as E
from .prf import prf, match_points

SETS = ("rebuilt", "gallery", "extended")


# ============================================================================ items
def collect(cfg):
    """every image of the chosen sets, with its ground truth"""
    bench, gal = Path(cfg["bench_dir"]), Path(cfg["gallery_dir"])
    data = Path(cfg.get("data_dir", "../data"))
    items = []
    if "rebuilt" in cfg["sets"]:
        from . import report as RP
        figures, _ = RP.load_chartbench(bench)
        for fid in figures:
            items.append({"set": "rebuilt", "name": fid, "image": bench / "outputs" / "rebuilt_plots" / f"{fid}_rebuilt.png",
                          "truth": json.loads((bench / "outputs" / "ground_truth" / f"{fid}.json").read_text()),
                          "figures": figures})
    if "gallery" in cfg["sets"]:
        if not (gal / "images").exists() and Path("gallery.zip").exists():
            import zipfile
            zipfile.ZipFile("gallery.zip").extractall(".")
        for p in sorted((gal / "images").glob("*.png")):
            items.append({"set": "gallery", "name": p.stem, "image": p,
                          "truth": json.loads((gal / "truth" / f"{p.stem}.json").read_text())})
    if "extended" in cfg["sets"]:
        if str(bench) not in sys.path:
            sys.path.insert(0, str(bench))
        from chartbench.profiles import PROFILES
        from chartbench import extended as XB
        for fid in XB.ids():
            gp = bench / "outputs" / "ground_truth" / f"{fid}.json"
            img = data / PROFILES[fid]["file"]
            if not gp.exists() or not img.exists():
                continue
            gt = json.loads(gp.read_text())
            items.append({"set": "extended", "name": fid, "image": img, "truth": gt, "profile": PROFILES[fid],
                          "gt_flagged": bool((gt.get("qa") or {}).get("flagged"))})
    return items


def seed_cache(items, out, old_dirs, log=print):
    """reuse VLM answers saved by earlier runs (Auto_VLM_Digitization_3D: <old>/<name>/vlm_spec.json,
    multi-plot figures: <old>/multi/<stem>/...). Only copied where this run has no answer yet."""
    n = 0
    for it in items:
        dst = Path(out) / it["set"] / it["name"]
        for old in map(Path, old_dirs or []):
            if it["set"] == "extended":
                src = old / "multi" / Path(it["image"]).stem
                if src.is_dir() and not dst.exists():
                    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.png", "*.csv"))
                    n += 1
                    break
            else:
                src = old / it["name"] / "vlm_spec.json"
                if src.exists() and not (dst / "vlm_spec.json").exists():
                    dst.mkdir(parents=True, exist_ok=True)
                    shutil.copy(src, dst / "vlm_spec.json")
                    n += 1
                    break
    log(f"VLM answers reused from earlier runs: {n}")
    return n


# ============================================================================ run
def run_item(item, reader, out, use_cache=True, check=True, panels=("extended",)):
    """one image -> {'result': pipeline output (or the combined panels), 'dir', 'panels'?, seconds}"""
    from .run import process_image
    from .panels import process_figure
    root = Path(out) / item["set"]
    t = time.time()
    try:
        if item["set"] in panels:
            r = process_figure(item["image"], reader, root, item["name"], use_cache, check)
        else:
            r = process_image(item["image"], reader, root, item["name"], use_cache, check)
    except Exception as e:                                  # one bad image must not stop the run
        d = root / item["name"]
        d.mkdir(parents=True, exist_ok=True)
        r = {"name": item["name"], "dir": str(d), "error": f"{type(e).__name__}: {e}",
             "result": {"status": "rejected", "status_reasons": [f"run failed: {type(e).__name__}: {e}"],
                        "series": {}, "chart_type": None, "warnings": [], "checks": []}}
    r["seconds"] = round(time.time() - t, 1)
    res = r["result"]
    if not res.get("chart_type"):
        res["chart_type"] = (res.get("spec") or {}).get("chart_type")
    return r


def vlm_seconds(run):
    if run.get("panels"):
        return round(sum(((p.get("spec_info") or {}).get("seconds") or 0) for p in run["panels"].values()), 1)
    return (run.get("spec_info") or {}).get("seconds")


# ============================================================================ extended scoring
def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _cat_map(names, cats):
    """extracted category names -> truth category index (exact, containment, fuzzy), else None"""
    nc = [_norm(c) for c in cats]
    out = []
    for n in names:
        k = _norm(n)
        j = nc.index(k) if k in nc else next((i for i, c in enumerate(nc) if k and c and (k in c or c in k)), None)
        if j is None and k:
            m = difflib.get_close_matches(k, nc, n=1, cutoff=0.6)
            j = nc.index(m[0]) if m else None
        out.append(j)
    return out


def _xy(d, cats=None):
    """extracted series (any extractor's format) -> x, y arrays in truth coordinates.
    Categorical truth: x = category index, found by name where the extractor gives names."""
    if d.get("values") is not None and d.get("categories") is not None and not d.get("x"):
        names, y = list(d["categories"]), list(d["values"])
        x = list(range(len(y)))
    else:
        x, y, names = list(d.get("x") or []), list(d.get("y") or []), d.get("categories")
    integer_x = all(isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer()) for v in x)
    if cats and names and len(names) == len(x) and integer_x:      # category indices: re-index by name
        m = _cat_map(names, cats)
        x = [mi if mi is not None else xi for mi, xi in zip(m, x)]
    try:
        x, y = np.asarray(x, float), np.asarray(y, float)
    except (TypeError, ValueError):
        return np.array([]), np.array([])
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def _iou(a, b):
    """a = (L, T, R, B) found frame, b = (L, R, T, B) profile frame"""
    ix = max(0, min(a[2], b[1]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[2]))
    A = (a[2] - a[0]) * (a[3] - a[1])
    B = (b[1] - b[0]) * (b[3] - b[2])
    return ix * iy / max(1, A + B - ix * iy)


def _candidates(profile, run):
    """truth panel id -> {extracted name: series dict}"""
    pans = profile["panels"]
    cand = {Pn["id"]: {} for Pn in pans}
    if run.get("panels"):
        for k, v in run["panels"].items():
            best = max(pans, key=lambda Pn: _iou(v["frame"], Pn["frame"]))
            for n, d in (v.get("series") or (v.get("result") or {}).get("series") or {}).items():
                cand[best["id"]][f"{k}/{n}" if n in cand[best["id"]] else n] = d
    else:
        main = max(pans, key=lambda Pn: abs((Pn["frame"][1] - Pn["frame"][0]) * (Pn["frame"][3] - Pn["frame"][2])))
        cand[main["id"]] = dict((run.get("result") or {}).get("series") or {})
    return cand


def _span(g, s):
    ys = g["y_axis_right"]["range"] if s.get("y_axis") == "right" and g.get("y_axis_right") else (g.get("y_axis") or {}).get("range")
    return abs(ys[1] - ys[0]) if ys else (float(np.ptp(s["y"])) or 1.0)


def _precision_items(s, g, ex, ey, thr, categorical, n_line=200):
    """(correct, total) extracted items of one series against its truth series"""
    if not len(ex):
        return 0, 0
    tx, ty = np.asarray(s["x"], float), np.asarray(s["y"], float)
    sp = _span(g, s)
    kind = s.get("kind")
    if categorical or kind in ("bar", "stacked"):
        good = 0
        for xv, yv in zip(ex, ey):
            k = np.where(np.abs(tx - xv) < 0.3)[0]
            if len(k):
                same = [j for j in k if np.sign(ty[j]) == np.sign(yv)] or list(k)
                good += int(min(abs(ty[j] - yv) for j in same) / sp * 100 <= thr)
        return good, len(ex)
    if kind == "line":
        if len(ex) > n_line:
            idx = np.linspace(0, len(ex) - 1, n_line).round().astype(int)
            ex, ey = ex[idx], ey[idx]
        o = np.argsort(tx)
        tol = 0.02 * (np.ptp(tx) or 1.0)
        inside = (ex >= tx.min() - tol) & (ex <= tx.max() + tol)
        err = np.abs(np.interp(ex, tx[o], ty[o]) - ey) / sp * 100
        return int((inside & (err <= thr)).sum()), len(ex)
    # markers / sticks: one-to-one pairing, x within half the typical marker spacing
    dx = np.diff(np.sort(np.unique(tx)))
    xtol = 0.5 * float(np.median(dx)) if len(dx) else 0.02 * (abs(tx[0]) or 1.0)

    def ok(i, j):
        e = abs(ty[i] - ey[j]) / sp * 100
        return (abs(tx[i] - ex[j]) <= xtol and e <= thr), abs(tx[i] - ex[j]) / (xtol or 1) + e
    return match_points(tx, ex, ok), len(ex)


def _names_only(cands, refs):
    """match_names without its last 'pair by order' step"""
    out, used = {}, set()
    for r in refs:
        nr = _norm(r)
        hit = next((c for c in cands if c not in used and _norm(c) == nr), None)
        hit = hit or next((c for c in cands if c not in used and nr and _norm(c) and (_norm(c) in nr or nr in _norm(c))), None)
        if hit is None:
            close = difflib.get_close_matches(nr, [_norm(c) for c in cands if c not in used], n=1, cutoff=0.5)
            hit = next((c for c in cands if c not in used and _norm(c) == close[0]), None) if close else None
        if hit is not None:
            out[r] = hit
            used.add(hit)
    return out


def _pair_series(g, truth, xy, categorical):
    """truth series -> extracted series: by name first; series whose names do not match (a figure
    without a legend, or names the VLM made up) are paired by the data - the pairing with the
    smallest mean error (one-to-one)."""
    out = _names_only(list(xy), truth)
    lt = [t for t in truth if t not in out]
    lc = [c for c in xy if c not in out.values()]
    if lt and lc:
        from scipy.optimize import linear_sum_assignment
        C = np.zeros((len(lt), len(lc)))
        for i, t in enumerate(lt):
            s = g["series"][t]
            tx, ty = np.asarray(s["x"], float), np.asarray(s["y"], float)
            sp = _span(g, s)
            for j, c in enumerate(lc):
                ex, ey = xy[c]
                errs = []
                for xv, yv in zip(tx, ty):
                    if categorical or s.get("kind") in ("bar", "stacked"):
                        k = np.where(np.abs(ex - xv) < 0.3)[0]
                        errs.append(min(abs(ey[k] - yv)) / sp * 100 if len(k) else 100.0)
                    else:
                        o = np.argsort(ex)
                        inside = ex.min() - 0.05 * (np.ptp(tx) or 1) <= xv <= ex.max() + 0.05 * (np.ptp(tx) or 1)
                        errs.append(min(abs(np.interp(xv, ex[o], ey[o]) - yv) / sp * 100, 100.0) if inside else 100.0)
                C[i, j] = np.mean(errs) if errs else 100.0
        r, c = linear_sum_assignment(C)
        out.update({lt[a]: lc[b] for a, b in zip(r, c)})
    return out


def score_extended(item, run, thr=5.0):
    from chartbench import extended as XB
    gt, F = item["truth"], item["profile"]
    cand = _candidates(F, run)
    pred, n_e, tp_e, mapping_txt = {"panels": {}}, 0, 0, {}
    for Pn in F["panels"]:
        g = gt["panels"][Pn["id"]]
        got = cand.get(Pn["id"], {})
        cats = g.get("categories")
        xy = {n: _xy(d, cats) for n, d in got.items()}
        xy = {n: v for n, v in xy.items() if len(v[0])}
        truth = [n for n, s in g["series"].items() if s.get("x")]
        mapping = _pair_series(g, truth, xy, bool(cats))                # truth name -> extracted name
        pred["panels"][Pn["id"]] = {"series": {t: {"x": xy[c][0].tolist(), "y": xy[c][1].tolist()} for t, c in mapping.items()}}
        for t, c in mapping.items():
            mapping_txt[f"{Pn['id']}/{t}"] = c
            a, b = _precision_items(g["series"][t], g, xy[c][0], xy[c][1], thr, bool(cats))
            tp_e += a; n_e += b
        for c in set(xy) - set(mapping.values()):                    # series the digitizer made up
            n_e += min(len(xy[c][0]), 200)
    s = XB.score(item["name"], gt, pred, thr)
    rows = pd.DataFrame([{"panel": r["panel"], "series": r["series"], "extracted as": mapping_txt.get(f"{r['panel']}/{r['series']}", "(not found)"),
                          "x": r["x"], "y truth": r["y_true"], "y extracted": r["y_model"],
                          "error % of y-span": round(r["error_pct_of_axis"], 2),
                          "hit (<= 5 %)": "yes" if r["hit"] else ("MISSING" if r["missing"] else "no")} for r in s["rows"]])
    n_t = s["n_points"]
    hits = int(sum(r["hit"] for r in s["rows"]))
    p = prf(hits, n_t, tp_e, n_e)
    found = rows["hit (<= 5 %)"] != "MISSING" if len(rows) else pd.Series(dtype=bool)
    row = {"truth points": n_t, "hit rate %": round(s["hit_rate"], 1), "accuracy %": round(s["accuracy"], 1),
           "coverage %": round(s["coverage"], 1),
           "mean error % (found)": round(rows.loc[found, "error % of y-span"].mean(), 2) if len(rows) and found.any() else None,
           "precision %": p["precision %"], "recall %": p["recall %"], "F1 %": p["F1 %"], "extracted items": p["extracted items"]}
    return row, rows


# ============================================================================ any set
def score_item(item, run, thr=5.0):
    """-> (summary row, points table). Same metrics for every set."""
    res = run["result"]
    base = {"set": item["set"], "figure": item["name"], "status": res.get("status"),
            "chart type (VLM)": res.get("chart_type"), "layout": run.get("layout", "single")}
    if item["set"] == "rebuilt":
        from . import report as RP
        from .prf import prf_rebuilt
        gt = item["truth"]
        mapping = E.match_names([n for n, d in res.get("series", {}).items() if d.get("x")], list(gt["series"]))
        res.setdefault("series", {})
        df = RP.point_table(item["name"], gt, res, mapping)
        sr = RP.summary_row(item["name"], df)
        pr = prf_rebuilt(item["name"], gt, res, mapping, thr)
        row = {"truth points": sr["truth points"], "hit rate %": sr["hit rate %"], "accuracy %": sr["accuracy %"],
               "coverage %": sr["coverage %"], "mean error % (found)": sr["mean error % (found points)"]} | \
              {k: pr.get(k) for k in ("precision %", "recall %", "F1 %", "extracted items")}
        return base | {"truth type": "rebuilt"} | row, df
    if item["set"] == "gallery":
        from . import gallery_eval as GE
        t = item["truth"]
        df = GE.score(t, res, thr)
        sm = GE.summary(item["name"], df, res.get("status"), t, res, thr)
        sm.pop("figure", None); sm.pop("status", None)
        sm["truth points"] = sm.pop("truth items", 0)
        if t.get("type") == "unsupported":
            sm["rejected correctly"] = res.get("status") == "rejected"
        return base | {"truth type": t.get("type")} | sm, df
    row, df = score_extended(item, run, thr)
    return base | {"truth type": item["truth"].get("layout"), "truth flagged in QA": item.get("gt_flagged")} | row, df


def means(table, by="set"):
    """mean metrics per set and overall, two ways:
    'ok plots'              - only plots the pipeline accepted (as in the per-set notebooks)
    'all supported plots'   - end to end: a rejected / failed plot scores 0 recall, 0 F1"""
    cols = ["hit rate %", "accuracy %", "coverage %", "precision %", "recall %", "F1 %"]
    t = table[(table["truth type"] != "unsupported") & (table["truth points"].fillna(0) > 0)].copy()
    full = t.copy()
    rej = full["status"] == "rejected"
    for c in ("hit rate %", "coverage %", "recall %", "F1 %", "accuracy %"):
        full.loc[rej, c] = full.loc[rej, c].fillna(0)
    out = []
    for label, d in (("ok plots", t[t["status"] == "ok"]), ("all supported plots", full)):
        g = d.groupby(by)[cols].mean().round(1)
        g["plots"] = d.groupby(by).size()
        allrow = d[cols].mean().round(1).to_frame("ALL").T
        allrow["plots"] = len(d)
        g = pd.concat([g, allrow])
        g.insert(0, "scope", label)
        out.append(g)
    return pd.concat(out)


class DrySpecReader:
    """dry run without a VLM: spec files (hand-written / generated), looked up in several folders.
    NOT a VLM - images without a spec file are reported as failed."""

    def __init__(self, dirs):
        self.dirs = [Path(d) for d in dirs]

    def read(self, image_path):
        from .spec import renormalize
        p = Path(image_path)
        for d in self.dirs:
            c = d / f"{p.stem}.json"
            if c.exists():
                dd = json.loads(c.read_text())
                return {"raw_axes": "(from file)", "raw_series": "(from file)", "spec": renormalize(dd.get("spec", dd))}
        raise FileNotFoundError(f"no spec file for {p.name}")


def titled_side_by_side(item, run, row, dest):
    """original | recreated with set, status and scores in the caption"""
    from PIL import Image
    from .run import side_by_side
    rec = run.get("recreated")
    if not rec or not Path(rec).exists():
        im = Image.open(item["image"]).convert("RGB")
        rec = Path(run["dir"]) / "no_recreation.png"
        Image.new("RGB", im.size, "white").save(rec)
    f = lambda k: "-" if row.get(k) is None or row.get(k) != row.get(k) else f"{row[k]:.0f}%"
    cap = f"{item['set']} / {item['name']}   [{row.get('status')}]"
    if row.get("truth type") == "unsupported":
        cap += "   unsupported plot: " + ("rejected correctly" if row.get("rejected correctly") else "NOT rejected")
    elif row.get("truth points"):
        cap += f"   hit {f('hit rate %')}   P {f('precision %')}   R {f('recall %')}   F1 {f('F1 %')}"
    return side_by_side(item["image"], rec, dest, cap)
