"""Tables, leaderboard, baseline and diagnostics built from the cached answers.

Nothing here runs a model: everything is computed from
outputs/model_results/<model>/<image_set>/<figure>.json and
outputs/ground_truth/<figure>.json, so the whole report can be rebuilt at any
time in seconds.

Saved files (all under outputs/):
  tables/<model>/<image_set>.json   one entry per figure: summary + every point
  tables/all_models.json            everything above in one file
  leaderboard.csv / diagnostics.csv
  report.html                       the same tables as a standalone page
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from . import config as cfgmod
from .digitize import Axis
from .figures import FIGURES, ORDER, axis_span, is_categorical
from .scoring import point_rows, score_figure, summarize

MODEL_LABELS = {
    "classical": "Classical digitizer (reference reader)",
    "qwen2.5-vl-3b": "Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b": "Qwen2.5-VL-7B-Instruct",
    "qwen3-vl-4b": "Qwen3-VL-4B-Instruct",
    "qwen3-vl-8b": "Qwen3-VL-8B-Instruct",
    "deplot": "DePlot (google/deplot)",
    "olmocr2-native": "olmOCR-2-7B, native OCR prompt",
    "olmocr2-json": "olmOCR-2-7B, JSON data prompt",
    "baseline-midline": "Baseline: flat line through the middle of each axis (no model)",
}
FIG_TITLES = {f: FIGURES[f]["title"] for f in ORDER}


def label(key):
    base = key.split("__")[0]
    extra = key[len(base):].replace("__", " ").strip()
    return MODEL_LABELS.get(base, key) + (f" [{extra}]" if extra else "")


# --------------------------------------------------------------------------- io
def _load(P, key, image_set, fig):
    p = Path(P.results) / key / image_set / f"{fig}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _gt(P, fig):
    return json.loads((Path(P.ground_truth) / f"{fig}.json").read_text())


def available_models(P):
    return sorted(p.name for p in Path(P.results).iterdir() if p.is_dir()) if Path(P.results).exists() else []


# --------------------------------------------------------------------- baseline
def midline_baseline(fig_id, gt):
    """A 'model' that reads nothing: a flat line across the full x-range at the
    middle of each series' y-axis. Any extractor worth using must beat it."""
    c = FIGURES[fig_id]["calib"]
    out = {"series": {}}
    if is_categorical(fig_id):   # bars / pie: one value per category at the middle of its own axis
        from .figures import series_axis
        key = {"left": "y", "right": "y_right", "upper": "y_upper", "lower": "y_lower"}
        for n, s in gt["series"].items():
            ys = []
            for x, y in zip(s["x"], s["y"]):
                ax = series_axis(fig_id, n, x, y)
                k = key[ax] if key[ax] in c else "y_lower"
                T, B = c["frame"][2], c["frame"][3]
                if ax == "upper":
                    B = c["split_px"]
                if ax in ("lower",) or (ax == "left" and k == "y_lower"):
                    T = c["split_px"]
                ys.append(float(Axis(*c[k]).to_value((T + B) / 2)))
            out["series"][n] = {"x": list(s["x"]), "y": ys}
        return out
    for n, s in gt["series"].items():
        ax = FIGURES[fig_id]["series"][n].get("axis", "left")
        Y = Axis(*c[{"left": "y", "right": "y_right", "upper": "y_upper", "lower": "y_lower"}[ax]])
        T, B = c["frame"][2], c["frame"][3]
        if ax == "upper":
            B = c["split_px"]
        if ax == "lower":
            T = c["split_px"]
        mid = float(Y.to_value((T + B) / 2))
        x = np.asarray(s["x"], float)
        out["series"][n] = {"x": [float(x.min()), float(x.max())], "y": [mid, mid]}
    return out


# ----------------------------------------------------------------------- tables
def model_tables(P, key, image_set="rebuilt", save=True):
    """{figure: {"summary": {...}, "rows": [...]}} for one model and image set."""
    out = {}
    for f in ORDER:
        gt = _gt(P, f)
        if key == "baseline-midline":
            pred, extra = midline_baseline(f, gt), {}
        else:
            a = _load(P, key, image_set, f)
            if a is None:
                continue
            pred = a["parsed"]
            extra = {"seconds": a.get("seconds"), "error": a.get("error"),
                     "series_returned": list(pred.get("series", {}).keys()),
                     "points_returned": int(sum(len(s.get("x", [])) for s in pred.get("series", {}).values()))}
        rows = point_rows(f, gt, pred)
        base = summarize(point_rows(f, gt, midline_baseline(f, gt)))
        out[f] = {"figure": f, "title": FIG_TITLES[f], "model": key, "image_set": image_set,
                  "truth": "exact ground truth (data the rebuilt image was drawn from)" if image_set == "rebuilt"
                           else "reference digitization of the published figure (approximate)",
                  "summary": {**summarize(rows), **extra, "baseline_accuracy": base["accuracy"],
                              "baseline_hit_rate": base["hit_rate"]},
                  "rows": rows}
    if save and out:
        d = Path(P.work) / "outputs" / "tables" / key
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{image_set}.json").write_text(json.dumps(out, indent=1, default=_json_default))
    return out


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "✓" if v else "✗"
    v = float(v)
    if v != v:
        return "—"
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.4g}"


CSS = """
<style>
.cb-wrap{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;font-size:12px;color:#1f2328}
.cb-head{font-size:15px;font-weight:600;margin:10px 0 2px}
.cb-sub{color:#57606a;margin-bottom:8px}
.cb-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.cb-card{border:1px solid #d0d7de;border-radius:6px;padding:8px;background:#fff;min-width:0}
.cb-card h4{margin:0 0 2px;font-size:13px}
.cb-card .t{color:#57606a;font-size:11px;margin-bottom:6px}
.cb-kpi{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px}
.cb-kpi div{background:#f6f8fa;border-radius:4px;padding:3px 6px}
.cb-kpi b{font-size:13px}
.cb-scroll{max-height:330px;overflow:auto;border-top:1px solid #eaeef2}
.cb-card table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
.cb-card th{position:sticky;top:0;background:#f6f8fa;text-align:right;padding:3px 5px;font-weight:600;border-bottom:1px solid #d0d7de}
.cb-card th:first-child,.cb-card td:first-child{text-align:left}
.cb-card td{text-align:right;padding:2px 5px;border-bottom:1px solid #f0f2f4;white-space:nowrap}
.cb-card tr.hit td{background:#eef9f0}
.cb-card tr.miss td{background:#fdf0f0}
.cb-card tr.none td{background:#f6f6f6;color:#8c959f}
.cb-card tr.ser td{background:#fff;font-weight:600;text-align:left;padding-top:6px;border-bottom:1px solid #d0d7de}
.cb-card tr.ser span{font-weight:400;color:#57606a}
.cb-card th:first-child,.cb-card td:first-child{text-align:right}
.cb-good{color:#1a7f37}.cb-bad{color:#cf222e}
</style>"""


def card_html(t):
    s = t["summary"]
    good = s["hit_rate"] > s["baseline_hit_rate"] and s["accuracy"] > s["baseline_accuracy"]
    rows = []
    current = None
    for r in t["rows"]:
        if r["series"] != current:  # one header row per series keeps the table narrow
            current = r["series"]
            sr = [x for x in t["rows"] if x["series"] == current]
            nh = sum(x["hit"] for x in sr)
            rows.append(f"<tr class='ser'><td colspan='5'>{html.escape(current)}"
                        f"<span> — {nh}/{len(sr)} within ±{cfgmod.CONFIG['scoring']['hit_threshold']:g}%</span></td></tr>")
        cls = "none" if r["missing"] else ("hit" if r["hit"] else "miss")
        rows.append(f"<tr class='{cls}'><td>{_fmt(r['x'])}</td>"
                    f"<td>{_fmt(r['y_true'])}</td><td>{_fmt(r['y_model'])}</td>"
                    f"<td>{_fmt(r['error_pct_of_axis'])}</td><td>{'—' if r['missing'] else _fmt(r['hit'])}</td></tr>")
    err = f"<div class='cb-bad'>error: {html.escape(str(s.get('error')))}</div>" if s.get("error") else ""
    return (f"<div class='cb-card'><h4>{t['figure']}</h4><div class='t'>{html.escape(t['title'])}</div>"
            f"<div class='cb-kpi'><div>accuracy <b>{_fmt(s['accuracy'])}</b></div>"
            f"<div>hit rate ≤{cfgmod.CONFIG['scoring']['hit_threshold']:g}% <b>{_fmt(s['hit_rate'])}%</b></div>"
            f"<div>coverage <b>{_fmt(s['coverage'])}%</b></div></div>"
            f"<div class='t'>no-model baseline: accuracy {_fmt(s['baseline_accuracy'])}, hit rate "
            f"{_fmt(s['baseline_hit_rate'])}% → <span class='{'cb-good' if good else 'cb-bad'}'>"
            f"{'beats' if good else 'does NOT beat'} baseline</span></div>{err}"
            f"<div class='cb-scroll'><table><tr><th>x</th><th>truth y</th><th>model y</th>"
            f"<th>error (% axis)</th><th>hit</th></tr>"
            + "".join(rows) + "</table></div></div>")


def grid_html(key, tables, image_set):
    if not tables:
        return f"<div class='cb-wrap'><div class='cb-sub'>No cached answers for <b>{html.escape(label(key))}</b> on the {image_set} images.</div></div>"
    hits = np.mean([t["summary"]["hit_rate"] for t in tables.values()])
    acc = np.mean([t["summary"]["accuracy"] for t in tables.values()])
    truth = next(iter(tables.values()))["truth"]
    cards = "".join(card_html(tables[f]) for f in ORDER if f in tables)
    return (CSS + f"<div class='cb-wrap'><div class='cb-head'>{html.escape(label(key))} — {image_set} images</div>"
            f"<div class='cb-sub'>Truth = {truth}. Mean over the figures: hit rate <b>{hits:.1f}%</b>, "
            f"accuracy <b>{acc:.1f}</b>. Green = within ±{cfgmod.CONFIG['scoring']['hit_threshold']:g}% of the axis span, "
            f"red = outside, grey = not returned.</div><div class='cb-grid'>{cards}</div></div>")


def show_model(P, key):
    """Display (in a notebook) the 2 x 3 grid of per-figure tables for a model."""
    from IPython.display import HTML, display
    S = cfgmod.CONFIG["scoring"]
    sets = [S["image_set_for_tables"]] + (["original"] if S["show_original_tables_too"] and
                                          S["image_set_for_tables"] != "original" else [])
    for image_set in sets:
        display(HTML(grid_html(key, model_tables(P, key, image_set), image_set)))


# -------------------------------------------------------------------- leaderboard
def leaderboard(P, keys):
    import pandas as pd
    rows = []
    for key in list(keys) + ["baseline-midline"]:
        rec = {"model": key, "label": label(key)}
        for image_set in ("rebuilt", "original"):
            t = model_tables(P, key, image_set, save=key != "baseline-midline")
            if not t:
                continue
            sm = [v["summary"] for v in t.values()]
            rec[f"hit_rate_{image_set}"] = np.mean([s["hit_rate"] for s in sm])
            rec[f"accuracy_{image_set}"] = np.mean([s["accuracy"] for s in sm])
            rec[f"coverage_{image_set}"] = np.mean([s["coverage"] for s in sm])
            if key != "baseline-midline":
                secs = [s.get("seconds") for s in sm if s.get("seconds") is not None]
                rec[f"sec_per_image_{image_set}"] = np.mean(secs) if secs else np.nan
            rec[f"figures_{image_set}"] = len(t)
        rows.append(rec)
    df = pd.DataFrame(rows).set_index("model")
    if "hit_rate_rebuilt" in df:
        df = df.sort_values("hit_rate_rebuilt", ascending=False)
    return df


def per_figure_matrix(P, keys, metric="hit_rate", image_set="rebuilt"):
    import pandas as pd
    data = {}
    for key in list(keys) + ["baseline-midline"]:
        t = model_tables(P, key, image_set, save=False)
        if t:
            data[label(key)] = {f: t[f]["summary"][metric] for f in t}
    return pd.DataFrame(data).T.reindex(columns=ORDER)


def save_all(P, keys):
    allt = {}
    for key in keys:
        allt[key] = {s: model_tables(P, key, s) for s in ("rebuilt", "original")}
    allt["baseline-midline"] = {s: model_tables(P, "baseline-midline", s, save=False) for s in ("rebuilt",)}
    p = Path(P.work) / "outputs" / "tables" / "all_models.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"config": cfgmod.CONFIG, "tables": allt}, indent=1, default=_json_default))
    return p


# --------------------------------------------------------------------- diagnostics
def _series_patterns(pred_series):
    """Flags that show a model produced a template instead of reading the plot."""
    x = np.asarray(pred_series.get("x", []), float); y = np.asarray(pred_series.get("y", []), float)
    if len(y) < 3:
        return {"constant_y": False, "even_steps_y": False, "even_x_grid": False}
    dy = np.diff(y); dx = np.diff(np.sort(x))
    span_y = np.ptp(y)
    constant = span_y == 0
    even_y = (not constant) and np.allclose(dy, dy[0], rtol=0.02, atol=1e-9)
    even_x = len(dx) > 1 and np.allclose(dx, dx[0], rtol=0.02, atol=1e-9)
    return {"constant_y": bool(constant), "even_steps_y": bool(even_y), "even_x_grid": bool(even_x)}


def diagnostics(P, keys):
    """Evidence for *why* a model scores the way it does (rebuilt images)."""
    import pandas as pd
    rows = []
    for key in keys:
        if key == "classical":
            continue
        n_ref = n_ret = n_series = n_const = n_even_y = n_marker = n_even_x_bad = same = 0
        xerr = []
        tokens = []
        for f in ORDER:
            a = _load(P, key, "rebuilt", f); o = _load(P, key, "original", f)
            if a is None:
                continue
            gt = _gt(P, f)
            rc = a.get("run_config") or {}
            if rc.get("visual_tokens"):
                tokens.append(rc["visual_tokens"])
            n_ref += sum(len(s["x"]) if FIGURES[f]["series"][n]["kind"] != "line" else cfgmod.CONFIG["scoring"]["line_points"]
                         for n, s in gt["series"].items())
            pred = a["parsed"].get("series", {})
            n_ret += sum(len(s.get("x", [])) for s in pred.values())
            if o is not None and pred and o["parsed"] == a["parsed"]:
                same += 1
            xs_span = abs(Axis(*FIGURES[f]["calib"]["x"]).to_value(FIGURES[f]["calib"]["frame"][1]) -
                          Axis(*FIGURES[f]["calib"]["x"]).to_value(FIGURES[f]["calib"]["frame"][0]))
            for n, s in pred.items():
                n_series += 1
                pat = _series_patterns(s)
                n_const += pat["constant_y"]; n_even_y += pat["even_steps_y"]
                if FIGURES[f]["series"].get(n, {}).get("kind") == "marker" and n in gt["series"]:
                    gx = np.asarray(gt["series"][n]["x"], float)
                    px = np.asarray(s.get("x", []), float)
                    if len(px):
                        n_marker += 1
                        gt_even = len(gx) > 2 and np.allclose(np.diff(gx), np.diff(gx)[0], rtol=0.02)
                        n_even_x_bad += pat["even_x_grid"] and not gt_even
                        xerr.extend(np.min(np.abs(gx[:, None] - px[None, :]), axis=1) / xs_span * 100)
        if n_series == 0 and n_ret == 0:
            rows.append({"model": key, "label": label(key), "points returned / truth points": f"0 / {n_ref}",
                         "note": "returned no data points at all"})
            continue
        rows.append({
            "model": key, "label": label(key),
            "points returned / truth points": f"{n_ret} / {n_ref}",
            "series with constant y (%)": 100 * n_const / max(n_series, 1),
            "series with perfectly even y-steps (%)": 100 * n_even_y / max(n_series, 1),
            "marker series answered on an invented even x-grid (%)": 100 * n_even_x_bad / max(n_marker, 1),
            "x-placement error of markers (% of x-axis)": float(np.mean(xerr)) if xerr else np.nan,
            "figures with identical answer for original & rebuilt": same,
            "visual tokens per image (min-max)": (f"{min(tokens)}-{max(tokens)}" if tokens else _tokens_for(key)),
        })
    return pd.DataFrame(rows).set_index("model")


def _tokens_for(key):
    """Token range for runs made before run_config was recorded (default settings)."""
    tk = visual_token_table(upscale=1.0)
    col = ("Qwen2.5-VL tokens (28 px each)" if key.startswith("qwen2.5") else
           "Qwen3-VL tokens (32 px each)" if key.startswith("qwen3") else
           "olmOCR tokens (1288 px input)" if key.startswith("olmocr") else None)
    if col is None:
        return "n/a (Pix2Struct patches)"
    return f"{tk[col].min()}-{tk[col].max()} (computed)"


def visual_token_table(upscale=None):
    """How many visual tokens each model gets per figure at the configured input size,
    and how big a marker is relative to one token."""
    import pandas as pd
    from .models import visual_tokens
    C = cfgmod.CONFIG
    up = C["vlm_input"]["upscale"] if upscale is None else upscale
    rows = []
    for f in ORDER:
        w, h = C["images"]["original_sizes_px"][f]
        W, H = round(w * up), round(h * up)
        q25 = visual_tokens(W, H, 28, max_pixels=C["vlm_input"]["qwen25_max_pixels"])
        q3 = visual_tokens(W, H, 32)
        s = C["vlm_input"]["olmocr_long_side"] / max(w, h)
        ol = visual_tokens(round(w * s), round(h * s), 28, max_pixels=C["vlm_input"]["qwen25_max_pixels"])
        ms = [v.get("ms") for v in FIGURES[f]["series"].values() if v.get("kind") == "marker" and v.get("ms")]
        marker_px = (np.mean(ms) * up) if ms else None
        rows.append({"figure": f, "image fed (px)": f"{W}x{H}",
                     "Qwen2.5-VL tokens (28 px each)": q25[1], "Qwen3-VL tokens (32 px each)": q3[1],
                     "olmOCR tokens (1288 px input)": ol[1],
                     "typical marker size (px)": round(marker_px, 1) if marker_px else "line plot",
                     "marker area / token area (Qwen2.5)": round((marker_px / 28) ** 2, 2) if marker_px else "—"})
    return pd.DataFrame(rows).set_index("figure")


# ---------------------------------------------------------------- overlay plots
MODEL_STYLE = {"qwen2.5-vl-3b": ("#1f77b4", "o"), "qwen2.5-vl-7b": ("#17becf", "s"), "qwen3-vl-4b": ("#d62728", "D"),
               "qwen3-vl-8b": ("#ff7f0e", "P"), "deplot": ("#9467bd", "v"), "olmocr2-json": ("#8c564b", "X"),
               "olmocr2-native": ("#7f7f7f", "x")}


def plot_extractions(P, keys, image_set="rebuilt", series_per_fig=None):
    """For each figure, the truth (black) and what every model returned, in data
    coordinates. One series per figure (the first, or the one given in
    series_per_fig) keeps the panels readable."""
    import matplotlib.pyplot as plt
    series_per_fig = series_per_fig or {}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, f in zip(axes.ravel(), ORDER):
        gt = _gt(P, f)
        name = series_per_fig.get(f, next(iter(gt["series"])))
        s = gt["series"][name]
        kind = FIGURES[f]["series"][name]["kind"]
        ax.plot(s["x"], s["y"], "k-" if kind == "line" else "k-o", lw=2.2, ms=5, label="truth", zorder=5)
        for key in keys:
            if key == "classical":
                continue
            a = _load(P, key, image_set, f)
            if not a or name not in a["parsed"].get("series", {}):
                continue
            p = a["parsed"]["series"][name]
            col, mk = MODEL_STYLE.get(key.split("__")[0], ("#999999", "."))
            o = np.argsort(p["x"])
            ax.plot(np.asarray(p["x"])[o], np.asarray(p["y"])[o], "-", marker=mk, color=col, lw=1, ms=4,
                    alpha=0.85, label=label(key))
        c = FIGURES[f]["calib"]; X = Axis(*c["x"])
        ax.set_xlim(sorted([float(X.to_value(c["frame"][0])), float(X.to_value(c["frame"][1]))]))
        if f == "ftir":
            ax.invert_xaxis()
        ax.set_title(f"{f}: series '{name}'", fontsize=11)
        ax.grid(alpha=0.3)
    h, l = axes[0, 0].get_legend_handles_labels()
    for ax in axes.ravel()[1:]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 6), fontsize=10, frameon=False)
    fig.suptitle(f"What the models returned vs. the truth ({image_set} images)", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig


# --------------------------------------------------------------------------- report
def write_report(P, keys, path=None):
    """Standalone HTML page with config, leaderboard, diagnostics and every table."""
    lb = leaderboard(P, keys)
    dg = diagnostics(P, keys)
    tk = visual_token_table()
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Chart extraction benchmark</title>"
             "<meta name='viewport' content='width=device-width,initial-scale=1'>" + CSS +
             "<style>body{margin:24px;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1f2328}"
             "table.lb{border-collapse:collapse;font-size:12px}table.lb td,table.lb th{border:1px solid #d0d7de;padding:4px 6px;text-align:right}"
             "table.lb th:first-child,table.lb td:first-child{text-align:left}@media(max-width:1100px){.cb-grid{grid-template-columns:1fr}}</style>"
             "</head><body><h2>Chart data extraction benchmark — 6 phosphorus-paper figures</h2>"]
    parts.append("<h3>Leaderboard (mean over 6 figures)</h3>" + lb.round(1).to_html(classes="lb"))
    parts.append("<h3>Diagnostics (rebuilt images)</h3>" + dg.round(1).to_html(classes="lb"))
    parts.append("<h3>Visual tokens per image</h3>" + tk.to_html(classes="lb"))
    for key in keys:
        for image_set in ("rebuilt", "original"):
            parts.append(grid_html(key, model_tables(P, key, image_set), image_set))
    parts.append("<h3>Configuration</h3><pre>" + html.escape(json.dumps(cfgmod.CONFIG, indent=1, default=str)) + "</pre></body></html>")
    path = Path(path or Path(P.work) / "outputs" / "report.html")
    path.write_text("".join(parts))
    lb.to_csv(Path(P.work) / "outputs" / "leaderboard.csv")
    dg.to_csv(Path(P.work) / "outputs" / "diagnostics.csv")
    return path


# ------------------------------------------------------------ presentation figures
def _out_of_range(fig_id, data):
    """How many of a model's points fall outside the plotted axis ranges
    (they exist in the answer but cannot appear on the re-plotted figure)."""
    c = FIGURES[fig_id]["calib"]
    if is_categorical(fig_id):   # bars / pie: out of range = unknown category or outside its own axis
        from .figures import series_axis
        n, ncat = 0, len(FIGURES[fig_id]["categories"])
        key = {"left": "y", "right": "y_right", "upper": "y_upper", "lower": "y_lower"}
        for name, s in (data or {}).get("series", {}).items():
            if name not in FIGURES[fig_id]["series"]:
                continue
            for x, y in zip(s.get("x", []), s.get("y", [])):
                try:
                    x, y = float(x), float(y)
                except (TypeError, ValueError):
                    continue
                if not (-0.5 < x < ncat - 0.5):
                    n += 1; continue
                ax = series_axis(fig_id, name, x, y)
                k = key[ax] if key[ax] in c else "y_lower"
                if k == "y_upper":
                    lo_v = float(Axis(*c[k]).to_value(c["split_px"])); hi_v = float(Axis(*c[k]).to_value(c["frame"][2]))
                else:
                    lo_v = float(Axis(*c[k]).to_value(c["frame"][3]))
                    hi_v = float(Axis(*c["y_upper"]).to_value(c["frame"][2])) if (k == "y_lower" and "y_upper" in c) \
                        else float(Axis(*c[k]).to_value(c["frame"][2]))
                n += not (min(lo_v, hi_v) <= y <= max(lo_v, hi_v))
        return n
    X = Axis(*c["x"]); L, R, T, B = c["frame"]
    xlo, xhi = sorted([float(X.to_value(L)), float(X.to_value(R))])
    n = 0
    for name, s in (data or {}).get("series", {}).items():
        if name not in FIGURES[fig_id]["series"]:
            continue
        ax = FIGURES[fig_id]["series"][name].get("axis", "left")
        Y = Axis(*c[{"left": "y", "right": "y_right", "upper": "y_upper", "lower": "y_lower"}[ax]])
        t, b = T, B
        if ax == "upper":
            b = c["split_px"]
        if ax == "lower":
            t = c["split_px"]
        ylo, yhi = sorted([float(Y.to_value(b)), float(Y.to_value(t))])
        for x, y in zip(s.get("x", []), s.get("y", [])):
            try:
                x, y = float(x), float(y)
            except (TypeError, ValueError):
                continue
            n += not (xlo <= x <= xhi and ylo <= y <= yhi)
    return n


def presentation_figure(P, key, fig_id, dpi=150):
    """Slide-ready figure for one model and one plot, rebuilt image only:
         rebuilt image (exact ground truth)  |  the model's values read from it, re-plotted
    The right-hand plot uses exactly the same axes, style and legend, so the only
    difference to the left-hand image is the data. Saved as
    outputs/presentation/<model>/<figure>.png."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from .render import render

    out_dir = Path(P.work) / "outputs" / "presentation" / key
    (out_dir / "replots").mkdir(parents=True, exist_ok=True)
    gt = _gt(P, fig_id)
    a = _load(P, key, "rebuilt", fig_id)
    data = a["parsed"] if a else {"series": {}}
    rp = out_dir / "replots" / f"{fig_id}_rebuilt_{key}.png"
    render(fig_id, data, rp, clip=True)
    sm = summarize(point_rows(fig_id, gt, data))
    n_ret = sum(len(s.get("x", [])) for s in data.get("series", {}).values())
    oor = _out_of_range(fig_id, data)
    note = (f"hit rate {sm['hit_rate']:.0f}%  |  mean error {100 - sm['accuracy']:.1f}% of axis  |  "
            f"{len(data.get('series', {}))}/{len(gt['series'])} series returned, {n_ret} points")
    if oor:
        note += f"  |  {oor} outside axes (not drawn)"
    if a is None:
        note = "no answer cached"
    elif n_ret == 0:
        note = "model returned no data points"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.9))
    for ax, img, title in ((axes[0], P.rebuilt_img(fig_id), "Rebuilt figure = exact ground truth\n(the image the model read)"),
                           (axes[1], rp, f"Values extracted by {label(key)}, re-plotted on the same axes\n{note}")):
        ax.imshow(mpimg.imread(img)); ax.set_axis_off()
        ax.set_title(title, fontsize=10.5, loc="left")
    fig.suptitle(f"{fig_id}: {FIG_TITLES[fig_id]}  —  {label(key)}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out_dir / f"{fig_id}.png"
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


BAR_COLORS = {  # fixed order, validated categorical palette; classical = neutral reference
    "classical": "#8a8983", "qwen2.5-vl-3b": "#2a78d6", "qwen2.5-vl-7b": "#eb6834",
    "qwen3-vl-4b": "#1baf7a", "deplot": "#eda100", "olmocr2-native": "#e87ba4",
    "olmocr2-json": "#008300", "qwen3-vl-8b": "#6250d6",
}


def error_bar_chart(P, keys, path=None, dpi=150):
    """Grouped bars: mean error (% of the y-axis span, missing points = 100 %) of
    every model on every plot, rebuilt images (exact ground truth). The black
    tick on each group is the no-model baseline (flat line through the middle)."""
    import matplotlib.pyplot as plt
    order = [k for k in BAR_COLORS if k in keys] + [k for k in keys if k not in BAR_COLORS]
    err = {k: {} for k in order}
    base = {}
    for f in ORDER:
        gt = _gt(P, f)
        base[f] = 100 - summarize(point_rows(f, gt, midline_baseline(f, gt)))["accuracy"]
        for k in order:
            a = _load(P, k, "rebuilt", f)
            if a is not None:
                err[k][f] = 100 - summarize(point_rows(f, gt, a["parsed"]))["accuracy"]
    n = len(order)
    width = 0.8 / n
    x = np.arange(len(ORDER))
    fig, ax = plt.subplots(figsize=(16, 6.5))
    for i, k in enumerate(order):
        vals = [err[k].get(f, np.nan) for f in ORDER]
        pos = x - 0.4 + width * (i + 0.5)
        bars = ax.bar(pos, vals, width * 0.92, color=BAR_COLORS.get(k, "#999999"), label=label(k), zorder=3)
        for p_, v in zip(pos, vals):
            if v == v:
                ax.text(p_, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#52514e")
    for xi, f in zip(x, ORDER):
        ax.plot([xi - 0.42, xi + 0.42], [base[f]] * 2, color="#0b0b0b", lw=1.6, ls=(0, (4, 2)), zorder=4,
                label="No-model baseline (flat line through the middle)" if xi == 0 else None)
    ax.set_xticks(x)
    short = {"ftir": "FTIR spectra", "pore": "Pore size (broken axis)", "methane": "Methane yield",
             "dosage": "Dosage (dual axis)", "n2": "N\u2082 isotherm", "kinetics": "Kinetics + fits"}
    ax.set_xticklabels([f"{f}\n{short.get(f, '')}" for f in ORDER], fontsize=10)
    ax.set_ylabel("mean error (% of y-axis span)\nmissing points count as 100 %", fontsize=10.5)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#8a8983"); ax.spines["bottom"].set_color("#8a8983")
    ax.set_title("Extraction error per model and plot — rebuilt images (exact ground truth). Lower is better.",
                 fontsize=12.5, loc="left")
    ax.legend(ncol=4, fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    path = Path(path or Path(P.work) / "outputs" / "presentation" / "error_by_model_and_plot.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    return fig, path


def presentation_all(P, keys):
    paths = {}
    for key in keys:
        for f in ORDER:
            if any(_load(P, key, s, f) for s in ("original", "rebuilt")):
                paths[(key, f)] = presentation_figure(P, key, f)
    return paths
