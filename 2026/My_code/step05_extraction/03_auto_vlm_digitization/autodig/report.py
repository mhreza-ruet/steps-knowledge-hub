"""Tables and figures for the notebook (kept here so the notebook stays short)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import evaluate as E


# ------------------------------------------------------------- what the VLM said
def spec_table(specs):
    """specs: {name: spec}. One row per image."""
    rows = []
    for n, s in specs.items():
        rows.append({"image": n, "chart_type": s["chart_type"],
                     "x title": s["x_axis"]["title"], "x ticks": " ".join(s["x_axis"]["tick_labels"]),
                     "y title": s["y_axis"]["title"], "y ticks": " ".join(s["y_axis"]["tick_labels"]),
                     "y broken": s["y_axis"].get("broken", False),
                     "right y ticks": " ".join(s["y_axis_right"]["tick_labels"]) if s.get("y_axis_right") else "",
                     "legend": s["legend"], "n series": len(s["series"])})
    return pd.DataFrame(rows).set_index("image")


def series_table(specs):
    rows = []
    for n, s in specs.items():
        for q in s["series"]:
            rows.append({"image": n, **q})
    return pd.DataFrame(rows)


def calibration_table(results):
    """How each axis was calibrated (VLM tick values matched to detected label positions)."""
    rows = []
    for n, r in results.items():
        for ax, inf in r["calibration_info"].items():
            rows.append({"image": n, "axis": ax, "method": inf.get("method"),
                         "VLM ticks": inf.get("n_values"), "labels found in pixels": inf.get("n_labels_found"),
                         "fit rms (px)": round(inf["rms_px"], 2) if inf.get("rms_px") is not None else None,
                         "scale": "log" if inf.get("log") else "linear"})
        if r["warnings"]:
            rows.append({"image": n, "axis": "WARNING", "method": "; ".join(r["warnings"])})
    return pd.DataFrame(rows)


def checks_table(spec_infos):
    """What the pixel cross-check changed in the VLM's answer."""
    rows = []
    for n, sp in spec_infos.items():
        for c in sp.get("checks") or ["(no disagreement between VLM and pixels)"]:
            rows.append({"image": n, "pixel check": c})
    return pd.DataFrame(rows)


def extraction_table(results):
    rows = []
    for n, r in results.items():
        said = {q["name"]: q.get("n_points") for q in r["spec"]["series"]}
        for s, d in r["series"].items():
            st = r["styles"].get(s, {})
            rows.append({"image": n, "series": s, "kind": d["kind"], "points found": len(d["x"]),
                         "VLM counted": said.get(s) if d["kind"] == "marker" else "",
                         "x range": f"{min(d['x']):.4g} .. {max(d['x']):.4g}" if d["x"] else "",
                         "y range": f"{min(d['y']):.4g} .. {max(d['y']):.4g}" if d["y"] else "",
                         "y axis": d["y_axis"], "colour from": st.get("target", {}).get("from", "")})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- scores
def score_table(scores, label="round trip"):
    rows = []
    for n, s in scores.items():
        rows.append({"image": n, "hit rate %": s["summary"]["hit_rate"], "accuracy %": s["summary"]["accuracy"],
                     "coverage %": s["summary"]["coverage"], "points": s["summary"]["n_points"],
                     "per series hit %": ", ".join(f"{p['series']}: {p['hit_rate']:.0f}" for p in s["per_series"])})
    df = pd.DataFrame(rows).set_index("image")
    df.loc["MEAN"] = [round(df["hit rate %"].mean(), 1), round(df["accuracy %"].mean(), 1),
                      round(df["coverage %"].mean(), 1), int(df["points"].sum()), ""]
    return df


def show_grid(paths, titles, ncols=3, width=18):
    n = len(paths)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(width, width / ncols * 0.75 * nrows))
    for ax in np.ravel(axs):
        ax.axis("off")
    for ax, p, t in zip(np.ravel(axs), paths, titles):
        ax.imshow(plt.imread(p)); ax.set_title(t, fontsize=11)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------- benchmark vs exact ground truth
def load_chartbench(bench_dir):
    bench_dir = Path(bench_dir)
    if str(bench_dir) not in sys.path:
        sys.path.insert(0, str(bench_dir))
    from chartbench.figures import FIGURES
    from chartbench.scoring import score_figure
    return FIGURES, score_figure


def bench_score(fig_id, ref, result, FIGURES, score_figure):
    """Score one pipeline result against chartbench reference data (exact GT for
    rebuilt plots, classical v1 data for originals). Series names from the VLM
    are mapped onto the benchmark names (fuzzy, then by order)."""
    names = list(FIGURES[fig_id]["series"])
    cand = {n: d for n, d in result["series"].items() if d["x"]}
    mapping = E.match_names(list(cand), names)
    pred = {"series": {gt: cand[c] for gt, c in mapping.items()}}
    sc = score_figure(fig_id, ref, pred)
    sc["name_mapping"] = mapping
    return sc


def leaderboard_with(bench_dir, row):
    f = Path(bench_dir) / "outputs" / "leaderboard.csv"
    if not f.exists():          # notebook 02's leaderboard is written in its section 4
        return pd.DataFrame([row])
    lb = pd.read_csv(f)
    lb = pd.concat([pd.DataFrame([row]), lb], ignore_index=True)
    return lb


def leaderboard_chart(lb, out=None, rebuilt_only=False):
    lb = lb.sort_values("hit_rate_rebuilt", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(lb) + 1.2))
    y = np.arange(len(lb))
    if rebuilt_only:
        cols = ["#2b6cb0" if str(m) == "vlm+digitizer" else "#a0aec0" for m in lb["model"]]
        ax.barh(y, lb["hit_rate_rebuilt"], 0.6, color=cols)
        for yy, v in zip(y, lb["hit_rate_rebuilt"]):
            ax.text(v + 1, yy, f"{v:.0f}", va="center", fontsize=9)
    else:
        ax.barh(y + 0.2, lb["hit_rate_rebuilt"], 0.4, label="rebuilt plots (exact ground truth)", color="#2b6cb0")
        ax.barh(y - 0.2, lb["hit_rate_original"], 0.4, label="original plots (vs hand-tuned digitizer)", color="#a0aec0")
        ax.legend(loc="lower right", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(lb["label"], fontsize=10)
    ax.set_xlabel("hit rate % on the rebuilt plots (points within 5 % of the axis span)"); ax.set_xlim(0, 110)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    return fig


def bench_series_table(scores):
    rows = []
    for fid, sc in scores.items():
        inv = {v: k for k, v in sc.get("name_mapping", {}).items()}
        for x in sc["series"]:
            rows.append({"figure": fid, "ground-truth series": x["series"],
                         "extracted as": sc.get("name_mapping", {}).get(x["series"], "(not found)"),
                         "hit rate %": round(x["hit_rate"], 1), "accuracy %": round(x["accuracy"], 1),
                         "coverage %": round(x["coverage"], 1), "points": x["n_points"]})
    return pd.DataFrame(rows)


def gt_overlay(fig_id, gt, res, mapping, out=None, title=None):
    """Ground truth (open black circles / grey line) vs extracted (coloured dots), one panel per y-axis."""
    keys = sorted({d["y_axis"] for d in res["series"].values() if d["x"]} or {"y_axis"})
    fig, axs = plt.subplots(1, len(keys), figsize=(6.2 * len(keys), 4.4), squeeze=False)
    for ax, key in zip(axs[0], keys):
        for gname, g in gt["series"].items():
            c = mapping.get(gname)
            d = res["series"].get(c) if c else None
            if d is not None and d["y_axis"] != key:
                continue
            if d is None and key != keys[0]:
                continue
            col = np.clip(np.array(res["styles"].get(c, {}).get("rgb", [80, 80, 80])) / 255, 0, 1) if c else (0.5, 0.5, 0.5)
            gx, gy = np.asarray(g["x"], float), np.asarray(g["y"], float)
            if len(gx) > 60:
                ax.plot(gx, gy, color="0.55", lw=4, alpha=0.5, label=f"truth: {gname}")
            else:
                ax.plot(gx, gy, "o", mfc="none", mec="black", ms=9, mew=1.2, label=f"truth: {gname}")
            if d is not None:
                if d["kind"] == "line":
                    ax.plot(d["x"], d["y"], "-", color=col, lw=1.4, label=f"extracted: {c}")
                else:
                    ax.plot(d["x"], d["y"], ".", color=col, ms=9, label=f"extracted: {c}")
        ax.set_title(f"{title or fig_id}  ({'right' if key == 'y_axis_right' else 'left'} y-axis)" if len(keys) > 1 else (title or fig_id))
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=130)
    return fig


# ------------------------------------------------------------- where do the errors come from?
SERIES_FIELDS = ("color", "marker", "marker_fill", "line", "y_axis")


def spec_diff(vlm, ref):
    """Field-by-field comparison of the VLM's spec with a hand-written reference spec.
    Returns a list of (field, VLM value, reference value) that differ."""
    from .calibrate import parse_number
    from .extract import colour_target, hue_dist
    d = []
    for ax in ("x_axis", "y_axis", "y_axis_right"):
        a, b = vlm.get(ax), ref.get(ax)
        if (a is None) != (b is None):
            d.append((ax, "present" if a else "none", "present" if b else "none"))
            continue
        if a is None:
            continue
        va = [parse_number(x) for x in a["tick_labels"]]
        vb = [parse_number(x) for x in b["tick_labels"]]
        if va != vb:
            d.append((f"{ax} ticks", " ".join(a["tick_labels"]), " ".join(b["tick_labels"])))
        if a["scale"] != b["scale"]:
            d.append((f"{ax} scale", a["scale"], b["scale"]))
        if ax == "y_axis" and bool(a.get("broken")) != bool(b.get("broken")) and b["tick_labels"]:
            d.append(("y_axis broken", a.get("broken"), b.get("broken")))
    if bool(vlm.get("legend")) != bool(ref.get("legend")):
        d.append(("legend", vlm.get("legend"), ref.get("legend")))
    if len(vlm["series"]) != len(ref["series"]):
        d.append(("number of series", len(vlm["series"]), len(ref["series"])))
    m = E.match_names([s["name"] for s in vlm["series"]], [s["name"] for s in ref["series"]])
    vs = {s["name"]: s for s in vlm["series"]}
    for rs in ref["series"]:
        v = vs.get(m.get(rs["name"]))
        if v is None:
            d.append((f"series '{rs['name']}'", "missing", "present"))
            continue
        for f in SERIES_FIELDS:
            a, b = v.get(f), rs.get(f)
            if f == "color":
                ta, tb = colour_target(a), colour_target(b)
                same = ta["kind"] == tb["kind"] and (ta["kind"] == "dark" or hue_dist(ta["hue"], tb["hue"]) <= 30)
            else:
                same = a == b
            if not same:
                d.append((f"'{rs['name']}' {f}", a, b))
    return d


def attribution_table(acc_vlm, acc_ref, diffs):
    """acc_*: {figure: hit rate}. Who is to blame for the missing hit rate?"""
    rows = []
    for fid in acc_vlm:
        hv, hr = acc_vlm[fid], acc_ref.get(fid)
        if hr is None:
            blame = "no reference spec"
        elif hv >= hr - 1:
            blame = "VLM spec as good as the reference" + ("; remaining loss = digitizer" if hr < 99 else "")
        else:
            blame = "VLM / prompt: the fields below differ" + ("; plus digitizer" if hr < 99 else "")
        rows.append({"figure": fid, "hit % with VLM spec": hv, "hit % with reference spec": hr,
                     "lost to VLM": None if hr is None else round(max(0.0, hr - hv), 1),
                     "lost to digitizer": None if hr is None else round(100 - hr, 1), "diagnosis": blame,
                     "fields that differ": "; ".join(f"{f}: VLM={a} / ref={b}" for f, a, b in diffs.get(fid, [])) or "-"})
    return pd.DataFrame(rows).set_index("figure")


# ------------------------------------------------------------- per-point tables + ground-truth check
def point_table(fid, gt, res, mapping):
    """One row per ground-truth point: truth, extracted (read off the extracted series at the
    truth x), error (% of the y-axis span), point accuracy, hit. Traced curves are compared at
    the 25 evenly spaced x of the benchmark scoring."""
    from chartbench.scoring import point_rows
    pred = {"series": {g: res["series"][c] for g, c in mapping.items() if c in res["series"]}}
    rows = point_rows(fid, gt, pred)
    from chartbench.figures import FIGURES as _F
    cats = _F[fid].get("categories")
    df = pd.DataFrame([{"series": r["series"], "extracted as": mapping.get(r["series"], "(not found)"),
                        "x": (cats[int(round(r["x"]))] if cats else r["x"]), "y truth": r["y_true"], "y extracted": r["y_model"],
                        "error % of y-span": round(r["error_pct_of_axis"], 2),
                        "point accuracy %": round(100 - r["error_pct_of_axis"], 2),
                        "hit (<= 5 %)": "yes" if r["hit"] else ("MISSING" if r["missing"] else "no")} for r in rows])
    return df


def summary_row(fid, df):
    n = len(df)
    return {"figure": fid, "truth points": n,
            "hit rate %": round(100 * (df["hit (<= 5 %)"] == "yes").mean(), 1) if n else float("nan"),
            "accuracy %": round(df["point accuracy %"].mean(), 1) if n else float("nan"),
            "coverage %": round(100 * (df["hit (<= 5 %)"] != "MISSING").mean(), 1) if n else float("nan"),
            "mean error % (found points)": round(df.loc[df["hit (<= 5 %)"] != "MISSING", "error % of y-span"].mean(), 2) if n else float("nan"),
            "missed points": int((df["hit (<= 5 %)"] != "yes").sum())}


def gt_on_image(fid, image, gt, res, FIGURES, df=None, out=None):
    """Ground truth placed on the input image with the benchmark's own pixel calibration
    (independent of our extraction): if the circles sit on the markers, the ground truth is right.
    Orange x = extracted points; red circle = truth point that was missed."""
    from chartbench.figures import series_axis
    F_ = FIGURES[fid]
    if F_.get("chart_type") == "pie":       # no axes: show the image with truth vs extracted shares
        im = plt.imread(str(image))
        fig, ax = plt.subplots(figsize=(7.5, 7.5 * im.shape[0] / im.shape[1]))
        ax.imshow(im); ax.axis("off")
        if df is not None:
            txt = "\n".join(f"{r['x']}: truth {r['y truth']:.4g}  extracted {r['y extracted'] if r['y extracted'] is None else round(r['y extracted'], 3)}"
                             for _, r in df.iterrows())
            ax.text(0.01, 0.01, txt, transform=ax.transAxes, fontsize=8, va="bottom",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))
        ax.set_title(f"{fid}: pie - truth vs extracted shares", fontsize=9)
        fig.tight_layout()
        if out:
            fig.savefig(out, dpi=120)
        return fig
    c = FIGURES[fid]["calib"]
    lin = lambda k, v: c[k][0] + (np.asarray(v, float) - c[k][1]) * (c[k][2] - c[k][0]) / (c[k][3] - c[k][1])
    key = {"left": "y", "right": "y_right", "upper": "y_upper", "lower": "y_lower"}
    im = plt.imread(str(image))
    fig, ax = plt.subplots(figsize=(7.5, 7.5 * im.shape[0] / im.shape[1]))
    ax.imshow(im); ax.axis("off")
    cats = F_.get("categories")

    def to_px(name, x, y):
        """pixel position of a truth point, with the benchmark's own calibration"""
        xv = float(x) if not isinstance(x, str) else float(cats.index(x))
        k = key.get(series_axis(fid, name, xv, y), "y")
        k = k if k in c else ("y_lower" if "y_lower" in c else "y")
        px = lin("x", xv) + (F_["series"][name].get("offset_px", 0) if cats else 0)
        return px, lin(k, y)

    for name, s in gt["series"].items():
        x, y = np.asarray(s["x"], float), np.asarray(s["y"], float)
        if len(x) > 60:                      # traced curve: show 25 points
            i = np.linspace(0, len(x) - 1, 25).astype(int); x, y = x[i], y[i]
        P = np.array([to_px(name, a, b) for a, b in zip(x, y)])
        ax.plot(P[:, 0], P[:, 1], "o", mfc="none", mec="black", ms=11, mew=1.4)
    for d in res["series"].values():
        ax.plot(d.get("px", []), d.get("py", []), "x", color="darkorange", ms=7, mew=2)
    if df is not None:
        for _, r in df.iterrows():
            if r["hit (<= 5 %)"] != "yes":
                px, py = to_px(r["series"], r["x"], r["y truth"])
                ax.plot(px, py, "o", mfc="none", mec="red", ms=16, mew=2.2)
    ax.set_title(f"{fid}: black o = ground truth on the image, orange x = extracted, red o = missed", fontsize=9)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120)
    return fig
