"""Synthetic test gallery for the new chart types - every image drawn from known data.

    python make_gallery.py            # writes images/, truth/, specs/ next to this file

For every image three files are written:
    images/<name>.png    the plot
    truth/<name>.json    the exact data it was drawn from (ground truth)
    specs/<name>.json    the TEXT an ideal VLM would read from it (tick labels as printed,
                         category / series names, colours) - used for dry runs without a
                         VLM (spec_source = "files"). It never contains pixel positions.

Several style variants per type (colormaps, colorbar side, edges, orientation,
notches, view angles, ...). The 'unknown_*' images are plot types the pipeline
does NOT support; their spec files deliberately carry a WRONG chart type (what a
confused VLM might answer) so the rejection gates are tested, not just the
VLM's 'other'.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

HERE = Path(__file__).resolve().parent
IMG, TRU, SPC = HERE / "images", HERE / "truth", HERE / "specs"
DPI = 100
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 11})


def _labels(texts):
    return [t.get_text().replace("−", "-") for t in texts if t.get_text().strip()]


def visible_ticks(ax, which):
    """tick labels as printed (only those inside the axis limits), in reading order"""
    ax.figure.canvas.draw()
    if which == "x":
        lo, hi = sorted(ax.get_xlim()); locs, texts = ax.get_xticks(), ax.get_xticklabels()
    elif which == "y":
        lo, hi = sorted(ax.get_ylim()); locs, texts = ax.get_yticks(), ax.get_yticklabels()
    else:
        lo, hi = sorted(ax.get_zlim()); locs, texts = ax.get_zticks(), ax.get_zticklabels()
    out = []
    for v, t in zip(locs, texts):
        s = t.get_text().replace("−", "-")
        if s.strip() and lo - 1e-9 * abs(hi - lo) <= v <= hi + 1e-9 * abs(hi - lo):
            out.append((v, s))
    rev = (which == "x" and ax.get_xlim()[0] > ax.get_xlim()[1]) or (which == "y" and ax.get_ylim()[0] > ax.get_ylim()[1])
    out = sorted(out, reverse=rev)
    return [s for _, s in out]


def cbar_ticks(cb):
    cb.ax.figure.canvas.draw()
    horiz = cb.orientation == "horizontal"
    lo, hi = cb.ax.get_xlim() if horiz else cb.ax.get_ylim()
    locs = cb.ax.get_xticks() if horiz else cb.ax.get_yticks()
    texts = cb.ax.get_xticklabels() if horiz else cb.ax.get_yticklabels()
    return [t.get_text().replace("−", "-") for v, t in sorted(zip(locs, texts), key=lambda p: p[0])
            if t.get_text().strip() and min(lo, hi) - 1e-9 <= v <= max(lo, hi) + 1e-9]


def save(fig, name, truth, spec):
    for d in (IMG, TRU, SPC):
        d.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / f"{name}.png", dpi=DPI)
    plt.close(fig)
    (TRU / f"{name}.json").write_text(json.dumps(truth, indent=1, default=float))
    (SPC / f"{name}.json").write_text(json.dumps(spec, indent=1, default=float))
    print("wrote", name)


def axis_spec(ax, which, title):
    return {"title": title, "tick_labels": visible_ticks(ax, which), "scale": "linear"}


R = np.random.default_rng(7)


# ===================================================================== heatmaps
def heatmaps():
    # H1 imshow + colorbar right + annotations
    rows = ["Fe", "Al", "Ca", "Mg", "Zn", "Cu"]
    cols = [f"S{i}" for i in range(1, 9)]
    v = np.round(R.uniform(0, 50, (6, 8)), 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(v, cmap="viridis", vmin=0, vmax=50, aspect="auto")
    ax.set_xticks(range(8), cols); ax.set_yticks(range(6), rows)
    for i in range(6):
        for j in range(8):
            ax.text(j, i, f"{v[i, j]:.0f}", ha="center", va="center", fontsize=8, color="w" if v[i, j] < 30 else "k")
    cb = fig.colorbar(im, ax=ax); cb.set_label("Adsorbed P (mg/g)")
    fig.tight_layout()
    save(fig, "heatmap_1", {"type": "heatmap", "rows": rows, "columns": cols, "values": v.tolist()},
         {"chart_type": "heatmap", "rows": rows, "columns": cols, "annotated": True,
          "colorbar": {"title": "Adsorbed P (mg/g)", "tick_labels": cbar_ticks(cb), "orientation": "vertical"}})
    # H2 diverging correlation matrix with white gaps
    n = 9
    names = ["pH", "EC", "TP", "TN", "OM", "Fe", "Al", "Ca", "Clay"]
    A = R.normal(size=(n, n)); C = np.corrcoef(A @ A.T + R.normal(size=(n, n)))
    C = np.round(C, 2)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    m = ax.pcolormesh(C, cmap="RdBu_r", vmin=-1, vmax=1, edgecolors="white", linewidth=1.5)
    ax.set_xticks(np.arange(n) + 0.5, names, rotation=45); ax.set_yticks(np.arange(n) + 0.5, names)
    ax.invert_yaxis(); ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(m, ax=ax, shrink=0.85)
    fig.tight_layout()
    save(fig, "heatmap_2", {"type": "heatmap", "rows": names, "columns": names, "values": C.tolist()},
         {"chart_type": "heatmap", "rows": names, "columns": names, "annotated": False,
          "colorbar": {"title": "", "tick_labels": cbar_ticks(cb), "orientation": "vertical"}})
    # H3 YlOrRd, horizontal colorbar below, months
    rows = ["2019", "2020", "2021", "2022", "2023"]
    cols = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    v = np.round(10 + 8 * np.sin(np.linspace(0, np.pi, 12))[None] + R.normal(0, 1.5, (5, 12)), 1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    im = ax.imshow(v, cmap="YlOrRd", vmin=5, vmax=22, aspect="auto")
    ax.set_xticks(range(12), cols, rotation=45, ha="right"); ax.set_yticks(range(5), rows)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.25, fraction=0.06); cb.set_label("Runoff P load (kg/ha)")
    fig.tight_layout()
    save(fig, "heatmap_3", {"type": "heatmap", "rows": rows, "columns": cols, "values": v.tolist()},
         {"chart_type": "heatmap", "rows": rows, "columns": cols, "annotated": False,
          "colorbar": {"title": "Runoff P load (kg/ha)", "tick_labels": cbar_ticks(cb), "orientation": "horizontal"}})
    # H4 jet, coarse, annotated with decimals, black cell borders
    rows = ["0-10 cm", "10-20 cm", "20-40 cm", "40-60 cm"]
    cols = ["Control", "Biochar", "Compost", "Manure", "NPK"]
    v = np.round(R.uniform(0.5, 4.5, (4, 5)), 2)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    m = ax.pcolormesh(v, cmap="jet", vmin=0, vmax=5, edgecolors="k", linewidth=0.8)
    ax.set_xticks(np.arange(5) + 0.5, cols); ax.set_yticks(np.arange(4) + 0.5, rows); ax.invert_yaxis()
    for i in range(4):
        for j in range(5):
            ax.text(j + 0.5, i + 0.5, f"{v[i, j]:.2f}", ha="center", va="center", fontsize=9, color="w")
    cb = fig.colorbar(m, ax=ax); cb.set_label("Olsen P (mg/kg)")
    fig.tight_layout()
    save(fig, "heatmap_4", {"type": "heatmap", "rows": rows, "columns": cols, "values": v.tolist()},
         {"chart_type": "heatmap", "rows": rows, "columns": cols, "annotated": True,
          "colorbar": {"title": "Olsen P (mg/kg)", "tick_labels": cbar_ticks(cb), "orientation": "vertical"}})


# ===================================================================== 2-D fields (surface seen from above)
def _f1(x, y):
    return np.exp(-((x - 1) ** 2 + (y - 0.5) ** 2) / 1.5) * 8 + 0.6 * x - 0.4 * y + 2


def _f2(x, y):
    return np.sin(x) * np.cos(y) * 3 + 0.2 * x * y


def fields():
    specs = [("field_1", _f1, (-2, 4), (-2, 3), "viridis", "contourf", 12, "vertical", True),
             ("field_2", _f2, (0, 6), (0, 5), "plasma", "pcolormesh", None, "vertical", False),
             ("field_3", _f1, (-2, 4), (-2, 3), "coolwarm", "imshow", None, "horizontal", False),
             ("field_4", _f2, (0, 6), (0, 5), "jet", "contourf", 8, "vertical", False)]
    for name, f, xr, yr, cmap, kind, nlev, orient, lines in specs:
        x = np.linspace(*xr, 200); y = np.linspace(*yr, 160)
        X, Y = np.meshgrid(x, y); Z = f(X, Y)
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        levels = None
        if kind == "contourf":
            cs = ax.contourf(X, Y, Z, levels=nlev, cmap=cmap)
            levels = cs.levels.tolist()
            if lines:
                ax.contour(X, Y, Z, levels=cs.levels, colors="k", linewidths=0.6)
            mappable = cs
        elif kind == "pcolormesh":
            mappable = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="gouraud")
        else:
            mappable = ax.imshow(Z, extent=(*xr, *yr), origin="lower", cmap=cmap, aspect="auto")
        ax.set_xlabel("Distance east (m)"); ax.set_ylabel("Distance north (m)")
        cb = fig.colorbar(mappable, ax=ax, orientation=orient, **({"pad": 0.18} if orient == "horizontal" else {}))
        cb.set_label("Soil P (mg/kg)")
        fig.tight_layout()
        gx = np.linspace(*xr, 41); gy = np.linspace(*yr, 41)
        GX, GY = np.meshgrid(gx, gy)
        save(fig, name, {"type": "field", "x": gx.tolist(), "y": gy.tolist(), "z": f(GX, GY).tolist(), "levels": levels,
                         "z_range": [float(Z.min()), float(Z.max())]},
             {"chart_type": "contour", "x_axis": axis_spec(ax, "x", "Distance east (m)"),
              "y_axis": axis_spec(ax, "y", "Distance north (m)"),
              "colorbar": {"title": "Soil P (mg/kg)", "tick_labels": cbar_ticks(cb), "orientation": orient}})


# ===================================================================== histograms
def _hist_truth(data, bins, density=False):
    h, e = np.histogram(data, bins=bins, density=density)
    return {"edges": e.tolist(), "heights": h.tolist()}


def histograms():
    # Hi1 single, black edges
    d = R.normal(25, 6, 400)
    fig, ax = plt.subplots(figsize=(6, 4.4))
    bins = np.linspace(5, 45, 21)
    ax.hist(d, bins=bins, color="tab:blue", edgecolor="black")
    ax.set_xlabel("Particle size (µm)"); ax.set_ylabel("Count")
    fig.tight_layout()
    save(fig, "hist_1", {"type": "histogram", "series": {"Count": _hist_truth(d, bins)}},
         {"chart_type": "histogram", "x_axis": axis_spec(ax, "x", "Particle size (µm)"), "y_axis": axis_spec(ax, "y", "Count"),
          "legend": False, "series": [{"name": "Count", "color": "blue", "style": "filled"}]})
    # Hi2 two overlapping, alpha 0.5, legend
    a = R.normal(1.2, 0.35, 500); b = R.normal(1.9, 0.4, 500)
    bins = np.linspace(0, 3.2, 33)
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.hist(a, bins=bins, alpha=0.5, color="tab:blue", label="Before")
    ax.hist(b, bins=bins, alpha=0.5, color="tab:orange", label="After")
    ax.legend(); ax.set_xlabel("TP (mg/L)"); ax.set_ylabel("Frequency")
    fig.tight_layout()
    save(fig, "hist_2", {"type": "histogram", "series": {"Before": _hist_truth(a, bins), "After": _hist_truth(b, bins)}},
         {"chart_type": "histogram", "x_axis": axis_spec(ax, "x", "TP (mg/L)"), "y_axis": axis_spec(ax, "y", "Frequency"),
          "legend": True, "series": [{"name": "Before", "color": "light blue", "style": "filled"},
                                     {"name": "After", "color": "light orange", "style": "filled"}]})
    # Hi3 density with KDE line
    d = np.r_[R.normal(3, 0.8, 300), R.normal(6, 1.0, 200)]
    bins = np.linspace(0, 9, 19)
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.hist(d, bins=bins, density=True, color="#5aa469", edgecolor="white")
    from scipy.stats import gaussian_kde
    xs = np.linspace(0, 9, 300); ax.plot(xs, gaussian_kde(d)(xs), color="darkred", lw=2)
    ax.set_xlabel("Grain size (φ)"); ax.set_ylabel("Density")
    fig.tight_layout()
    save(fig, "hist_3", {"type": "histogram", "series": {"Density": _hist_truth(d, bins, True)}},
         {"chart_type": "histogram", "x_axis": axis_spec(ax, "x", "Grain size (φ)"), "y_axis": axis_spec(ax, "y", "Density"),
          "legend": False, "series": [{"name": "Density", "color": "green", "style": "filled"}]})
    # Hi4 step outlines, two series
    a = R.gamma(2, 2, 600); b = R.gamma(4, 1.3, 600)
    bins = np.linspace(0, 16, 17)
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.hist(a, bins=bins, histtype="step", color="red", lw=2, label="Site A")
    ax.hist(b, bins=bins, histtype="step", color="blue", lw=2, label="Site B")
    ax.legend(); ax.set_xlabel("Rainfall (mm/day)"); ax.set_ylabel("Days")
    fig.tight_layout()
    save(fig, "hist_4", {"type": "histogram", "series": {"Site A": _hist_truth(a, bins), "Site B": _hist_truth(b, bins)}},
         {"chart_type": "histogram", "x_axis": axis_spec(ax, "x", "Rainfall (mm/day)"), "y_axis": axis_spec(ax, "y", "Days"),
          "legend": True, "series": [{"name": "Site A", "color": "red", "style": "step"},
                                     {"name": "Site B", "color": "blue", "style": "step"}]})


# ===================================================================== box plots
def _stats(d, whis=1.5):
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    iqr = q3 - q1
    lo = d[d >= q1 - whis * iqr].min(); hi = d[d <= q3 + whis * iqr].max()
    fl = d[(d < lo) | (d > hi)]
    return {"q1": q1, "median": med, "q3": q3, "whislo": lo, "whishi": hi, "fliers": sorted(fl.tolist()), "mean": float(d.mean())}


def boxes():
    cats = ["Control", "BC300", "BC500", "BC700", "Fe-BC"]
    data = [np.r_[R.normal(m, s, 40), R.normal(m + 3.5 * s, s / 3, 2)] for m, s in zip([12, 15, 19, 23, 30], [2, 2.5, 3, 2.2, 4])]
    # B1 matplotlib default
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.boxplot(data, tick_labels=cats)
    ax.set_ylabel("P removal (%)")
    fig.tight_layout()
    save(fig, "box_1", {"type": "box", "orientation": "vertical", "series": {"P removal (%)": {"categories": cats, "stats": [_stats(d) for d in data]}}},
         {"chart_type": "box", "orientation": "vertical", "categories": cats, "y_axis": axis_spec(ax, "y", "P removal (%)"),
          "x_axis": {"title": "", "tick_labels": cats, "scale": "linear"},
          "legend": False, "series": [{"name": "P removal (%)", "color": "black"}]})
    # B2 filled colours + mean markers
    fig, ax = plt.subplots(figsize=(6, 4.4))
    bp = ax.boxplot(data, tick_labels=cats, patch_artist=True, showmeans=True, medianprops={"color": "black"})
    for p, c in zip(bp["boxes"], ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3"]):
        p.set_facecolor(c)
    ax.set_ylabel("P removal (%)"); ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    save(fig, "box_2", {"type": "box", "orientation": "vertical", "series": {"P removal (%)": {"categories": cats, "stats": [_stats(d) for d in data]}}},
         {"chart_type": "box", "orientation": "vertical", "categories": cats, "y_axis": axis_spec(ax, "y", "P removal (%)"),
          "x_axis": {"title": "", "tick_labels": cats, "scale": "linear"},
          "legend": False, "series": [{"name": "P removal (%)", "color": "various"}]})
    # B3 horizontal
    c4 = ["Sand", "Loam", "Clay", "Peat"]
    d4 = [R.lognormal(np.log(m), 0.35, 50) for m in (5, 9, 14, 22)]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.boxplot(d4, tick_labels=c4, orientation="horizontal", patch_artist=True,
               boxprops={"facecolor": "lightsteelblue"}, medianprops={"color": "navy", "lw": 2})
    ax.set_xlabel("Sorption capacity (mg/g)")
    fig.tight_layout()
    save(fig, "box_3", {"type": "box", "orientation": "horizontal", "series": {"Sorption capacity (mg/g)": {"categories": c4, "stats": [_stats(d) for d in d4]}}},
         {"chart_type": "box", "orientation": "horizontal", "categories": c4[::-1], "x_axis": axis_spec(ax, "x", "Sorption capacity (mg/g)"),
          "y_axis": {"title": "", "tick_labels": c4[::-1], "scale": "linear"},
          "legend": False, "series": [{"name": "Sorption capacity (mg/g)", "color": "light blue"}]})
    # B4 grouped two series + legend
    c3 = ["Spring", "Summer", "Autumn"]
    ga = [R.normal(m, 1.2, 30) for m in (4, 6, 5)]; gb = [R.normal(m, 1.0, 30) for m in (3, 4.2, 3.6)]
    fig, ax = plt.subplots(figsize=(6, 4.4))
    pa = ax.boxplot(ga, positions=np.arange(3) * 3 - 0.45, widths=0.7, patch_artist=True, boxprops={"facecolor": "tab:orange"},
                    medianprops={"color": "k"})
    pb = ax.boxplot(gb, positions=np.arange(3) * 3 + 0.45, widths=0.7, patch_artist=True, boxprops={"facecolor": "tab:green"},
                    medianprops={"color": "k"})
    ax.set_xticks(np.arange(3) * 3, c3); ax.set_ylabel("DRP (µg/L)")
    ax.legend([pa["boxes"][0], pb["boxes"][0]], ["Inflow", "Outflow"])
    fig.tight_layout()
    save(fig, "box_4", {"type": "box", "orientation": "vertical", "series": {"Inflow": {"categories": c3, "stats": [_stats(d) for d in ga]},
                                                                            "Outflow": {"categories": c3, "stats": [_stats(d) for d in gb]}}},
         {"chart_type": "box", "orientation": "vertical", "categories": c3, "y_axis": axis_spec(ax, "y", "DRP (µg/L)"),
          "x_axis": {"title": "", "tick_labels": c3, "scale": "linear"},
          "legend": True, "series": [{"name": "Inflow", "color": "orange"}, {"name": "Outflow", "color": "green"}]})
    # B5 notched, wide fliers
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.boxplot(data[:4], tick_labels=cats[:4], notch=True, flierprops={"marker": "x"})
    ax.set_ylabel("P removal (%)")
    fig.tight_layout()
    save(fig, "box_5", {"type": "box", "orientation": "vertical", "series": {"P removal (%)": {"categories": cats[:4], "stats": [_stats(d) for d in data[:4]]}}},
         {"chart_type": "box", "orientation": "vertical", "categories": cats[:4], "y_axis": axis_spec(ax, "y", "P removal (%)"),
          "x_axis": {"title": "", "tick_labels": cats[:4], "scale": "linear"},
          "legend": False, "series": [{"name": "P removal (%)", "color": "black"}]})


# ===================================================================== polar
def _polar_spec(ax, series, kind, rtitle=""):
    ax.figure.canvas.draw()
    off, dr = np.degrees(ax.get_theta_offset()), ax.get_theta_direction()
    items = [((off + dr * np.degrees(v)) % 360, t.get_text()) for v, t in zip(ax.get_xticks(), ax.get_xticklabels()) if t.get_text().strip()]
    ang = [t for _, t in sorted(items)]          # as met counter-clockwise from 3 o'clock
    lo, hi = ax.get_ylim()
    rl = [t.get_text().replace("−", "-") for v, t in zip(ax.get_yticks(), ax.get_yticklabels())
          if t.get_text().strip() and lo - 1e-9 <= v <= hi + 1e-9]
    return {"chart_type": "polar", "kind": kind, "angle_labels": ang, "radial_labels": rl, "radial_title": rtitle,
            "legend": len(series) > 1, "series": series}


def polars():
    th = np.linspace(0, 2 * np.pi, 361)
    # P1 two curves, default orientation
    r1 = 2 + np.cos(3 * th); r2 = 1.5 + 0.8 * np.sin(th)
    fig = plt.figure(figsize=(5.2, 5)); ax = fig.add_subplot(projection="polar")
    ax.plot(th, r1, color="tab:red", lw=2, label="Rose"); ax.plot(th, r2, color="tab:blue", lw=2, label="Limaçon")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1))
    fig.tight_layout()
    save(fig, "polar_1", {"type": "polar", "series": {"Rose": {"theta_deg": np.degrees(th).tolist(), "r": r1.tolist(), "kind": "line"},
                                                      "Limaçon": {"theta_deg": np.degrees(th).tolist(), "r": r2.tolist(), "kind": "line"}}},
         _polar_spec(ax, [{"name": "Rose", "color": "red", "marker": "none", "line": "solid"},
                          {"name": "Limaçon", "color": "blue", "marker": "none", "line": "solid"}], "line"))
    # P2 compass scatter: 0 at North, clockwise
    t2 = np.radians(R.uniform(0, 360, 40)); rr = R.uniform(1, 9, 40)
    fig = plt.figure(figsize=(5.2, 5)); ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.scatter(t2, rr, color="tab:green", s=30); ax.set_ylim(0, 10)
    fig.tight_layout()
    save(fig, "polar_2", {"type": "polar", "series": {"Wind": {"theta_deg": np.degrees(t2).tolist(), "r": rr.tolist(), "kind": "marker"}}},
         _polar_spec(ax, [{"name": "Wind", "color": "green", "marker": "circle", "line": "none"}], "scatter"))
    # P3 polar bars (rose)
    n = 16; tb = np.arange(n) * 2 * np.pi / n; hb = np.round(R.uniform(2, 12, n), 1)
    fig = plt.figure(figsize=(5.2, 5)); ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.bar(tb, hb, width=2 * np.pi / n * 0.9, color="tab:purple", edgecolor="k", alpha=0.85)
    fig.tight_layout()
    save(fig, "polar_3", {"type": "polar", "series": {"Frequency": {"theta_deg": np.degrees(tb).tolist(), "r": hb.tolist(), "kind": "bar"}}},
         _polar_spec(ax, [{"name": "Frequency", "color": "purple", "marker": "none", "line": "none"}], "bar"))
    # P4 radial axis not starting at 0, markers + line
    t4 = np.radians(np.arange(0, 360, 30)); r4 = 5 + 3 * np.abs(np.sin(t4)) + R.normal(0, 0.3, len(t4))
    fig = plt.figure(figsize=(5.2, 5)); ax = fig.add_subplot(projection="polar")
    ax.plot(np.r_[t4, t4[:1]], np.r_[r4, r4[:1]], "o-", color="tab:orange")
    ax.set_ylim(3, 9)
    fig.tight_layout()
    save(fig, "polar_4", {"type": "polar", "series": {"Yield": {"theta_deg": np.degrees(t4).tolist(), "r": r4.tolist(), "kind": "marker"}}},
         _polar_spec(ax, [{"name": "Yield", "color": "orange", "marker": "circle", "line": "solid"}], "scatter"))


# ===================================================================== stacked / horizontal bars
def stackeds():
    cats = ["2019", "2020", "2021", "2022", "2023"]
    ser = {"Particulate P": R.uniform(5, 15, 5), "Dissolved P": R.uniform(3, 10, 5), "Organic P": R.uniform(1, 6, 5)}
    cols = ["#4c72b0", "#dd8452", "#55a868"]
    fig, ax = plt.subplots(figsize=(6, 4.4)); bot = np.zeros(5)
    for (n, v), c in zip(ser.items(), cols):
        ax.bar(cats, v, bottom=bot, color=c, label=n, width=0.6); bot += v
    ax.legend(); ax.set_ylabel("P load (t/yr)")
    fig.tight_layout()
    save(fig, "stacked_1", {"type": "stacked", "orientation": "vertical", "categories": cats, "series": {k: v.tolist() for k, v in ser.items()}},
         {"chart_type": "stacked_bar", "orientation": "vertical", "categories": cats, "x_axis": {"title": "", "tick_labels": cats, "scale": "linear"},
          "y_axis": axis_spec(ax, "y", "P load (t/yr)"), "legend": True,
          "series": [{"name": n, "color": c} for n, c in zip(ser, ["blue", "orange", "green"])]})
    # S2 horizontal stacked, 4 series
    cats = ["Corn", "Soybean", "Wheat", "Pasture", "Forest", "Urban"]
    ser = {"Runoff": R.uniform(1, 5, 6), "Leaching": R.uniform(0.5, 3, 6), "Erosion": R.uniform(1, 6, 6), "Other": R.uniform(0.2, 1.5, 6)}
    colsn = ["red", "gold", "teal", "gray"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4)); left = np.zeros(6)
    for (n, v), c in zip(ser.items(), ["tab:red", "gold", "teal", "tab:gray"]):
        ax.barh(cats, v, left=left, color=c, label=n, height=0.6); left += v
    ax.invert_yaxis(); ax.legend(loc="lower right"); ax.set_xlabel("P loss (kg/ha)")
    fig.tight_layout()
    save(fig, "stacked_2", {"type": "stacked", "orientation": "horizontal", "categories": cats, "series": {k: v.tolist() for k, v in ser.items()}},
         {"chart_type": "stacked_bar", "orientation": "horizontal", "categories": cats, "x_axis": axis_spec(ax, "x", "P loss (kg/ha)"),
          "y_axis": {"title": "", "tick_labels": cats, "scale": "linear"}, "legend": True,
          "series": [{"name": n, "color": c} for n, c in zip(ser, colsn)]})
    # S3 100 % stacked with black edges
    cats = ["Ca-P", "Fe-P", "Al-P", "Org-P"]
    raw = R.uniform(1, 5, (3, 4)); pct = 100 * raw / raw.sum(0)
    names = ["Labile", "Moderately labile", "Stable"]
    fig, ax = plt.subplots(figsize=(6, 4.4)); bot = np.zeros(4)
    for n, v, c in zip(names, pct, ["#a6cee3", "#1f78b4", "#08306b"]):
        ax.bar(cats, v, bottom=bot, color=c, edgecolor="black", label=n, width=0.55); bot += v
    ax.set_ylim(0, 100); ax.set_ylabel("Fraction (%)"); ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13), frameon=False)
    fig.tight_layout()
    save(fig, "stacked_3", {"type": "stacked", "orientation": "vertical", "categories": cats, "series": dict(zip(names, pct.tolist()))},
         {"chart_type": "stacked_bar", "orientation": "vertical", "categories": cats, "x_axis": {"title": "", "tick_labels": cats, "scale": "linear"},
          "y_axis": axis_spec(ax, "y", "Fraction (%)"), "legend": True,
          "series": [{"name": n, "color": c} for n, c in zip(names, ["light blue", "blue", "navy"])]})
    # S4 horizontal grouped (not stacked)
    cats = ["Struvite", "Vivianite", "HAP", "Ca-def"]
    a = R.uniform(20, 80, 4); b = R.uniform(20, 80, 4)
    y = np.arange(4)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(y - 0.2, a, height=0.4, color="tab:blue", label="Lab"); ax.barh(y + 0.2, b, height=0.4, color="tab:orange", label="Field")
    ax.set_yticks(y, cats); ax.invert_yaxis(); ax.legend(); ax.set_xlabel("Recovery (%)")
    fig.tight_layout()
    save(fig, "hbar_1", {"type": "stacked", "orientation": "horizontal", "stacked": False, "categories": cats,
                         "series": {"Lab": a.tolist(), "Field": b.tolist()}},
         {"chart_type": "bar", "orientation": "horizontal", "categories": cats, "x_axis": axis_spec(ax, "x", "Recovery (%)"),
          "y_axis": {"title": "", "tick_labels": cats, "scale": "linear"}, "legend": True,
          "series": [{"name": "Lab", "color": "blue"}, {"name": "Field", "color": "orange"}]})


# ===================================================================== 3-D
def screen_order_3d(ax, which):
    """3-D tick labels in the order a reader meets them: left -> right across the
    image (the vertical z-axis: bottom -> top)"""
    from mpl_toolkits.mplot3d import proj3d
    ax.figure.canvas.draw()
    lims = {w: sorted(getattr(ax, f"get_{w}lim")()) for w in "xyz"}
    mid = {w: 0.5 * sum(lims[w]) for w in "xyz"}
    lo, hi = lims[which]
    items = []
    for v, t in zip(getattr(ax, f"get_{which}ticks")(), getattr(ax, f"get_{which}ticklabels")()):
        s = t.get_text().replace("\u2212", "-")
        if not s.strip() or not (lo - 1e-9 * abs(hi - lo) <= v <= hi + 1e-9 * abs(hi - lo)):
            continue
        p = dict(mid); p[which] = v
        sx, sy, _ = proj3d.proj_transform(p["x"], p["y"], p["z"], ax.get_proj())
        items.append((sx, sy, s))
    key = (lambda q: q[1]) if which == "z" else (lambda q: q[0])
    return [q[2] for q in sorted(items, key=key)]


def proj_pixels(ax, X, Y, Z):
    """image pixel positions (x right, y down) of 3-D data points - for checking a camera fit"""
    from mpl_toolkits.mplot3d import proj3d
    fig = ax.figure
    fig.canvas.draw()
    xs, ys, _ = proj3d.proj_transform(np.asarray(X, float), np.asarray(Y, float), np.asarray(Z, float), ax.get_proj())
    d = ax.transData.transform(np.c_[xs, ys])
    H = fig.get_size_inches()[1] * DPI
    return np.c_[d[:, 0], H - d[:, 1]].tolist()


def _box_check(ax):
    lo = [sorted(getattr(ax, f"get_{w}lim")()) for w in "xyz"]
    pts = np.array([[a, b, c] for a in lo[0] for b in lo[1] for c in lo[2]])
    return {"xyz": pts.tolist(), "px": proj_pixels(ax, *pts.T), "limits": lo}


def _axes3(ax, titles, which=("x", "y", "z")):
    return {f"{w}_axis": {"title": t, "tick_labels": screen_order_3d(ax, w), "scale": "linear"} for w, t in zip(which, titles)}


def bars3d():
    for name, cmode, proj, view in (("bar3d_1", "single", "persp", (30, -60)), ("bar3d_2", "rows", "ortho", (25, -50)),
                                    ("bar3d_3", "single", "persp", (35, -130))):
        xs = ["0.5", "1", "2", "5"]; ys = ["pH 4", "pH 6", "pH 8", "pH 10", "pH 12"]
        Hh = np.round(R.uniform(5, 40, (len(xs), len(ys))), 1)
        fig = plt.figure(figsize=(6.4, 5)); ax = fig.add_subplot(projection="3d", proj_type=proj)
        X, Y = np.meshgrid(np.arange(len(xs)), np.arange(len(ys)), indexing="ij")
        pal = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        col = ["tab:blue"] * X.size if cmode == "single" else [pal[i] for i in X.ravel()]
        ax.bar3d(X.ravel() - 0.3, Y.ravel() - 0.3, 0, 0.6, 0.6, Hh.ravel(), color=col, shade=True)
        ax.set_xticks(range(len(xs)), xs); ax.set_yticks(range(len(ys)), ys)
        ax.set_xlabel("Dose (g/L)"); ax.set_ylabel("Initial pH"); ax.set_zlabel("Removal (%)")
        ax.view_init(*view)
        fig.tight_layout()
        sp = {"chart_type": "bar3d", **_axes3(ax, ["Dose (g/L)", "Initial pH", "Removal (%)"]),
              "series": [{"name": "Removal (%)", "color": "blue" if cmode == "single" else "various"}]}
        save(fig, name, {"type": "bar3d", "x_categories": xs, "y_categories": ys, "heights": Hh.tolist(),
                         "camera_check": _box_check(ax),
                         "top_px": proj_pixels(ax, X.ravel(), Y.ravel(), Hh.ravel())}, sp)


def scatters3d():
    n = 60
    x = R.uniform(0, 10, n); y = R.uniform(0, 5, n); z = 2 + 0.5 * x + 1.2 * y + R.normal(0, 1, n)
    # C1 colour = z with colorbar
    fig = plt.figure(figsize=(6.6, 5)); ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(x, y, z, c=z, cmap="viridis", s=25, depthshade=False)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.12); cb.set_label("P uptake (mg)")
    ax.set_xlabel("Biochar (t/ha)"); ax.set_ylabel("Lime (t/ha)"); ax.set_zlabel("P uptake (mg)")
    fig.tight_layout()
    save(fig, "scatter3d_1", {"type": "scatter3d", "camera_check": _box_check(ax), "series": {"P uptake (mg)": {"x": x.tolist(), "y": y.tolist(), "z": z.tolist()}}},
         {"chart_type": "scatter3d", **_axes3(ax, ["Biochar (t/ha)", "Lime (t/ha)", "P uptake (mg)"]),
          "colorbar": {"title": "P uptake (mg)", "tick_labels": cbar_ticks(cb), "orientation": "vertical", "same_as": "z"},
          "stems": False, "legend": False, "series": [{"name": "P uptake (mg)", "color": "colormap", "marker": "circle"}]})
    # C2 plain two series - depth not recoverable -> flagged
    x2 = R.uniform(0, 10, 30); y2 = R.uniform(0, 5, 30); z2 = R.uniform(0, 20, 30)
    fig = plt.figure(figsize=(6.4, 5)); ax = fig.add_subplot(projection="3d")
    ax.scatter(x[:30], y[:30], z[:30], color="tab:red", label="Field", depthshade=False)
    ax.scatter(x2, y2, z2, color="tab:blue", marker="^", label="Pot", depthshade=False)
    ax.legend(); ax.set_xlabel("Biochar (t/ha)"); ax.set_ylabel("Lime (t/ha)"); ax.set_zlabel("P uptake (mg)")
    fig.tight_layout()
    save(fig, "scatter3d_2", {"type": "scatter3d", "camera_check": _box_check(ax), "series": {"Field": {"x": x[:30].tolist(), "y": y[:30].tolist(), "z": z[:30].tolist()},
                                                              "Pot": {"x": x2.tolist(), "y": y2.tolist(), "z": z2.tolist()}}},
         {"chart_type": "scatter3d", **_axes3(ax, ["Biochar (t/ha)", "Lime (t/ha)", "P uptake (mg)"]), "colorbar": None,
          "stems": False, "legend": True, "series": [{"name": "Field", "color": "red", "marker": "circle"},
                                                     {"name": "Pot", "color": "blue", "marker": "triangle_up"}]})
    # C3 stems to the floor
    m = 18
    fig = plt.figure(figsize=(6.4, 5)); ax = fig.add_subplot(projection="3d")
    ax.stem(x[:m], y[:m], z[:m], linefmt="C0-", markerfmt="C0o", basefmt=" ", bottom=0)
    ax.set_zlim(0, None)
    ax.set_xlabel("Biochar (t/ha)"); ax.set_ylabel("Lime (t/ha)"); ax.set_zlabel("P uptake (mg)")
    fig.tight_layout()
    save(fig, "scatter3d_3", {"type": "scatter3d", "camera_check": _box_check(ax), "series": {"P uptake (mg)": {"x": x[:m].tolist(), "y": y[:m].tolist(), "z": z[:m].tolist()}}},
         {"chart_type": "scatter3d", **_axes3(ax, ["Biochar (t/ha)", "Lime (t/ha)", "P uptake (mg)"]), "colorbar": None,
          "stems": True, "legend": False, "series": [{"name": "P uptake (mg)", "color": "blue", "marker": "circle"}]})


def surfaces3d():
    for name, f, xr, yr, cmap, view, cbar, kind in (("surface3d_1", _f1, (-2, 4), (-2, 3), "viridis", (30, -60), True, "surface"),
                                                     ("surface3d_2", _f2, (0, 6), (0, 5), "coolwarm", (40, -120), True, "surface"),
                                                     ("surface3d_3", _f1, (-2, 4), (-2, 3), None, (30, -60), False, "wireframe")):
        x = np.linspace(*xr, 60); y = np.linspace(*yr, 50); X, Y = np.meshgrid(x, y); Z = f(X, Y)
        fig = plt.figure(figsize=(6.6, 5)); ax = fig.add_subplot(projection="3d")
        if kind == "surface":
            s = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True)
        else:
            s = ax.plot_wireframe(X, Y, Z, color="tab:blue", rstride=4, cstride=4)
        cb = fig.colorbar(s, ax=ax, shrink=0.7, pad=0.12) if cbar else None
        if cb is not None:
            cb.set_label("Soil P (mg/kg)")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("Soil P (mg/kg)")
        ax.view_init(*view)
        fig.tight_layout()
        gx = np.linspace(*xr, 31); gy = np.linspace(*yr, 31); GX, GY = np.meshgrid(gx, gy)
        save(fig, name, {"type": "surface3d", "camera_check": _box_check(ax), "x": gx.tolist(), "y": gy.tolist(), "z": f(GX, GY).tolist(),
                         "z_range": [float(Z.min()), float(Z.max())]},
             {"chart_type": "surface3d", "kind": kind, **_axes3(ax, ["x (m)", "y (m)", "Soil P (mg/kg)"]),
              "colorbar": ({"title": "Soil P (mg/kg)", "tick_labels": cbar_ticks(cb), "orientation": "vertical", "same_as": "z"} if cb else None),
              "series": [{"name": "Soil P (mg/kg)", "color": cmap or "blue"}]})


# ===================================================================== unsupported types (rejection tests)
def unknowns():
    # violin -> a confused VLM says "box"
    d = [R.normal(m, 1.5, 200) for m in (4, 6, 5)]
    fig, ax = plt.subplots(figsize=(6, 4.4)); ax.violinplot(d, showmedians=True)
    ax.set_xticks([1, 2, 3], ["A", "B", "C"]); ax.set_ylabel("TP (mg/L)"); fig.tight_layout()
    save(fig, "unknown_violin", {"type": "unsupported"},
         {"chart_type": "box", "orientation": "vertical", "categories": ["A", "B", "C"], "y_axis": axis_spec(ax, "y", "TP (mg/L)"),
          "x_axis": {"title": "", "tick_labels": ["A", "B", "C"], "scale": "linear"}, "legend": False,
          "series": [{"name": "TP (mg/L)", "color": "blue"}]})
    # radar / spider -> "polar"
    labs = ["pH", "EC", "TP", "TN", "OM", "CEC"]; v = R.uniform(0.3, 1, 6)
    ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    fig = plt.figure(figsize=(5.2, 5)); ax = fig.add_subplot(projection="polar")
    ax.plot(np.r_[ang, ang[:1]], np.r_[v, v[:1]], color="tab:red"); ax.fill(ang, v, color="tab:red", alpha=0.25)
    ax.set_xticks(ang, labs); ax.set_yticklabels([]); ax.grid(True); fig.tight_layout()
    save(fig, "unknown_radar", {"type": "unsupported"},
         {"chart_type": "polar", "kind": "line", "angle_labels": [labs[k] for k in np.argsort(np.degrees(ang) % 360)], "radial_labels": [], "legend": False,
          "series": [{"name": "Score", "color": "red", "marker": "none", "line": "solid"}]})
    # network graph -> "scatter"
    pts = R.uniform(0, 1, (14, 2))
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for i in range(14):
        for j in range(i + 1, 14):
            if R.random() < 0.18:
                ax.plot(*pts[[i, j]].T, color="gray", lw=1)
    ax.scatter(*pts.T, s=300, color="tab:cyan", edgecolor="k", zorder=3)
    for i, p in enumerate(pts):
        ax.text(*p, str(i), ha="center", va="center", zorder=4)
    ax.axis("off"); fig.tight_layout()
    save(fig, "unknown_network", {"type": "unsupported"},
         {"chart_type": "scatter", "x_axis": {"title": "", "tick_labels": ["0.0", "0.5", "1.0"], "scale": "linear"},
          "y_axis": {"title": "", "tick_labels": ["0.0", "0.5", "1.0"], "scale": "linear", "broken": False}, "y_axis_right": None,
          "legend": False, "series": [{"name": "nodes", "color": "cyan", "marker": "circle", "marker_fill": "filled", "line": "none"}]})
    # venn -> "pie"
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    for c, xy in (("tab:red", (0.4, 0.5)), ("tab:blue", (0.62, 0.5)), ("tab:green", (0.51, 0.32))):
        ax.add_patch(plt.Circle(xy, 0.22, color=c, alpha=0.35))
    for t, xy in (("Fe", (0.3, 0.6)), ("Al", (0.72, 0.6)), ("Ca", (0.51, 0.2))):
        ax.text(*xy, t, fontsize=14, ha="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.85); ax.set_aspect("equal"); ax.axis("off"); fig.tight_layout()
    save(fig, "unknown_venn", {"type": "unsupported"},
         {"chart_type": "pie", "slices": [{"name": "Fe", "color": "red", "printed_value": None},
                                          {"name": "Al", "color": "blue", "printed_value": None},
                                          {"name": "Ca", "color": "green", "printed_value": None}]})
    # micrograph-like texture -> "heatmap"
    from scipy.ndimage import gaussian_filter
    tex = gaussian_filter(R.normal(size=(300, 400)), 3) + 0.3 * gaussian_filter(R.normal(size=(300, 400)), 15)
    fig, ax = plt.subplots(figsize=(6, 4.6)); ax.imshow(tex, cmap="gray"); ax.axis("off")
    ax.plot([300, 380], [280, 280], color="w", lw=4); ax.text(340, 270, "2 µm", color="w", ha="center")
    fig.tight_layout()
    save(fig, "unknown_micrograph", {"type": "unsupported"},
         {"chart_type": "heatmap", "rows": ["1", "2", "3"], "columns": ["1", "2", "3"], "annotated": False,
          "colorbar": {"title": "", "tick_labels": ["0", "1"], "orientation": "vertical"}})
    # ternary-like triangle -> "scatter"
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]]); ax.plot(*tri.T, color="k")
    for k in range(1, 5):
        f = k / 5
        ax.plot([f, 0.5 + f / 2], [0, (1 - f) * np.sqrt(3) / 2], color="lightgray", lw=0.8)
        ax.plot([f / 2, 1 - f / 2], [f * np.sqrt(3) / 2] * 2, color="lightgray", lw=0.8)
    p = R.dirichlet([2, 3, 4], 25); xy = np.c_[p[:, 1] + p[:, 2] / 2, p[:, 2] * np.sqrt(3) / 2]
    ax.scatter(*xy.T, color="tab:blue"); ax.text(-0.05, -0.06, "Sand"); ax.text(0.95, -0.06, "Silt"); ax.text(0.45, 0.9, "Clay")
    ax.set_aspect("equal"); ax.axis("off"); fig.tight_layout()
    save(fig, "unknown_ternary", {"type": "unsupported"},
         {"chart_type": "scatter", "x_axis": {"title": "", "tick_labels": ["0", "50", "100"], "scale": "linear"},
          "y_axis": {"title": "", "tick_labels": ["0", "50", "100"], "scale": "linear", "broken": False}, "y_axis_right": None,
          "legend": False, "series": [{"name": "soils", "color": "blue", "marker": "circle", "marker_fill": "filled", "line": "none"}]})
    # table rendered as an image -> "bar"
    fig, ax = plt.subplots(figsize=(6, 3)); ax.axis("off")
    tb = ax.table(cellText=[[f"{v:.2f}" for v in R.uniform(0, 10, 4)] for _ in range(5)],
                  colLabels=["Fe", "Al", "Ca", "Mg"], rowLabels=[f"Site {i}" for i in range(1, 6)], loc="center")
    tb.scale(1, 1.6); fig.tight_layout()
    save(fig, "unknown_table", {"type": "unsupported"},
         {"chart_type": "bar", "x_axis": {"title": "", "tick_labels": ["Fe", "Al", "Ca", "Mg"], "scale": "linear"},
          "y_axis": {"title": "", "tick_labels": ["0", "5", "10"], "scale": "linear", "broken": False}, "y_axis_right": None,
          "legend": False, "categories": [{"name": n, "y_axis": "left"} for n in ["Fe", "Al", "Ca", "Mg"]],
          "series": [{"name": "value", "color": "blue"}]})
    # 2x2 multi-panel figure -> "line"
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 4.8))
    for k, a in enumerate(axs.ravel()):
        t = np.linspace(0, 10, 50); a.plot(t, np.sin(t + k) * (k + 1), color=f"C{k}"); a.set_title(f"({'abcd'[k]})", loc="left")
    fig.tight_layout()
    save(fig, "unknown_multipanel", {"type": "unsupported"},
         {"chart_type": "line", "x_axis": {"title": "", "tick_labels": ["0", "5", "10"], "scale": "linear"},
          "y_axis": {"title": "", "tick_labels": ["-1", "0", "1"], "scale": "linear", "broken": False}, "y_axis_right": None,
          "legend": False, "series": [{"name": "curve", "color": "blue", "marker": "none", "marker_fill": "filled", "line": "solid"}]})
    # dendrogram -> "line"
    from scipy.cluster.hierarchy import dendrogram, linkage
    fig, ax = plt.subplots(figsize=(6, 4.4)); dendrogram(linkage(R.normal(size=(12, 4)), "ward"), ax=ax, color_threshold=0)
    ax.set_ylabel("Distance"); fig.tight_layout()
    save(fig, "unknown_dendrogram", {"type": "unsupported"},
         {"chart_type": "line", "x_axis": {"title": "", "tick_labels": [], "scale": "linear"},
          "y_axis": axis_spec(ax, "y", "Distance") | {"broken": False}, "y_axis_right": None,
          "legend": False, "series": [{"name": "tree", "color": "blue", "marker": "none", "marker_fill": "filled", "line": "solid"}]})
    # schematic / flowchart -> honest "other"
    fig, ax = plt.subplots(figsize=(6, 3.5)); ax.axis("off")
    for i, t in enumerate(["Wastewater", "Adsorption", "Desorption", "Struvite"]):
        ax.add_patch(plt.Rectangle((i * 0.26, 0.4), 0.2, 0.2, fc="lightyellow", ec="k")); ax.text(i * 0.26 + 0.1, 0.5, t, ha="center", va="center", fontsize=9)
        if i < 3:
            ax.annotate("", (i * 0.26 + 0.26, 0.5), (i * 0.26 + 0.2, 0.5), arrowprops={"arrowstyle": "->"})
    ax.set_xlim(-0.02, 1.02); fig.tight_layout()
    save(fig, "unknown_schematic", {"type": "unsupported"}, {"chart_type": "other"})


if __name__ == "__main__":
    heatmaps(); fields(); histograms(); boxes(); polars(); stackeds(); bars3d(); scatters3d(); surfaces3d(); unknowns()
