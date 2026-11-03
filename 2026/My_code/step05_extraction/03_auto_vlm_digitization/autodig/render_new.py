"""Recreate the step-5 chart types from the extracted data (for the side-by-side check).

Only extracted numbers, the VLM's text (titles, names, tick labels) and measured
colours are used - nothing is copied from the image.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .calibrate import parse_number


def _fig(res, dpi=100, projection=None):
    W, H = res.get("size", (640, 480))
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection=projection)
    return fig, ax


def _nums(labels):
    v = [parse_number(x) for x in labels or []]
    return [x for x in v if x is not None]


def _finish(fig, out_png, dpi=100):
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def _cmap(res):
    t = res.get("cmap_table")
    if not t:
        return "viridis", None, None
    from matplotlib.colors import LinearSegmentedColormap
    v = np.array(t["values"]); c = np.array(t["rgb"]) / 255
    o = np.argsort(v); v, c = v[o], c[o]
    pos = (v - v[0]) / max(v[-1] - v[0], 1e-12)
    return LinearSegmentedColormap.from_list("measured", list(zip(pos, c))), float(v[0]), float(v[-1])


def recreate_new(res, out_png, dpi=100):
    ct = res.get("chart_type")
    spec = res.get("spec", {})
    f = {"heatmap": _heatmap, "contour": _field, "histogram": _hist, "box": _box, "stacked_bar": _stacked,
         "polar": _polar, "bar3d": _bar3d, "scatter3d": _scatter3d, "surface3d": _surface3d}.get(ct)
    if ct == "bar" and res.get("orientation") == "horizontal":
        f = _stacked
    if f is None:
        return None
    return f(res, spec, out_png, dpi)


def _heatmap(res, spec, out_png, dpi):
    fig, ax = _fig(res, dpi)
    cm, lo, hi = _cmap(res)
    v = np.array(res["values"], float)
    im = ax.imshow(v, cmap=cm, vmin=lo, vmax=hi, aspect="auto")
    ax.set_xticks(range(len(res["columns"])), res["columns"], rotation=45 if len(res["columns"]) > 6 else 0)
    ax.set_yticks(range(len(res["rows"])), res["rows"])
    if spec.get("annotated") or v.size <= 30:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                ax.text(j, i, f"{v[i, j]:.3g}", ha="center", va="center", fontsize=7, color="w")
    cb = fig.colorbar(im, ax=ax, orientation=(spec.get("colorbar") or {}).get("orientation") or "vertical")
    cb.set_label((spec.get("colorbar") or {}).get("title", ""))
    return _finish(fig, out_png, dpi)


def _field(res, spec, out_png, dpi):
    fig, ax = _fig(res, dpi)
    cm, lo, hi = _cmap(res)
    Z = np.array([[np.nan if v is None else v for v in row] for row in res["z"]], float)
    if res.get("levels"):
        m = ax.contourf(res["x"], res["y"], Z, levels=res["levels"], cmap=cm)
    else:
        m = ax.pcolormesh(res["x"], res["y"], Z, cmap=cm, vmin=lo, vmax=hi, shading="auto")
    ax.set_xlabel(spec["x_axis"]["title"]); ax.set_ylabel(spec["y_axis"]["title"])
    cb = fig.colorbar(m, ax=ax, orientation=(spec.get("colorbar") or {}).get("orientation") or "vertical")
    cb.set_label((spec.get("colorbar") or {}).get("title", ""))
    return _finish(fig, out_png, dpi)


def _hist(res, spec, out_png, dpi):
    fig, ax = _fig(res, dpi)
    for name, d in res["series"].items():
        c = np.array(d.get("rgb", [100, 100, 200])) / 255
        L, R, Hh = np.array(d["left"]), np.array(d["right"]), np.array(d["height"])
        if d.get("style") == "step":
            xs = np.r_[[x for a, b in zip(L, R) for x in (a, b)]]; ys = np.r_[[h for h in Hh for _ in (0, 1)]]
            ax.plot(xs, ys, color=c, lw=2, label=name)
        else:
            ax.bar(L, Hh, width=R - L, align="edge", color=c, edgecolor="k", lw=0.4, label=name, alpha=0.8 if len(res["series"]) > 1 else 1)
    ax.set_xlabel(spec["x_axis"]["title"]); ax.set_ylabel(spec["y_axis"]["title"])
    if len(res["series"]) > 1:
        ax.legend()
    return _finish(fig, out_png, dpi)


def _box(res, spec, out_png, dpi):
    fig, ax = _fig(res, dpi)
    names = list(res["series"])
    cats = list(spec.get("categories") or [])
    horiz = res.get("orientation") == "horizontal"
    for k, n in enumerate(names):
        d = res["series"][n]
        stats, pos = [], []
        for c, st in zip(d["categories"], d["stats"]):
            if None in (st["q1"], st["q3"]):
                continue
            ci = cats.index(c) if c in cats else len(pos)
            pos.append(ci + (k - (len(names) - 1) / 2) * 0.8 / max(1, len(names)))
            stats.append({"q1": st["q1"], "q3": st["q3"], "med": st["median"] if st["median"] is not None else (st["q1"] + st["q3"]) / 2,
                          "whislo": st["whislo"], "whishi": st["whishi"], "fliers": st["fliers"], "label": c})
        if stats:
            if horiz:
                pos = [len(cats) - 1 - p for p in pos]            # first category at the top
            ax.bxp(stats, positions=pos, widths=0.6 / max(1, len(names)), orientation="horizontal" if horiz else "vertical",
                   patch_artist=True, boxprops={"facecolor": f"C{k}" if len(names) > 1 else "white"}, medianprops={"color": "k"})
    ticks = list(range(len(cats)))
    if horiz:
        ax.set_yticks(ticks, cats[::-1]); ax.set_xlabel(spec["x_axis"]["title"])
    else:
        ax.set_xticks(ticks, cats); ax.set_ylabel(spec["y_axis"]["title"])
    if len(names) > 1:
        from matplotlib.patches import Patch
        ax.legend([Patch(color=f"C{k}") for k in range(len(names))], names)
    return _finish(fig, out_png, dpi)


def _stacked(res, spec, out_png, dpi):
    fig, ax = _fig(res, dpi)
    cats = list(res.get("categories") or spec.get("categories") or [])
    horiz = res.get("orientation") == "horizontal"
    stacked = res.get("stacked", False)
    n = len(res["series"])
    base = np.zeros(len(cats))
    for k, (name, d) in enumerate(res["series"].items()):
        v = np.zeros(len(cats))
        for c, val in zip(d["categories"], d["values"]):
            if c in cats:
                v[cats.index(c)] = val
        col = np.array(res.get("styles", {}).get(name, {}).get("rgb", [120, 120, 200])) / 255
        pos = np.arange(len(cats)) if stacked else np.arange(len(cats)) + (k - (n - 1) / 2) * 0.8 / n
        w = 0.6 if stacked else 0.8 / n
        if horiz:
            ax.barh(pos, v, left=base if stacked else 0, height=w, color=col, label=name)
        else:
            ax.bar(pos, v, bottom=base if stacked else 0, width=w, color=col, label=name)
        if stacked:
            base = base + v
    if horiz:
        ax.set_yticks(range(len(cats)), cats); ax.invert_yaxis(); ax.set_xlabel(spec["x_axis"]["title"])
    else:
        ax.set_xticks(range(len(cats)), cats); ax.set_ylabel(spec["y_axis"]["title"])
    ax.legend(fontsize=8)
    return _finish(fig, out_png, dpi)


def _polar(res, spec, out_png, dpi):
    fig, ax = _fig(res, dpi, projection="polar")
    cal = res["calibration"]
    ax.set_theta_offset(np.radians(cal["phi0"])); ax.set_theta_direction(cal["direction"])
    for k, (name, d) in enumerate(res["series"].items()):
        th, r = np.radians(np.array(d["theta_deg"])), np.array(d["r"])
        if d["kind"] == "bar":
            w = 2 * np.pi / max(len(th), 1) * 0.9
            ax.bar(th, r - res["r_limits"][0], bottom=res["r_limits"][0], width=w, color=f"C{k}", edgecolor="k", alpha=0.85, label=name)
        elif d["kind"] == "marker":
            ax.scatter(th, r, s=20, color=f"C{k}", label=name)
        else:
            ax.scatter(th, r, s=3, color=f"C{k}", label=name)
    ax.set_ylim(*res["r_limits"])
    if len(res["series"]) > 1:
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1), fontsize=8)
    return _finish(fig, out_png, dpi)


def _axes3_labels(ax, spec):
    ax.set_xlabel(spec["x_axis"]["title"]); ax.set_ylabel(spec["y_axis"]["title"]); ax.set_zlabel(spec["z_axis"]["title"])


def _bar3d(res, spec, out_png, dpi):
    fig, ax = _fig(res, dpi, projection="3d")
    cam = res.get("camera", {})
    xl, yl = cam.get("x_labels", []), cam.get("y_labels", [])
    for row in res.get("table", []):
        i, j = xl.index(row["x_category"]), yl.index(row["y_category"])
        ax.bar3d(i - 0.3, j - 0.3, 0, 0.6, 0.6, row["height"], color="tab:blue", shade=True)
    ax.set_xticks(range(len(xl)), xl); ax.set_yticks(range(len(yl)), yl)
    _axes3_labels(ax, spec)
    return _finish(fig, out_png, dpi)


def _scatter3d(res, spec, out_png, dpi):
    fig, ax = _fig(res, dpi, projection="3d")
    for k, (name, d) in enumerate(res["series"].items()):
        ax.scatter(d["x"], d["y"], d["z"], color=f"C{k}", label=name, depthshade=False)
    _axes3_labels(ax, spec)
    if res.get("flag"):
        ax.set_title("depth guessed (flagged)", fontsize=9, color="red")
    if len(res["series"]) > 1:
        ax.legend(fontsize=8)
    return _finish(fig, out_png, dpi)


def _surface3d(res, spec, out_png, dpi):
    fig, ax = _fig(res, dpi, projection="3d")
    d = res["series"].get("surface", {})
    if len(d.get("x", [])) >= 3:
        x, y, z = np.array(d["x"]), np.array(d["y"]), np.array(d["z"])
        k = np.arange(len(x)) if len(x) <= 4000 else np.random.default_rng(0).choice(len(x), 4000, replace=False)
        cm, lo, hi = _cmap(res)
        ax.scatter(x[k], y[k], z[k], c=z[k], cmap=cm if res.get("cmap_table") else "viridis", s=2)
    _axes3_labels(ax, spec)
    if res.get("flag"):
        ax.set_title("depth guessed (flagged)", fontsize=9, color="red")
    return _finish(fig, out_png, dpi)
