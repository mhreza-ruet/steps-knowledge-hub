"""Recreate the step-5 chart types from the extracted data (for the side-by-side check).

Only extracted numbers, the VLM's text (titles, names, tick labels) and measured
colours are used - nothing is copied from the image.
"""
from __future__ import annotations

import numpy as np
import sys as _sys
import matplotlib
if "ipykernel" not in _sys.modules:      # scripts: no screen needed. In a notebook keep its inline backend,
    matplotlib.use("Agg")                # otherwise plt.show() there would display nothing
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


def _box_axes(fig, box, size):
    """axes placed on a pixel box (x0, y0, x1, y1) of the input image"""
    W, H = size
    x0, y0, x1, y1 = box
    return fig.add_axes([x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H])


def _tick_values(labels):
    out = []
    for t in labels or []:
        v = parse_number(t)
        if v is not None:
            out.append((v, str(t).replace("-", "\u2212") if str(t).startswith("-") else str(t)))
    return out


def _colorbar_like_input(fig, mappable, res, spec):
    """colour bar where it is in the input, with the input's own tick labels and range"""
    cinfo = res.get("colorbar") or {}
    cbs = spec.get("colorbar") or {}
    if not cinfo.get("box"):
        cb = fig.colorbar(mappable, ax=fig.axes[0])
    else:
        x0, y0, x1, y1 = cinfo["box"]
        cax = _box_axes(fig, (x0, y0, x1, y1), res.get("size", (640, 480)))
        cb = fig.colorbar(mappable, cax=cax, orientation=cinfo.get("orient", "vertical"))
        if cinfo.get("label_side") in ("left", "top"):
            (cax.yaxis if cinfo.get("orient") == "vertical" else cax.xaxis).set_ticks_position(cinfo["label_side"])
    tv = _tick_values(cinfo.get("tick_labels") or cbs.get("tick_labels"))
    if tv:
        lo, hi = sorted(cinfo.get("ends") or [min(v for v, _ in tv), max(v for v, _ in tv)])
        tv = [(v, t) for v, t in tv if lo - 1e-9 * abs(hi - lo) <= v <= hi + 1e-9 * abs(hi - lo)]
        cb.set_ticks([v for v, _ in tv], labels=[t for _, t in tv])
    cb.set_label(cbs.get("title", ""))
    return cb


def _heatmap(res, spec, out_png, dpi):
    """the cells at the input's place and size, its colour bar range and ticks, cell gaps / borders,
    printed numbers (as printed, in their measured colour) and tick-label rotation"""
    W, H = res.get("size", (640, 480))
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    v = np.array(res["values"], float)
    nr, nc = v.shape
    ax = _box_axes(fig, res["region"], (W, H)) if res.get("region") else fig.add_subplot(111)
    cm, lo, hi = _cmap(res)
    ends = (res.get("colorbar") or {}).get("ends")
    if ends:
        lo, hi = sorted(ends)
    gap = res.get("cell_gap") or {}
    kw = {"edgecolors": tuple(np.array(gap["rgb"]) / 255), "linewidth": gap["width_px"] * 72 / dpi} if gap else {}
    m = ax.pcolormesh(np.arange(nc + 1), np.arange(nr + 1), v, cmap=cm, vmin=lo, vmax=hi, **kw)
    ax.set_xlim(0, nc); ax.set_ylim(nr, 0)
    rot = (res.get("tick_rotation") or {}).get("x", 0)
    ax.set_xticks(np.arange(nc) + 0.5, res["columns"], rotation=rot, ha="right" if rot == 45 else "center",
                  rotation_mode="anchor" if rot == 45 else "default")
    ax.set_yticks(np.arange(nr) + 0.5, res["rows"])
    if gap and gap.get("colour") == "white":
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    ann = res.get("annotation") or {}
    if ann.get("present"):
        d = ann.get("decimals")
        if d is None:
            d = 0 if np.nanmin(np.abs(v)) >= 10 else 2
        cols = ann.get("rgb") or {}
        for i in range(nr):
            for j in range(nc):
                t = (ann.get("text") or [[None] * nc] * nr)[i][j] or f"{v[i, j]:.{d}f}"
                c = cols.get(f"{i},{j}")
                c = tuple(np.array(c) / 255) if c else ("w" if np.mean(np.array(res["cell_rgb"][i][j])) < 128 else "k")
                ax.text(j + 0.5, i + 0.5, t, ha="center", va="center", fontsize=8, color=c)
    _colorbar_like_input(fig, m, res, spec)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def _field(res, spec, out_png, dpi):
    """filled contours (stepped colour bar) or a smooth colour map, on the input's frame, with
    the input's axis range / ticks, colour-bar range / ticks and its contour lines"""
    from scipy import ndimage as ndi
    from matplotlib.colors import ListedColormap, BoundaryNorm
    W, H = res.get("size", (640, 480))
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    fr = res.get("frame")
    ax = _box_axes(fig, (fr["left"], fr["top"], fr["right"], fr["bottom"]), (W, H)) if fr else fig.add_subplot(111)
    x, y = np.array(res["x"]), np.array(res["y"])
    Z = np.array([[np.nan if q is None else q for q in row] for row in res["z"]], float)
    if np.isnan(Z).any():                       # fill gaps (labels, lines) from the nearest sample
        idx = ndi.distance_transform_edt(np.isnan(Z), return_distances=False, return_indices=True)
        Z = Z[tuple(idx)]
    lim = res.get("axis_limits") or {}
    if lim:                                     # the samples start one frame-line inside: reach the frame
        x = np.r_[lim["x"][0], x, lim["x"][1]] if (x[0] - lim["x"][0]) * (x[-1] - x[0]) > 0 else x
        if len(x) == Z.shape[1] + 2:
            Z = np.c_[Z[:, :1], Z, Z[:, -1:]]
        y = np.r_[lim["y"][0], y, lim["y"][1]] if (y[0] - lim["y"][0]) * (y[-1] - y[0]) > 0 else y
        if len(y) == Z.shape[0] + 2:
            Z = np.r_[Z[:1], Z, Z[-1:]]
    cm, lo, hi = _cmap(res)
    ends = (res.get("colorbar") or {}).get("ends")
    if ends:
        lo, hi = sorted(ends)
    lev = res.get("levels")
    if lev and len(lev) >= 3:
        t = res["cmap_table"]
        mids = 0.5 * (np.array(lev[:-1]) + np.array(lev[1:]))
        tv, tc = np.array(t["values"]), np.array(t["rgb"]) / 255
        cols = [tc[np.argmin(np.abs(tv - mval))] for mval in mids]
        cmap = ListedColormap(cols); norm = BoundaryNorm(lev, cmap.N)
        Zs = ndi.gaussian_filter(Z, 1.2)        # smooth level boundaries (the samples are quantised)
        m = ax.contourf(x, y, Zs, levels=lev, cmap=cmap, norm=norm, extend="neither")
        cl = res.get("contour_lines")
        if cl:
            ax.contour(x, y, Zs, levels=lev, colors=[tuple(np.array(cl["rgb"]) / 255)], linewidths=max(0.5, cl["width_px"] * 72 / dpi))
    else:
        m = ax.pcolormesh(x, y, Z, cmap=cm, vmin=lo, vmax=hi, shading="gouraud")
    if lim:
        ax.set_xlim(*lim["x"]); ax.set_ylim(*lim["y"])
    for k, setter in (("x_axis", ax.set_xticks), ("y_axis", ax.set_yticks)):
        tv = _tick_values(spec.get(k, {}).get("tick_labels"))
        if tv:
            setter([v for v, _ in tv], labels=[t for _, t in tv])
    if lim:
        ax.set_xlim(*lim["x"]); ax.set_ylim(*lim["y"])
    ax.set_xlabel(spec["x_axis"]["title"]); ax.set_ylabel(spec["y_axis"]["title"])
    _colorbar_like_input(fig, m, res, spec)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def _frame_limits(res, key, lo_px, hi_px, labels=None):
    """axis range of the input = the calibrated value at the frame edges (snapped onto a printed
    tick value when within 1.5 % of the span)"""
    m = (res.get("calibration") or {}).get(key)
    if not m:
        return None
    m = m[0]
    f = lambda p: (p - m["a"]) / m["b"]
    v = [f(lo_px), f(hi_px)]
    if m.get("log"):
        v = [10 ** q for q in v]
    t = [q for q, _ in _tick_values(labels)]
    span = abs(v[1] - v[0]) or 1
    return [min(t, key=lambda q: abs(q - e)) if t and min(abs(q - e) for q in t) <= 0.015 * span else e for e in v]


def _axes_like_frame(fig, res, spec):
    """axes on the input's frame, with its x / y range and printed ticks (regular font)"""
    fr = res.get("frame")
    W, H = res.get("size", (640, 480))
    ax = _box_axes(fig, (fr["left"], fr["top"], fr["right"], fr["bottom"]), (W, H)) if fr else fig.add_subplot(111)
    if fr:
        xl = _frame_limits(res, "x_axis", fr["left"], fr["right"], spec["x_axis"].get("tick_labels"))
        yl = _frame_limits(res, "y_axis", fr["bottom"], fr["top"], spec["y_axis"].get("tick_labels"))
        for lim, setlim, setticks, k in ((xl, ax.set_xlim, ax.set_xticks, "x_axis"), (yl, ax.set_ylim, ax.set_yticks, "y_axis")):
            tv = _tick_values(spec[k].get("tick_labels"))
            if tv:
                setticks([q for q, _ in tv], labels=[t for _, t in tv])
            if lim:
                setlim(*lim)
    ax.set_xlabel(spec["x_axis"].get("title", "")); ax.set_ylabel(spec["y_axis"].get("title", ""))
    return ax


def _hist(res, spec, out_png, dpi):
    """bins drawn from the extracted edges / heights with the measured colours, transparency
    (overlaps blend as in the input), bar edge lines, step outlines, overlay curves and legend"""
    from matplotlib.patches import Patch
    W, H = res.get("size", (640, 480))
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = _axes_like_frame(fig, res, spec)
    tr = res.get("transparency") or {}
    alpha = tr.get("alpha", 1.0)
    order = tr.get("draw_order") or []
    names = sorted(res["series"], key=lambda n: order.index(n) if n in order else -1)
    handles = {}
    for name in names:
        d = res["series"][name]
        seen = np.array(d.get("rgb", [100, 100, 200]), float)
        L, R, Hh = np.array(d["left"]), np.array(d["right"]), np.array(d["height"])
        if d.get("style") == "step":
            c = tuple(seen / 255)
            lw = (d.get("line_px") or 2) * 72 / dpi
            edges = np.r_[L, R[-1]] if len(L) else L
            ok = len(L) and np.allclose(L[1:], R[:-1], atol=0.02 * (R - L).mean())
            if ok:
                ax.stairs(Hh, edges, color=c, lw=lw, baseline=0)          # a step histogram closes down to 0 at both ends
            else:
                for l_, r_, h_ in zip(L, R, Hh):
                    ax.plot([l_, l_, r_, r_], [0, h_, h_, 0], color=c, lw=lw)
            handles[name] = Patch(facecolor="none", edgecolor=c, lw=lw, label=name)
        else:
            a_ = alpha if name in order else 1.0
            base = np.clip((seen - (1 - a_) * 255) / a_, 0, 255) / 255     # the colour before transparency
            e = d.get("edge_rgb")
            ax.bar(L, Hh, width=R - L, align="edge", color=tuple(base), alpha=a_,
                   edgecolor=tuple(np.array(e) / 255) if e is not None else "none", lw=0.8 if e is not None else 0)
            handles[name] = Patch(facecolor=tuple(base), alpha=a_, edgecolor="none", label=name)
    for o in res.get("overlay_lines") or []:
        x, y = np.array(o["x"]), np.array(o["y"])
        k = np.argsort(x)
        ax.plot(x[k], y[k], color=tuple(np.array(o["rgb"]) / 255), lw=max(1.0, (o.get("width_px", 2) - 1) * 0.72))
    lg = res.get("legend_box")
    if lg or len(res["series"]) > 1:
        ax.legend(handles=[handles[n] for n in res["series"] if n in handles], loc=(lg or {}).get("loc", "best"))
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def _rgb(c, default):
    return tuple(np.clip(np.asarray(c, float) / 255, 0, 1)) if c is not None else default


def _like_input_axes(fig, res):
    """put the plot frame where it is in the input image"""
    fr, (W, H) = res.get("frame"), res.get("size", (640, 480))
    if fr:
        fig.axes[0].set_position([fr["left"] / W, 1 - fr["bottom"] / H, (fr["right"] - fr["left"]) / W, (fr["bottom"] - fr["top"]) / H])


def _box(res, spec, out_png, dpi):
    """boxes at the measured positions and widths, drawn with the measured fill, edge, median and
    mean-marker colours, notches and outlier markers"""
    fig, ax = _fig(res, dpi)
    names = list(res["series"])
    cats = list(spec.get("categories") or [])
    horiz = res.get("orientation") == "horizontal"
    handles = []
    for k, n in enumerate(names):
        d = res["series"][n]
        draw = d.get("draw") or [{}] * len(d["stats"])
        first = None
        for c, st, sty in zip(d["categories"], d["stats"], draw):
            if None in (st["q1"], st["q3"]):
                continue
            ci = cats.index(c) if c in cats else 0
            pos = sty.get("pos", ci + (k - (len(names) - 1) / 2) * 0.8 / max(1, len(names)))
            width = sty.get("width") or 0.6 / max(1, len(names))
            fill = _rgb(sty.get("fill_rgb"), (1, 1, 1))
            edge = _rgb(sty.get("edge_rgb"), (0, 0, 0))
            med = _rgb(sty.get("median_rgb"), edge)
            mlw = float(np.clip(0.72 * max(1.0, (sty.get("median_lw_px") or 2) - 0.6), 1.0, 4.0))
            fl = sty.get("flier") or {}
            fcol = _rgb(fl.get("rgb"), edge)
            fmk = fl.get("marker", "o")
            b = {"q1": st["q1"], "q3": st["q3"], "med": st["median"] if st["median"] is not None else (st["q1"] + st["q3"]) / 2,
                 "whislo": st["whislo"], "whishi": st["whishi"], "fliers": st["fliers"], "label": c}
            if st.get("mean") is not None:
                b["mean"] = st["mean"]
            if sty.get("notch"):                          # matplotlib's notch half-height: 1.57 IQR / sqrt(n); n unknown -> use the drawn look
                iqr = st["q3"] - st["q1"]
                b["cilo"], b["cihi"] = b["med"] - 0.25 * iqr, b["med"] + 0.25 * iqr
            bp = ax.bxp([b], positions=[pos], widths=width, orientation="horizontal" if horiz else "vertical", patch_artist=True,
                        shownotches=bool(sty.get("notch")), showmeans="mean" in b,
                        boxprops={"facecolor": fill, "edgecolor": edge}, whiskerprops={"color": edge}, capprops={"color": edge},
                        medianprops={"color": med, "lw": mlw},
                        meanprops={"marker": sty.get("mean_marker", "^"), "markerfacecolor": _rgb(sty.get("mean_rgb"), (0.2, 0.6, 0.2)),
                                   "markeredgecolor": _rgb(sty.get("mean_rgb"), (0.2, 0.6, 0.2))},
                        flierprops={"marker": fmk, "markeredgecolor": fcol,
                                    "markerfacecolor": fcol if (fl.get("filled") and fmk not in ("x", "+")) else "none"})
            first = first or bp["boxes"][0]
        if first is not None:
            handles.append(first)
    clim = res.get("category_limits")
    if horiz:
        ax.set_yticks(range(len(cats)), cats); ax.set_xlabel(spec["x_axis"]["title"])
        if clim:
            ax.set_ylim(*(clim[::-1] if res.get("category_axis_reversed") else clim))
        if res.get("value_limits"):
            ax.set_xlim(*res["value_limits"])
    else:
        ax.set_xticks(range(len(cats)), cats); ax.set_ylabel(spec["y_axis"]["title"])
        if clim:
            ax.set_xlim(*(clim[::-1] if res.get("category_axis_reversed") else clim))
        if res.get("value_limits"):
            ax.set_ylim(*res["value_limits"])
    if res.get("grid"):
        ax.grid(axis="x" if horiz else "y", alpha=0.5); ax.set_axisbelow(True)
    if len(names) > 1:
        ax.legend(handles, names, loc="upper right")
    _like_input_axes(fig, res)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


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


def _chain(th, r, rmax):
    """order the points of a traced curve along the curve (nearest neighbour walk in the
    plane) and cut it where it jumps - a rose or a closed loop has several r per angle"""
    x, y = r * np.cos(th), r * np.sin(th)
    n = len(x)
    if n < 3:
        return [np.arange(n)]
    left = set(range(1, n)); path = [0]
    while left:
        i = path[-1]
        j = min(left, key=lambda k: (x[k] - x[i]) ** 2 + (y[k] - y[i]) ** 2)
        path.append(j); left.remove(j)
    path = np.array(path)
    step = np.hypot(np.diff(x[path]), np.diff(y[path]))
    cut = step > max(5 * np.median(step), 0.04 * rmax)
    parts = np.split(path, np.where(cut)[0] + 1)
    # close the loop if the walk ends next to where it started
    if len(parts) == 1 and np.hypot(x[path[-1]] - x[path[0]], y[path[-1]] - y[path[0]]) <= max(5 * np.median(step), 0.04 * rmax):
        parts = [np.r_[path, path[:1]]]
    return parts


MPL_MK = {"circle": "o", "square": "s", "triangle": "^", "triangle_up": "^", "triangle_down": "v", "diamond": "D",
          "star": "*", "cross": "x", "x": "x", "plus": "+"}


def _polar(res, spec, out_png, dpi):
    """the circle where it is in the input, its zero direction / turning direction, the printed
    angle labels and radial tick labels (grid circles at those values), the radial labels along
    the measured direction, measured series colours / styles and the measured bar bin width"""
    W, H = res.get("size", (640, 480))
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    c = res.get("circle")
    if c:
        ax = fig.add_axes([(c["cx"] - c["R"]) / W, 1 - (c["cy"] + c["R"]) / H, 2 * c["R"] / W, 2 * c["R"] / H], projection="polar")
    else:
        ax = fig.add_subplot(111, projection="polar")
    cal = res["calibration"]
    # zero direction in (-180, 180] (an offset of 359.8 deg makes matplotlib < 3.11 drop the grid
    # circles and the outer ring); within 1.5 deg of a multiple of 45 it is that multiple
    off = (float(cal["phi0"]) + 180.0) % 360.0 - 180.0
    if abs(off - 45 * round(off / 45)) <= 1.5:
        off = 45.0 * round(off / 45)
    ax.set_theta_offset(np.radians(off)); ax.set_theta_direction(int(cal["direction"]))
    rlo, rhi = res.get("r_limits_draw") or res["r_limits"]
    for k, (name, d) in enumerate(res["series"].items()):
        th, r = np.radians(np.array(d["theta_deg"])), np.array(d["r"])
        col = _rgb(d.get("rgb"), f"C{k}")
        if d["kind"] == "bar":
            bw = (res.get("bar_bins") or {}).get("bin_width_deg")
            w = np.radians(bw) if bw else 2 * np.pi / max(len(th), 1) * 0.9
            ax.bar(th, r - rlo, bottom=rlo, width=w, color=col, edgecolor="k", lw=0.8, label=name)
        elif d["kind"] == "marker":
            mk = MPL_MK.get(str(d.get("marker") or "circle").lower(), "o")
            if d.get("line", "none") not in ("none", ""):     # markers joined by a line, in angle order, closed if all round
                o = np.argsort(th); tt, rr = th[o], r[o]
                if len(tt) > 2 and (2 * np.pi - (tt[-1] - tt[0])) <= 1.5 * np.median(np.diff(tt)):
                    tt, rr = np.r_[tt, tt[:1]], np.r_[rr, rr[:1]]
                ax.plot(tt, rr, color=col, marker=mk, ls="--" if d["line"] == "dashed" else "-", ms=6, label=name)
            else:
                ax.plot(th, r, ls="none", marker=mk, color=col, ms=6, label=name)
        else:
            first = True
            for part in _chain(th, r - rlo, rhi - rlo):
                ax.plot(th[part], r[part], color=col, lw=1.8, ls="--" if d.get("line") == "dashed" else "-", label=name if first else None)
                first = False
    ax.set_ylim(rlo, rhi)
    from .polar import parse_angle
    al = res.get("angle_labels_used") or spec.get("angle_labels") or []
    av = [(parse_angle(t)[0], t) for t in al]
    av = [(v, t) for v, t in av if v is not None]
    if av:
        ax.set_thetagrids([v for v, _ in av], labels=[t for _, t in av])
    tv = [(v, t) for v, t in _tick_values(spec.get("radial_labels")) if rlo - 1e-9 <= v <= rhi + 1e-9]
    if tv:
        ax.set_rticks([v for v, _ in tv], labels=[t for _, t in tv])
    if res.get("radial_label_angle_deg") is not None:
        ax.set_rlabel_position(res["radial_label_angle_deg"])
    if len(res["series"]) > 1:
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1), fontsize=8)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def _axes3_labels(ax, spec):
    ax.set_xlabel(spec["x_axis"]["title"]); ax.set_ylabel(spec["y_axis"]["title"]); ax.set_zlabel(spec["z_axis"]["title"])


def _apply_view(ax, res):
    """same view as the input: elevation, azimuth, projection, box proportions and limits found
    by the camera fit (axes3d.fit_view_camera)"""
    v = (res.get("camera") or {}).get("view") or {}
    lim = v.get("limits") or (res.get("camera") or {}).get("limits")
    if lim:
        cam = res.get("camera") or {}
        lim = {k: list(lim[k]) for k in "xyz"}
        for k in "xyz":
            # a tick printed in the input is inside its axis: a fitted limit a hair short of the
            # first / last tick value (fit noise) would drop that tick from the redraw
            vals = [float(t) for t in (cam.get(f"{k}_values") or [])]
            lo, hi = min(lim[k]), max(lim[k]); sp = hi - lo
            for t in vals:
                if lo - 0.015 * sp <= t < lo:
                    lo = t - 1e-6 * sp
                if hi < t <= hi + 0.015 * sp:
                    hi = t + 1e-6 * sp
            lim[k] = [lo, hi] if lim[k][0] <= lim[k][1] else [hi, lo]
        ax.set_xlim(*lim["x"]); ax.set_ylim(*lim["y"]); ax.set_zlim(*lim["z"])
    if v:
        ax.set_box_aspect(v.get("box_aspect", (4, 4, 3)))
        if v.get("proj_type") == "ortho":
            ax.set_proj_type("ortho")
        else:
            ax.set_proj_type("persp", focal_length=v.get("focal_length") or 1.0)
        ax.view_init(v["elev"], v["azim"])
    return bool(v)


def _place_like_input(fig, ax, res, iters=4, to_draw=None):
    """move / scale the 3-D axes so the box lands where it is in the input image"""
    cam = res.get("camera") or {}
    if "P" not in cam:
        return
    from mpl_toolkits.mplot3d import proj3d
    from .axes3d import project
    P = np.asarray(cam["P"], float)
    lim = ((cam.get("view") or {}).get("limits")) or cam["limits"]
    C = np.array([[a, b, c] for a in lim["x"] for b in lim["y"] for c in lim["z"]], float)
    target = project(P, C)
    if to_draw is not None:                          # the redraw uses mirrored coordinates
        C = np.array([to_draw(c) for c in C], float)
    W, H = fig.get_size_inches() * fig.dpi
    for _ in range(iters):
        fig.canvas.draw()
        xs, ys, _ = proj3d.proj_transform(C[:, 0], C[:, 1], C[:, 2], ax.get_proj())
        d = ax.transData.transform(np.c_[xs, ys])
        got = np.c_[d[:, 0], H - d[:, 1]]
        # target = k * got + t (least squares)
        A = np.zeros((2 * len(got), 3)); bb = np.zeros(2 * len(got))
        A[0::2, 0], A[0::2, 1], bb[0::2] = got[:, 0], 1, target[:, 0]
        A[1::2, 0], A[1::2, 2], bb[1::2] = got[:, 1], 1, target[:, 1]
        (k, tx, ty), *_ = np.linalg.lstsq(A, bb, rcond=None)
        pos = ax.get_position()
        x0, x1 = pos.x0 * W, pos.x1 * W
        y0, y1 = H - pos.y1 * H, H - pos.y0 * H              # pixel rows (down)
        nx0, nx1 = k * x0 + tx, k * x1 + tx
        ny0, ny1 = k * y0 + ty, k * y1 + ty
        ax.set_position([nx0 / W, (H - ny1) / H, (nx1 - nx0) / W, (ny1 - ny0) / H])
        if abs(k - 1) < 1e-3 and abs(tx) < 0.3 and abs(ty) < 0.3:
            break


def _mpl_face_factors():
    """matplotlib's lighting factor for the 6 faces of a 3-D bar (-z, +z, -y, +y, -x, +x), computed
    with matplotlib's own shading code (the light is fixed in data space, not in the view)"""
    import mpl_toolkits.mplot3d.art3d as A3
    cub = np.array([[(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)], [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
                    [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
                    [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)], [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]], float)
    c = A3._shade_colors(np.array([[0.5, 0.5, 0.5, 1.0]] * 6), A3._generate_normals(cub), None)
    return np.asarray(c)[:, :3].mean(1) / 0.5


def _mpl_shade_factor():
    return float(_mpl_face_factors()[1])


def _choose_mirror(res, view):
    """The same picture is seen from azimuth a with the category order as fitted, or from a + 180
    with both category orders reversed. The light of the redraw is fixed, so pick the one whose
    lit faces have the brightness order of the input (top / x-facing / y-facing side)."""
    s2f = (res.get("fit") or {}).get("shade_to_face") or {}
    obs = {v: int(k) for k, v in s2f.items()}             # face -> brightness rank in the input
    if not view or len(obs) < 3:
        return False
    f = _mpl_face_factors()

    def predicted(azim):
        e, a = np.radians(view["elev"]), np.radians(azim)
        vx, vy = np.cos(e) * np.cos(a), np.cos(e) * np.sin(a)
        return {"top": f[1], "x-side": f[5] if vx > 0 else f[4], "y-side": f[3] if vy > 0 else f[2]}

    def agree(p):
        faces = ["top", "x-side", "y-side"]
        return sum(1 for i in range(3) for j in range(i + 1, 3)
                   if np.sign(p[faces[i]] - p[faces[j]]) == np.sign(obs[faces[i]] - obs[faces[j]]))
    return agree(predicted(view["azim"] + 180)) > agree(predicted(view["azim"]))


def _paint_far_to_near(ax, drawn):
    """matplotlib orders separate 3-D collections by their mean depth, which puts bars of a grid in
    front of nearer ones.  Bars on a regular floor grid never overlap in x-y, so their correct drawing
    order is simply the distance of the bar centre along the horizontal viewing direction (measured
    in the plot box's own proportions); draw far bars first and switch matplotlib's own ordering off."""
    if not drawn:
        return
    e, a = np.radians(ax.elev), np.radians(ax.azim)
    asp = np.asarray(ax.get_box_aspect(), float)
    sx = (np.diff(ax.get_xlim3d())[0] or 1.0) / asp[0]
    sy = (np.diff(ax.get_ylim3d())[0] or 1.0) / asp[1]
    dx, dy = np.cos(e) * np.cos(a) / sx, np.cos(e) * np.sin(a) / sy       # towards the viewer
    ax.computed_zorder = False
    order = sorted(drawn, key=lambda t: t[0] * dx + t[1] * dy)
    for k, (_, _, pc) in enumerate(order):
        pc.set_zorder(1 + k * 1e-3)


def _bar3d(res, spec, out_png, dpi):
    W, H = res.get("size", (640, 480))
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    cam = res.get("camera", {})
    xl, yl = cam.get("x_labels", []), cam.get("y_labels", [])
    xv, yv = cam.get("x_values", list(range(len(xl)))), cam.get("y_values", list(range(len(yl))))
    w = float(res.get("bar_width_grid_units", 0.6))
    k_top = _mpl_shade_factor()
    zb = 0.0
    view = dict((cam.get("view") or {}))
    nx, ny = len(xl), len(yl)
    mirror = _choose_mirror(res, view)
    if mirror:                                      # same picture, light falling as in the input
        fx = lambda a: (nx - 1) - a
        fy = lambda b: (ny - 1) - b
        view["azim"] = view["azim"] + 180
        lim = dict(view.get("limits") or cam["limits"])
        view["limits"] = {"x": sorted(fx(v) for v in lim["x"]), "y": sorted(fy(v) for v in lim["y"]), "z": lim["z"]}
        res = dict(res, camera=dict(cam, view=view))
        xv, yv = [fx(v) for v in xv], [fy(v) for v in yv]
    else:
        fx = fy = (lambda q: q)
    rows = [r for r in res.get("table", []) if r.get("height") is not None]
    drawn = []
    for row in rows:
        a, b = fx(row["x_pos"]), fy(row["y_pos"])
        rgb = np.asarray(row.get("rgb") or [31, 119, 180], float) / 255
        base = np.clip(rgb / k_top, 0, 1)            # the measured top colour, before matplotlib's lighting
        pc = ax.bar3d(a - w / 2, b - w / 2, zb, w, w, row["height"] - zb, color=tuple(base), shade=True)
        drawn.append((a, b, pc))
    for row in res.get("table", []):                 # hidden bars: dashed outline of the largest possible height
        if row.get("height") is None and "at most" in str(row.get("note", "")):
            ub = float(str(row["note"]).split("at most")[-1])
            a, b = fx(row["x_pos"]), fy(row["y_pos"])
            xs_ = [a - w / 2, a + w / 2, a + w / 2, a - w / 2, a - w / 2]; ys_ = [b - w / 2, b - w / 2, b + w / 2, b + w / 2, b - w / 2]
            ax.plot(xs_, ys_, [ub] * 5, color="gray", ls=":", lw=0.8, zorder=0.5)
    ax.set_xticks(xv, xl); ax.set_yticks(yv, yl)
    _axes3_labels(ax, spec)
    _apply_view(ax, res)
    _paint_far_to_near(ax, drawn)
    _place_like_input(fig, ax, dict(res, camera=cam), to_draw=lambda c: (fx(c[0]), fy(c[1]), c[2]))
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def _fig3d_like_input(res, dpi):
    W, H = res.get("size", (640, 480))
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax


def _ticks3d(ax, spec):
    for k, setter in (("x_axis", ax.set_xticks), ("y_axis", ax.set_yticks), ("z_axis", ax.set_zticks)):
        tv = _tick_values((spec.get(k) or {}).get("tick_labels"))
        if tv:
            setter([v for v, _ in tv], labels=[t for _, t in tv])


def _finish3d(fig, ax, res, spec, out_png, dpi, mappable=None):
    _axes3_labels(ax, spec)
    _ticks3d(ax, spec)
    _apply_view(ax, res)
    _place_like_input(fig, ax, res)
    if mappable is not None and (res.get("colorbar") or {}).get("box"):
        _colorbar_like_input(fig, mappable, res, spec)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def _scatter3d(res, spec, out_png, dpi):
    """every point drawn in the colour it has in the input (per point when colour = value), with
    the series' marker, in the input's view, so the redraw can be checked point by point"""
    fig, ax = _fig3d_like_input(res, dpi)
    for k, (name, d) in enumerate(res["series"].items()):
        if not d.get("x"):
            continue
        mk = MPL_MK.get(str(d.get("marker") or "circle").lower().replace("triangle_up", "triangle"), "o")
        ms = (1.05 * max(4.0, d.get("marker_px", 6)) * 72 / dpi) ** 2       # scatter size = (diameter in pt)^2
        each = d.get("rgb_each")
        col = [tuple(np.array(c) / 255) for c in each] if each else _rgb(d.get("rgb"), f"C{k}")
        ax.scatter(d["x"], d["y"], d["z"], c=col, marker=mk, s=ms, label=name, depthshade=False,
                   edgecolors="none")
        if res.get("stems") or (res.get("spec") or {}).get("stems"):
            z0 = min((res.get("camera") or {}).get("z_values") or [0])
            zb = 0.0 if min((res.get("camera") or {}).get("z_values") or [0]) <= 0 else z0
            for x, y, z, c in zip(d["x"], d["y"], d["z"], col if isinstance(col, list) else [col] * len(d["x"])):
                ax.plot([x, x], [y, y], [zb, z], color=c, lw=1.2)
    if len(res["series"]) > 1:
        ax.legend(fontsize=8)
    mappable = None
    if res.get("cmap_table"):
        import matplotlib.cm as mcm
        from matplotlib.colors import Normalize
        cm, lo, hi = _cmap(res)
        ends = (res.get("colorbar") or {}).get("ends")
        if ends:
            lo, hi = sorted(ends)
        mappable = mcm.ScalarMappable(norm=Normalize(lo, hi), cmap=cm)
    return _finish3d(fig, ax, res, spec, out_png, dpi, mappable)


def _surface3d(res, spec, out_png, dpi):
    """A coloured surface z = f(x, y) is redrawn as one whole, continuous sheet:
      1. the samples are binned on a regular x-y grid over the axis range, one robust (median)
         z per cell - this removes the per-pixel colour-quantisation noise;
      2. cells with no sample (parts hidden behind a fold, or never matched) are filled from the
         surrounding cells (linear inside the sampled area, nearest outside), so the sheet has no
         holes - the CSV keeps only the measured samples;
      3. drawn with plot_surface and the measured colour scale (same range as the input's colour
         bar), so the colouring follows z exactly as in the input.
    A wireframe (no colour bar) is redrawn from its traced lines."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from scipy.interpolate import griddata
    from scipy import ndimage as ndi
    fig, ax = _fig3d_like_input(res, dpi)
    d = res["series"].get("surface", {})
    cam = res.get("camera") or {}
    mappable = None
    if res.get("cmap_table") and len(d.get("x", [])) >= 20:
        import matplotlib.cm as mcm
        from matplotlib.colors import Normalize
        x, y, z = np.array(d["x"]), np.array(d["y"]), np.array(d["z"])
        xv = cam.get("x_values") or [x.min(), x.max()]; yv = cam.get("y_values") or [y.min(), y.max()]
        n = 45
        gx = np.linspace(min(xv), max(xv), n); gy = np.linspace(min(yv), max(yv), n)
        ix = np.clip(np.round((x - gx[0]) / (gx[1] - gx[0])).astype(int), 0, n - 1)
        iy = np.clip(np.round((y - gy[0]) / (gy[1] - gy[0])).astype(int), 0, n - 1)
        GZ = np.full((n, n), np.nan)
        order = np.lexsort((z, iy * n + ix))
        keys = (iy * n + ix)[order]; zs = z[order]
        starts = np.r_[0, np.where(np.diff(keys))[0] + 1]; ends = np.r_[starts[1:], len(keys)]
        for a_, b_ in zip(starts, ends):
            GZ.flat[keys[a_]] = np.median(zs[a_:b_])
        GX, GY = np.meshgrid(gx, gy)
        ok = ~np.isnan(GZ)
        if ok.sum() >= 4:
            pts = np.c_[GX[ok], GY[ok]]
            fill = griddata(pts, GZ[ok], (GX, GY), method="linear")
            near = griddata(pts, GZ[ok], (GX, GY), method="nearest")
            GZ = np.where(ok, GZ, np.where(np.isnan(fill), near, fill))
            GZ = ndi.gaussian_filter(GZ, 0.8, mode="nearest")
        cm, lo, hi = _cmap(res)
        ends_ = (res.get("colorbar") or {}).get("ends")
        if ends_:
            lo, hi = sorted(ends_)
        ax.plot_surface(GX, GY, GZ, cmap=cm, vmin=lo, vmax=hi, rstride=1, cstride=1, linewidth=0.15,
                        edgecolor=(1, 1, 1, 0.15), antialiased=True)
        mappable = mcm.ScalarMappable(norm=Normalize(lo, hi), cmap=cm)
    elif res.get("draw_lines"):
        col = _rgb(res.get("line_rgb"), "C0")
        ax.add_collection3d(Line3DCollection(res["draw_lines"], colors=[col], linewidths=1.2))
    elif len(d.get("x", [])) >= 3:
        ax.scatter(d["x"], d["y"], d["z"], s=2, color=_rgb(res.get("line_rgb"), "C0"))
    return _finish3d(fig, ax, res, spec, out_png, dpi, mappable)
