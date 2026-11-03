"""Recreate a chart from the extracted data, with nothing typed by hand.

Everything comes from the pipeline result:
  canvas size  -> image size            plot-area box -> detected frame
  axis limits  -> calibration at the frame edges
  tick labels / titles / series names / markers -> VLM spec
  series colours -> median colour of the pixels that were extracted
  legend position -> detected legend box (if any)

A sidecar <png>.spec.json (the spec used) is written next to the image so the
recreated plot can also be re-read without the VLM (FileSpecReader).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .calibrate import AxisMap, parse_number

MPL_MARKER = {"circle": "o", "square": "s", "triangle_up": "^", "triangle_down": "v",
              "diamond": "D", "star": "*", "cross": "x", "none": None}


def _maps(res, key):
    return [AxisMap(**m) for m in res["calibration"].get(key, [])]


def _limits(m, p0, p1):
    """Values at the two frame edges (p0 = left/bottom pixel, p1 = right/top pixel)."""
    return float(m.to_value(p0)), float(m.to_value(p1))


def _ticks(ax_spec, lo, hi, log=False, edge_tol=0.03):
    """VLM tick labels that fall on this axis. A label just outside the frame
    edge (the frame usually sits exactly on the first/last tick, and the
    calibration is a fraction of a pixel off) is kept and the limit moved to it."""
    f = (lambda v: np.log10(v)) if log else (lambda v: v)
    flo, fhi = f(lo), f(hi)
    span = abs(fhi - flo)
    vals, labs = [], []
    for s in (ax_spec or {}).get("tick_labels", []):
        v = parse_number(s)
        if v is None or (log and v <= 0):
            continue
        fv = f(v)
        if min(flo, fhi) - edge_tol * span <= fv <= max(flo, fhi) + edge_tol * span:
            vals.append(v); labs.append(s)
            if fv < min(flo, fhi):
                if flo < fhi: lo = v
                else: hi = v
            if fv > max(flo, fhi):
                if flo < fhi: hi = v
                else: lo = v
    return vals, labs, lo, hi


def _style_axis(ax, which, ax_spec, lo, hi, relative, fs, log):
    """Returns the tick labels actually drawn."""
    setl = ax.set_xlim if which == "x" else ax.set_ylim
    if log:
        (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    setter = ax.set_xticks if which == "x" else ax.set_yticks
    labeler = ax.set_xticklabels if which == "x" else ax.set_yticklabels
    drawn = []
    if relative:
        setter([])
    else:
        v, drawn, lo, hi = _ticks(ax_spec, lo, hi, log)
        if v:
            setter(v); labeler(drawn, fontsize=fs)
    setl(lo, hi)
    if log:
        (ax.xaxis if which == "x" else ax.yaxis).set_minor_locator(matplotlib.ticker.NullLocator())
    return drawn


def recreate(res, out_png, dpi=100, title_fs=None, show=False):
    """res: pipeline() output. Writes out_png (+ out_png.spec.json), returns the path."""
    if res.get("status") == "rejected":
        W, H = res.get("size", (640, 480))
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        import textwrap
        fig.text(0.5, 0.5, "REJECTED - not extracted\n\n" + textwrap.fill("; ".join(res.get("status_reasons", [])), 60),
                 ha="center", va="center", fontsize=10, color="firebrick")
        fig.savefig(out_png, dpi=dpi); plt.close(fig)
        return out_png
    from .render_new import recreate_new
    if res.get("chart_type") in ("heatmap", "contour", "histogram", "box", "stacked_bar", "polar", "bar3d", "scatter3d", "surface3d") \
            or (res.get("chart_type") == "bar" and res.get("orientation") == "horizontal"):
        return recreate_new(res, out_png, dpi)
    if res.get("chart_type") == "bar":
        return recreate_bars(res, out_png, dpi)
    if res.get("chart_type") == "pie":
        return recreate_pie(res, out_png, dpi)
    spec, fr = res["spec"], res["frame"]
    W, H = res["size"]
    fs = title_fs or max(8, round(H / 34))
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    L, R, T, B = fr["left"], fr["right"], fr["top"], fr["bottom"]
    X = _maps(res, "x_axis")
    Y = _maps(res, "y_axis")
    YR = _maps(res, "y_axis_right")
    if not X or not Y:
        fig.text(0.5, 0.5, "axis calibration failed", ha="center")
        fig.savefig(out_png, dpi=dpi); plt.close(fig)
        return out_png
    xrel = res["calibration_info"].get("x_axis", {}).get("relative", False)
    yrel = res["calibration_info"].get("y_axis", {}).get("relative", False)
    x0, x1 = _limits(X[0], L, R)

    # ---- y segments (one for a normal axis, two for a broken axis), bottom -> top
    def segs(maps):
        out = []
        for m in maps:
            lo_px = max(T, m.lo_px) if m.lo_px is not None else T
            hi_px = min(B, m.hi_px) if m.hi_px is not None else B
            out.append((m, hi_px, lo_px))        # (map, bottom pixel, top pixel)
        return sorted(out, key=lambda s: -s[1])
    ysegs = segs(Y)
    axes = []
    drawn = {"x_axis": [], "y_axis": [], "y_axis_right": []}
    gap = 6 if len(ysegs) > 1 else 0
    for k, (m, pb, pt) in enumerate(ysegs):
        pb2 = pb - (gap / 2 if k > 0 else 0)
        pt2 = pt + (gap / 2 if k < len(ysegs) - 1 else 0)
        ax = fig.add_axes([L / W, 1 - pb2 / H, (R - L) / W, (pb2 - pt2) / H])
        y0, y1 = _limits(m, pb, pt)
        drawn["y_axis"] += _style_axis(ax, "y", spec["y_axis"], y0, y1, yrel, fs, m.log)
        dx = _style_axis(ax, "x", spec["x_axis"], x0, x1, xrel, fs, X[0].log)
        if k == 0:
            drawn["x_axis"] = dx
        if k > 0:
            ax.set_xticklabels([]); ax.spines["bottom"].set_visible(False); ax.tick_params(axis="x", length=0)
        if k < len(ysegs) - 1:
            ax.spines["top"].set_visible(False)
        ax.tick_params(direction="out", labelsize=fs, width=max(1, fr["line_width"] / 2))
        for s in ax.spines.values():
            s.set_linewidth(max(1, fr["line_width"] / 1.5))
        axes.append((ax, m))
    base = axes[0][0]
    base.set_xlabel(spec["x_axis"].get("title", ""), fontsize=fs + 1, fontweight="bold")
    fig.text(max(0.01, (L - 0.11 * W) / W) if False else 0.012, 1 - ((T + B) / 2) / H,
             spec["y_axis"].get("title", ""), rotation=90, va="center", ha="left", fontsize=fs + 1, fontweight="bold")
    axr = None
    if YR:
        axr = base.twinx()
        top_ax = axes[-1][0]
        axr.set_position([L / W, 1 - B / H, (R - L) / W, (B - T) / H])
        r0, r1 = _limits(YR[0], B, T)
        drawn["y_axis_right"] = _style_axis(axr, "y", spec.get("y_axis_right"), r0, r1, False, fs, YR[0].log)
        axr.set_ylabel(spec["y_axis_right"].get("title", ""), fontsize=fs + 1, fontweight="bold")
        axr.tick_params(labelsize=fs)

    # ---- data
    handles = []
    for s in spec["series"]:
        name = s["name"]
        d = res["series"].get(name)
        if not d or not d["x"]:
            continue
        st = res["styles"].get(name, {})
        col = np.clip(np.array(st.get("rgb", [0, 0, 0])) / 255.0, 0, 1)
        mk = MPL_MARKER.get(s["marker"]) if d["kind"] == "marker" else None
        ls = {"solid": "-", "dashed": "--"}.get(s["line"], "none")
        if d["kind"] == "line":
            ls = "--" if s["line"] == "dashed" else "-"
        # marker size from the measured marker thickness (largest inscribed radius, px);
        # matplotlib markersize (pt) per inscribed radius at 100 dpi, measured once per shape
        if mk:
            k_ms = {"o": 1.31, "s": 1.25, "^": 2.0, "v": 2.0, "D": 1.28, "*": 2.5, "x": 5.0}.get(mk, 1.3)
            r_ = st.get("marker_r") or st.get("marker_px", 9) / 2.6
            ms = float(np.clip(k_ms * r_ * 100 / dpi, 3, 16))
        else:
            ms = 0
        lw = float(np.clip(st["line_px"] * 0.72 * 100 / dpi, 0.8, 5)) if st.get("line_px") else 1.6
        kw = dict(color=col, marker=mk or "", linestyle=ls, markersize=ms, linewidth=lw, label=name,
                  markerfacecolor="none" if s["marker_fill"] == "hollow" else col)
        x = np.asarray(d["x"]); y = np.asarray(d["y"])
        if d["y_axis"] == "y_axis_right" and axr is not None:
            h, = axr.plot(x, y, **kw)
        elif len(axes) == 1:
            h, = axes[0][0].plot(x, y, **kw)
        else:   # broken axis: each point to its segment
            segi = np.asarray(d.get("segment", [0] * len(x)))
            h = None
            for k, (ax, m) in enumerate(axes):
                idx = [i for i in range(len(x)) if Y.index(m) == segi[i]] if "segment" in d else list(range(len(x)))
                if idx:
                    hh, = ax.plot(x[idx], y[idx], **kw)
                    h = h or hh
        if h is not None:
            handles.append(h)

    # ---- legend: always drawn. Where the input had legend boxes, each box is
    # redrawn at the same place, with the same entries, columns and text size.
    # Otherwise matplotlib puts it where it covers the fewest data points.
    if handles:
        byname = {h.get_label(): h for h in handles}
        placed = set()
        lays = [dict(l, series=list(l["series"])) for l in (res.get("legend_layout") or []) if l["series"]]
        if lays:
            # series whose legend entry was not recognised (e.g. a superscript in its name)
            # join the top legend box, in the VLM's order
            inbox = {n for l in lays for n in l["series"]}
            order = [q["name"] for q in spec["series"]]
            top = min(lays, key=lambda l: l["box"][1])
            top["series"] = [n for n in order if n in set(top["series"]) | ({h.get_label() for h in handles} - inbox)]
        for lay in lays:
            hs = [byname[n] for n in lay["series"] if n in byname]
            if not hs:
                continue
            x0b, y0b = lay["box"][0], lay["box"][1]
            lfs = float(np.clip(lay["text_px"] * 0.9 * 100 / dpi, 6, fs + 4))
            fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(x0b / W, 1 - y0b / H), ncol=lay["ncol"],
                       fontsize=lfs, frameon=True, edgecolor="black", fancybox=False, framealpha=1.0,
                       borderaxespad=0, handlelength=1.6, columnspacing=1.0)
            placed.update(h.get_label() for h in hs)
        rest = [h for h in handles if h.get_label() not in placed]
        if rest:
            axes[-1][0].legend(handles=rest, loc="best", fontsize=fs - 1, frameon=True, edgecolor="black",
                               fancybox=False, framealpha=1.0)

    out_png = str(out_png)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    # sidecar: the spec of what was DRAWN (labels outside the recreated frame are dropped)
    side = json.loads(json.dumps(spec))
    for key, labs in drawn.items():
        if side.get(key) and side[key].get("tick_labels") and not (key == "x_axis" and xrel) and not (key == "y_axis" and yrel):
            side[key]["tick_labels"] = [t for t in side[key]["tick_labels"] if t in labs]
    Path(out_png).with_suffix(".spec.json").write_text(json.dumps(side, indent=1))
    return out_png


# ============================================================================ bars
def _segments(maps, T, B):
    out = []
    for m in maps:
        lo_px = max(T, m.lo_px) if m.lo_px is not None else T
        hi_px = min(B, m.hi_px) if m.hi_px is not None else B
        out.append((m, hi_px, lo_px))          # (map, bottom pixel, top pixel)
    return sorted(out, key=lambda s: -s[1])


def recreate_bars(res, out_png, dpi=100):
    """Grouped bars at the measured positions: same canvas, frame, y calibration
    (broken / right axes too), category names and series styles from the VLM,
    fill colours and bar widths / offsets measured in the pixels."""
    spec, fr = res["spec"], res["frame"]
    W, H = res["size"]
    fs = max(8, round(H / 34))
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    L, R, T, B = fr["left"], fr["right"], fr["top"], fr["bottom"]
    Y, YR = _maps(res, "y_axis"), _maps(res, "y_axis_right")
    cats = res.get("categories") or []
    cen = res.get("category_centres_px") or []
    if not Y or not cats or len(cen) < 1:
        fig.text(0.5, 0.5, "nothing to draw (no calibration or no bars)", ha="center")
        fig.savefig(out_png, dpi=dpi); plt.close(fig); return str(out_png)
    # category index <-> pixel (straight line through the measured group centres)
    if len(cen) >= 2:
        k, c0 = np.polyfit(np.arange(len(cen)), cen, 1)
    else:
        k, c0 = (R - L) / 2, cen[0]
    px2x = lambda p: (p - c0) / k
    xlo, xhi = px2x(L), px2x(R)
    cat_axis = res.get("category_axis") or ["left"] * len(cats)
    segs = _segments(Y, T, B)
    axes, gap = [], 6 if len(segs) > 1 else 0
    for i, (m, pb, pt) in enumerate(segs):
        pb2 = pb - (gap / 2 if i > 0 else 0); pt2 = pt + (gap / 2 if i < len(segs) - 1 else 0)
        ax = fig.add_axes([L / W, 1 - pb2 / H, (R - L) / W, (pb2 - pt2) / H])
        y0, y1 = _limits(m, pb, pt)
        _style_axis(ax, "y", spec["y_axis"], y0, y1, False, fs, m.log)
        ax.set_xlim(xlo, xhi)
        if i > 0:
            ax.spines["bottom"].set_visible(False); ax.tick_params(axis="x", bottom=False, labelbottom=False)
        if i < len(segs) - 1:
            ax.spines["top"].set_visible(False)
        ax.patch.set_alpha(0)
        axes.append((ax, m, pb, pt))
    base = axes[0][0]
    base.set_xticks(range(len(cats)))
    longest = max(len(c) for c in cats)
    rot = 0 if longest * fs * 0.62 < 0.9 * abs(k) else (30 if longest * fs * 0.62 < 3 * abs(k) else 45)
    base.set_xticklabels(cats, rotation=rot, ha="right" if rot else "center", rotation_mode="anchor",
                         fontsize=fs if rot == 0 else max(7, min(fs, 0.42 * abs(k))))
    axr = None
    if YR:
        axr = fig.add_axes([L / W, 1 - B / H, (R - L) / W, (B - T) / H]); axr.patch.set_alpha(0)
        r0, r1 = _limits(YR[0], B, T)
        axr.yaxis.tick_right(); axr.yaxis.set_label_position("right")
        _style_axis(axr, "y", spec.get("y_axis_right"), r0, r1, False, fs, YR[0].log)
        axr.set_xlim(xlo, xhi); axr.set_xticks([])
        for k_, sp_ in axr.spines.items():
            sp_.set_visible(k_ == "right")
        axr.set_ylabel(spec["y_axis_right"].get("title", ""), fontsize=fs + 1, fontweight="bold")
    unit = 1 / abs(k)
    handles = []
    from matplotlib.patches import Patch
    for q in spec["series"]:
        name = q["name"]; d = res["series"].get(name)
        st = res["styles"].get(name)
        if not d or not st:
            continue
        col = np.clip(np.array(st["rgb"]) / 255, 0, 1)
        pale = st.get("outline", (col.max() - col.min()) / max(col.max(), 1e-6) < 0.35)   # measured bar outline
        kw = dict(color=col, edgecolor="black" if pale else "none", lw=0.8, hatch="///" if q.get("hatched") else None)
        w = st.get("width_px", 10) * unit; off = st.get("offset_px", 0) * unit
        for x, y, e in zip(d["x"], d["y"], d.get("yerr") or [0] * len(d["x"])):
            xc = x + off
            if cat_axis[int(x)] == "right" and axr is not None:
                targets = [axr]
            else:   # every left segment draws the part of the bar it can show
                targets = [a[0] for a in axes]
            for t in targets:
                t.bar(xc, y, width=w, zorder=3, **kw)
                if e:
                    t.errorbar(xc, y, yerr=e, fmt="none", ecolor="black", elinewidth=1.2, capsize=3, zorder=4)
        handles.append(Patch(facecolor=col, edgecolor="black" if pale else col, hatch="///" if q.get("hatched") else None,
                             label=name))
    fig.text(0.012, 1 - ((T + B) / 2) / H, spec["y_axis"].get("title", ""), rotation=90, va="center", ha="left",
             fontsize=fs + 1, fontweight="bold")
    if len(handles) > 1 or spec.get("legend"):
        bx = res.get("legend_boxes") or []
        if bx:
            x0b, y0b = min(b[0] for b in bx), min(b[1] for b in bx)
            horiz = len(bx) == 1 and (bx[0][2] - bx[0][0]) > 2.5 * (bx[0][3] - bx[0][1])
            fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x0b / W, 1 - y0b / H), fontsize=fs - 1,
                       ncol=len(handles) if horiz else 1, frameon=True, edgecolor="black", fancybox=False,
                       borderaxespad=0)
        else:
            room = W - R - 10                       # legend outside the plot, to the right
            # width of one legend row in px ~ (0.76 * characters + 3.5) * fontsize
            lfs = max(6, min(fs - 1, room / (0.76 * max(len(h.get_label()) for h in handles) + 3.5)))
            for _ in range(8):               # shrink until the legend fits on the canvas
                lg = fig.legend(handles=handles, loc="center left", bbox_to_anchor=((R + 8) / W, 0.5), fontsize=lfs,
                                frameon=False)
                fig.canvas.draw()
                if lg.get_window_extent().x1 <= W - 2:
                    break
                lg.remove(); lfs *= 0.85
    fig.savefig(str(out_png), dpi=dpi); plt.close(fig)
    Path(str(out_png)).with_suffix(".spec.json").write_text(json.dumps(spec, indent=1))
    return str(out_png)


# ============================================================================ pie
def recreate_pie(res, out_png, dpi=100):
    """Tilted pie with the measured perspective (apex and four half-radii),
    slices clockwise from 12 o'clock with the extracted shares, colours from the
    pixels, name + percentage labels."""
    from matplotlib.patches import Polygon
    from .pie import pie_point
    W, H = res["size"]
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    cal = res.get("pie_calibration")
    d = res["series"].get("share (%)")
    if not cal or not d:
        ax.text(W / 2, H / 2, "no pie found", ha="center"); fig.savefig(str(out_png), dpi=dpi); plt.close(fig)
        return str(out_png)
    cx, cy = cal["apex"]
    vals = np.array(d["y"], float); tot = vals.sum() or 1
    rgb = res["styles"]["share (%)"]["slice_rgb"]
    depth, shade = cal.get("depth", 0), cal.get("side_shade", 0.55)
    if depth:   # 3D: side wall below the front half of the face, slice by slice, darker
        a0 = np.pi / 2
        for name, v in zip(d["categories"], vals):
            a1 = a0 - 2 * np.pi * v / tot
            th = np.linspace(min(a0, a1), max(a0, a1), 240)
            th = th[np.sin(th) <= 0.001]
            if len(th) >= 2:
                x, y = pie_point(cal, th)
                col = np.clip(np.array(rgb.get(name, [128, 128, 128])) / 255 * shade, 0, 1)
                ax.add_patch(Polygon(np.r_[np.c_[x, y], np.c_[x[::-1], y[::-1] + depth]], closed=True,
                                     facecolor=col, edgecolor=col, lw=0.4))
            a0 = a1
    a0 = np.pi / 2
    fs = max(9, round(H / 28))
    for name, v in zip(d["categories"], vals):
        a1 = a0 - 2 * np.pi * v / tot
        th = np.linspace(a1, a0, max(3, int(abs(a0 - a1) * 120)))
        x, y = pie_point(cal, th)
        col = np.clip(np.array(rgb.get(name, [128, 128, 128])) / 255, 0, 1)
        ax.add_patch(Polygon(np.r_[[[cx, cy]], np.c_[x, y]], closed=True, facecolor=col, edgecolor=col, lw=0.5))
        mid = 0.5 * (a0 + a1)
        lx, ly = pie_point(cal, np.array([mid]))
        inside = v / tot > 0.25
        f = 0.55 if inside else 1.18
        tx, ty = cx + (lx[0] - cx) * f, cy + (ly[0] - cy) * f
        ax.text(tx, ty, f"{name}\n{v:.4g}%", ha="center", va="center", fontsize=fs * 0.95, fontweight="bold",
                color="white" if inside and col.mean() < 0.5 else "black")
        a0 = a1
    fig.savefig(str(out_png), dpi=dpi); plt.close(fig)
    Path(str(out_png)).with_suffix(".spec.json").write_text(json.dumps(res["spec"], indent=1))
    return str(out_png)
