"""Generic rebuild of the extended figures (``profiles.py``): one canvas the size of the
original, every panel / inset drawn at its own pixel frame with its own axes, ticks,
titles, legend and annotations, from a dataset in the panel format of ``engine.py``.

The rebuilt image becomes the test image; the dataset behind it is its exact ground
truth. Same conventions as ``render.py`` (DPI 100, sizes given in pixels).
"""
from __future__ import annotations

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # silence missing-font notes

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .digitize import Axis
from .render import SANS, SERIF, DPI, _installed, _px

MK = {"o": "o", "circle": "o", "s": "s", "square": "s", "^": "^", "triangle": "^", "v": "v", "d": "D", "D": "D",
      "diamond": "D", "p": "p", "pentagon": "p", "*": "*", "x": "x", "+": "+", ".": "."}


class Canvas2:
    def __init__(self, F):
        self.F = F
        self.W, self.H = F["size"]
        st = F.get("style", {})
        fam = _installed(SERIF if st.get("font", "sans") == "serif" else SANS)
        self.weight = st.get("weight", "normal")
        matplotlib.rcParams.update({"font.family": fam, "mathtext.fontset": "custom", "mathtext.rm": fam,
                                    "mathtext.it": fam + ":italic", "mathtext.bf": fam + ":bold",
                                    "axes.unicode_minus": False, "font.weight": self.weight,
                                    "axes.labelweight": self.weight, "hatch.linewidth": 0.8})
        self.fig = Figure(figsize=(self.W / DPI, self.H / DPI), dpi=DPI)
        self.fig.patch.set_facecolor("white")

    def axes_at(self, L, R, T, B, **kw):
        return self.fig.add_axes([L / self.W, 1 - B / self.H, (R - L) / self.W, (B - T) / self.H], **kw)

    def text(self, x, y, s, size_px, **kw):
        kw.setdefault("ha", "center"); kw.setdefault("va", "center")
        return self.fig.text(x / self.W, 1 - y / self.H, s, fontsize=_px(size_px), **kw)


def _lim(P, key, lo_px, hi_px):
    A = Axis(*P[key][:4], log=bool(P.get(key + "_log", False)))
    return float(A.to_value(lo_px)), float(A.to_value(hi_px))


def _style_axes(cv, ax, P, st):
    lw = P.get("frame_lw", st.get("frame_lw", 1.2))
    for k, sp in ax.spines.items():
        sp.set_linewidth(_px(lw)); sp.set_color(P.get("frame_color", "black"))
        if k in P.get("hide_spines", []):
            sp.set_visible(False)
    tp = P.get("tick_px", st.get("tick_px", 14))
    tdir = P.get("tick_dir", st.get("tick_dir", "in"))
    tl = P.get("tick_len", st.get("tick_len", 5))
    for which in ("x", "y"):
        ax.tick_params(axis=which, which="major", direction=tdir, length=_px(tl), width=_px(lw),
                       labelsize=_px(tp), pad=_px(P.get("tick_pad", 3)))
        ax.tick_params(axis=which, which="minor", direction=tdir, length=_px(tl * 0.6), width=_px(lw))
    xp = P.get("xtick_px", st.get("xtick_px"))
    if xp:
        ax.tick_params(axis="x", which="major", labelsize=_px(xp))
    if P.get("ticks_top_right", st.get("ticks_top_right", False)):
        ax.tick_params(top=True, right=not P.get("y_right"))


def _set_ticks(ax, which, vals, labels, weight, rotation=0):
    if vals is None:
        return
    if which == "x":
        ax.set_xticks(vals)
        ax.set_xticklabels(labels if labels is not None else [_fmt(v) for v in vals], rotation=rotation,
                           fontweight=weight, ha="right" if rotation else "center")
    else:
        ax.set_yticks(vals)
        ax.set_yticklabels(labels if labels is not None else [_fmt(v) for v in vals], fontweight=weight)


def _fmt(v):
    v = float(v)
    return f"{int(round(v))}" if abs(v - round(v)) < 1e-9 else f"{v:g}"


def _series_style(s):
    return {"color": s.get("color", "black"), "lw": _px(s.get("lw", 1.5)), "ls": s.get("ls", "-")}


def _draw_series(ax, P, name, s, d, bars_ctx):
    k = s["kind"]
    if not d or not d.get("x"):
        return
    x = np.asarray(d["x"], float); y = np.asarray(d["y"], float)
    idx = x.copy()
    if P.get("cat_pos"):                                   # category index -> axis units at the drawn position
        X = Axis(*P["x"][:4])
        x = np.interp(x, np.arange(len(P["cat_pos"])), X.to_value(np.asarray(P["cat_pos"], float)))
    col = s.get("color", "black")
    if k == "line":
        if s.get("fill_to") is not None:                   # filled spectra (EDS): area under the trace
            ax.fill_between(x, s["fill_to"], y, color=col, lw=0, zorder=2.5)
        if len(x) > 2:
            dx = np.diff(x); big = np.where(np.abs(dx) > max(5 * np.median(np.abs(dx)), 1e-12) )[0]
            if len(big):                                   # a gap in the trace is not drawn across
                x = np.insert(x, big + 1, np.nan); y = np.insert(y, big + 1, np.nan)
        ax.plot(x, y, color=col, lw=_px(s.get("lw", 1.5)), ls=s.get("ls", "-"), zorder=3, clip_on=True,
                solid_capstyle="butt")
    elif k == "marker":
        mk = MK.get(s.get("marker", "o"), "o")
        ms = _px(s.get("ms_px", 8))
        face = "none" if s.get("hollow") else s.get("face", col)
        if s.get("line_color") or s.get("connect"):
            ax.plot(x, y, color=s.get("line_color", col), lw=_px(s.get("lw", 1.2)), zorder=2)
        ax.plot(x, y, ls="none", marker=mk, ms=ms, mfc=face, mec=s.get("edge", col),
                mew=_px(s.get("mew", 1.2)), zorder=4, clip_on=False)
        if d.get("yerr") and any(e > 0 for e in d["yerr"]):
            ax.errorbar(x, y, yerr=np.asarray(d["yerr"], float), fmt="none", ecolor=s.get("err_color", col),
                        elinewidth=_px(1.2), capsize=_px(s.get("cap_px", 6) / 2), capthick=_px(1.2), zorder=3.5)
    elif k in ("bar", "stacked", "pattern_stack"):
        unit = bars_ctx["unit"]
        w = s.get("width_px", 20) * unit
        xc = x + s.get("offset_px", 0.0) * unit
        bottom = None
        if k in ("stacked", "pattern_stack"):
            bottom = np.asarray([b if b is not None else 0.0 for b in d.get("bottom", [0] * len(x))], float)
        hatch = s.get("hatch")
        if k == "pattern_stack":
            w = P.get("bar_width_px", 15) * unit
            xc = x
        if s.get("gradients"):                            # left->right colour gradient per bar (as drawn)
            for j, (xx, yy) in enumerate(zip(xc, y)):
                c0, c1 = s["gradients"][int(round(idx[j])) % len(s["gradients"])]
                g = np.linspace(0, 1, 64)[None, :, None]
                im = (np.array(c0)[None, None, :] * (1 - g) + np.array(c1)[None, None, :] * g) / 255.0
                b0 = 0 if bottom is None else bottom[j]
                ax.imshow(im, extent=(xx - w / 2, xx + w / 2, b0, b0 + yy), aspect="auto", zorder=3,
                          interpolation="bilinear")
            ax.set_aspect("auto")
            if d.get("yerr"):
                e = np.asarray(d["yerr"], float); ok = e > 0
                ax.errorbar(xc[ok], y[ok], yerr=e[ok], fmt="none", ecolor="black", elinewidth=_px(1.2),
                            capsize=_px(s.get("cap_px", 8) / 2), capthick=_px(1.2), zorder=4)
            return
        if s.get("hatch_per_bar"):
            for j, (xx, yy) in enumerate(zip(xc, y)):
                hb = s["hatch_per_bar"][int(round(idx[j])) % len(s["hatch_per_bar"])]
                ax.bar(xx, yy, width=w, color=s.get("fill", col), edgecolor=s.get("edge", "black"),
                       lw=_px(s.get("edge_lw", 1.2)), hatch=hb, zorder=3)
            if d.get("yerr"):
                e = np.asarray(d["yerr"], float); ok = e > 0
                ax.errorbar(xc[ok], y[ok], yerr=e[ok], fmt="none", ecolor="black", elinewidth=_px(1.2),
                            capsize=_px(s.get("cap_px", 8) / 2), capthick=_px(1.2), zorder=4)
            return
        fills = s.get("fills")
        ax.bar(xc, y, width=w, bottom=bottom, color=fills if fills else s.get("fill", col),
               edgecolor=(s.get("hatch_color") if hatch else s.get("edge") or "none"),
               lw=_px(s.get("edge_lw", 1.0)) if (s.get("edge") or hatch) else 0, hatch=hatch, zorder=3)
        if hatch and s.get("edge"):
            ax.bar(xc, y, width=w, bottom=bottom, color="none", edgecolor=s["edge"], lw=_px(s.get("edge_lw", 1.0)), zorder=3.1)
        if d.get("yerr"):
            top = y + (bottom if bottom is not None else 0)
            e = np.asarray(d["yerr"], float)
            ok = e > 0
            if ok.any():
                ax.errorbar(xc[ok], top[ok], yerr=e[ok], fmt="none", ecolor=s.get("err_color", "black"),
                            elinewidth=_px(1.2), capsize=_px(s.get("cap_px", 8) / 2), capthick=_px(1.2), zorder=4)
    elif k == "stick":
        base = s.get("base", 0.0)
        ax.vlines(x, base, y, color=col, lw=_px(s.get("lw", 1.2)), zorder=3)


def _legend(cv, ax, P, st):
    L = P.get("legend")
    if not L:
        return
    hs = []
    for name in L.get("order", list(P["series"])):
        s = P["series"][name]
        lab = s.get("label", name)
        k = s["kind"]
        if k in ("bar", "stacked"):
            hs.append(Rectangle((0, 0), 1, 1, facecolor=s.get("fill", s.get("color")), edgecolor=s.get("hatch_color") or s.get("edge") or "none",
                                hatch=s.get("hatch"), lw=_px(1.0), label=lab))
        elif k == "marker":
            hs.append(Line2D([], [], color=s.get("line_color", s.get("color")), lw=_px(s.get("lw", 1.2)) if (s.get("line_color") or s.get("connect")) else 0,
                             marker=MK.get(s.get("marker", "o"), "o"), ms=_px(s.get("ms_px", 8)),
                             mfc="none" if s.get("hollow") else s.get("color"), mec=s.get("color"), label=lab))
        else:
            hs.append(Line2D([], [], color=s.get("color"), lw=_px(s.get("lw", 1.5)), ls=s.get("ls", "-"), label=lab))
    W, H = cv.W, cv.H
    x0, y0 = L["xy"]                                   # upper-left corner of the legend box (px)
    lg = cv.fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(x0 / W, 1 - y0 / H), ncol=L.get("ncol", 1),
                       fontsize=_px(L.get("size_px", st.get("legend_px", 13))), frameon=L.get("frame", False),
                       handlelength=L.get("handlelength", 2.0), columnspacing=1.0, borderaxespad=0, borderpad=0.3,
                       labelspacing=L.get("labelspacing", 0.3))
    if L.get("frame"):
        lg.get_frame().set_edgecolor("black"); lg.get_frame().set_linewidth(_px(1.0))
    for t in lg.get_texts():
        t.set_fontweight(L.get("weight", st.get("weight", "normal")))


def render_panel_px(cv, P, data, st):
    """axes broken into segments (x and / or y): drawn in PIXEL space - every value goes through the
    panel's piecewise pixel map, ticks are placed at their pixel positions and labelled with their
    values, break marks are drawn where segments meet. The picture matches the original exactly."""
    from .engine import axis_of
    L, R, T, B = P["frame"]
    X, Y = axis_of(P, "x"), axis_of(P, "y")
    ax = cv.axes_at(L, R, T, B)
    ax.set_xlim(L, R); ax.set_ylim(B, T)
    _style_axes(cv, ax, P, st)
    xt, xl = [], []
    for g in P.get("x_segments") or [{"px": (L, R), "map": P["x"], "ticks": P.get("x_ticks", [])}]:
        for v in g.get("ticks", []):
            xt.append(float(Axis(*g["map"]).to_pixel(v))); xl.append(_fmt(v))
    yt, yl = [], []
    for g in P.get("y_segments") or [{"rows": (T, B), "map": P["y"], "ticks": P.get("y_ticks", [])}]:
        for v in g.get("ticks", []):
            yt.append(float(Axis(*g["map"]).to_pixel(v))); yl.append(_fmt(v))
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontweight=cv.weight)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontweight=cv.weight)
    tpx = _px(P.get("title_px", st.get("title_px", 16)))
    if P.get("x_title"):
        ax.set_xlabel(P["x_title"], fontsize=tpx, fontweight=cv.weight)
    if P.get("y_title"):
        ax.set_ylabel(P["y_title"], fontsize=tpx, fontweight=cv.weight)
    pd = (data or {}).get("series", {})
    for name, s in P["series"].items():
        d = pd.get(name)
        if not d or not d.get("x"):
            continue
        px = np.asarray(X.to_pixel(np.asarray(d["x"], float)), float)
        py = np.asarray(Y.to_pixel(np.asarray(d["y"], float)), float)
        col = s.get("color", "black")
        if s["kind"] == "line" or s.get("connect"):
            ax.plot(px, py, color=s.get("line_color", col), lw=_px(s.get("lw", 1.4)), zorder=2)
        if s["kind"] == "marker":
            ax.plot(px, py, ls="none", marker=MK.get(s.get("marker", "o"), "o"), ms=_px(s.get("ms_px", 8)),
                    mfc="none" if s.get("hollow") else col, mec=col, mew=_px(s.get("mew", 1.2)), zorder=4, clip_on=False)
    # white gaps + break marks where the segments meet
    for g in (P.get("x_segments") or [])[:-1]:
        xb = max(g["px"]) + P.get("x_break_gap", 4) / 2
        for yy in (B, T):
            cv.fig.add_artist(Rectangle(((xb - 4) / cv.W, 1 - (yy + 3) / cv.H), 8 / cv.W, 6 / cv.H, facecolor="white", edgecolor="none", zorder=11))
            for dx in (-3, 3):
                cv.fig.add_artist(Line2D([(xb + dx - 3) / cv.W, (xb + dx + 3) / cv.W], [1 - (yy + 6) / cv.H, 1 - (yy - 6) / cv.H],
                                         color="black", lw=_px(1.2), zorder=12))
    for g in (P.get("y_segments") or [])[:-1]:
        yb = max(g["rows"])
        for xx in (L, R):
            cv.fig.add_artist(Rectangle(((xx - 3) / cv.W, 1 - (yb + 4) / cv.H), 6 / cv.W, 8 / cv.H, facecolor="white", edgecolor="none", zorder=11))
            for dy in (-3, 3):
                cv.fig.add_artist(Line2D([(xx - 6) / cv.W, (xx + 6) / cv.W], [1 - (yb + dy + 3) / cv.H, 1 - (yb + dy - 3) / cv.H],
                                         color="black", lw=_px(1.2), zorder=12))
    _legend(cv, ax, P, st)
    if P.get("label"):
        cv.text(P["label"]["xy"][0], P["label"]["xy"][1], P["label"]["text"], P["label"].get("size_px", st.get("title_px", 16)),
                ha="left", fontweight=P["label"].get("weight", cv.weight))
    return ax


def render_panel(cv, P, data, st):
    if P.get("pixel_space"):
        return render_panel_px(cv, P, data, st)
    L, R, T, B = P["frame"]
    pd = data.get("series", {}) if data else {}
    broken = "split_px" in P or bool(P.get("y_segments"))
    axes = []
    if P.get("y_segments"):
        segs = P["y_segments"]
        g = P.get("break_gap_px", 0)
        for j, sg in enumerate(segs):
            t, b = sg["rows"]
            a_ = cv.axes_at(L, R, t + (g / 2 if j else 0), b - (g / 2 if j < len(segs) - 1 else 0))
            A = Axis(*sg["map"])
            a_.set_ylim(float(A.to_value(b - (g / 2 if j < len(segs) - 1 else 0))), float(A.to_value(t + (g / 2 if j else 0))))
            if j:
                a_.spines["top"].set_visible(False)
            if j < len(segs) - 1:
                a_.spines["bottom"].set_visible(False)
                a_.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            axes.append(a_)
        ax = axes[-1]
    elif "split_px" in P:
        g = P.get("break_gap_px", 8)
        sp = P["split_px"]
        ax_up = cv.axes_at(L, R, T, sp - g / 2)
        ax_lo = cv.axes_at(L, R, sp + g / 2, B)
        ax_up.set_ylim(*_lim(P, "y_upper", sp - g / 2, T)); ax_lo.set_ylim(*_lim(P, "y_lower", B, sp + g / 2))
        ax_up.spines["bottom"].set_visible(False); ax_lo.spines["top"].set_visible(False)
        ax_up.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        axes = [ax_up, ax_lo]
        ax = ax_lo
    else:
        ax = cv.axes_at(L, R, T, B)
        axes = [ax]
        if P.get("y"):
            ax.set_ylim(*_lim(P, "y", B, T))
        if P.get("y_log"):
            ax.set_yscale("log")
    for a in axes:
        a.set_facecolor(P.get("face", "none"))
        if P.get("grid"):
            a.grid(True, color="#D0D0D0", lw=_px(0.8), zorder=0)
            a.set_axisbelow(True)
        if P.get("x"):
            a.set_xlim(*_lim(P, "x", L, R))
            if P.get("x_log"):
                a.set_xscale("log")
        _style_axes(cv, a, P, st)
    if P.get("y_segments"):
        for a_, sg in zip(axes, P["y_segments"]):
            _set_ticks(a_, "y", sg.get("ticks"), sg.get("ticklabels"), cv.weight)
    elif broken:
        _set_ticks(ax_up, "y", P.get("y_upper_ticks"), P.get("y_upper_ticklabels"), cv.weight)
        _set_ticks(ax_lo, "y", P.get("y_lower_ticks"), P.get("y_lower_ticklabels"), cv.weight)
    else:
        _set_ticks(ax, "y", P.get("y_ticks"), P.get("y_ticklabels"), cv.weight)
    xt_ = P.get("x_ticks")
    if xt_ is not None and P.get("cat_pos") and P.get("categories"):     # category indices -> drawn positions
        Xa = Axis(*P["x"][:4])
        xt_ = [float(Xa.to_value(np.interp(v, np.arange(len(P["cat_pos"])), P["cat_pos"]))) for v in xt_]
    _set_ticks(ax, "x", xt_, P.get("x_ticklabels"), cv.weight, P.get("x_rotation", 0))
    if P.get("no_y_ticks"):
        for a in axes:
            a.set_yticks([])
    if P.get("no_x_labels"):
        ax.set_xticklabels([])
    axr = None
    if P.get("y_right"):
        axr = ax.twinx() if not broken else axes[-1].twinx()
        if broken:
            axr.set_position(cv.axes_at(L, R, T, B).get_position()); cv.fig.axes[-1].remove()
        axr.set_ylim(*_lim(P, "y_right", B, T))
        _style_axes(cv, axr, P, st)
        _set_ticks(axr, "y", P.get("y_right_ticks"), P.get("y_right_ticklabels"), cv.weight)
        if P.get("y_right_title"):
            axr.set_ylabel(P["y_right_title"], fontsize=_px(P.get("title_px", st.get("title_px", 16))),
                           color=P.get("y_right_color", "black"), fontweight=cv.weight)
        if P.get("y_right_color"):
            axr.tick_params(axis="y", colors=P["y_right_color"]); axr.spines["right"].set_color(P["y_right_color"])
    tpx = _px(P.get("title_px", st.get("title_px", 16)))
    if P.get("x_title"):
        ax.set_xlabel(P["x_title"], fontsize=tpx, fontweight=cv.weight)
    if P.get("y_title"):
        (axes[0] if not broken else cv.fig).__class__  # noqa
        if broken:
            cv.text(L - P.get("y_title_dx", 55), (T + B) / 2, P["y_title"], P.get("title_px", st.get("title_px", 16)),
                    rotation=90, fontweight=cv.weight)
        else:
            ax.set_ylabel(P["y_title"], fontsize=tpx, fontweight=cv.weight, color=P.get("y_color", "black"))
    if P.get("y_color"):
        for a in axes:
            a.tick_params(axis="y", colors=P["y_color"]); a.spines["left"].set_color(P["y_color"])
    # bars: value units per pixel along x
    bars_ctx = {}
    if P.get("x"):
        X = Axis(*P["x"][:4])
        bars_ctx["unit"] = abs(X.v1 - X.v0) / abs(X.p1 - X.p0)
    for name, s in P["series"].items():
        d = pd.get(name)
        if P.get("category_axis") and d and d.get("x") and axr is not None:
            for side, tg in (("left", [ax]), ("right", [axr])):
                keep = [j for j, xv in enumerate(d["x"]) if P["category_axis"].get(int(round(xv)), "left") == side]
                dd = {k: ([v[j] for j in keep] if isinstance(v, list) else v) for k, v in d.items()}
                for a in tg:
                    _draw_series(a, P, name, s, dd, bars_ctx)
            continue
        if s.get("axis") == "right" and axr is not None:
            tgt = [axr]
        elif broken:
            tgt = axes
        else:
            tgt = [ax]
        for a in tgt:
            _draw_series(a, P, name, s, d, bars_ctx)
    if P.get("y_segments"):
        for a in axes:
            a.set_xlim(*_lim(P, "x", L, R))
            if P.get("grid"):
                a.grid(True, axis="y", color="#D0D0D0", lw=_px(1.0)); a.set_axisbelow(True)
    elif broken:
        for a in axes:
            a.set_xlim(*_lim(P, "x", L, R))
        sp = P["split_px"]
        for x in (L, R):
            for dy in (-P.get("break_gap_px", 8) / 2, P.get("break_gap_px", 8) / 2):
                cv.fig.add_artist(Line2D([(x - 7) / cv.W, (x + 7) / cv.W], [1 - (sp + dy + 4) / cv.H, 1 - (sp + dy - 4) / cv.H],
                                         color="black", lw=_px(1.4), zorder=12))
    for a in P.get("vlines", []):                      # dashed separators drawn in the original (data x)
        for a2 in (axes if broken else []):
            if a2 is not ax:
                a2.axvline(a["x"], color=a.get("color", "black"), ls=a.get("ls", "--"), lw=_px(a.get("lw", 1.2)), zorder=1)
        ax.axvline(a["x"], color=a.get("color", "black"), ls=a.get("ls", "--"), lw=_px(a.get("lw", 1.2)), zorder=1)
    for a in P.get("hlines", []):
        ax.axhline(a["y"], color=a.get("color", "black"), ls=a.get("ls", "--"), lw=_px(a.get("lw", 1.2)), zorder=1)
    for t, xy, sz, col in P.get("texts", []):              # text written inside the panel (peak names, printed values)
        cv.text(xy[0], xy[1], t, sz, ha="center", color=col, fontweight="bold")
    _legend(cv, ax, P, st)
    if P.get("label"):
        lx, ly, ls_ = P["label"]["xy"][0], P["label"]["xy"][1], P["label"].get("size_px", st.get("title_px", 16))
        cv.text(lx, ly, P["label"]["text"], ls_, ha="left", fontweight=P["label"].get("weight", cv.weight))
    return ax


def render_figure(F, data, out_path):
    """data: {"panels": {panel_id: {"series": {...}}}} (engine.extract_figure output / ground truth)"""
    cv = Canvas2(F)
    st = F.get("style", {})
    for P in F["panels"]:
        if P.get("engine") == "autodig":
            continue
        render_panel(cv, P, (data.get("panels") or {}).get(P["id"], {}), st)
    for P in F["panels"]:
        for t in P.get("tables", []):                      # printed fit-parameter tables
            x0, x1, y0, y1 = t["box"]
            cv.fig.add_artist(Rectangle((x0 / cv.W, 1 - y1 / cv.H), (x1 - x0) / cv.W, (y1 - y0) / cv.H, facecolor="white",
                                        edgecolor="#888888", lw=_px(1.0), zorder=8))
            lines = t["text"].split("\n")
            for i, ln_ in enumerate(lines):
                cv.text(x0 + 4, y0 + (i + 0.6) * (y1 - y0) / len(lines), ln_, t.get("size_px", 13), ha="left",
                        color=t.get("color", "black"), family="monospace").set_zorder(9)
    fl = F.get("fig_legend")
    if fl:
        hs = []
        for lab, kind, col in fl["entries"]:
            if kind == "marker":
                hs.append(Line2D([], [], ls="none", marker="s", ms=_px(8), color=col, label=lab))
            else:
                hs.append(Line2D([], [], color=col, lw=_px(1.6), ls="--" if kind == "dashed" else "-", label=lab))
        cv.fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(fl["xy"][0] / cv.W, 1 - fl["xy"][1] / cv.H),
                      ncol=fl.get("ncol", len(hs)), fontsize=_px(fl.get("size_px", 16)), frameon=False, borderaxespad=0,
                      columnspacing=1.0, handlelength=2.0, prop={"weight": F.get("style", {}).get("weight", "normal"),
                                                                 "size": _px(fl.get("size_px", 16))})
    for a in F.get("annotations", []):
        kw = {"family": a["font"]} if a.get("font") else {}
        cv.text(a["xy"][0], a["xy"][1], a["text"], a.get("size_px", 12), ha=a.get("ha", "left"), multialignment="center",
                color=a.get("color", "black"), fontweight=a.get("weight", "normal"), rotation=a.get("rotation", 0), **kw)
    cv.fig.savefig(out_path, dpi=DPI, facecolor="white")
    if any(P.get("engine") == "autodig" for P in F["panels"]):
        _paste_autodig(F, data, out_path)
    return out_path


def _paste_autodig(F, data, out_path):
    """3-D / contour panels are redrawn by autodig's renderer (same view, same colour map) and
    pasted at their place on the canvas"""
    import json, sys, tempfile
    from pathlib import Path
    from PIL import Image
    root = Path(__file__).resolve().parents[2] / "03_auto_vlm_digitization"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from autodig.render_new import recreate
    base = Image.open(out_path).convert("RGB")
    for P in F["panels"]:
        if P.get("engine") != "autodig":
            continue
        res = (data.get("panels") or {}).get(P["id"], {}).get("autodig_full")
        if res is None:
            continue
        tmp = tempfile.mktemp(suffix=".png")
        recreate(res, tmp)
        im = Image.open(tmp).convert("RGB").resize(base.size)
        x0, y0, x1, y1 = P.get("region", (0, 0) + tuple(base.size))
        base.paste(im.crop((x0, y0, x1, y1)), (x0, y0))
    base.save(out_path)
