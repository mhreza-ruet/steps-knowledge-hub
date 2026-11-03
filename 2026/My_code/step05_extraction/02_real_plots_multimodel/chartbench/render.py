"""Rebuild each of the six figures from a dataset (series x/y values).

Every rebuilt figure is drawn on a canvas with the *same pixel size* as the
original, with the plot frame at the *same pixel position* (taken from
``figures.FIGURES[...]['calib']``), the same axis limits, tick spacing, tick
direction, fonts (Times-like serif or Arial-like sans), colours, markers,
line widths, legends and annotation text. So the only thing that differs from
the original is the data - which is exactly what the extraction benchmark
needs.

``render(fig_id, dataset, out_path)`` works with any dataset that follows the
``{"series": {name: {"x": [...], "y": [...], "yerr": [...]}}}`` shape, so the
same function draws (a) the reference rebuild from the classical digitizer's
data and (b) a "what would this model's data look like" rebuild for any VLM.
"""
from __future__ import annotations

import matplotlib
import numpy as np

from matplotlib.figure import Figure  # no pyplot: never touches the notebook's backend
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

from .digitize import Axis
from .figures import FIGURES

_CLIP = [False]  # True when drawing a model's extraction: values outside the axes are not drawn

SERIF = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
SANS = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
DPI = 100
PX = 72.0 / DPI  # points per pixel


def _installed(families):
    """First family in the list that is actually installed (Times New Roman /
    Arial on macOS; their metric-compatible Liberation clones on Linux)."""
    from matplotlib import font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    for f in families:
        if f in names:
            return f
    return families[-1]


def _px(v):
    """pixels -> points (font sizes, line widths, marker sizes)."""
    return v * PX


class Canvas:
    def __init__(self, fig_id, family, weight="normal"):
        F = FIGURES[fig_id]
        self.F, self.c = F, F["calib"]
        self.W, self.H = F["size"]
        fam = _installed(family)
        matplotlib.rcParams.update({
            "font.family": fam, "mathtext.fontset": "custom",
            "mathtext.rm": fam, "mathtext.it": fam + ":italic",
            "mathtext.bf": fam + ":bold",
            "axes.unicode_minus": False,
            "font.weight": weight, "axes.labelweight": weight,
        })
        self.fig = Figure(figsize=(self.W / DPI, self.H / DPI), dpi=DPI)
        self.fig.patch.set_facecolor("white")

    def axes_at(self, left, right, top, bottom, **kw):
        W, H = self.W, self.H
        return self.fig.add_axes([left / W, 1 - bottom / H, (right - left) / W, (bottom - top) / H], **kw)

    def text(self, x, y, s, size_px, **kw):
        kw.setdefault("ha", "center"); kw.setdefault("va", "center")
        return self.fig.text(x / self.W, 1 - y / self.H, s, fontsize=_px(size_px), **kw)

    def save(self, path):
        self.fig.savefig(path, dpi=DPI, facecolor="white")


def _limits(c, key, lo_px, hi_px):
    A = Axis(*c[key])
    return float(A.to_value(lo_px)), float(A.to_value(hi_px))


def _style_frame(ax, lw_px, color="black"):
    for sp in ax.spines.values():
        sp.set_linewidth(_px(lw_px)); sp.set_color(color)


def _ticks(ax, which, direction, major_len, minor_len, width_px, label_px, weight="normal", pad_px=3):
    ax.tick_params(axis=which, which="major", direction=direction, length=_px(major_len), width=_px(width_px),
                   labelsize=_px(label_px), pad=_px(pad_px))
    ax.tick_params(axis=which, which="minor", direction=direction, length=_px(minor_len), width=_px(width_px))
    labels = ax.get_xticklabels() if which == "x" else ax.get_yticklabels()
    for t in labels:
        t.set_fontweight(weight)


def _series(data, name):
    s = (data or {}).get("series", {}).get(name)
    if not s or not s.get("x"):
        return None
    x = np.asarray(s["x"], float); y = np.asarray(s["y"], float)
    ok = ~(np.isnan(x) | np.isnan(y))
    e = np.asarray(s["yerr"], float)[ok] if s.get("yerr") is not None and len(s["yerr"]) == len(x) else None
    return x[ok], y[ok], e


# =============================================================================
# 1. FTIR
# =============================================================================
def _ftir(data, out):
    cv = Canvas("ftir", SERIF)
    c = cv.c; L, R, T, B = c["frame"]
    ax = cv.axes_at(L, R, T, B)
    ax.set_xlim(*_limits(c, "x", L, R)); ax.set_ylim(*_limits(c, "y", B, T))
    _style_frame(ax, 1.2)
    ax.set_xticks([4000, 3500, 3000, 2500, 2000, 1500, 1000, 500])
    ax.set_yticks([])
    _ticks(ax, "x", "out", 6, 3, 1.1, 19.5, pad_px=4)
    ax.tick_params(axis="x", top=False)
    # dashed reference lines (positions read off the original)
    for xv in (3427, 1629, 1347, 1090, 790, 617):
        ax.axvline(xv, color="#333333", lw=_px(1.0), ls=(0, (4, 3)), zorder=1)
    for name in ("D-HZO-P", "D-HZO", "Diatomite"):
        s = _series(data, name)
        if s is None:
            continue
        st = FIGURES["ftir"]["series"][name]
        ax.plot(s[0], s[1], color=st["color"], lw=_px(1.3), zorder=3)
    ax.set_xlabel("")
    cv.text(258, 420, r"Wavenumber (cm$^{-1}$)", 20.5)
    cv.text(14, 190, "Transmittance (%)", 20.5, rotation=90)
    cv.text(72, 44, "(c)", 25, fontweight="bold")
    for (x, y, s) in [(224, 87, "D-HZO-P"), (223, 184, "D-HZO"), (224, 284, "Diatomite"),
                      (366, 34, "1139"), (428, 34, "793"), (88, 230, "O-H"), (313, 232, r"H$_2$O"),
                      (394, 232, r"PO$_4^{3-}$"), (455, 234, "P-O"), (363, 122, "Zr-OH"), (430, 122, "Si-O"),
                      (394, 280, "Si-O"), (87, 346, "3427"), (312, 349, "1629"), (394, 349, "1090"),
                      (459, 342, "617")]:
        cv.text(x, y, s, 19)
    cv.save(out)


# =============================================================================
# 2. Pore-size distribution with a broken y-axis
# =============================================================================
def _pore(data, out):
    cv = Canvas("pore", SERIF)
    c = cv.c; L, R, T, B = c["frame"]; sp = c["split_px"]
    gap = 6  # pixels of white between the two left-axis segments
    axU = cv.axes_at(L, R, T, sp)
    axL = cv.axes_at(L, R, sp, B)
    xl = _limits(c, "x", L, R)
    for ax in (axU, axL):
        ax.set_xlim(*xl); _style_frame(ax, 1.2); ax.patch.set_alpha(0)
        ax.tick_params(right=False, top=False)
    axU.set_ylim(*_limits(c, "y_upper", sp, T)); axL.set_ylim(*_limits(c, "y_lower", B, sp))
    axU.spines["bottom"].set_visible(False); axL.spines["top"].set_visible(False)
    axU.spines["left"].set_bounds(_limits(c, "y_upper", sp - gap / 2, T)[0], axU.get_ylim()[1])
    axL.spines["left"].set_bounds(axL.get_ylim()[0], _limits(c, "y_lower", sp + gap / 2, T)[0])
    axU.set_yticks([0.03, 0.06, 0.09, 0.12, 0.15]); axU.set_xticks([])
    axL.set_yticks([0, 0.0003, 0.0006, 0.0009, 0.0012]); axL.set_xticks([0, 4, 8, 12, 16, 20])
    axU.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    axL.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))
    for ax in (axU, axL):
        _ticks(ax, "y", "out", 7, 3, 1.1, 19, pad_px=3)
    _ticks(axL, "x", "out", 7, 3, 1.1, 19, pad_px=4)
    # axis-break marks
    for dy in (-3, 3):
        cv.fig.add_artist(Line2D([(L - 9) / cv.W, (L + 9) / cv.W], [1 - (sp + dy + 4) / cv.H, 1 - (sp + dy - 4) / cv.H],
                                 color="black", lw=_px(1.2)))
    # dashed markers of the two highlighted pores
    axU.axvline(3.49, color="#222222", lw=_px(1.2), ls=(0, (4, 3)))
    axL.axvline(12.05, color="#222222", lw=_px(1.2), ls=(0, (4, 3)), ymax=0.97)
    for name, ax in (("Diatomite", axL), ("D-HZO", axU)):
        s = _series(data, name)
        if s is None:
            continue
        st = FIGURES["pore"]["series"][name]
        ax.plot(s[0], s[1], color="black", lw=_px(1.2), zorder=2)
        ax.plot(s[0], s[1], ls="none", marker=st["marker"], ms=_px(st["ms"] * 1.25), color=st["color"],
                mec=st["color"], zorder=3, clip_on=_CLIP[0])
    # legend (drawn by hand at the original pixel position)
    for yy, name in ((93, "Diatomite"), (115, "D-HZO")):
        st = FIGURES["pore"]["series"][name]
        cv.fig.add_artist(Line2D([332 / cv.W, 372 / cv.W], [1 - yy / cv.H] * 2, color="black", lw=_px(1.2)))
        cv.fig.add_artist(Line2D([352 / cv.W], [1 - yy / cv.H], ls="none", marker=st["marker"],
                                 ms=_px(st["ms"] * 1.25), color=st["color"]))
        cv.text(383, yy, "Diatoimte" if name == "Diatomite" else name, 18.5, ha="left")
    cv.text(320, 415, "Pore diameter (nm)", 20.5)
    cv.text(18, 192, r"dV/dD (cm$^3$/g)", 20.5, rotation=90)
    cv.text(133, 47, "(e)", 22, fontweight="bold")
    cv.text(152, 135, "3.49", 18.5)
    cv.text(392, 311, "12.05", 18.5)
    cv.save(out)


# =============================================================================
# 3. Methane yield with error bars
# =============================================================================
def _methane(data, out):
    cv = Canvas("methane", SERIF, "bold")
    c = cv.c; L, R, T, B = c["frame"]
    ax = cv.axes_at(L, R, T, B)
    ax.set_xlim(*_limits(c, "x", L, R)); ax.set_ylim(*_limits(c, "y", B, T))
    _style_frame(ax, 2.2)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30]); ax.set_yticks([30, 60, 90, 120, 150, 180])
    ax.set_yticks(np.arange(15, 181, 30), minor=True)
    _ticks(ax, "x", "in", 7, 4, 1.6, 26, weight="bold", pad_px=7)
    _ticks(ax, "y", "in", 7, 4, 1.6, 26, weight="bold", pad_px=6)
    ax.tick_params(top=False, right=False, which="both")
    handles = []
    for name in ("CT", "RM", "BM"):
        s = _series(data, name)
        st = FIGURES["methane"]["series"][name]
        if s is not None:
            x, y, e = s
            if e is not None:
                ax.errorbar(x, y, yerr=e, fmt="none", ecolor=st["err_color"], elinewidth=_px(1.1),
                            capsize=_px(4.5), capthick=_px(1.1), zorder=2)
            ax.plot(x, y, color=st["color"], lw=_px(st["lw"]), zorder=3)
            ax.plot(x, y, ls="none", marker=st["marker"], ms=_px(st["ms"]), color=st["color"],
                    mec=st["color"], zorder=4, clip_on=_CLIP[0])
        handles.append(Line2D([], [], color=st["color"], lw=_px(st["lw"]), marker=st["marker"], ms=_px(st["ms"])))
    for (lx, tx, name) in ((240, 268, "CT"), (323, 352, "RM"), (416, 445, "BM")):
        st = FIGURES["methane"]["series"][name]
        yy = 1 - 39 / cv.H
        cv.fig.add_artist(Line2D([(lx - 18) / cv.W, (lx + 18) / cv.W], [yy, yy], color=st["color"], lw=_px(st["lw"])))
        cv.fig.add_artist(Line2D([lx / cv.W], [yy], ls="none", marker=st["marker"], ms=_px(st["ms"]), color=st["color"]))
        cv.text(tx, 39, name, 21, ha="left", fontweight="bold")
    cv.text(347, 454, "Time (days)", 29, fontweight="bold")
    cv.text(19, 206, "Methane yield (mL/g VSS)", 28.5, fontweight="bold", rotation=90)
    cv.text(128, 41, "(a)", 25, fontweight="bold")
    cv.save(out)


# =============================================================================
# 4. Dosage, dual y-axes
# =============================================================================
def _shadow_legend(cv, box, entries, size_px):
    """Origin-style legend: grey frame + solid black drop shadow."""
    x0, x1, y0, y1 = box
    W, H = cv.W, cv.H
    cv.fig.add_artist(Rectangle((x0 / W + 5 / W, 1 - (y1 + 5) / H), (x1 - x0) / W, (y1 - y0) / H, color="black", zorder=8))
    cv.fig.add_artist(Rectangle((x0 / W, 1 - y1 / H), (x1 - x0) / W, (y1 - y0) / H, facecolor="white",
                                edgecolor="#555555", lw=_px(1.2), zorder=9))
    for i, (label, st) in enumerate(entries):
        yy = y0 + (i + 0.5) * (y1 - y0) / len(entries)
        cv.fig.add_artist(Line2D([(x0 + 4) / W, (x0 + 56) / W], [1 - yy / H] * 2, color=st["color"], lw=_px(2.0), zorder=10))
        cv.fig.add_artist(Line2D([(x0 + 30) / W], [1 - yy / H], ls="none", marker=st["marker"], ms=_px(st["ms"]),
                                 color=st["color"], mfc="white" if st.get("hollow") else st["color"], zorder=11))
        t = cv.text(x0 + 60, yy + 1, label, size_px, ha="left", fontweight="normal")
        t.set_zorder(12)


def _dosage(data, out):
    cv = Canvas("dosage", SANS, "bold")
    c = cv.c; L, R, T, B = c["frame"]
    ax = cv.axes_at(L, R, T, B)
    ax2 = ax.twinx()
    ax.set_xlim(*_limits(c, "x", L, R)); ax.set_ylim(*_limits(c, "y", B, T)); ax2.set_ylim(*_limits(c, "y_right", B, T))
    for a in (ax, ax2):
        _style_frame(a, 2.2)
    ax.set_xticks([0, 1, 2, 3, 4, 5]); ax.set_xticks(np.arange(0.5, 5.01, 1), minor=True)
    ax.set_yticks(np.arange(0, 181, 20)); ax2.set_yticks(np.arange(0, 91, 10)); ax2.set_yticks(np.arange(5, 91, 10), minor=True)
    _ticks(ax, "x", "out", 7, 4, 1.8, 27, weight="bold", pad_px=5)
    _ticks(ax, "y", "out", 7, 4, 1.8, 27, weight="bold", pad_px=4)
    _ticks(ax2, "y", "out", 7, 4, 1.8, 27, weight="bold", pad_px=4)
    ax.tick_params(top=False, which="both")
    # grid: dashed at every 20 (left) = 10 % (right), dotted in between
    for yv in np.arange(20, 180, 20):
        ax.axhline(yv, color="#9E9E9E", lw=_px(1.6), ls=(0, (6, 4)), zorder=0)
    for yv in np.arange(10, 180, 20):
        ax.axhline(yv, color="#C8C8C8", lw=_px(1.2), ls=(0, (1.2, 4)), zorder=0)
    for name, a in (("N removal efficiency", ax2), ("P removal efficiency", ax2),
                    ("N adsorption capacity", ax), ("P adsorption capacity", ax)):
        s = _series(data, name)
        if s is None:
            continue
        st = FIGURES["dosage"]["series"][name]
        a.plot(s[0], s[1], color=st["color"], lw=_px(st["lw"]), marker=st["marker"], ms=_px(st["ms"]),
               mfc="white" if st.get("hollow") else st["color"], mec=st["color"], mew=_px(1.3), clip_on=_CLIP[0], zorder=3)
    S = FIGURES["dosage"]["series"]
    _shadow_legend(cv, (303, 563, 104, 162), [("N removal efficiency", S["N removal efficiency"]),
                                               ("P removal efficiency", S["P removal efficiency"])], 24)
    _shadow_legend(cv, (272, 547, 308, 366), [("N adsorption capacity", S["N adsorption capacity"]),
                                               ("P adsorption capacity", S["P adsorption capacity"])], 24)
    cv.text(348, 501, "Dosage (g/L)", 31, fontweight="bold")
    cv.text(20, 232, "Adsorption capacity (mg/g)", 28.5, fontweight="bold", rotation=90)
    cv.text(669, 232, "Removal efficiency (%)", 28.5, fontweight="bold", rotation=90)
    cv.save(out)


# =============================================================================
# 5. N2 isotherm
# =============================================================================
def _n2(data, out):
    cv = Canvas("n2", SANS)
    c = cv.c; L, R, T, B = c["frame"]
    ax = cv.axes_at(L, R, T, B)
    ax.set_xlim(*_limits(c, "x", L, R)); ax.set_ylim(*_limits(c, "y", B, T))
    _style_frame(ax, 1.6, "#3C3C3C")
    ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_yticks([0, 10, 20, 30, 40])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    _ticks(ax, "x", "out", 5, 3, 1.3, 20.5, pad_px=3)
    _ticks(ax, "y", "out", 5, 3, 1.3, 20.5, pad_px=3)
    ax.tick_params(top=False, right=False)
    st = FIGURES["n2"]["series"]["adsorption"]
    a = _series(data, "adsorption"); d = _series(data, "desorption")
    if a is not None:
        ax.plot(a[0], a[1], color=st["color"], lw=_px(st["lw"]), marker="s", ms=_px(st["ms"] * 1.2), clip_on=_CLIP[0])
    if d is not None:
        xd, yd = d[0], d[1]
        if a is not None and len(a[0]):  # the loop closes onto the adsorption branch
            xa, ya = a[0], a[1]
            k = np.argmin(np.abs(xa - xd.min()))
            xd = np.r_[xa[k], xd, xa[-1]]; yd = np.r_[ya[k], yd, ya[-1]]
            o = np.argsort(xd); xd, yd = xd[o], yd[o]
        ax.plot(xd, yd, color=st["color"], lw=_px(st["lw"]), marker="s", ms=_px(st["ms"] * 1.2), clip_on=_CLIP[0])
    cv.text(260, 352, r"Pressure ($\mathbf{\mathit{P}/\mathit{P}_0}$)", 21, fontweight="bold")
    cv.text(22, 155, r"N$_\mathbf{2}$ uptaken (cm$^\mathbf{3}$/g)", 21, fontweight="bold", rotation=90)
    cv.text(21, 15, "(d)", 21)
    cv.save(out)


# =============================================================================
# 6. Kinetics: data + fitted curves
# =============================================================================
def _kinetics(data, out):
    cv = Canvas("kinetics", SANS, "bold")
    c = cv.c; L, R, T, B = c["frame"]
    ax = cv.axes_at(L, R, T, B)
    ax.set_xlim(*_limits(c, "x", L, R)); ax.set_ylim(*_limits(c, "y", B, T))
    _style_frame(ax, 2.8)
    ax.set_xticks(np.arange(0, 15, 2)); ax.set_xticks(np.arange(1, 14, 2), minor=True)
    ax.set_yticks(np.arange(0, 121, 20)); ax.set_yticks(np.arange(10, 131, 20), minor=True)
    _ticks(ax, "x", "out", 8, 5, 2.2, 30, weight="bold", pad_px=5)
    _ticks(ax, "y", "out", 8, 5, 2.2, 30, weight="bold", pad_px=4)
    ax.tick_params(top=False, right=False, which="both")
    S = FIGURES["kinetics"]["series"]
    for name in ("Pseudo-first-order", "Pseudo-second-order", "Intraparticle diffusion"):
        s = _series(data, name)
        if s is not None:
            ax.plot(s[0], s[1], color=S[name]["color"], lw=_px(S[name]["lw"]), zorder=3)
    s = _series(data, "PO4 adsorption capacity")
    if s is not None:
        ax.plot(s[0], s[1], ls="none", marker="^", ms=_px(11), color="black", zorder=4, clip_on=_CLIP[0])
    # framed legend with drop shadow at the original position
    x0, x1, y0, y1 = 322, 602, 193, 298
    W, H = cv.W, cv.H
    cv.fig.add_artist(Rectangle(((x0 + 5) / W, 1 - (y1 + 5) / H), (x1 - x0) / W, (y1 - y0) / H, color="black", zorder=7))
    cv.fig.add_artist(Rectangle((x0 / W, 1 - y1 / H), (x1 - x0) / W, (y1 - y0) / H, facecolor="white",
                                edgecolor="#555555", lw=_px(1.2), zorder=8))
    rows = [("PO$_4^{3-}$ adsorption capacity", None), ("Pseudo-first-order", "#FF0000"),
            ("Pseudo-second-order", "#0000FF"), ("Intraparticle diffusion", "#FF00FF")]
    for i, (label, col) in enumerate(rows):
        yy = (208, 237, 262, 287)[i]
        if col is None:
            cv.fig.add_artist(Line2D([347 / W], [1 - yy / H], ls="none", marker="^", ms=_px(11), color="black", zorder=9))
        else:
            cv.fig.add_artist(Line2D([325 / W, 370 / W], [1 - yy / H] * 2, color=col, lw=_px(2.8), zorder=9))
        t = cv.text(375, yy, label, 22.5, ha="left", fontweight="normal"); t.set_zorder(10)
    cv.text(374, 527, "Reaction time (h)", 31, fontweight="bold")
    cv.text(28, 255, "Adsorption capacity (mg/g)", 29, fontweight="bold", rotation=90)
    cv.text(21, 20, "(b)", 30, fontweight="bold")
    cv.save(out)


RENDERERS = {"ftir": _ftir, "pore": _pore, "methane": _methane, "dosage": _dosage, "n2": _n2, "kinetics": _kinetics}


def render(fig_id, dataset, out_path, clip=False):
    """clip=True: points outside the original axis ranges are not drawn (used
    when re-plotting a model's extraction, which may contain wild values)."""
    _CLIP[0] = clip
    try:
        with matplotlib.rc_context():  # fonts/weights set per figure are restored afterwards
            RENDERERS[fig_id](dataset, str(out_path))
    finally:
        _CLIP[0] = False
    return out_path


# =============================================================================
# 7-10. Bar charts
# =============================================================================
def _bar_values(data, name, n):
    """(x indices, heights, errors) of one bar series; missing categories are skipped."""
    s = (data or {}).get("series", {}).get(name)
    if not s or not s.get("x"):
        return [], [], []
    xs, ys = np.asarray(s["x"], float), np.asarray(s["y"], float)
    es = np.asarray(s["yerr"], float) if s.get("yerr") is not None and len(s["yerr"]) == len(xs) else np.zeros(len(xs))
    ok = ~(np.isnan(xs) | np.isnan(ys)) & (xs > -0.5) & (xs < n - 0.5)
    return xs[ok], ys[ok], np.nan_to_num(es[ok])


def _draw_bars(cv, ax, fig_id, data, names, value_ok=None, err=True, err_lw=2.0, cap_px=11, zorder=3):
    """Bars of the given series on one axes (x in category index units)."""
    F = FIGURES[fig_id]
    c = F["calib"]; X = Axis(*c["x"])
    unit = 1.0 / abs(X.p1 - X.p0) * abs(X.v1 - X.v0)        # value units per pixel along x
    n = len(F["categories"])
    for name in names:
        st = F["series"][name]
        xs, ys, es = _bar_values(data, name, n)
        for x, y, e in zip(xs, ys, es):
            if value_ok is not None and not value_ok(int(round(x)), y):
                continue
            xc = x + st["offset_px"] * unit
            w = st["width_px"] * unit
            ax.bar(xc, y, width=w, color=st["color"], edgecolor=st.get("edge") or "none",
                   lw=_px(1.0) if st.get("edge") else 0, hatch=st.get("hatch"), zorder=zorder, clip_on=True)
            if st.get("hatch"):
                ax.patches[-1].set_edgecolor(st.get("hatch_color", "#555555"))
                ax.bar(xc, y, width=w, color="none", edgecolor=st.get("edge") or "none", lw=_px(1.0), zorder=zorder + 0.1)
            if err and e > 0:
                ax.errorbar(xc, y, yerr=e, fmt="none", ecolor="black", elinewidth=_px(err_lw),
                            capsize=_px(cap_px / 2), capthick=_px(err_lw), zorder=zorder + 1)


def _legend_box(cv, box, entries, size_px, horizontal=True, patch_w=40, patch_h=20, frame_color="black", starts=None):
    """Legend drawn at its original pixel box: [(label, style), ...]."""
    x0, x1, y0, y1 = box
    W, H = cv.W, cv.H
    cv.fig.add_artist(Rectangle((x0 / W, 1 - y1 / H), (x1 - x0) / W, (y1 - y0) / H, facecolor="white",
                                edgecolor=frame_color, lw=_px(1.2), zorder=9))
    n = len(entries)
    for i, (label, st) in enumerate(entries):
        if starts is not None:
            ex0 = starts[i]; yy = (y0 + y1) / 2
        elif horizontal:
            ex0 = x0 + 8 + i * (x1 - x0 - 8) / n; yy = (y0 + y1) / 2
        else:
            ex0 = x0 + 8; yy = y0 + (i + 0.5) * (y1 - y0) / n
        r = Rectangle((ex0 / W, 1 - (yy + patch_h / 2) / H), patch_w / W, patch_h / H, facecolor=st["color"],
                      edgecolor=st.get("edge") or st["color"], lw=_px(1.0), hatch=st.get("hatch"), zorder=10)
        if st.get("hatch"):
            r.set_edgecolor(st.get("hatch_color", "#555555"))
        cv.fig.add_artist(r)
        if st.get("hatch"):
            cv.fig.add_artist(Rectangle((ex0 / W, 1 - (yy + patch_h / 2) / H), patch_w / W, patch_h / H,
                                        facecolor="none", edgecolor="black", lw=_px(1.0), zorder=10.5))
        t = cv.text(ex0 + patch_w + 6, yy + 1, label, size_px, ha="left", fontweight="normal")
        t.set_zorder(11)


def _break_marks(cv, x, y, color="black", lw=1.6, gap=8):
    for dy in (-gap / 2, gap / 2):
        cv.fig.add_artist(Line2D([(x - 8) / cv.W, (x + 8) / cv.W], [1 - (y + dy + 5) / cv.H, 1 - (y + dy - 5) / cv.H],
                                 color=color, lw=_px(lw), zorder=12))


def _ph(data, out):
    cv = Canvas("ph", SANS, "bold")
    c = cv.c; L, R, T, B = c["frame"]
    ax = cv.axes_at(L, R, T, B)
    ax.set_xlim(*_limits(c, "x", L, R)); ax.set_ylim(*_limits(c, "y", B, T))
    _style_frame(ax, 2.0)
    ax.set_yticks(range(0, 9)); ax.set_yticks(np.arange(0.5, 8, 1), minor=True)
    ax.set_xticks(range(3)); ax.set_xticklabels(FIGURES["ph"]["categories"])
    _ticks(ax, "y", "out", 10, 6, 1.6, 27, weight="bold", pad_px=4)
    _ticks(ax, "x", "out", 5, 0, 1.6, 34, weight="bold", pad_px=6)
    ax.tick_params(top=False, right=False, which="both")
    _draw_bars(cv, ax, "ph", data, ["pH"])
    cv.text(30, 245, "pH", 38, fontweight="bold", rotation=90)
    cv.text(19, 40, "(a)", 28, fontweight="normal")
    cv.save(out)


def _broken_bar_axes(cv, fig_id, L, R, T, B, sp, gap):
    c = FIGURES[fig_id]["calib"]
    axU = cv.axes_at(L, R, T, sp - gap / 2)
    axL = cv.axes_at(L, R, sp + gap / 2, B)
    for a in (axU, axL):
        a.set_xlim(*_limits(c, "x", L, R)); a.patch.set_alpha(0)
    axU.set_ylim(*_limits(c, "y_upper", sp - gap / 2, T)); axL.set_ylim(*_limits(c, "y_lower", B, sp + gap / 2))
    axU.spines["bottom"].set_visible(False); axL.spines["top"].set_visible(False)
    axU.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    return axU, axL


def _concentration(data, out):
    fid = "concentration"
    cv = Canvas(fid, SANS, "bold")
    c = cv.c; L, R, T, B = c["frame"]; sp = c["split_px"]
    axU, axL = _broken_bar_axes(cv, fid, L, R, T, B, sp, 4)
    for a in (axU, axL):
        _style_frame(a, 2.2)
        a.tick_params(right=False, top=False, which="both")
    axU.set_yticks([100, 110]); axU.set_yticks([95, 105], minor=True)
    axL.set_yticks([0, 10, 20]); axL.set_yticks([5, 15], minor=True)
    axL.set_xticks(range(4)); axL.set_xticklabels(FIGURES[fid]["categories"], rotation=18, ha="right",
                                                   rotation_mode="anchor")
    for a in (axU, axL):
        _ticks(a, "y", "out", 10, 6, 1.6, 27, weight="bold", pad_px=4)
    _ticks(axL, "x", "out", 6, 0, 1.6, 29, weight="bold", pad_px=6)
    names = list(FIGURES[fid]["series"])
    lower_top = float(Axis(*c["y_lower"]).to_value(sp))
    _draw_bars(cv, axU, fid, data, names, value_ok=lambda i, y: y > lower_top)
    _draw_bars(cv, axL, fid, data, names)
    _break_marks(cv, L, sp); _break_marks(cv, R, sp)
    S = FIGURES[fid]["series"]
    _legend_box(cv, (157, 651, 12, 52), [(n, S[n]) for n in names], 23, patch_w=48, patch_h=26, frame_color="#444444",
                starts=[162, 252, 402])
    cv.text(36, 228, "Concentration (g/kg)", 36, fontweight="bold", rotation=90)
    cv.text(18, 18, "(b)", 27, fontweight="normal")
    cv.save(out)


def _concentration2(data, out):
    fid = "concentration2"
    MAG, BLU = "#FF00FF", "#0000FF"
    cv = Canvas(fid, SANS, "bold")
    c = cv.c; L, R, T, B = c["frame"]; sp = c["split_px"]
    axU, axL = _broken_bar_axes(cv, fid, L, R, T, B, sp, 4)
    axR = cv.axes_at(L, R, T, B); axR.patch.set_alpha(0)
    axR.set_xlim(*_limits(c, "x", L, R)); axR.set_ylim(*_limits(c, "y_right", B, T))
    for a in (axU, axL):
        _style_frame(a, 2.0, MAG)
        a.spines["right"].set_visible(False); a.spines["top"].set_color("black"); a.spines["bottom"].set_color("black")
        a.tick_params(right=False, top=False, which="both", colors=MAG)
    for k, sp_ in axR.spines.items():
        sp_.set_linewidth(_px(2.0)); sp_.set_color(BLU if k == "right" else "black")
        if k == "left":
            sp_.set_visible(False)
    axR.yaxis.tick_right(); axR.set_xticks([])
    axR.set_yticks(range(0, 21, 4)); axR.set_yticks(range(2, 20, 4), minor=True)
    _ticks(axR, "y", "out", 9, 5, 1.6, 26, weight="bold", pad_px=4)
    axR.tick_params(axis="y", which="both", colors=BLU, left=False)
    axU.set_yticks([90, 100, 110, 120]); axU.set_yticks([85, 95, 105, 115], minor=True)
    axL.set_yticks([0, 5]); axL.set_yticks([2.5], minor=True)
    for a in (axU, axL):
        _ticks(a, "y", "out", 7, 4, 1.6, 26, weight="bold", pad_px=4)
        a.tick_params(axis="y", which="both", colors=MAG)
    axL.set_xticks(range(4)); axL.set_xticklabels(["Vitamin C", "Chlorophyll", "Soluble protein", "Soluble suger"],
                                                   rotation=15, ha="right", rotation_mode="anchor")
    _ticks(axL, "x", "out", 5, 0, 1.6, 25, weight="bold", pad_px=5)
    axL.tick_params(axis="x", colors="black")
    names = list(FIGURES[fid]["series"])
    lower_top = float(Axis(*c["y_lower"]).to_value(sp))
    left_cat = lambda i: FIGURES[fid]["category_axis"].get(i, "left") == "left"
    _draw_bars(cv, axU, fid, data, names, value_ok=lambda i, y: left_cat(i) and y > lower_top)
    _draw_bars(cv, axL, fid, data, names, value_ok=lambda i, y: left_cat(i))
    _draw_bars(cv, axR, fid, data, names, value_ok=lambda i, y: not left_cat(i))
    _break_marks(cv, L, sp, color=MAG)
    # dashed boxes + arrows telling which axis each group uses
    W, H = cv.W, cv.H
    for (x0, x1, y0, y1), col in (((126.5, 334.5, 78.5, 424), MAG), ((370.5, 583.5, 77.5, 424), BLU)):
        cv.fig.add_artist(Rectangle((x0 / W, 1 - y1 / H), (x1 - x0) / W, (y1 - y0) / H, facecolor="none",
                                    edgecolor=col, lw=_px(1.8), ls=(0, (5, 3)), zorder=7))
    from matplotlib.patches import FancyArrowPatch
    for (xa, ya, xb, yb), col in (((160, 78, 112, 67), MAG), ((550, 77, 597, 67), BLU)):
        cv.fig.add_artist(FancyArrowPatch((xa / W, 1 - ya / H), (xb / W, 1 - yb / H), arrowstyle="-|>",
                                          mutation_scale=_px(14), color=col, lw=_px(1.6), zorder=8))
    S = FIGURES[fid]["series"]
    _legend_box(cv, (135, 580, 31, 62), [(n, S[n]) for n in names], 21, patch_w=41, patch_h=22, frame_color="#444444",
                starts=[137, 236, 391])
    cv.text(30, 225, "Concentration (mg/kg)", 32, fontweight="bold", rotation=90)
    cv.text(676, 225, "Content (%)", 32, fontweight="bold", rotation=90)
    cv.save(out)


def _climet(data, out):
    fid = "climet"
    cv = Canvas(fid, SERIF)
    c = cv.c; L, R, T, B = c["frame"]
    ax = cv.axes_at(L, R, T, B)
    ax.set_xlim(*_limits(c, "x", L, R)); ax.set_ylim(*_limits(c, "y", B, T))
    for k, sp_ in ax.spines.items():
        sp_.set_visible(k == "bottom")
    ax.spines["bottom"].set_color("#949495"); ax.spines["bottom"].set_linewidth(_px(1.5))
    ax.set_yticks(range(0, 101, 10)); ax.set_yticklabels([f"{v}%" for v in range(0, 101, 10)])
    for v in range(10, 101, 10):
        ax.axhline(v, color="#A6A6A6", lw=_px(1.3), zorder=1)
    ax.set_xticks(np.arange(-0.5, 18, 1)); ax.set_xticklabels([])
    ax.set_xticks(range(18), minor=True)
    ax.set_xticklabels(FIGURES[fid]["categories"], minor=True, rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", which="major", direction="out", length=_px(6), width=_px(1.3), color="#949495")
    ax.tick_params(axis="x", which="minor", length=0, labelsize=_px(23), pad=_px(4))
    ax.tick_params(axis="y", length=0, labelsize=_px(23), pad=_px(8))
    # Scenario 1: solid; Scenario 2: vertical gradient (pink at the top -> dark red at the bottom)
    F = FIGURES[fid]; X = Axis(*c["x"]); unit = abs(X.v1 - X.v0) / abs(X.p1 - X.p0)
    from matplotlib.colors import LinearSegmentedColormap
    for name in F["series"]:
        st = F["series"][name]
        xs, ys, _ = _bar_values(data, name, 18)
        for x, y in zip(xs, ys):
            xc, w = x + st["offset_px"] * unit, st["width_px"] * unit
            if "color_bottom" in st:
                cmap = LinearSegmentedColormap.from_list("g", [st["color_bottom"], st["color"]])
                ax.imshow(np.linspace(0, 1, 100)[:, None], extent=(xc - w / 2, xc + w / 2, 0, y), origin="lower",
                          cmap=cmap, aspect="auto", zorder=3, interpolation="bilinear")
            else:
                ax.bar(xc, y, width=w, color=st["color"], zorder=3)
    ax.set_xlim(*_limits(c, "x", L, R)); ax.set_ylim(*_limits(c, "y", B, T))
    W, H = cv.W, cv.H
    cv.fig.add_artist(Rectangle((16 / W, 1 - 743 / H), (1117 - 16) / W, (743 - 14) / H, facecolor="none",
                                edgecolor="#9F9F9F", lw=_px(1.2), zorder=0))
    for yy, name in ((358, "Scenario 1"), (398, "Scenario 2")):
        st = F["series"][name]
        cv.fig.add_artist(Rectangle((977 / W, 1 - (yy + 6) / H), 11 / W, 12 / H, color=st["color"], zorder=5))
        cv.text(995, yy, name, 23, ha="left")
    cv.save(out)


# =============================================================================
# 11-12. 3D pie charts
# =============================================================================
def _shade(hex_color, f):
    from matplotlib.colors import to_rgb
    return tuple(max(0.0, min(1.0, v * f)) for v in to_rgb(hex_color))


def _pie3d(fig_id, data, out):
    """Tilted pie: top face from the four half-radii of the original (perspective),
    slices clockwise from 12 o'clock in the order of `categories`, a darker side
    wall below the front half, and name + percentage labels at the original
    positions (bold sans)."""
    from matplotlib.patches import Polygon
    from .digitize import pie_point
    F = FIGURES[fig_id]
    cv = Canvas(fig_id, SANS, "bold")
    W, H = cv.W, cv.H
    ax = cv.fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    cal = F["calib"]["pie"]; cx, cy = cal["apex"]; depth = cal["depth"]
    names = F["categories"]
    s = (data or {}).get("series", {}).get(F["value_series"], {"x": [], "y": []})
    val = {int(round(x)): max(0.0, float(y)) for x, y in zip(s.get("x", []), s.get("y", [])) if y == y}
    vals = np.array([val.get(i, 0.0) for i in range(len(names))]); tot = vals.sum() or 1.0
    starts, a0 = [], np.pi / 2
    for v in vals:
        a1 = a0 - 2 * np.pi * v / tot
        starts.append((a0, a1)); a0 = a1
    # side wall (front half, angles between 180 and 360 deg)
    for (a0, a1), n in zip(starts, names):
        lo, hi = min(a0, a1), max(a0, a1)
        th = np.linspace(lo, hi, 200)
        th = th[np.sin(th) <= 0.001]
        if len(th) < 2:
            continue
        x, y = pie_point(cal, th)
        poly = np.r_[np.c_[x, y], np.c_[x[::-1], y[::-1] + depth]]
        ax.add_patch(Polygon(poly, closed=True, facecolor=_shade(F["slices"][n]["color"], 0.5),
                             edgecolor=_shade(F["slices"][n]["color"], 0.45), lw=0.3, zorder=2))
    for (a0, a1), n in zip(starts, names):
        th = np.linspace(a1, a0, max(3, int(abs(a0 - a1) * 120)))
        x, y = pie_point(cal, th)
        poly = np.r_[[[cx, cy]], np.c_[x, y]]
        ax.add_patch(Polygon(poly, closed=True, facecolor=F["slices"][n]["color"],
                             edgecolor=F["slices"][n]["color"], lw=0.5, zorder=3))
    for i, n in enumerate(names):
        sl = F["slices"][n]; lx, ly = sl["label_xy"]
        v = vals[i]
        txt = f"{v:.4g}%" if v >= 1 else f"{v:.4g}%"
        ax.text(lx, ly - 16, n, fontsize=_px(23), fontweight="bold", color=sl["label_color"], ha="center",
                va="center", zorder=5)
        ax.text(lx, ly + 16, txt, fontsize=_px(23), fontweight="bold", color=sl["label_color"], ha="center",
                va="center", zorder=5)
    ax.text(4, 22, F["panel_label"], fontsize=_px(36), fontweight="bold", family=_installed(SERIF), ha="left", va="center")
    ax.plot([W - 2, W - 2], [0, H], color="black", lw=_px(3))
    cv.save(out)


RENDERERS.update({"ph": _ph, "concentration": _concentration, "concentration2": _concentration2,
                  "climet": _climet, "pie1": lambda d, o: _pie3d("pie1", d, o),
                  "pie2": lambda d, o: _pie3d("pie2", d, o)})
