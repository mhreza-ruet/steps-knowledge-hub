"""line / marker plots (single panel)"""
from .profiles import add

RED = {"hue": 0, "hue_tol": 18, "min_sat": 0.4, "min_val": 0.4}
NAVY = {"rgb": (40, 84, 156), "tol": 70, "min_sat": 60}
BLK = {"dark": True, "max_gray": 90, "max_sat": 50}


def mk(match, color, marker, ms, **kw):
    d = dict(kind="marker", match=match, color=color, marker=marker, ms_px=ms, connect=True, lw=1.4)
    d.update(kw)
    return d


def ln(match, color, lw=2.0, **kw):
    d = dict(kind="line", match=match, color=color, lw=lw)
    d.update(kw)
    return d


# ------------------------------------------------------------- soluble polysaccharides / proteins (CT, RM, BM)
X1 = (141.0, 5, 483.0, 30)          # line_1: tick marks every 5 days (68.3 px), no labels
for fid, fr, y, yt, yticks, lab, xmap, xt, XS in [
        ("line_1", (72.5, 496.5, 26.0, 266.0), (265.83, 0, 25.83, 150), "Soluble polysaccharides (mg COD/L)",
         [0, 30, 60, 90, 120, 150], "(a)", "LINE1_X", None, [0, 1, 3, 5, 7, 15, 30]),
        ("line_2", (69.0, 493.5, 11.5, 251.5), (203.6, 200, 11.2, 1000), "Soluble proteins (mg COD/L)",
         [200, 400, 600, 800, 1000], "(b)", (69.0, 0, 480.0, 30), [0, 5, 10, 15, 20, 25, 30], [0, 1, 3, 5, 7, 15, 30])]:
    add(fid, file=f"test_data/{fid}.png", size=None, chart_type="line_marker_errorbar",
        title="Three series of markers joined by lines, with error bars" + (" (x tick labels missing in the original: x in days, read from the tick marks of the matching panel (b))" if xmap == "LINE1_X" else ""),
        style={"font": "serif", "weight": "bold", "tick_px": 14, "title_px": 14, "tick_dir": "in", "tick_len": 4},
        panels=[dict(id="main", role="main", frame=fr, x=X1 if xmap == "LINE1_X" else xmap, y=y, y_ticks=yticks,
                     x_ticks=xt if xt else [0, 5, 10, 15, 20, 25, 30], no_x_labels=xt is None,
                     x_title=None if xt is None else "Time (days)", y_title=yt, x_unit="days",
                     label={"text": lab, "xy": (fr[0] + 10, fr[2] + 22), "size_px": 20},
                     exclude=[(fr[1] - 170, fr[1] - 2, fr[2] + 2, fr[2] + 30), (fr[0] + 3, fr[0] + 45, fr[2] + 5, fr[2] + 40)],
                     legend={"xy": (fr[1] - 160, fr[2] + 6), "ncol": 3, "size_px": 14, "weight": "bold", "handlelength": 1.6},
                     series={"CT": mk(BLK, "#333333", "s", 8, err=True, expected_x=XS, outline_match=True, min_score=0.3),
                             "RM": mk(NAVY, "#28549C", "o", 9, err=True, expected_x=XS),
                             "BM": mk(RED, "#E42424", "^", 9, err=True, expected_x=XS)})])

T3 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 26, 30]
add("line_3", file="test_data/line_3.png", size=None, chart_type="line_marker_errorbar",
    title="Methane yield over time, three reactors (markers + lines, error bars)",
    style={"font": "sans", "weight": "bold", "tick_px": 20, "title_px": 22, "tick_dir": "out", "tick_len": 5, "frame_lw": 1.4},
    panels=[dict(id="main", role="main", frame=(93.5, 554.5, 10.5, 364.5), x=(93.45, 0, 539.7, 30), y=(310.26, 20, 37.9, 120),
                 x_ticks=[0, 5, 10, 15, 20, 25, 30], y_ticks=[20, 40, 60, 80, 100, 120], x_title="Time (days)",
                 y_title="Methane yield (%)", label={"text": "(a)", "xy": (100, 26), "size_px": 22},
                 exclude=[(100, 420, 15, 45)], legend={"xy": (165, 14), "ncol": 3, "size_px": 18, "weight": "bold", "handlelength": 1.8},
                 series={"MBC": mk(BLK, "#333333", "s", 11, err=True, expected_x=T3),
                         "FBM": mk(NAVY, "#28549C", "o", 12, err=True, expected_x=T3),
                         "SBM": mk(RED, "#E42424", "^", 12, err=True, expected_x=T3)})])

# letters printed next to the points (drawn again as annotations, excluded from the series)
LET = {"line_4": [(124, 208, "i"), (159, 177, "g"), (199, 127, "e"), (235, 102, "d"), (272, 73, "c"), (311, 25, "a"),
                  (351, 61, "b"), (388, 140, "f"), (426, 188, "gh")],
       "line_5": [(118, 76, "a"), (160, 115, "b"), (196, 172, "c"), (231, 209, "d"), (267, 229, "e"), (305, 248, "f"),
                  (345, 259, "g"), (382, 275, "h"), (417, 283, "i")]}
for fid, fr, x, y, col, match, mkr, hol, name, lab, lx in [
        ("line_4", (86.5, 459.0, 15.0, 315.5), (86.4, 2, 459.26, 12), (292.36, 50, 15.07, 110), "#E81818", RED, "s", False,
         "NH$_4^+$ adsorption capacity", "(a)", (95, 280)),
        ("line_5", (81.5, 454.0, 18.0, 318.5), (81.21, 2, 454.29, 12), (295.57, 50, 17.43, 110), "#1818E8",
         {"hue": 240, "hue_tol": 20, "min_sat": 0.4}, "^", True, "PO$_4^{3-}$ adsorption capacity", "(b)", (90, 284))]:
    add(fid, file=f"test_data/{fid}.png", size=None, chart_type="line_marker",
        title="Adsorption capacity vs pH, one series (markers + line, letters on the points)",
        style={"font": "sans", "weight": "bold", "tick_px": 17, "title_px": 18, "tick_dir": "out", "tick_len": 4},
        panels=[dict(id="main", role="main", frame=fr, x=x, y=y, x_ticks=[2, 4, 6, 8, 10, 12],
                     y_ticks=[50, 60, 70, 80, 90, 100, 110], x_title="pH", y_title="Adsorption capacity (mg/g)",
                     label={"text": lab, "xy": (2, fr[2] + 6), "size_px": 20, "weight": "bold"},
                     exclude=[(fr[0] + 3, fr[0] + 230, fr[3] - 48, fr[3] - 3)] + [(x0 - 8, x0 + 9, y0 - 7, y0 + 7) for x0, y0, _ in LET[fid]],
                     legend={"xy": (lx[0], lx[1] - 8), "size_px": 14, "weight": "normal", "handlelength": 1.6},
                     series={name: mk(match, col, mkr, 9 if hol else 7, hollow=hol, expected_x=[3, 4, 5, 6, 7, 8, 9, 10, 11], on_line=hol)})],
        annotations=[{"text": t, "xy": (x0, y0), "ha": "center", "size_px": 15, "color": col} for x0, y0, t in LET[fid]])

for fid, fr, x, y, specs, title in [
        ("line_6", (82.0, 458.5, 12.0, 316.0), (82.0, 0, 458.62, 14), (314.95, 0, 11.88, 1.0),
         [("NH$_4^+$", RED, "#F01010"), ("NH$_3$", {"hue": 230, "hue_tol": 20, "min_sat": 0.4}, "#1010F0")],
         "Ammonium / ammonia mole fractions vs pH (two curves)"),
        ("line_7", (79.5, 454.5, 14.0, 317.0), (79.71, 0, 454.79, 14), (315.98, 0, 13.69, 1.0),
         [("H$_3$PO$_4$", {"hue": 300, "hue_tol": 15, "min_sat": 0.5}, "#FF00FF"),
          ("H$_2$PO$_4^-$", {"hue": 30, "hue_tol": 12, "min_sat": 0.5}, "#FF8C00"),
          ("HPO$_4^{2-}$", {"hue": 240, "hue_tol": 12, "min_sat": 0.5}, "#0000FF"),
          ("PO$_4^{3-}$", {"hue": 0, "hue_tol": 10, "min_sat": 0.5}, "#FF0000")],
         "Phosphate species mole fractions vs pH (four curves)")]:
    add(fid, file=f"test_data/{fid}.png", size=None, chart_type="line",
        title=title,
        style={"font": "sans", "weight": "bold", "tick_px": 17, "title_px": 18, "tick_dir": "out", "tick_len": 4},
        panels=[dict(id="main", role="main", frame=fr, x=x, y=y, x_ticks=list(range(0, 15, 2)),
                     y_ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], y_ticklabels=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
                     x_title="pH", y_title="Mole fractions",
                     label={"text": "(c)" if fid == "line_6" else "(d)", "xy": (2, fr[2] + 6), "size_px": 20},
                     exclude=([(160, 205, 26, 56), (155, 200, 280, 305)] if fid == "line_6" else
                              [(172, 245, 24, 56), (308, 385, 24, 56), (146, 214, 218, 252), (344, 400, 216, 250)]),
                     series={n: ln(m, c, lw=2.2, samples=200) for n, m, c in specs})],
        annotations=([{"text": "NH$_4^+$", "xy": (178, 38), "size_px": 17, "weight": "bold"},
                      {"text": "NH$_3$", "xy": (172, 290), "size_px": 17, "weight": "bold"}] if fid == "line_6" else
                     [{"text": "H$_2$PO$_4^-$", "xy": (140, 45), "size_px": 17, "weight": "bold", "color": "#FF8C00"},
                      {"text": "HPO$_4^{2-}$", "xy": (272, 45), "size_px": 17, "weight": "bold", "color": "#0000FF"},
                      {"text": "H$_3$PO$_4$", "xy": (138, 238), "size_px": 17, "weight": "bold", "color": "#FF00FF"},
                      {"text": "PO$_4^{3-}$", "xy": (330, 240), "size_px": 17, "weight": "bold", "color": "#0000FF"}]))

# ------------------------------------------------------------- P extraction efficiency vs L/S (several acids)
LS = [5, 10, 15, 20, 25]


def hm(match, color, marker, ms=8, **kw):
    return mk(match, color, marker, ms, hollow=True, expected_x=LS, outline_match=True, lw=1.2, face="none", **kw)


add("line_9", file="test_data/line_9.png", size=None, chart_type="line_marker",
    title="Phosphorus extraction efficiency vs liquid/solid ratio, six extractants (hollow markers + lines)",
    style={"font": "serif", "weight": "normal", "tick_px": 20, "title_px": 22, "tick_dir": "in", "tick_len": 5},
    panels=[dict(id="main", role="main", frame=(60.5, 549.5, 7.5, 369.5), x=(111.4, 5, 500.2, 25), y=(363.5, 0, 66.5, 10),
                 x_ticks=LS, y_ticks=[0, 2, 4, 6, 8, 10, 12], x_title="L/S", y_title="PEE [%]",
                 label={"text": "(a)", "xy": (66, 22), "size_px": 22}, exclude=[(65, 545, 10, 60)],
                 legend={"xy": (130, 10), "ncol": 3, "size_px": 20, "handlelength": 2.6, "columnspacing": 0.5},
                 series={"0.1 N": hm({"dark": True, "max_gray": 130, "max_sat": 40}, "#000000", "o"),
                         "0.2 N": hm({"hue": 0, "hue_tol": 15, "min_sat": 0.25, "min_val": 0.3}, "#C81818", "s"),
                         "0.5 N": hm({"hue": 247, "hue_tol": 8, "min_sat": 0.3, "min_val": 0.3}, "#2808F8", "^"),
                         "1.0 N": hm({"rgb": (80, 112, 112), "tol": 45}, "#507070", "v"),
                         "2.0 N": hm({"hue": 287, "hue_tol": 10, "min_sat": 0.3, "min_val": 0.3}, "#C808F8", "D"),
                         "Water": hm({"hue": 60, "hue_tol": 12, "min_sat": 0.25, "min_val": 0.3}, "#686818", "<")})])

add("line_10", file="test_data/line_10.png", size=None, chart_type="line_marker",
    title="Phosphorus extraction efficiency vs liquid/solid ratio, five acid strengths (hollow markers + lines)",
    style={"font": "serif", "weight": "normal", "tick_px": 20, "title_px": 22, "tick_dir": "in", "tick_len": 5},
    panels=[dict(id="main", role="main", frame=(64.5, 546.5, 10.5, 368.5), x=(209.5, 10, 498.5, 25), y=(368.5, 0, 27.5, 100),
                 x_ticks=LS, y_ticks=[0, 20, 40, 60, 80, 100], x_title="L/S", y_title="PEE [%]",
                 label={"text": "(b)", "xy": (68, 26), "size_px": 22}, exclude=[(105, 545, 12, 62)],
                 legend={"xy": (210, 12), "ncol": 3, "size_px": 20, "handlelength": 2.6},
                 series={"0.1 N": hm({"rgb": (128, 50, 50), "tol": 45}, "#803232", "s", line_color="#000000"),
                         "0.2 N": hm({"hue": 28, "hue_tol": 10, "min_sat": 0.25, "min_val": 0.3}, "#D87828", "o"),
                         "0.5 N": hm({"dark": True, "max_gray": 80, "max_sat": 30}, "#373737", "^", line_color="#000000"),
                         "1.0 N": hm({"rgb": (80, 112, 112), "tol": 45}, "#507070", "v", line_color="#2808F8"),
                         "2.0 N": hm({"hue": 290, "hue_tol": 14, "min_sat": 0.3, "min_val": 0.3}, "#DC40F0", "D", line_color="#C80860")})])

add("line_14", file="test_data/line_14.png", size=None, chart_type="line_marker",
    title="Removal efficiency vs dosage, Fe and Al (hollow markers + lines)",
    style={"font": "serif", "weight": "normal", "tick_px": 21, "title_px": 23, "tick_dir": "in", "tick_len": 5},
    panels=[dict(id="main", role="main", frame=(90.0, 674.0, 17.5, 444.0), x=(101.26, 0, 662.9, 0.5), y=(423.95, 0, 16.38, 100),
                 x_ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], x_ticklabels=["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"],
                 y_ticks=[0, 20, 40, 60, 80, 100], x_title="Dosage [g/mL]", y_title="Removal efficiency [%]",
                 exclude=[(270, 500, 20, 60)], legend={"xy": (282, 20), "ncol": 2, "size_px": 21, "handlelength": 2.4},
                 series={"Fe": mk({"hue": 0, "hue_tol": 15, "min_sat": 0.45}, "#CC2424", "o", 15, hollow=True, face="none",
                                  expected_x=[0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5], outline_match=True, lw=1.2),
                         "Al": mk({"dark": True, "max_gray": 90, "max_sat": 40}, "#000000", "^", 15, hollow=True, face="none",
                                  expected_x=[0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5], outline_match=True, lw=1.2)})])

# ------------------------------------------------------------- measured points + fitted model curves
def fitfig(fid, fr, x, y, xt, yt, xtitle, label, pts_name, pts_match, pts_marker, pts_ms, curves, legend_box, hollow=False,
           pts_color="#000000", xs=None, duplicate_of=None):
    legend_xy = (legend_box[0] + 2, legend_box[2] + 2)
    add(fid, file=f"test_data/{fid}.png", size=None, chart_type="scatter+fit_curves", duplicate_of=duplicate_of,
        title="Measured points with fitted model curves",
        style={"font": "sans", "weight": "bold", "tick_px": 22, "title_px": 26, "tick_dir": "out", "tick_len": 6, "frame_lw": 2.0},
        panels=[dict(id="main", role="main", frame=fr, x=x, y=y, x_ticks=xt, y_ticks=yt, x_title=xtitle,
                     y_title="Adsorption capacity (mg/g)", label={"text": label, "xy": (2, fr[2] + 4), "size_px": 26},
                     exclude=[(legend_box[0] - 3, legend_box[1] + 3, legend_box[2] - 3, legend_box[3] + 3)],
                     legend={"xy": legend_xy, "size_px": 16, "frame": True, "weight": "normal", "handlelength": 2.2},
                     series={pts_name: dict(kind="marker", match=pts_match, marker=pts_marker, ms_px=pts_ms, hollow=hollow,
                                            color=pts_color, face="none" if hollow else pts_color, expected_x=xs),
                             **{n: dict(kind="line", match=m, color=c, lw=3.0, samples=150, max_jump=40) for n, (m, c) in curves.items()}})])


KT = [0.5, 1, 3, 5, 7, 9, 11, 13]          # sampling times of the kinetics experiments (h)
IC = [10, 20, 40, 60, 80, 100, 120, 140]   # initial concentrations of the isotherm experiments (mg/L)
RC = ({"hue": 0, "hue_tol": 15, "min_sat": 0.5}, "#FF0000")
BC = ({"hue": 240, "hue_tol": 15, "min_sat": 0.5}, "#0000FF")
MC = ({"hue": 300, "hue_tol": 15, "min_sat": 0.5}, "#FF00FF")
fitfig("linefit_1", (112.5, 671.5, 22.0, 472.0), (112.29, 0, 672.08, 14), (471.67, 0, 21.33, 140), list(range(0, 15, 2)),
       list(range(0, 141, 20)), "Reaction time (h)", "(a)", "NH$_4^+$ adsorption capacity",
       {"dark": True, "max_gray": 80, "max_sat": 40}, "s", 9,
       {"Pseudo-first-order": RC, "Pseudo-second-order": BC, "Intraparticle diffusion": MC}, (342, 635, 230, 344), xs=KT)
fitfig("linefit_2", (112.0, 665.5, 25.5, 470.5), (111.71, 0, 665.84, 160), (470.62, 0, 24.75, 140), list(range(0, 161, 20)),
       list(range(0, 141, 20)), "Initial concentration (mg/L)", "(a)", "NH$_4^+$ adsorption capacity",
       {"dark": True, "max_gray": 80, "max_sat": 40}, "s", 9, {"Langmuir": RC, "Freundlich": BC}, (339, 630, 361, 448), xs=IC)
fitfig("linefit_3", (113.5, 672.5, 11.0, 461.5), (113.29, 0, 673.08, 14), (460.77, 0, 45.8, 120), list(range(0, 15, 2)),
       list(range(0, 121, 20)), "Reaction time (h)", "(b)", "PO$_4^{3-}$ adsorption capacity",
       {"dark": True, "max_gray": 80, "max_sat": 40}, "^", 11,
       {"Pseudo-first-order": RC, "Pseudo-second-order": BC, "Intraparticle diffusion": MC}, (344, 642, 209, 324), xs=KT,
       duplicate_of="kinetics")
fitfig("linefit_4", (98.5, 666.0, 26.5, 483.0), (98.84, 0, 665.71, 160), (482.5, 0, 26.2, 90), list(range(0, 161, 20)),
       list(range(0, 91, 10)), "Initial concentration (mg/L)", "(b)", "PO$_4^{3-}$ adsorption capacity",
       {"dark": True, "max_gray": 150, "max_sat": 40}, "^", 12, {"Langmuir": RC, "Freundlich": BC}, (333, 636, 374, 463), hollow=True,
       pts_color="#555555", xs=IC)

# ------------------------------------------------------------- broken axes (drawn in pixel space)
T11 = [5, 10, 15, 30, 60, 120, 1440]
XSEG11 = [{"px": (57, 299), "map": (57, 0, 277.5, 120), "ticks": [0, 20, 40, 60, 80, 100, 120]},
          {"px": (305, 544.5), "map": (352.5, 1200, 544.5, 1500), "ticks": [1200, 1350, 1500]}]
add("line_11", file="test_data/line_11.png", size=None, chart_type="line_marker_broken_axes",
    title="Extracted phosphorus vs time, NaOH and H2SO4 (x axis and y axis both broken)",
    style={"font": "serif", "weight": "normal", "tick_px": 23, "title_px": 25, "tick_dir": "in", "tick_len": 5},
    panels=[dict(id="main", role="main", frame=(57.0, 544.5, 9.5, 366.5), pixel_space=True, x_segments=XSEG11,
                 y_segments=[{"rows": (9.5, 259), "map": (205.8, 15, 14.2, 35), "ticks": [10, 15, 20, 25, 30, 35]},
                             {"rows": (259, 366.5), "map": (350.75, 0, 279.25, 4), "ticks": [0, 2, 4]}],
                 x_title="Time [min]", y_title="Extracted amount of P [mg/g]", label={"text": "(c)", "xy": (64, 26), "size_px": 22},
                 exclude=[(62, 120, 12, 42), (150, 470, 12, 70)],
                 legend={"xy": (155, 16), "ncol": 2, "size_px": 18, "handlelength": 2.6},
                 series={"1.0 N NaOH": dict(kind="marker", match={"hue": 240, "hue_tol": 18, "min_sat": 0.3}, color="#1414E6",
                                            marker="^", ms_px=10, hollow=True, connect=True, lw=1.2, outline_match=True, expected_x=T11),
                         "0.2 N H$_2$SO$_4$": dict(kind="marker", match={"hue": 0, "hue_tol": 15, "min_sat": 0.3}, color="#E61414",
                                                    marker="o", ms_px=10, hollow=True, connect=True, lw=1.2, outline_match=True, expected_x=T11)})])

XSEG12 = [{"px": (58.5, 300), "map": (58.5, 0, 278.5, 120), "ticks": [0, 20, 40, 60, 80, 100, 120]},
          {"px": (306, 545.5), "map": (324.5, 1200, 545.5, 1500), "ticks": [1200, 1350, 1500]}]
M12 = [("Ca", {"dark": True, "max_gray": 90, "max_sat": 40}, "#000000", "o"),
       ("Fe", {"hue": 2, "hue_tol": 14, "min_sat": 0.4}, "#E61414", "s"),
       ("Mg", {"hue": 240, "hue_tol": 14, "min_sat": 0.4}, "#1414E6", "^"),
       ("Al", {"rgb": (80, 112, 112), "tol": 45}, "#507070", "v"),
       ("Zn", {"hue": 300, "hue_tol": 18, "min_sat": 0.4}, "#E614E6", "d"),
       ("Cu", {"hue": 60, "hue_tol": 14, "min_sat": 0.4}, "#808000", "<")]
add("line_12", file="test_data/line_12.png", size=None, chart_type="line_marker_broken_x",
    title="Extracted metals vs time, six metals (x axis broken)",
    style={"font": "serif", "weight": "normal", "tick_px": 23, "title_px": 25, "tick_dir": "in", "tick_len": 5},
    panels=[dict(id="main", role="main", frame=(58.5, 545.5, 12.5, 366.5), pixel_space=True, x_segments=XSEG12,
                 y=(356.92, 0, 11.96, 35), y_ticks=list(range(0, 36, 5)),
                 x_title="Time [min]", y_title="Extracted amount of metals [mg/g]", label={"text": "(d)", "xy": (64, 30), "size_px": 22},
                 exclude=[(62, 120, 14, 45), (180, 460, 14, 60)],
                 legend={"xy": (185, 14), "ncol": 3, "size_px": 17, "handlelength": 2.4},
                 series={n: dict(kind="marker", match=m, color=c, marker=mk_, ms_px=9, hollow=True, connect=True, lw=1.2, outline_match=True, expected_x=T11)
                         for n, m, c, mk_ in M12})])

PH13 = [1.41, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10, 11]
add("line_13", file="test_data/line_13.png", size=None, chart_type="line_marker_broken_y",
    title="P, Al and Fe concentration vs pH (y axis broken)",
    style={"font": "serif", "weight": "normal", "tick_px": 21, "title_px": 26, "tick_dir": "in", "tick_len": 5},
    panels=[dict(id="main", role="main", frame=(105.0, 671.0, 20.5, 439.5), pixel_space=True, x=(128.5, 1, 647.5, 12),
                 x_ticks=list(range(1, 13)),
                 y_segments=[{"rows": (20.5, 230), "map": (182.9, 1200, 19.1, 1500), "ticks": [1200, 1300, 1400, 1500]},
                             {"rows": (230, 439.5), "map": (423.4, 0, 261.85, 300), "ticks": [0, 100, 200, 300]}],
                 x_title="pH", y_title="Concentration [mg/L]", label={"text": "(a)", "xy": (112, 38), "size_px": 26},
                 exclude=[(110, 165, 22, 56), (460, 668, 22, 64), (100, 115, 220, 240), (660, 675, 220, 240)],
                 legend={"xy": (480, 24), "ncol": 3, "size_px": 21, "handlelength": 2.4},
                 series={"P": dict(kind="marker", match={"dark": True, "max_gray": 90, "max_sat": 40}, color="#000000", marker="s",
                                   ms_px=10, hollow=True, connect=True, lw=1.2, outline_match=True, expected_x=PH13),
                         "Al": dict(kind="marker", match={"hue": 2, "hue_tol": 14, "min_sat": 0.4}, color="#C81818", marker="o",
                                    ms_px=10, hollow=True, connect=True, lw=1.2, outline_match=True, expected_x=PH13),
                         "Fe": dict(kind="marker", match={"hue": 240, "hue_tol": 14, "min_sat": 0.4}, color="#1818C8", marker="^",
                                    ms_px=10, hollow=True, connect=True, lw=1.2, outline_match=True, expected_x=PH13 + [12])})])
