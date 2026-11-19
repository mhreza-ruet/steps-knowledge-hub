"""figures with several panels or an inset (one image, several plots)"""
from .profiles import add

# ---------------------------------------------------------------- line plot + inset bar chart
add("nutrient_energy", file="Nutrient-energy.png", size=(524, 446), chart_type="line+inset_bar",
    title="Water flux over 5 cycles (lines) with an inset bar chart of water volume per cycle",
    duplicates=["test_data/line_8.png"],
    style={"font": "sans", "weight": "bold", "tick_px": 19, "title_px": 20, "tick_dir": "out", "tick_len": 5, "frame_lw": 1.6},
    panels=[
        dict(id="main", role="main", frame=(61.5, 514.5, 15.5, 385.5),
             x=(132.0, 4, 496.0, 24), y=(385.17, 0, 14.17, 10),
             x_ticks=[0, 4, 8, 12, 16, 20, 24], y_ticks=[0, 2, 4, 6, 8, 10],
             x_title="Time (h)", y_title="Average Water Flux (LMH)",
             exclude=[(255, 520, 10, 225), (70, 512, 315, 368)],
             legend={"xy": (72, 318), "ncol": 3, "size_px": 18, "weight": "bold", "handlelength": 2.2,
                     "order": ["Cycle 1", "Cycle 3", "Cycle 2", "Cycle 4", "Cycle 5"]},
             series={
                 "Cycle 1": dict(kind="line", match={"dark": True, "max_gray": 70}, color="#000000", lw=4, frame_pad=5),
                 "Cycle 2": dict(kind="line", match={"rgb": (204, 20, 20), "tol": 70}, color="#D01414", lw=4),
                 "Cycle 3": dict(kind="line", match={"rgb": (36, 12, 252), "tol": 45}, color="#240CFC", lw=4),
                 "Cycle 4": dict(kind="line", match={"rgb": (60, 110, 12), "tol": 50}, color="#3C6E0C", lw=4),
                 "Cycle 5": dict(kind="line", match={"rgb": (108, 12, 252), "tol": 40}, color="#6C0CFC", lw=4),
             }),
        dict(id="inset_bar", role="inset", frame=(309.5, 514.5, 15.5, 182.0),
             categories=["1", "2", "3", "4", "5"], x=(330.0, 0, 493.5, 4), y=(181.7, 0, 32.7, 80),
             x_ticks=[0, 1, 2, 3, 4], x_ticklabels=["1", "2", "3", "4", "5"], y_ticks=[0, 20, 40, 60, 80],
             x_title="Cycle", y_title="Water Volume (mL)", tick_px=16, title_px=17, face="white",
             series={"Water volume": dict(kind="bar", match={"rgb": (100, 170, 240), "tol": 60}, width_px=19,
                                          color="#64AAF0", fill="#64AAF0", edge="#000000")}),
    ])

# ---------------------------------------------------------------- bars + markers (dual axis) + inset bar chart
add("nutrient", file="Nutrient.png", size=None, chart_type="bar+marker_dual_axis+inset_bar",
    title="Specific production rate (bars, left axis) and collected volume (markers, right axis), inset bar chart of SER",
    duplicates=["test_data/bar_7.png"],
    style={"font": "sans", "weight": "bold", "tick_px": 20, "title_px": 20, "tick_dir": "out", "tick_len": 5, "frame_lw": 1.5},
    panels=[
        dict(id="main", role="main", frame=(71.5, 544.0, 14.5, 399.0), categories=["Closed-loop", "Open-loop"],
             x=(189.0, 0, 426.0, 1), y=(398.48, 0, 13.69, 40), y_right=(398.83, 0, 13.83, 50),
             y_ticks=[0, 8, 16, 24, 32, 40], y_right_ticks=[0, 10, 20, 30, 40, 50], x_ticks=[0, 1],
             x_ticklabels=["Closed-loop", "Open-loop"], xtick_px=24,
             y_title="Specific Production Rate (L m$^{-3}$d$^{-1}$)", y_right_title="Collected Volume (mL)",
             y_right_color="#D82030", label={"text": "C", "xy": (86, 32), "size_px": 26},
             exclude=[(230, 544, 10, 225), (80, 110, 18, 48)],
             series={"Specific production rate": dict(kind="bar", match={"rgb": (73, 104, 229), "tol": 50}, width_px=97, err=True,
                                                       color="#4968E5", fill="#4968E5", edge="#3040A0"),
                     "Collected volume": dict(kind="marker", axis="right", match={"rgb": (202, 53, 52), "tol": 70},
                                              marker="p", ms_px=20, hollow=True, err=True, color="#CA3534", mew=2.5)}),
        dict(id="inset_bar", role="inset", frame=(284.0, 496.5, 14.5, 187.5), categories=["Closed-loop", "Open-loop"],
             x=(337.0, 0, 443.5, 1), y=(185.9, 0, 32.5, 8), y_ticks=[0, 2, 4, 6, 8], x_ticks=[0, 1],
             x_ticklabels=["Closed-loop", "Open-loop"], y_title="SER (%)", tick_px=15, title_px=13, face="white",
             series={"SER": dict(kind="bar", match={"rgb": (201, 57, 88), "tol": 50}, width_px=55, err=True,
                                 color="#C93958", fill="#C93958", edge="#000000")}),
    ])

# ---------------------------------------------------------------- 2x2 panels: isotherm points + two fitted models
LANG_EQ = "y = (a*b*x^(1-c))/(1 + b*x^(1-c))"
FREU_EQ = "y = a*x^(b*x^(-c))"
BIO_PANELS = [
    # id, frame, x map, y map, x ticks, y ticks, panel label xy, Langmuir table box, Freundlich table box, printed params
    ("a", (66.5, 694.5, 45.0, 484.0), (66.5, 0.1, 694.5, 0.7), (484.0, 0, 45.0, 12), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
     list(range(0, 13, 2)), (360, 66), (202, 514, 195, 315), (204, 511, 327, 472),
     {"Langmuir": {"a": 0.588, "b": None, "c": 0.933, "R-Square": 0}, "Freundlich": {"a": 0.023, "b": -14.616, "c": -0.888, "R-Square": 0.922}}),
    ("b", (784.0, 1412.0, 45.0, 484.0), (784.0, 0, 1412.0, 4), (484.0, 0, 45.0, 4), [0, 1, 2, 3, 4], [0, 1, 2, 3, 4],
     (1097, 66), (1011, 1297, 213, 331), (1012, 1297, 345, 461),
     {"Langmuir": {"a": 4.078, "b": 0.764, "c": -0.715, "R-Square": 0.965}, "Freundlich": {"a": 2.539, "b": 0.267, "c": 496.841, "R-Square": 0.552}}),
    ("c", (66.5, 694.5, 547.0, 985.0), (66.5, 0, 694.5, 3), (985.0, 0, 547.0, 6), [0, 1, 2, 3], list(range(0, 7)),
     (360, 567), (310, 593, 716, 850), (310, 593, 862, 976),
     {"Langmuir": {"a": 6.022, "b": 1.425, "c": 0.076, "R-Square": 0.97906}, "Freundlich": {"a": 3.512, "b": 0.375, "c": 0.2223, "R-Square": 0.979}}),
    ("d", (784.0, 1412.0, 547.0, 985.0), (784.0, 0, 1412.0, 2), (985.0, 0, 547.0, 8), [0, 1, 2], [0, 2, 4, 6, 8],
     (1097, 569), (1052, 1340, 734, 859), (1054, 1340, 864, 976),
     {"Langmuir": {"a": 80.855, "b": 0.069, "c": 0.059, "R-Square": 0.997}, "Freundlich": {"a": 5.241, "b": 0.888, "c": 0.001, "R-Square": 0.997}}),
]


def _table_text(name, eq, p):
    rows = [f"Model      {name}", f"Equation   {eq}"] + [f"{k:<10} {'-' if v is None else v}" for k, v in p.items()]
    return "\n".join(rows)


add("biosorption", file="Biosorption.png", size=None, chart_type="panels_2x2:scatter+fit_curves",
    title="Adsorption isotherms at 10/20/30/40 °C: four panels, each with data points and Langmuir / Freundlich fits "
          "(fit parameters printed in each panel)",
    duplicates=["test_data/stacked_linefit_1.png"],
    style={"font": "sans", "weight": "normal", "tick_px": 18, "title_px": 20, "tick_dir": "out", "tick_len": 5, "frame_lw": 1.2},
    fig_legend={"xy": (350, 4), "ncol": 3, "size_px": 20, "entries": [("Experimental data", "marker", "#000000"),
                                                                    ("Langmuir modelling", "line", "#FF0000"),
                                                                    ("Freundlich modelling", "dashed", "#0000FF")]},
    panels=[dict(id=pid, role="panel", frame=fr, x=x, y=y, x_ticks=xt, y_ticks=yt,
                 x_title="Ce" if pid == "a" else None, y_title=None,
                 label={"text": f"({pid})", "xy": (lab[0] - 16, lab[1]), "size_px": 24},
                 exclude=[tl, tf, (lab[0] - 25, lab[0] + 25, lab[1] - 15, lab[1] + 15)],
                 printed=prm, printed_equations={"Langmuir": LANG_EQ, "Freundlich": FREU_EQ},
                 tables=[{"box": tl, "text": _table_text("Langmuir", LANG_EQ, prm["Langmuir"]), "color": "#FF0000"},
                         {"box": tf, "text": _table_text("Freundlich", FREU_EQ, prm["Freundlich"]), "color": "#0000FF"}],
                 series={"Experimental data": dict(kind="marker", match={"dark": True, "max_gray": 90, "max_sat": 40},
                                                   color="#000000", marker="s", ms_px=7, min_area_frac=0.4),
                         "Langmuir modelling": dict(kind="line", match={"hue": 0, "hue_tol": 15, "min_sat": 0.5}, color="#FF0000",
                                                    lw=1.5, samples=200, max_jump=40, frame_pad=0, gap_fill_px=60),
                         "Freundlich modelling": dict(kind="line", match={"hue": 235, "hue_tol": 20, "min_sat": 0.5}, color="#0000FF",
                                                      lw=1.5, ls="--", samples=200, max_jump=60, gap_fill_px=60, frame_pad=0)})
            for pid, fr, x, y, xt, yt, lab, tl, tf, prm in BIO_PANELS],
    annotations=[{"text": "q$_e$ (mg/g)", "xy": (16, 465), "size_px": 22, "rotation": 90, "ha": "center"},
                 {"text": "C$_e$ (mg/L)", "xy": (740, 1043), "size_px": 22, "ha": "center"},
                 {"text": "Fig. 4.  Adsorption isotherms for the biosorption of Cu$^{2+}$ by Desmodesmus sp. CHX1 and 10 °C (a), "
                          "20 °C (b), 30 °C (c) and 40 °C (d).", "xy": (715, 1092), "size_px": 19, "ha": "center", "font": "serif"}])

# ---------------------------------------------------------------- 7 stacked panels sharing one x axis (XRD + reference cards)
SB = [12, 95.5, 178, 261.5, 344.5, 427.5, 510.5, 593.5]
S_Y = [(83.25, 0, 13.75, 6000), (170.0, 0, 104.0, 4000), (253.0, 0, 187.0, 4000), (336.0, 0, 270.0, 4000), (420.0, 0, 359.5, 4000),
       (504.0, 0, 434.0, 100), (587.0, 0, 517.4, 100)]
S_T = [[0, 3000, 6000]] + [[0, 2000, 4000]] * 4 + [[0, 50, 100]] * 2
S_NAME = ["pH=8.0, Mg:P:N=1:1:1", "pH=9.5, Mg:P:N=1.4:1:2", "pH=9.5, Mg:P:N=1.4:1:1.5", "pH=9.5, Mg:P:N=1.4:1:1",
          "pH=9.5, Mg:P:N=1:1:1", "Struvite, PDF#71-2089", "NaCl, PDF#05-0628"]
S_TXT = [[(444, 516, 18, 38), (532, 661, 19, 38)], [(427, 498, 100, 121), (515, 663, 100, 121)],
         [(410, 481, 187, 204), (497, 661, 187, 204)], [(429, 500, 267, 287), (517, 662, 267, 287)],
         [(445, 516, 350, 370), (533, 662, 351, 370)], [(458, 665, 434, 453)], [(475, 659, 518, 536)]]


def _sus_panel(i):
    T, B = SB[i], SB[i + 1]
    stick = i >= 5
    txt = S_TXT[i]
    s = (dict(kind="stick", match={"dark": True, "max_gray": 195, "max_sat": 40}, color="#000000", lw=1.0, qa_limit_px=6) if stick else
         dict(kind="line", match={"dark": True, "max_gray": 140, "max_sat": 40}, color="#000000", lw=1.0, spectrum=True,
              peaks="up", samples=900, frame_pad=5, x_px=(103, 664)))
    return dict(id=f"p{i + 1}", role="panel", frame=(95.0, 667.0, T, B), x=(94.84, 10, 667.38, 50), y=S_Y[i], y_ticks=S_T[i],
                x_ticks=list(range(10, 51, 5)), no_x_labels=i < 6, x_title="2 Theta [deg]" if i == 6 else None,
                exclude=[(a - 3, b + 3, c - 3, d + 7) for a, b, c, d in txt], tick_dir="in", tick_px=17, title_px=22,
                panel_title=S_NAME[i],
                label={"text": S_NAME[i].replace(", Mg", "  Mg") if i < 5 else S_NAME[i], "xy": (txt[0][0], (txt[0][2] + txt[0][3]) / 2),
                       "size_px": 17, "weight": "normal"},
                series={("Reference pattern" if stick else "XRD pattern"): s})


add("sustainable", file="Sustainable.png", size=None, chart_type="stacked_panels:spectrum+reference_sticks",
    title="XRD patterns of precipitates at five pH / Mg:P:N conditions plus struvite and NaCl reference cards: "
          "seven stacked panels sharing the 2-theta axis, each with its own intensity axis",
    duplicates=["test_data/stacked_spectrum_3.png"],
    style={"font": "serif", "weight": "normal", "tick_px": 17, "title_px": 22, "tick_dir": "in", "tick_len": 4, "frame_lw": 1.2},
    panels=[_sus_panel(i) for i in range(7)],
    annotations=[{"text": "Intensity [Count]", "xy": (14, 303), "size_px": 23, "rotation": 90, "ha": "center"}])

# ---------------------------------------------------------------- two EDS spectra side by side
def _eds(pid, fr, x, y, yt, txt, title, ann):
    return dict(id=pid, role="panel", frame=fr, x=x, y=y, x_ticks=[1, 2, 3, 4], y_ticks=yt, panel_title=title, grid=True,
                exclude=[(a - 2, b + 2, c - 2, d + 2) for a, b, c, d in txt], tick_px=13, tick_dir="out",
                series={"EDS spectrum": dict(kind="line", match={"hue": 0, "hue_tol": 15, "min_sat": 0.5, "min_val": 0.25},
                                             color="#B01010", lw=1.2, spectrum=True, peaks="up", samples=600, fill_to=0)},
                texts=ann)


NE2_L = [(112, 212, 53, 70), (49, 68, 83, 100), (61, 94, 154, 176), (99, 113, 154, 172), (113, 152, 154, 173), (141, 156, 198, 216),
         (162, 172, 198, 216), (174, 226, 198, 216), (38, 60, 14, 36)]
NE2_R = [(332, 448, 53, 70), (289, 308, 78, 96), (306, 340, 130, 152), (347, 357, 130, 148), (358, 378, 130, 148), (377, 411, 130, 149),
         (383, 399, 198, 216), (405, 416, 198, 216), (417, 469, 198, 216)]
add("nutrient_energy_2", file="Nutrient-energy-2.png", size=None, chart_type="panels_1x2:spectrum",
    title="EDS spectra of standard and recovered struvite: two panels side by side, each with its own y axis "
          "(element peaks O, Mg, P; printed Mg / P percentages)",
    duplicates=["test_data/stacked_spectrum_1.png"],
    style={"font": "sans", "weight": "normal", "tick_px": 13, "title_px": 13, "tick_dir": "out", "tick_len": 4, "frame_lw": 1.0},
    panels=[_eds("standard", (24.0, 239.5, 13.0, 364.0), (67.65, 1, 232.35, 4), (364.71, 0, 37.67, 35), list(range(0, 36, 5)), NE2_L,
                 "Standard", [("C", (48, 26), 20, "#000000"), ("Standard", (162, 62), 20, "#A01010"), ("O", (58, 92), 20, "#A01010"),
                              ("Mg 9.8%", (106, 165), 19, "#A01010"), ("P 13.4%", (183, 207), 19, "#A01010")]),
            _eds("recovered", (264.0, 475.0, 13.0, 363.5), (307.6, 1, 471.4, 4), (364.43, 0, 45.57, 50), list(range(0, 51, 10)), NE2_R,
                 "Recovered", [("Recovered", (390, 62), 20, "#A01010"), ("O", (298, 87), 20, "#A01010"),
                               ("Mg 12.7%", (358, 140), 19, "#A01010"), ("P 15.2%", (426, 207), 19, "#A01010")])],
    printed_values={"standard": {"Mg (%)": 9.8, "P (%)": 13.4}, "recovered": {"Mg (%)": 12.7, "P (%)": 15.2}},
    annotations=[{"text": "cps/eV", "xy": (4, 6), "size_px": 10}, {"text": "cps/eV", "xy": (246, 6), "size_px": 10}])

# ---------------------------------------------------------------- two XRD patterns stacked, shared x axis, own y axes
add("nutrient_energy_3", file="Nutrient-energy-3.png", size=None, chart_type="stacked_panels:spectrum",
    title="XRD of recovered (upper panel) and standard (lower panel) struvite: two stacked panels sharing the 2-theta axis",
    duplicates=["test_data/stacked_spectrum_2.png"],
    style={"font": "sans", "weight": "bold", "tick_px": 21, "title_px": 22, "tick_dir": "in", "tick_len": 5, "frame_lw": 1.5},
    panels=[dict(id="recovered", role="panel", frame=(82.5, 511.0, 14.5, 207.0), x=(83.41, 10, 511.02, 70), y=(207.0, 0, 14.5, 2500),
                 y_ticks=[500, 1000, 1500, 2000, 2500], x_ticks=list(range(10, 71, 10)), no_x_labels=True,
                 hide_spines=["bottom"], label={"text": "D", "xy": (92, 30), "size_px": 22},
                 exclude=[(212, 508, 16, 66), (86, 118, 16, 50)],
                 series={"Recovered Crystal Peak": dict(kind="line", match={"dark": True, "max_gray": 130, "max_sat": 40},
                                                        color="#222222", lw=1.2, spectrum=True, peaks="up", samples=800,
                                                        frame_pad=1, x_px=(93, 505),
                                                        exclude=[(82, 512, 12, 18), (82, 512, 206, 212)])}),
            dict(id="standard", role="panel", frame=(82.5, 511.0, 207.0, 362.0), x=(83.41, 10, 511.02, 70), y=(360.5, 0, 206.5, 2000),
                 y_ticks=[0, 500, 1000, 1500, 2000], y_color="#E01010", x_ticks=list(range(10, 71, 10)), x_title="Two Theta Degree",
                 series={"Standard Crystal Peak": dict(kind="line", match={"hue": 0, "hue_tol": 15, "min_sat": 0.45},
                                                       color="#E01010", lw=1.2, spectrum=True, peaks="up", samples=800,
                                                       frame_pad=4, x_px=(86, 508))})],
    fig_legend={"xy": (215, 17), "ncol": 1, "size_px": 19, "entries": [("Stardard Crystal Peak", "line", "#E01010"),
                                                                     ("Recovered Crystal Peak", "line", "#222222")]},
    annotations=[{"text": "Intensity (Counts)", "xy": (14, 190), "size_px": 22, "rotation": 90, "ha": "center", "weight": "bold"}])
