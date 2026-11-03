"""Registry of the real figures in ``../data`` (6 line/scatter plots, 4 bar charts, 2 pie charts).

For each figure this file holds three things, all derived from the original
image's own pixels (frame lines, tick marks, colors were measured, not guessed):

* ``calib``  - plot-frame box and the pixel->value map of every axis
               (two tick marks per axis), used by the classical digitizer *and*
               by the renderer, so a rebuilt plot puts every axis at exactly the
               same pixel position as the original.
* ``series`` - how each data series looks (color, marker, line) and how the
               classical digitizer should find it.
* ``prompt`` - the plain-language description handed to every VLM (axis titles,
               series names and how to recognise them). It never contains data
               values.

Image file names are the user's original names; ``short`` ids are used for
output files.
"""
from __future__ import annotations

FIGURES = {
    # ------------------------------------------------------------------ FTIR
    "ftir": {
        "file": "FTIR.png",
        "title": "FTIR spectra (stacked, offset curves, no y-ticks)",
        "chart_type": "spectrum",
        "size": (493, 437),
        "calib": {
            "frame": (35, 481, 11, 370),            # left, right, top, bottom (px)
            "x": (35, 4000, 469, 500),              # px0, val0, px1, val1
            "y": (370, 0, 11, 100),                 # arbitrary units: % of plot height
        },
        "series": {
            "D-HZO-P":   {"kind": "line", "color": "#0000FF", "hue": 240, "lw": 1.0},
            "D-HZO":     {"kind": "line", "color": "#00FF00", "hue": 120, "lw": 1.0},
            "Diatomite": {"kind": "line", "color": "#FF0000", "hue": 0,   "lw": 1.0},
        },
        "digitize": {"inner": (37, 480, 13, 369), "line_samples": 220},
        "x_unit": "cm-1", "y_unit": "a.u. (% of plot height)",
        "prompt": (
            "This is an FTIR spectrum figure with three vertically offset curves: a blue curve labelled "
            "'D-HZO-P' (top), a green curve labelled 'D-HZO' (middle) and a red curve labelled 'Diatomite' "
            "(bottom). The x-axis is 'Wavenumber (cm-1)' running from 4000 on the left to about 400 on the "
            "right. The y-axis 'Transmittance (%)' has NO numeric ticks, so report every y value as a "
            "percentage of the plot-frame height: 0 = bottom edge of the plot frame, 100 = top edge."
        ),
        "sample_hint": "Give each curve sampled every 200 cm-1 from 4000 down to 400.",
    },
    # ------------------------------------------------------------- pore size
    "pore": {
        "file": "Pore.png",
        "title": "Pore-size distribution (broken y-axis)",
        "chart_type": "scatter_line_broken_axis",
        "size": (554, 431),
        "calib": {
            "frame": (99, 539, 15, 369),
            "x": (99, 0, 451, 16),
            "y_upper": (159, 0.03, 51, 0.12),
            "y_lower": (363, 0.0, 208, 0.0012),
            "split_px": 189,   # rows above belong to the upper axis segment
        },
        "series": {
            "Diatomite": {"kind": "marker", "axis": "lower", "color": "#008000", "hue": 120,
                          "marker": "v", "ms": 9, "line_color": "#000000", "lw": 1.0},
            "D-HZO":     {"kind": "marker", "axis": "upper", "color": "#FF0000", "hue": 0,
                          "marker": "^", "ms": 9, "line_color": "#000000", "lw": 1.0},
        },
        "digitize": {"inner": (101, 538, 17, 368), "exclude": [(300, 380, 70, 140)],
                     "open_size": 4, "min_area": 10, "sat": 0.35},
        "x_unit": "nm", "y_unit": "cm3/g",
        "prompt": (
            "This is a pore-size distribution plot, x-axis 'Pore diameter (nm)' (0 to 20). The y-axis "
            "'dV/dD (cm3/g)' is BROKEN into two segments: the upper segment runs from about 0.01 to 0.15 and "
            "the lower segment runs from 0.0000 to about 0.0013. Series: 'D-HZO' (red upward triangles, all in "
            "the upper segment) and 'Diatomite' (green downward triangles, all in the lower segment)."
        ),
        "sample_hint": "List every marker of each series.",
    },
    # --------------------------------------------------------------- methane
    "methane": {
        "file": "Methane.png",
        "title": "Cumulative methane yield with error bars",
        "chart_type": "scatter_line_errorbar",
        "size": (603, 470),
        "calib": {
            "frame": (99, 593, 14, 388),
            "x": (99.5, 0, 577, 30),
            "y": (387.5, 0, 77, 150),
        },
        "series": {
            "CT": {"kind": "marker", "color": "#404040", "dark": True, "marker": "s", "ms": 9.5, "lw": 1.6,
                   "err_color": "#808080"},
            "RM": {"kind": "marker", "color": "#2E489F", "hue": 226, "marker": "o", "ms": 10, "lw": 1.6,
                   "err_color": "#8A9AC8"},
            "BM": {"kind": "marker", "color": "#EC1F24", "hue": 358, "marker": "^", "ms": 10, "lw": 1.6,
                   "err_color": "#F08A8C"},
        },
        "digitize": {"inner": (102, 591, 17, 386), "exclude": [(215, 500, 17, 60), (102, 160, 17, 60)],
                     "open_size": 5, "min_area": 15, "sat": 0.35,
                     "expected_x": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 26, 30],
                     "error_bars": True, "marker_r": 6},
        "x_unit": "days", "y_unit": "mL/g VSS",
        "prompt": (
            "This is a line plot with markers and error bars. x-axis 'Time (days)' (0 to 30), y-axis "
            "'Methane yield (mL/g VSS)' (0 to 180). Series: 'CT' (dark grey squares), 'RM' (blue circles), "
            "'BM' (red triangles)."
        ),
        "sample_hint": "List every marker (the mean value, not the error-bar ends) of each series.",
    },
    # ---------------------------------------------------------------- dosage
    "dosage": {
        "file": "Dosage.png",
        "title": "Dosage effect, dual y-axes",
        "chart_type": "scatter_line_dual_axis",
        "size": (687, 526),
        "calib": {
            "frame": (95, 599, 29, 436),
            "x": (118.5, 0, 571.5, 5),
            "y": (436, 0, 74.5, 160),           # left axis
            "y_right": (436, 0, 74.5, 80),      # right axis
        },
        "series": {
            "N adsorption capacity": {"kind": "marker", "axis": "left", "color": "#000000", "dark": True,
                                      "marker": "s", "ms": 7.5, "lw": 2.6},
            "P adsorption capacity": {"kind": "marker", "axis": "left", "color": "#0000FF", "hue": 240,
                                      "marker": "^", "ms": 8, "lw": 2.6},
            "N removal efficiency":  {"kind": "marker", "axis": "right", "color": "#FF00FF", "hue": 300,
                                      "marker": "s", "ms": 7.5, "lw": 2.6, "hollow": True},
            "P removal efficiency":  {"kind": "marker", "axis": "right", "color": "#FF0000", "hue": 0,
                                      "marker": "^", "ms": 8.5, "lw": 2.6, "hollow": True},
        },
        "digitize": {"inner": (98, 597, 32, 434), "exclude": [(310, 595, 100, 175), (285, 590, 315, 385)],
                     "open_size": 5, "min_area": 20, "sat": 0.2,
                     "expected_x": [0.1, 0.2, 0.4, 1, 2, 3, 4, 5]},
        "x_unit": "g/L", "y_unit": "mg/g (left) / % (right)",
        "prompt": (
            "This plot has TWO y-axes. x-axis 'Dosage (g/L)' (0 to 5). Left y-axis 'Adsorption capacity "
            "(mg/g)' (0 to 180) is used by 'N adsorption capacity' (black filled squares) and 'P adsorption "
            "capacity' (blue filled triangles). Right y-axis 'Removal efficiency (%)' (0 to 90) is used by "
            "'N removal efficiency' (magenta hollow squares) and 'P removal efficiency' (red hollow triangles). "
            "Report each series in the units of its own axis."
        ),
        "sample_hint": "List every marker of each series.",
    },
    # -------------------------------------------------------------------- N2
    "n2": {
        "file": "N2.png",
        "title": "N2 adsorption-desorption isotherm (hysteresis loop)",
        "chart_type": "scatter_line_hysteresis",
        "size": (469, 367),
        "calib": {
            "frame": (70, 449, 4, 309),
            "x": (70.3, 0, 373, 0.8),
            "y": (308.5, 0, 37.5, 40),
        },
        "series": {
            "adsorption":  {"kind": "marker", "color": "#0A0AEC", "hue": 240, "marker": "s", "ms": 5.5, "lw": 1.6},
            "desorption":  {"kind": "marker", "color": "#0A0AEC", "hue": 240, "marker": "s", "ms": 5.5, "lw": 1.6},
        },
        "digitize": {"inner": (73, 448, 6, 306), "open_size": 5, "min_area": 8, "sat": 0.4, "fill": False,
                     "branch_split_x": 0.47},
        "x_unit": "P/P0", "y_unit": "cm3/g",
        "prompt": (
            "This is a nitrogen adsorption-desorption isotherm. x-axis 'Pressure (P/P0)' (0 to 1.0), y-axis "
            "'N2 uptaken (cm3/g)' (0 to about 45). Both branches are blue squares joined by blue lines and form a "
            "hysteresis loop: call the LOWER branch 'adsorption' and the UPPER branch 'desorption' (they "
            "coincide below about P/P0 = 0.45)."
        ),
        "sample_hint": "List every marker of each branch.",
    },
    # -------------------------------------------------------------- kinetics
    "kinetics": {
        "file": "Kinetics.png",
        "title": "Adsorption kinetics: data points + three fitted model curves",
        "chart_type": "scatter_plus_fit_curves",
        "size": (701, 554),
        "calib": {
            "frame": (117, 678, 11, 464),
            "x": (117.5, 0, 636.5, 13),
            "y": (463, 0, 46.5, 120),
        },
        "series": {
            "PO4 adsorption capacity": {"kind": "marker", "color": "#000000", "dark": True, "marker": "^",
                                        "ms": 10, "lw": 0},
            "Pseudo-first-order":      {"kind": "line", "color": "#FF0000", "hue": 0, "lw": 2.4},
            "Pseudo-second-order":     {"kind": "line", "color": "#0000FF", "hue": 240, "lw": 2.4},
            "Intraparticle diffusion": {"kind": "line", "color": "#FF00FF", "hue": 300, "lw": 2.4},
        },
        "digitize": {"inner": (120, 675, 14, 461), "exclude": [(330, 670, 200, 330)],
                     "open_size": 6, "min_area": 20, "sat": 0.5,
                     "expected_x": [0.5, 1, 3, 5, 7, 9, 11, 13], "line_samples": 120,
                     "markers_through_curves": True},
        "x_unit": "h", "y_unit": "mg/g",
        "prompt": (
            "x-axis 'Reaction time (h)' (0 to 14), y-axis 'Adsorption capacity (mg/g)' (0 to 130). It shows "
            "measured data 'PO4 adsorption capacity' (black triangles) and three fitted model curves: "
            "'Pseudo-first-order' (red), 'Pseudo-second-order' (blue) and 'Intraparticle diffusion' (magenta)."
        ),
        "sample_hint": "List every black triangle, and give each curve sampled at x = 0, 1, 2, ..., 13 (stop where a curve leaves the plot).",
    },
    # ================================================================ BAR CHARTS
    # Bar data: x = category index (0, 1, 2, ... in the order of "categories"),
    # y = bar height in the units of the axis the bar is read against,
    # yerr = half-length of the error bar (0 if none).
    # ------------------------------------------------------------------ pH
    "ph": {
        "file": "pH.png",
        "title": "pH of three treatments (single-series bar chart, error bars)",
        "chart_type": "bar",
        "size": (687, 535),
        "categories": ["CK", "OPT-HC", "OPT-HC-N/P"],
        "calib": {
            "frame": (83.5, 652.5, 14.5, 474.5),
            "x": (178.5, 0, 557.5, 2),              # category centres (tick marks under the x-axis)
            "y": (474.5, 0, 41.5, 8),
        },
        "series": {
            "pH": {"kind": "bar", "fill_rgb": (255, 206, 148), "color": "#FFCE94", "tol": 40,
                   "offset_px": 0.0, "width_px": 76, "edge": "#000000"},
        },
        "digitize": {"exclude": [], "error_bars": True},
        "x_unit": "treatment", "y_unit": "pH",
        "prompt": (
            "This is a bar chart with ONE series: the pH (y-axis 'pH', 0 to 8) of three treatments on the "
            "x-axis: 'CK', 'OPT-HC' and 'OPT-HC-N/P' (one light-orange bar each, with a small black error bar "
            "on top)."
        ),
        "sample_hint": "Give the height of every bar (the top of the bar, not the end of the error bar).",
    },
    # ---------------------------------------------------- soil concentration
    "concentration": {
        "file": "concentration.png",
        "title": "Soil nutrients (grouped bars, 3 series, broken y-axis, error bars)",
        "chart_type": "bar_grouped_broken_axis",
        "size": (698, 559),
        "categories": ["Organic matter", "Total N", "Total P", "Total K"],
        "calib": {
            "frame": (119.5, 680.5, 8.5, 461.5),
            "x": (189.5, 0, 609.5, 3),
            "y_upper": (128.5, 100, 36, 110),        # segment above the break
            "y_lower": (461.5, 0, 275.5, 20),        # segment below the break
            "split_px": 226,
        },
        "series": {
            "CK":         {"kind": "bar", "fill_rgb": (255, 206, 148), "color": "#FFCE94", "tol": 40,
                           "offset_px": -37.5, "width_px": 37.5, "edge": "#000000"},
            "OPT-HC":     {"kind": "bar", "fill_rgb": (156, 214, 156), "color": "#9CD69C", "tol": 40,
                           "offset_px": 0.0, "width_px": 37.5, "edge": "#000000", "hatch": "///", "hatch_color": "#5E9A5E"},
            "OPT-HC-N/P": {"kind": "bar", "fill_rgb": (198, 182, 214), "color": "#C6B6D6", "tol": 30,
                           "offset_px": 37.5, "width_px": 37.5, "edge": "#000000", "hatch": "\\\\\\", "hatch_color": "#7E6E8E"},
        },
        "digitize": {"exclude": [(150, 660, 10, 58)], "error_bars": True},
        "x_unit": "nutrient", "y_unit": "g/kg",
        "prompt": (
            "This is a grouped bar chart. y-axis 'Concentration (g/kg)'. The y-axis is BROKEN: the lower part runs "
            "from 0 to about 25 and the upper part from about 92 to 112. x-axis categories: 'Organic matter', "
            "'Total N', 'Total P', 'Total K'. Three series (one bar each per category, left to right): 'CK' "
            "(light orange), 'OPT-HC' (light green, hatched) and 'OPT-HC-N/P' (light purple, hatched). Each bar "
            "has a black error bar. The 'Organic matter' bars reach the upper part of the axis."
        ),
        "sample_hint": "Give the height of every bar (the top of the bar, not the end of the error bar) in g/kg.",
    },
    # --------------------------------------------- plant quality, dual axis
    "concentration2": {
        "file": "concentration2.png",
        "title": "Plant quality (grouped bars, broken left axis + right axis, error bars)",
        "chart_type": "bar_grouped_dual_axis",
        "size": (701, 523),
        "categories": ["Vitamin C", "Chlorophyll", "Soluble protein", "Soluble sugar"],
        "category_axis": {0: "left", 1: "left", 2: "right", 3: "right"},
        "calib": {
            "frame": (107.5, 601.5, 27, 425.5),
            "x": (169.5, 0, 539.5, 3),
            "y_upper": (153.5, 100, 43.5, 120),     # left axis, above the break (mg/kg)
            "y_lower": (425.5, 0, 295.5, 5),        # left axis, below the break (mg/kg)
            "split_px": 267,
            "y_right": (425.5, 0, 27, 20),          # right axis (%)
        },
        "series": {
            "CK":         {"kind": "bar", "fill_rgb": (254, 205, 147), "color": "#FECD93", "tol": 40,
                           "offset_px": -24.5, "width_px": 24.5, "edge": "#000000"},
            "OPT-HC":     {"kind": "bar", "fill_rgb": (155, 213, 155), "color": "#9BD59B", "tol": 40,
                           "offset_px": 0.0, "width_px": 24.5, "edge": "#000000", "hatch": "///", "hatch_color": "#5E9A5E"},
            "OPT-HC-N/P": {"kind": "bar", "fill_rgb": (198, 180, 213), "color": "#C6B4D5", "tol": 30,
                           "offset_px": 24.5, "width_px": 24.5, "edge": "#000000", "hatch": "\\\\\\", "hatch_color": "#7E6E8E"},
        },
        "digitize": {"exclude": [(128, 590, 28, 66)], "error_bars": True},
        "x_unit": "quality index", "y_unit": "mg/kg (left) / % (right)",
        "prompt": (
            "This is a grouped bar chart with TWO y-axes. Left y-axis (magenta) 'Concentration (mg/kg)' is BROKEN: "
            "lower part 0 to about 6, upper part about 82 to 122; it is used by the categories 'Vitamin C' and "
            "'Chlorophyll' (inside the magenta dashed box). Right y-axis (blue) 'Content (%)' runs from 0 to 20 and "
            "is used by 'Soluble protein' and 'Soluble sugar' (inside the blue dashed box; printed as 'Soluble "
            "suger'). Three series, one bar each per category, left to right: 'CK' (light orange), 'OPT-HC' "
            "(light green, hatched), 'OPT-HC-N/P' (light purple, hatched), all with black error bars. Report each "
            "bar in the units of its own axis."
        ),
        "sample_hint": "Give the height of every bar (the top of the bar, not the end of the error bar).",
    },
    # ------------------------------------------------- LCA impact categories
    "climet": {
        "file": "climet.png",
        "title": "Impact categories, 2 scenarios (18 categories, 0-100 %)",
        "chart_type": "bar_grouped",
        "size": (1133, 757),
        "categories": ["Climate change", "Ozone depletion", "Terrestrial acidification", "Freshwater eutrophication",
                       "Marine eutrophication", "Human toxicity", "Photochemical oxidant formation",
                       "Particulate matter formation", "Terrestrial ecotoxicity", "Freshwater ecotoxicity",
                       "Marine ecotoxicity", "Ionising radiation", "Agricultural land occupation",
                       "Urban land occupation", "Natural land transformation", "Water depletion",
                       "Metal depletion", "Fossil depletion"],
        "calib": {
            "frame": (120, 940, 39, 464.5),          # no frame box: 0 % axis line + 100 % grid line
            "x": (143.25, 0, 916.75, 17),
            "y": (464.5, 0, 39.5, 100),
        },
        "series": {
            "Scenario 1": {"kind": "bar", "fill_rgb": (56, 109, 17), "color": "#386D11", "tol": 45,
                           "offset_px": -6.75, "width_px": 12.5, "edge": None},
            "Scenario 2": {"kind": "bar", "hue": 0, "hue_tol": 14, "min_sat": 0.25, "min_val": 0.45,
                           "color": "#E29A99", "color_bottom": "#A2302E",          # vertical gradient
                           "offset_px": 6.75, "width_px": 12.5, "edge": None},
        },
        "digitize": {"exclude": [(960, 1110, 330, 420)], "error_bars": False, "outline_px": 0.0},
        "x_unit": "impact category", "y_unit": "% (relative)",
        "prompt": (
            "This is a grouped bar chart of 18 impact categories (x-axis labels written at an angle, from "
            "'Climate change' on the left to 'Fossil depletion' on the right). y-axis from 0% to 100%. Two series: "
            "'Scenario 1' (dark green bars, left bar of each pair) and 'Scenario 2' (red/pink bars, right bar of "
            "each pair)."
        ),
        "sample_hint": "Give the height of every bar in percent (0 to 100) for all 18 categories.",
    },
    # ================================================================ PIE CHARTS
    # Pie data: one series "share (%)", x = slice index (order of "categories" =
    # clockwise from 12 o'clock), y = percentage. The slices' printed labels are
    # the reference values; the classical digitizer measures the slice angles.
    "pie1": {
        "file": "pie_1.png",
        "title": "Cost share, 3D pie (a) - 4 slices",
        "chart_type": "pie",
        "size": (664, 387),
        "categories": ["Separator", "Carbon cloth", "Carbon brush", "Others"],
        "value_series": "share (%)",
        "printed_values": [62.5, 6.6, 12.9, 18.0],
        "calib": {
            "frame": (0, 100, 0, 100), "x": (0, 0, 100, 100), "y": (100, 0, 0, 100),   # virtual 0-100 % axis
            "pie": {"apex": (364, 161), "left": 270, "right": 269, "up": 106, "down": 151, "depth": 59},
        },
        "slices": {
            "Separator":    {"rgb": (20, 0, 230),  "color": "#1400E6", "label_xy": (380, 254), "label_color": "white"},
            "Carbon cloth": {"rgb": (210, 20, 20), "color": "#D21414", "label_xy": (78, 293), "label_color": "black"},
            "Carbon brush": {"rgb": (160, 240, 70), "color": "#A0F046", "label_xy": (78, 160), "label_color": "black"},
            "Others":       {"rgb": (170, 0, 210), "color": "#AA00D2", "label_xy": (209, 33), "label_color": "black"},
        },
        "series": {"share (%)": {"kind": "pie"}},
        "panel_label": "(a)",
        "digitize": {"tol": 90},
        "x_unit": "slice", "y_unit": "%",
        "prompt": (
            "This is a 3D pie chart with four slices, each labelled with its name and percentage: 'Separator' "
            "(blue), 'Carbon cloth' (red), 'Carbon brush' (yellow-green) and 'Others' (purple)."
        ),
        "sample_hint": "Give the percentage of every slice.",
    },
    "pie2": {
        "file": "pie2.png",
        "title": "Cost share, 3D pie (b) - 4 slices",
        "chart_type": "pie",
        "size": (665, 400),
        "categories": ["Anode", "Cathode", "Separator", "Others"],
        "value_series": "share (%)",
        "printed_values": [14.08, 70.42, 1.408, 14.08],
        "calib": {
            "frame": (0, 100, 0, 100), "x": (0, 0, 100, 100), "y": (100, 0, 0, 100),
            "pie": {"apex": (330, 171), "left": 270, "right": 270, "up": 105, "down": 154, "depth": 57},
        },
        "slices": {
            "Anode":     {"rgb": (20, 0, 230),   "color": "#1400E6", "label_xy": (442, 34), "label_color": "black"},
            "Cathode":   {"rgb": (240, 240, 40), "color": "#F0F028", "label_xy": (343, 253), "label_color": "black"},
            "Separator": {"rgb": (210, 20, 20),  "color": "#D21414", "label_xy": (82, 86), "label_color": "black"},
            "Others":    {"rgb": (160, 240, 70), "color": "#A0F046", "label_xy": (213, 34), "label_color": "black"},
        },
        "series": {"share (%)": {"kind": "pie"}},
        "panel_label": "(b)",
        "digitize": {"tol": 90},
        "x_unit": "slice", "y_unit": "%",
        "prompt": (
            "This is a 3D pie chart with four slices, each labelled with its name and percentage: 'Anode' (blue), "
            "'Cathode' (yellow), 'Separator' (a very thin red slice) and 'Others' (light green)."
        ),
        "sample_hint": "Give the percentage of every slice.",
    },
}

ORDER = ["ftir", "pore", "methane", "dosage", "n2", "kinetics",
         "ph", "concentration", "concentration2", "climet", "pie1", "pie2"]


def y_axes(fig_id):
    """Names of the y-axes of a figure -> calibration key."""
    c = FIGURES[fig_id]["calib"]
    out = {}
    for k in ("y", "y_right", "y_upper", "y_lower"):
        if k in c:
            out[{"y": "left", "y_right": "right", "y_upper": "upper", "y_lower": "lower"}[k]] = k
    return out


def series_axis(fig_id, name, x=None, y=None):
    """Which y-axis a series (or one point of it) is read against:
    'left' | 'right' | 'upper' | 'lower'. Bar charts can assign an axis per
    category (dual-axis bar charts); on a broken axis the value decides the
    segment."""
    F = FIGURES[fig_id]
    s = F["series"][name]
    ax = s.get("axis", "left")
    if x is not None and "category_axis" in F:
        ax = F["category_axis"].get(int(round(float(x))), ax)
    c = F["calib"]
    if ax == "left" and "y_upper" in c and "y" not in c and y is not None:
        from .digitize import Axis
        top_of_lower = float(Axis(*c["y_lower"]).to_value(c["split_px"]))
        ax = "upper" if y > top_of_lower else "lower"
    return ax


def axis_span(fig_id, name, x=None, y=None):
    """Value span of the axis a series (point) is plotted against (used to normalise errors)."""
    c = FIGURES[fig_id]["calib"]
    ax = series_axis(fig_id, name, x, y)
    if ax == "left" and "y" not in c and "y_lower" in c:
        ax = "lower"
    key = {"left": "y", "right": "y_right", "upper": "y_upper", "lower": "y_lower"}[ax]
    if key not in c:
        key = "y"
    p0, v0, p1, v1 = c[key]
    T, B = c["frame"][2], c["frame"][3]
    if ax == "upper":
        B = c["split_px"]
    if ax == "lower":
        T = c["split_px"]
    per_px = (v1 - v0) / (p1 - p0)
    return abs(per_px * (B - T))


def is_categorical(fig_id):
    return "categories" in FIGURES[fig_id]
