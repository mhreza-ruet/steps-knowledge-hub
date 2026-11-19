"""scatter plots and spectra"""
from .profiles import add

BLK = {"dark": True, "max_gray": 100, "max_sat": 45}


def pct(fr):
    """no numeric y ticks: y as % of the plot-frame height (0 = bottom, 100 = top)"""
    return (fr[3], 0, fr[2], 100)


# ------------------------------------------------------------- PCA score plots
for fid, fr, x, y, xt, yt, xl, yl, ttl in [
        ("scatter_1", (70.0, 554.0, 10.0, 458.0), (124.0, -40, 501.0, 30), (439.77, -50, 10.23, 50), list(range(-50, 41, 10)),
         list(range(-50, 51, 10)), "PC1(32.49%)", "PC2(23.72%)", None),
        ("scatter_2", (55.0, 506.0, 10.0, 457.0), (55.07, -6, 507.02, 14), (439.62, -8, 9.49, 8), list(range(-6, 15, 2)),
         list(range(-8, 9, 2)), "PC1(41.09%)", "PC2(28.88%)", None)]:
    add(fid, file=f"test_data/{fid}.png", size=None, chart_type="scatter",
        title="PCA score plot: three groups of two points (b / s), dashed guide lines",
        style={"font": "serif", "weight": "bold", "tick_px": 15, "title_px": 18, "tick_dir": "in", "tick_len": 4},
        panels=[dict(id="main", role="main", frame=fr, x=x, y=y, x_ticks=xt, y_ticks=yt, x_title=xl, y_title=yl,
                     vlines=[{"x": 0, "ls": ":", "lw": 1.0}], hlines=[{"y": 0, "ls": ":", "lw": 1.0}],
                     legend={"xy": (fr[1] + 8, fr[2] + 5), "size_px": 16, "weight": "normal"} if fid == "scatter_2" else None,
                     series={"MBC": dict(kind="marker", match={"rgb": (12, 60, 132), "tol": 55}, color="#0C3C84", marker="o", ms_px=10),
                             "FBM": dict(kind="marker", match={"rgb": (225, 10, 10), "tol": 45, "min_sat": 150}, color="#E10A0A",
                                         marker="o", ms_px=10),
                             "SBM": dict(kind="marker", match={"rgb": (60, 180, 60), "tol": 60}, color="#3CB43C", marker="o", ms_px=10)})])

add("scatter_3", file="test_data/scatter_3.png", size=None, chart_type="scatter_timeseries",
    title="Current over 5 cycles (dense hollow-square markers, cycle separators)",
    style={"font": "sans", "weight": "bold", "tick_px": 19, "title_px": 22, "tick_dir": "out", "tick_len": 5, "frame_lw": 1.5},
    panels=[dict(id="main", role="main", frame=(54.0, 527.5, 14.5, 399.5), x=(54.32, 0, 527.14, 30), y=(399.5, 0, 69.5, 6),
                 x_ticks=list(range(0, 31, 3)), y_ticks=list(range(0, 8)), x_title="Time (day)", y_title="Current (mA)",
                 label={"text": "A", "xy": (64, 35), "size_px": 24}, exclude=[(56, 525, 360, 397), (58, 90, 18, 55)],
                 vlines=[{"x": v, "ls": "--", "lw": 1.3} for v in (5.0, 10.0, 16.0, 22.0)],
                 series={"Current": dict(kind="marker", match={"dark": True, "max_gray": 110, "max_sat": 40}, color="#000000",
                                         marker="s", ms_px=12, hollow=True, face="none", mew=1.6)})],
    annotations=[{"text": f"Cycle {i + 1}", "xy": (x0, 380), "size_px": 15, "weight": "bold", "ha": "center"}
                 for i, x0 in enumerate((85, 165, 252, 337, 450))])

add("scatter_4", file="test_data/scatter_4.png", size=None, chart_type="scatter_categorical",
    title="NH4-N specific RSF per cycle (one hollow-square marker per category)",
    style={"font": "sans", "weight": "bold", "tick_px": 16, "title_px": 15, "tick_dir": "out", "tick_len": 4, "frame_lw": 1.3},
    panels=[dict(id="main", role="main", frame=(80.5, 586.5, 21.5, 229.5), categories=[f"Cycle {i}" for i in range(1, 6)],
                 cat_pos=[121.0, 222.0, 323.0, 424.0, 526.0], x=(121.0, 0, 222.0, 1), y=(228.75, 0, 21.75, 2.0),
                 y_ticks=[0.0, 0.5, 1.0, 1.5, 2.0], y_ticklabels=["0.0", "0.5", "1.0", "1.5", "2.0"],
                 x_ticks=[0, 1, 2, 3, 4], x_ticklabels=[f"Cycle {i}" for i in range(1, 6)], y_title="NH$_4$-N Specific RSF (g L$^{-1}$)",
                 label={"text": "C", "xy": (90, 40), "size_px": 22}, exclude=[(85, 120, 25, 55), (180, 470, 25, 55)],
                 legend={"xy": (184, 28), "size_px": 15},
                 series={"NH4-N Specific RSF": dict(kind="marker", match={"dark": True, "max_gray": 110, "max_sat": 40},
                                                    color="#000000", marker="s", ms_px=12, hollow=True, face="none",
                                                    expected_x=[0, 1, 2, 3, 4])})])

add("scatter_5", file="test_data/scatter_5.png", size=None, chart_type="scatter_timeseries_errorbar",
    title="Effluent PO4-P over 5 cycles, anode (markers with error bars) and cathode (x as % of the axis: no numeric x ticks)",
    style={"font": "sans", "weight": "bold", "tick_px": 17, "title_px": 18, "tick_dir": "out", "tick_len": 4, "frame_lw": 1.3},
    panels=[dict(id="main", role="main", frame=(60.5, 488.5, 12.0, 359.5), x=(60.5, 0, 488.5, 100), y=(359.05, 0, 11.37, 90),
                 x_unit="% of the x axis", y_ticks=list(range(0, 91, 15)),
                 x_ticks=[(v - 60.5) / 4.28 for v in (98, 186, 264, 357, 438)], x_ticklabels=[f"Cycle {i}" for i in range(1, 6)],
                 y_title="Effluent PO$_4$-P (mg L$^{-1}$)", label={"text": "A", "xy": (70, 30), "size_px": 20},
                 exclude=[(300, 486, 15, 80), (64, 90, 15, 45)],
                 vlines=[{"x": (v - 60.5) / 4.28, "ls": "--", "lw": 1.3} for v in (136, 208, 297, 382)],
                 legend={"xy": (318, 20), "size_px": 17, "labelspacing": 0.9},
                 series={"Anode PO$_4$-P": dict(kind="marker", match={"rgb": (204, 30, 40), "tol": 70, "min_sat": 80}, color="#CC1E28",
                                                marker="o", ms_px=13, hollow=True, face="none", err=True, mew=2.0),
                         "Cathode PO$_4$-P": dict(kind="marker", match={"dark": True, "max_gray": 100, "max_sat": 40}, color="#000000",
                                                  marker="s", ms_px=9, hollow=True, face="none", frame_pad=0,
                                                  expected_x_from="Anode PO$_4$-P", exclude=[(60, 490, 12, 335)])})])

# ------------------------------------------------------------- spectra
fr1 = (42.5, 700.0, 11.0, 625.0)
add("spectrum_1", file="test_data/spectrum_1.png", size=None, chart_type="spectrum_offset",
    title="FTIR before / after treatment (two offset curves, peak labels, no y ticks: y in % of frame height)",
    style={"font": "serif", "weight": "bold", "tick_px": 17, "title_px": 19, "tick_dir": "in", "tick_len": 5},
    panels=[dict(id="main", role="main", frame=fr1, x=(42.46, 4000, 699.79, 500), y=pct(fr1), no_y_ticks=True,
                 x_ticks=list(range(4000, 499, -500)), x_title="Wave number (cm$^{-1}$)", y_title="Transmission (%)",
                 label={"text": "(a)", "xy": (430, 45), "size_px": 22}, exclude=[(45, 260, 13, 95), (400, 470, 25, 65)],
                 legend={"xy": (55, 38), "size_px": 16},
                 series={"Before treatment": dict(kind="line", match={"dark": True, "max_gray": 90, "max_sat": 40}, color="#000000",
                                                  lw=1.6, samples=400, max_jump=30),
                         "After treatment": dict(kind="line", match={"hue": 0, "hue_tol": 12, "min_sat": 0.5}, color="#FF0000",
                                                 lw=1.6, samples=400, max_jump=30)})])

add("spectrum_2", file="test_data/spectrum_2.png", size=None, chart_type="spectrum",
    title="XRD before / after treatment (two curves, sharp peaks)",
    style={"font": "serif", "weight": "bold", "tick_px": 15, "title_px": 19, "tick_dir": "out", "tick_len": 4},
    panels=[dict(id="main", role="main", frame=(87.0, 674.0, 14.0, 629.0), x=(87.0, 0, 639.0, 80), y=(628.7, 0, 13.7, 200),
                 x_ticks=[0, 20, 40, 60, 80], y_ticks=[0, 50, 100, 150, 200], x_title="2$\\theta$ (°)", y_title="Intensity  (a.u.)",
                 label={"text": "(b)", "xy": (290, 45), "size_px": 22},
                 exclude=[(95, 290, 20, 112), (285, 332, 244, 270), (375, 422, 336, 360), (405, 485, 28, 90)],
                 legend={"xy": (108, 25), "size_px": 16},
                 series={"Before treatment": dict(kind="line", match={"dark": True, "max_gray": 90, "max_sat": 40}, color="#000000",
                                                  lw=1.5, samples=500, max_jump=40),
                         "After treatment": dict(kind="line", match={"hue": 0, "hue_tol": 12, "min_sat": 0.5}, color="#FF0000",
                                                 lw=1.5, samples=500, spectrum=True, peaks="up")})],
    annotations=[{"text": "32.06", "xy": (308, 257), "ha": "center", "size_px": 15, "weight": "bold"},
                 {"text": "45.01", "xy": (398, 348), "ha": "center", "size_px": 15, "weight": "bold"}])

fr3 = (38.5, 450.5, 12.5, 344.5)
add("spectrum_3", file="test_data/spectrum_3.png", size=None, chart_type="spectrum",
    title="FTIR, one curve with band labels (no y ticks: y in % of frame height)",
    style={"font": "sans", "weight": "bold", "tick_px": 19, "title_px": 22, "tick_dir": "out", "tick_len": 4},
    panels=[dict(id="main", role="main", frame=fr3, x=(38.0, 4000, 439.5, 500), y=pct(fr3), no_y_ticks=True,
                 x_ticks=list(range(4000, 499, -500)), x_title="Wavenumber (cm$^{-1}$)", y_title="Transmittance (%)",
                 series={"Spectrum": dict(kind="line", match={"hue": 300, "hue_tol": 20, "min_sat": 0.4}, color="#E040E0",
                                          lw=2.0, samples=400, max_jump=40)})],
    annotations=[{"text": t, "xy": xy, "size_px": 13, "ha": "center"} for t, xy in
                 (("C-H", (155, 112)), ("C=O", (238, 162)), ("N-H", (238, 182)), ("C-H", (282, 118)), ("C-N", (326, 225)),
                  ("O-H, H-bonds", (98, 318)))])

TXT = {"spectrum_4": [(290, 335, 13, 35), (388, 434, 22, 44), (221, 266, 102, 124), (160, 236, 238, 274), (244, 269, 122, 155)],
       "spectrum_5": [(407, 455, 22, 44), (144, 282, 41, 68), (120, 226, 240, 267), (183, 239, 266, 304)],
       "spectrum_6": [(202, 273, 49, 79)]}
BASE_ONLY = {"spectrum_4": 285, "spectrum_5": 300, "spectrum_6": 250}   # the background line lies below this row


def _a(t, box, size=19):
    return {"text": t, "xy": ((box[0] + box[1]) / 2, (box[2] + box[3]) / 2), "size_px": size, "weight": "bold", "ha": "center"}


ANN = {"spectrum_4": [_a("C-C", TXT["spectrum_4"][0]), _a("C1s", TXT["spectrum_4"][1]), _a("C-N", TXT["spectrum_4"][2]),
                      _a("O-C=O", TXT["spectrum_4"][3])],
       "spectrum_5": [_a("N1s", TXT["spectrum_5"][0]), _a("Amides-N", TXT["spectrum_5"][1]), _a("Pyrrole-N", TXT["spectrum_5"][2])],
       "spectrum_6": [_a("P$_2$O7$^{4-}$", TXT["spectrum_6"][0])]}
for fid, fr, x, xt, curves, ttl, lab in [
        ("spectrum_4", (37.5, 440.0, 9.5, 333.5), (64.21, 294, 439.67, 280), list(range(294, 279, -2)),
         [("Raw", {"hue": 300, "hue_tol": 20, "min_sat": 0.4}, "#E040E0", 1.0, True),
          ("Envelope", {"hue": 240, "hue_tol": 14, "min_sat": 0.5}, "#2020E0", 2.0, False),
          ("C-C", {"hue": 355, "hue_tol": 10, "min_sat": 0.5}, "#E00C24", 2.0, False),
          ("C-N", {"hue": 28, "hue_tol": 10, "min_sat": 0.5}, "#FF8424", 2.0, False),
          ("O-C=O", {"hue": 120, "hue_tol": 20, "min_sat": 0.5}, "#24E424", 2.0, False),
          ("Background", {"dark": True, "max_gray": 80, "max_sat": 40}, "#000000", 2.0, False)],
         "XPS C1s with fitted components", "(b)"),
        ("spectrum_5", (42.0, 460.5, 8.5, 345.5), (41.6, 410, 390.65, 395), [410, 405, 400, 395],
         [("Raw", {"hue": 300, "hue_tol": 12, "min_sat": 0.5}, "#FC0CFC", 1.2, True),
          ("Envelope", {"hue": 355, "hue_tol": 10, "min_sat": 0.5}, "#E40C24", 2.0, False),
          ("Pyrrole-N", {"hue": 35, "hue_tol": 12, "min_sat": 0.5}, "#FF9010", 2.0, False),
          ("Background", {"dark": True, "max_gray": 80, "max_sat": 40}, "#000000", 2.0, False)],
         "XPS N1s with fitted components", "(c)"),
        ("spectrum_6", (38.5, 441.0, 10.0, 333.5), (38.3, 145, 441.3, 125), [145, 140, 135, 130, 125],
         [("Raw", {"hue": 240, "hue_tol": 12, "min_sat": 0.5}, "#0C0CFC", 1.2, True),
          ("Fit", {"hue": 0, "hue_tol": 12, "min_sat": 0.5}, "#F01010", 2.0, False),
          ("Background", {"dark": True, "max_gray": 80, "max_sat": 40}, "#000000", 2.0, False)],
         "XPS P2p (P2O7 4-) with fit", "(d)")]:
    add(fid, file=f"test_data/{fid}.png", size=None, chart_type="spectrum_fit",
        title=ttl + " (no y ticks: y in % of frame height)",
        style={"font": "sans", "weight": "bold", "tick_px": 19, "title_px": 22, "tick_dir": "out", "tick_len": 5, "frame_lw": 1.5},
        panels=[dict(id="main", role="main", frame=fr, x=x, y=pct(fr), no_y_ticks=True, x_ticks=xt,
                     x_title="Binding Energy (eV)", y_title="Intensity (a.u.)",
                     label={"text": lab, "xy": (2, 12), "size_px": 20, "weight": "normal"},
                     exclude=TXT[fid],
                     series={n: dict(kind="line", match=m, color=c, lw=w, samples=400, max_jump=60 if raw else (10 if n == "Background" else 30),
                                     **({"vert_run": 3, "qa_keep_small": True} if raw else {}),
                                     **({"prefer": "bottom", "exclude": [(fr[0], fr[1], fr[2], BASE_ONLY[fid])]} if n == "Background" else {}))
                             for n, m, c, w, raw in curves})],
        annotations=ANN[fid])
