from .profiles import add

GRAD = {"nonwhite": 225}

add("bar_1", file="test_data/bar_1.png", size=(549, 413), chart_type="bar",
    title="Single-series bar chart with error bars (gradient fills)",
    style={"font": "serif", "weight": "bold", "tick_px": 26, "title_px": 28, "tick_dir": "in", "tick_len": 6},
    panels=[dict(id="main", role="main", frame=(85.5, 536.5, 18.5, 372.5),
                 categories=["MBC", "FBM", "SBM"], x=(178.0, 0, 443.5, 2), y=(371.95, 0, 17.88, 15),
                 y_ticks=[0, 3, 6, 9, 12, 15], x_ticks=[0, 1, 2], x_ticklabels=["MBC", "FBM", "SBM"],
                 y_title="R$_\\mathregular{m}$ (mL CH$_\\mathregular{4}$/g VSS·d$^\\mathregular{-1}$)",
                 label={"text": "(b)", "xy": (104, 60)},
                 exclude=[(95, 135, 40, 80)],
                 series={"Rm": dict(kind="bar", match=GRAD, width_px=67, err=True, color="#888888",
                                    fill="#909090", qa_limit_px=8,
                                    gradients=[((60, 60, 60), (200, 200, 200)), ((40, 60, 150), (180, 190, 230)), ((235, 30, 40), (245, 170, 170))])})])

GR3 = [((60, 60, 60), (200, 200, 200)), ((40, 60, 150), (180, 190, 230)), ((235, 30, 40), (245, 170, 170))]
for fid, fr, y, cx, w, lab, yt, yt_ticks, base in [
        ("bar_2", (83.5, 565.5, 16.5, 395.0), (394.88, 0, 16.45, 50), (164.5, 485.5), 82, "(c)", "F420 (IU/L)", [0, 10, 20, 30, 40, 50], 0),
        ("bar_3", (90.5, 573.5, 15.5, 395.0), (394.88, 70, 15.45, 120), (171.5, 493.5), 82, "(d)", "INT-ETS (%)", [70, 80, 90, 100, 110, 120], 70)]:
    add(fid, file=f"test_data/{fid}.png", size=None, chart_type="bar",
        title="Single-series bar chart with error bars (gradient fills)",
        style={"font": "serif", "weight": "bold", "tick_px": 26, "title_px": 28, "tick_dir": "in", "tick_len": 6},
        panels=[dict(id="main", role="main", frame=fr, categories=["MBC", "FBM", "SBM"], x=(cx[0], 0, cx[1], 2), y=y,
                     y_ticks=yt_ticks, x_ticks=[0, 1, 2], x_ticklabels=["MBC", "FBM", "SBM"], y_title=yt,
                     label={"text": lab, "xy": (fr[0] + 18, fr[2] + 30)}, exclude=[(fr[0] + 5, fr[0] + 60, fr[2] + 10, fr[2] + 55)],
                     series={yt.split(" ")[0]: dict(kind="bar", match=GRAD, width_px=w, err=True, color="#888888",
                                                    base=base, gradients=GR3, qa_limit_px=8)})])

add("bar_5", file="test_data/bar_5.png", size=None, chart_type="bar", duplicate_of="ph",
    title="pH of three treatments (single series, error bars)",
    style={"font": "sans", "weight": "bold", "tick_px": 27, "xtick_px": 36, "title_px": 40, "tick_dir": "out", "tick_len": 5},
    panels=[dict(id="main", role="main", frame=(88.5, 657.5, 11.5, 471.0), categories=["CK", "OPT-HC", "OPT-HC-N/P"],
                 x=(183.5, 0, 562.5, 2), y=(470.53, 0, 38.8, 8), y_ticks=list(range(9)), x_ticks=[0, 1, 2],
                 x_ticklabels=["CK", "OPT-HC", "OPT-HC-N/P"], y_title="pH",
                 label={"text": "(a)", "xy": (3, 16), "weight": "normal", "size_px": 24},
                 series={"pH": dict(kind="bar", match={"rgb": (255, 207, 149), "tol": 45}, width_px=78, err=True,
                                    color="#FFCF95", fill="#FFCF95", edge="#000000")})])

add("bar_6", file="test_data/bar_6.png", size=None, chart_type="bar",
    title="Degree of disintegration, two tests (single series, error bars)",
    style={"font": "serif", "weight": "normal", "tick_px": 19, "title_px": 19, "tick_dir": "out", "tick_len": 5},
    panels=[dict(id="main", role="main", frame=(56.0, 463.5, 9.5, 338.0), categories=["Control test", "CER-assistant test"],
                 x=(158.0, 0, 361.5, 1), y=(338.29, 0, 9.21, 50), y_ticks=[0, 10, 20, 30, 40, 50], x_ticks=[0, 1],
                 x_ticklabels=["Control test", "CER-assistant test"], y_title="DD$_\\mathregular{M}$ (%)",
                 label={"text": "(a)", "xy": (2, 50)},
                 series={"DDM": dict(kind="bar", match={"rgb": (139, 253, 253), "tol": 50}, width_px=43, err=True,
                                     color="#8BFDFD", fill="#8BFDFD", edge="#000000")})])

CYAN = dict(match={"rgb": (139, 253, 253), "tol": 50}, color="#8BFDFD", fill="#8BFDFD", edge="#000000")
MAROON = dict(match={"rgb": (86, 10, 12), "tol": 50}, color="#560A0C", fill="#560A0C", edge="#000000")

add("bar_grouped_5", file="test_data/bar_grouped_5.png", size=None, chart_type="bar_grouped",
    title="Two series, two tests (grouped bars, error bars)",
    style={"font": "serif", "weight": "normal", "tick_px": 20, "title_px": 20, "tick_dir": "out", "tick_len": 5},
    panels=[dict(id="main", role="main", frame=(60.5, 471.0, 10.5, 341.5), categories=["Control test", "CER-assistant test"],
                 x=(163.5, 0, 368.5, 1), y=(341.44, 0, 10.44, 400), y_ticks=list(range(0, 401, 50)), x_ticks=[0, 1],
                 x_ticklabels=["Control test", "CER-assistant test"], y_title="Concentrations (mg/L)",
                 label={"text": "(b)", "xy": (2, 30)}, exclude=[(100, 230, 25, 110)],
                 legend={"xy": (106, 28), "size_px": 20, "frame": False},
                 series={"NH$_4^+$-N": dict(kind="bar", offset_px=-21, width_px=42, err=True, **CYAN),
                         "TP": dict(kind="bar", offset_px=21, width_px=42, err=True, **MAROON)})])

add("bar_grouped_6", file="test_data/bar_grouped_6.png", size=None, chart_type="bar_grouped_dual_axis",
    title="TOC (left axis) and TN (right axis) for three solutions (grouped bars, error bars)",
    style={"font": "serif", "weight": "normal", "tick_px": 20, "title_px": 22, "tick_dir": "out", "tick_len": 5},
    panels=[dict(id="main", role="main", frame=(75.5, 597.5, 13.5, 433.5),
                 categories=["Deionized water", "HCl solution", "NaCl solution"],
                 x=(163.0, 0, 511.0, 2), y=(433.66, 0, 13.8, 200), y_right=(433.76, 0, 13.74, 900),
                 y_ticks=list(range(0, 201, 20)), y_right_ticks=list(range(0, 901, 100)), x_ticks=[0, 1, 2],
                 x_ticklabels=["Deionized water", "HCl solution", "NaCl solution"],
                 y_title="TOC (mg/L)", y_right_title="TN (mg/L)", xtick_px=22,
                 label={"text": "(a)", "xy": (2, 40)}, exclude=[(100, 230, 25, 110)],
                 legend={"xy": (104, 30), "size_px": 22, "frame": False},
                 series={"TOC": dict(kind="bar", axis="left", offset_px=-26.5, width_px=53, err=True, **CYAN),
                         "TN": dict(kind="bar", axis="right", offset_px=26.5, width_px=53, err=True, **MAROON)})])

ORANGE = dict(match={"rgb": (254, 205, 147), "tol": 40}, color="#FECD93", fill="#FECD93", edge="#000000")
GREEN_H = dict(match={"rgb": (155, 213, 155), "tol": 40}, color="#9BD59B", fill="#9BD59B", edge="#000000",
               hatch="///", hatch_color="#5E9A5E")
PURPLE_H = dict(match={"rgb": (198, 180, 213), "tol": 30}, color="#C6B4D5", fill="#C6B4D5", edge="#000000",
                hatch="\\\\\\", hatch_color="#7E6E8E")

add("bar_grouped_3", file="test_data/bar_grouped_3.png", size=None, chart_type="bar_grouped_dual_axis",
    title="Plant height (left axis) and fresh weight (right axis), three treatments",
    style={"font": "sans", "weight": "bold", "tick_px": 24, "title_px": 30, "tick_dir": "out", "tick_len": 5, "frame_lw": 1.5},
    panels=[dict(id="main", role="main", frame=(92.0, 597.5, 21.5, 429.0), categories=["Plant height", "Plant fresh weight"],
                 category_axis={0: "left", 1: "right"},
                 x=(209.0, 0, 472.0, 1), y=(428.25, 0, 33.25, 35), y_right=(428.84, 0, 21.16, 24),
                 y_ticks=list(range(0, 36, 5)), y_right_ticks=list(range(0, 25, 4)), x_ticks=[0, 1],
                 x_ticklabels=["Plant height", "Plant fresh weight"], y_title="Height (cm)", y_right_title="Weight (g)",
                 y_color="#FF00FF", y_right_color="#0000FF", xtick_px=24,
                 label={"text": "(a)", "xy": (2, 22), "size_px": 28, "weight": "normal"}, exclude=[(120, 570, 25, 65)],
                 legend={"xy": (126, 30), "ncol": 3, "size_px": 18, "frame": True, "weight": "normal"},
                 series={"CK": dict(kind="bar", offset_px=-52, width_px=52, err=True, **ORANGE),
                         "OPT-HC": dict(kind="bar", offset_px=0, width_px=52, err=True, **GREEN_H),
                         "OPT-HC-N/P": dict(kind="bar", offset_px=52, width_px=52, err=True, **PURPLE_H)})])

add("bar_4", file="test_data/bar_4.png", size=None, chart_type="bar",
    title="Removal efficiency, prediction vs validation (outlined bars, one dotted)",
    style={"font": "serif", "weight": "bold", "tick_px": 18, "title_px": 18, "tick_dir": "out", "tick_len": 5},
    panels=[dict(id="main", role="main", frame=(88.5, 603.5, 8.0, 469.5), categories=["Predition", "Validation"],
                 x=(218.0, 0, 475.0, 1), y=(469.26, 0, 7.9, 100), y_ticks=list(range(0, 101, 20)), x_ticks=[0, 1],
                 x_ticklabels=["Predition", "Validation"], y_title="Removal efficiency (%)",
                 exclude=[(190, 500, 38, 53)],
                 series={"Removal efficiency": dict(kind="bar", match={"outline": True}, width_px=80, err=True,
                                                    color="#000000", fill="#FFFFFF", edge="#000000", qa=False,
                                                    hatch_per_bar=[None, ".."])})],
    annotations=[{"text": "a", "xy": (218, 47), "ha": "center", "size_px": 18, "weight": "bold"},
                 {"text": "a", "xy": (475, 47), "ha": "center", "size_px": 18, "weight": "bold"}])

# ------------------------------------------------------------- broken y axes
B8 = [("COD", (129, 158, 140)), ("NH4-N Removal", (205, 55, 84)), ("NH4-N Recovery", (222, 142, 141)),
      ("Struvite", (213, 117, 33)), ("Treated Centrate", (62, 63, 190))]
add("bar_8", file="test_data/bar_8.png", size=None, chart_type="bar_broken_axis",
    title="Specific energy consumption per normalized unit (one bar per unit, broken y axis, error bars)",
    style={"font": "sans", "weight": "bold", "tick_px": 19, "title_px": 20, "tick_dir": "out", "tick_len": 5, "frame_lw": 1.5},
    panels=[dict(id="main", role="main", frame=(87.0, 618.5, 19.0, 447.0), categories=[n for n, _ in B8],
                 x=(165.5, 0, 540.0, 4), y_upper=(162.8, 5, 18.95, 8), y_lower=(446.0, 0, 245.2, 2), split_px=190, break_gap_px=6,
                 y_upper_ticks=[5, 6, 7, 8], y_lower_ticks=[0.0, 0.5, 1.0, 1.5, 2.0],
                 y_lower_ticklabels=["0.0", "0.5", "1.0", "1.5", "2.0"], x_ticks=[], x_title="Normalized Unit",
                 y_title="SEC (kWh kg$^{-1}$ or m$^{-3}$)", y_title_dx=62,
                 label={"text": "B", "xy": (96, 24), "size_px": 22}, exclude=[(95, 560, 20, 100)],
                 legend={"xy": (198, 22), "ncol": 2, "size_px": 17, "order": [B8[0][0], B8[3][0], B8[4][0], B8[2][0], B8[1][0]]},
                 series={n: dict(kind="bar", match={"rgb": c, "tol": 40}, at=[i], width_px=48, err=True,
                                 color="#%02x%02x%02x" % c, fill="#%02x%02x%02x" % c) for i, (n, c) in enumerate(B8)})])

add("bar_grouped_1", file="test_data/bar_grouped_1.png", size=None, chart_type="bar_grouped_broken_axis",
    title="Available N / P / K under three treatments (grouped bars, broken y axis, error bars)",
    style={"font": "sans", "weight": "bold", "tick_px": 22, "title_px": 30, "tick_dir": "out", "tick_len": 6, "frame_lw": 1.5},
    panels=[dict(id="main", role="main", frame=(128.5, 668.5, 11.5, 446.5), categories=["Available N", "Available P", "Available K"],
                 x=(218.5, 0, 578.0, 2), y_upper=(147.08, 1200, 23.58, 1400), y_lower=(445.45, 0, 262.3, 300), split_px=197,
                 break_gap_px=6, y_upper_ticks=[1200, 1300, 1400], y_upper_ticklabels=["1,200", "1,300", "1,400"],
                 y_lower_ticks=[0, 100, 200, 300], x_ticks=[0, 1, 2], x_ticklabels=["Available N", "Available P", "Available K"],
                 x_rotation=12, xtick_px=24, y_title="Concentration (mg/g)", y_title_dx=105, exclude=[(160, 666, 13, 56)],
                 legend={"xy": (165, 14), "ncol": 3, "size_px": 18, "frame": True, "weight": "normal"},
                 series={"CK": dict(kind="bar", offset_px=-48, width_px=46, err=True, **ORANGE),
                         "OPT-HC": dict(kind="bar", offset_px=0, width_px=46, err=True, **GREEN_H),
                         "OPT-HC-N/P": dict(kind="bar", offset_px=48, width_px=46, err=True, **PURPLE_H)})])

# ------------------------------------------------------------- literature comparison: 23 scenarios, positive and negative bars
G7 = [107, 155, 204, 252, 301, 397, 446, 494, 543, 591, 688, 737, 785, 834, 931, 980, 1029, 1077, 1125, 1173, 1222, 1318, 1367]
G7_LAB = ["AD (Malabar)", "AD (alternate)", "AD + dry. (alternate)", "Lime (North Head)", "AD (Bondi)",
          "AD + co-inc. (cement)", "AD + co-inc. (MSW)", "AD + inc.", "AD (agri.)", "AD + comp.",
          "AD", "Inc.", "Pyr. (full reuse)", "Pyr. (partial reuse)",
          "Dry. (cement)", "AD + dry. (cement)", "AD", "AD + comp.", "Lime", "AD + dry. (agri.)", "Dry. (agri.)",
          "AD + pyr.", "Pyr."]
G7_STUDY = [0] * 5 + [1] * 5 + [2] * 4 + [3] * 7 + [4] * 2
G7_STUDIES = ["Peters and Lundie, 2001", "Poulsen and Hansen, 2003", "Hospido et al., 2005", "Peters and Rowley, 2009",
              "Cao and Pawlowski, 2013"]
add("bar_grouped_7", file="test_data/bar_grouped_7.png", size=None, chart_type="bar_grouped_negative",
    title="Net energy consumption and GWP of 23 sludge-treatment scenarios from five studies (paired bars, positive and negative)",
    style={"font": "sans", "weight": "normal", "tick_px": 15, "title_px": 15, "tick_dir": "out", "tick_len": 0, "frame_lw": 0.8},
    panels=[dict(id="main", role="main", frame=(86.0, 1397.0, 11.0, 601.0), categories=[f"{G7_LAB[i]} [{G7_STUDIES[G7_STUDY[i]]}]" for i in range(23)],
                 cat_pos=G7, x=(107, 0, 155, 1), y=(601.36, -5000, 9.27, 5000), y_ticks=list(range(-5000, 5001, 1000)), x_ticks=[],
                 y_title="GWP (kg-CO$_2$e/t-DS) or Net Energy Consumption (kWh/t-DS)", grid=True, frame_color="#BBBBBB",
                 vlines=[{"x": (v - 107) / 48.6, "ls": "--", "lw": 0.8} for v in (347, 652, 898, 1265)],
                 series={"Net Energy Consumption (kWh/t-DS)": dict(kind="bar", match={"dark": True, "max_gray": 80, "max_sat": 40},
                                                                   width_px=12, offset_px=0, color="#111111", fill="#111111"),
                         "GWP (kg CO2e/t-DS)": dict(kind="bar", match={"ink": 200, "close": 3}, width_px=11, offset_px=12,
                                                    color="#444444", fill="#FFFFFF", hatch="////", hatch_color="#444444",
                                                    edge="#444444", minus=["Net Energy Consumption (kWh/t-DS)"])})],
    fig_legend={"xy": (540, 703), "ncol": 1, "size_px": 15,
                "entries": [("Net Energy Consumption (kWh/t-DS)", "line", "#111111"), ("GWP (kg CO$_2$e/t-DS)", "line", "#777777")]},
    annotations=[{"text": t, "xy": (x0, 645), "size_px": 15, "ha": "center"} for t, x0 in
                 zip(G7_STUDIES, (224, 506, 763, 1082, 1327))] +
                [{"text": G7_LAB[i].replace(" (", "\n(").replace(" + ", " +\n"), "xy": (G7[i] + 6, 255 if i in (1, 2, 3, 15, 17) else 345),
                  "size_px": 13, "ha": "center"} for i in range(23)])

# ------------------------------------------------------------- feasibility (LCA): two panels, y axes broken twice / once
# patterned bars stand among many black text labels written in the bar columns: the pixel QA counts
# that text as 'missing', so it only flags these series when more than 60 % of their pixels disagree
HTP = dict(kind="bar", match={"dark": True, "max_gray": 70, "max_sat": 40}, width_px=12, offset_px=0, color="#111111",
           fill="#111111", touch_base=True, max_row_gap=2, label="Human Toxicity Potential (HTP)")
TETP = dict(kind="bar", match={"ink": 190, "close": 2, "close_h": 14, "drop_solid": True}, width_px=12, offset_px=12, color="#555555", fill="#FFFFFF",
            hatch="////", hatch_color="#555555", edge="#333333", touch_base=True, max_row_gap=2, qa_share=0.6, label="Terrestrial Ecotoxicity Potential (TETP)")
AETP = dict(kind="bar", match={"ink": 190, "close": 2, "close_h": 14, "drop_solid": True}, width_px=12, offset_px=24, color="#555555", fill="#FFFFFF",
            hatch="..", hatch_color="#555555", edge="#333333", touch_base=True, max_row_gap=2, qa_share=0.6, label="Aquatic Ecotoxicity Potential (AETP)")
FA_POS = [115, 178, 241, 304, 367, 430, 493, 619, 746, 809, 872, 935, 998, 1061, 1188, 1251, 1314, 1377]
FA_LAB = ["Dry. (cement)", "AD + dry. (cement)", "Lime", "AD", "AD + comp.", "AD + dry. (agri.)", "Dry. (agri.)",
          "AD", "AD (dewater.) + inc.", "AD (agri.)", "AD (dewater.) (agri.)", "AD + dry.", "AD (lf.)", "AD (dewater.) (lf.)",
          "AD + dry. + gas. (biomass cogen.)", "AD + dry + gas.", "AD + dry.", "Dry."]
FA_STUDY = ["Peters and Rowley, 2009"] * 7 + ["Carballa et al., 2011"] + ["Xu et al., 2011"] * 6 + ["Buonocore et al., 2018"] * 4
FB_POS = [345.5, 525.5, 615.5]
FB_LAB = ["Carb. [Wang et al., 2013]", "Co-inc. [Usapein and Chavalparit, 2017]", "Comp. [Usapein and Chavalparit, 2017]"]
add("feasibility", file="feasibility.png", size=None, chart_type="panels:bar_grouped_multi_broken_axis",
    title="Toxicity potentials (HTP / TETP / AETP) of sludge-treatment scenarios from several studies: two panels, "
          "y axes broken into segments, positive and negative bars",
    duplicates=["test_data/stacked_bar_grouped_1.png"],
    style={"font": "sans", "weight": "normal", "tick_px": 16, "title_px": 16, "tick_dir": "out", "tick_len": 0, "frame_lw": 0.8},
    panels=[
        dict(id="toxicity_HTP_TETP_AETP", role="panel", frame=(95.0, 1420.0, 8.0, 468.0), categories=[f"{a} [{b}]" for a, b in zip(FA_LAB, FA_STUDY)],
             cat_pos=FA_POS, x=(115, 0, 178, 1), x_ticks=[], grid=True, frame_color="#BBBBBB", break_gap_px=4,
             y_segments=[{"rows": (8.0, 142.0), "map": (123.0, 100, 8.0, 400), "ticks": [100, 200, 300, 400]},
                         {"rows": (142.0, 333.0), "map": (238.0, 0, 161.0, 10), "ticks": [-10, -5, 0, 5, 10]},
                         {"rows": (333.0, 468.0), "map": (352.5, -100, 467.5, -400), "ticks": [-400, -300, -200, -100]}],
             y_title="HTP, TETP, AETP (kg 1,4-DCBe/t-DS)", y_title_dx=78,
             vlines=[{"x": (v - 115) / 63.0, "ls": "--", "lw": 0.8} for v in (543.5, 697.5, 1134)],
             series={"HTP": HTP, "TETP": TETP, "AETP": AETP}),
        dict(id="toxicity_TETP_AETP", role="panel", frame=(306.5, 667.0, 549.0, 941.5), categories=FB_LAB, cat_pos=FB_POS,
             x=(345.5, 0, 525.5, 1), x_ticks=[], grid=True, frame_color="#BBBBBB", break_gap_px=4,
             y_segments=[{"rows": (549.0, 725.0), "map": (706.0, 100000, 549.0, 500000), "ticks": [100000, 200000, 300000, 400000, 500000],
                          "ticklabels": ["100,000", "200,000", "300,000", "400,000", "500,000"]},
                         {"rows": (725.0, 941.5), "map": (824.0, 0, 745.0, 20000), "ticks": [-30000, -20000, -10000, 0, 10000, 20000],
                          "ticklabels": ["-30,000", "-20,000", "-10,000", "0", "10,000", "20,000"]}],
             y_title="TETP, AETP (kg TEG/t-DS)", y_title_dx=95,
             vlines=[{"x": (453 - 345.5) / 180.0, "ls": "--", "lw": 0.8}],
             series={"TETP": dict(TETP, offset_px=0), "AETP": dict(AETP, offset_px=12)})],
    fig_legend={"xy": (875, 855), "ncol": 1, "size_px": 16,
                "entries": [("Human Toxicity Potential (HTP)", "line", "#111111"), ("Terrestrial Ecotoxicity Potential (TETP)", "line", "#666666"),
                            ("Aquatic Ecotoxicity Potential (AETP)", "dashed", "#666666")]},
    annotations=[{"text": t, "xy": (x0, 488), "size_px": 16, "ha": "center"} for t, x0 in
                 (("Peters and Rowley, 2009", 310), ("Carballa et al., 2011", 619), ("Xu et al., 2011", 918), ("Buonocore et al., 2018", 1280))] +
                [{"text": "Wang et al.,\n2013", "xy": (350, 975), "size_px": 16, "ha": "center"},
                 {"text": "Usapein and Chavalparit,\n2017", "xy": (555, 975), "size_px": 16, "ha": "center"}])
