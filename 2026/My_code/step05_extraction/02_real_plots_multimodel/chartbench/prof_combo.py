"""bars + markers on two y-axes (bar/scatter combinations)"""
from .profiles import add

def _px2i(px, p0=108.0, step=(553.0 - 108.0) / 32):
    return round((px - p0) / step, 3)

CYC = [(18, 0, 110), (61, 60, 190), (73, 104, 229), (103, 170, 250), (125, 223, 234)]
AT = [range(0, 6), range(6, 12), range(12, 19), range(19, 26), range(26, 33)]
add("barscatter_1", file="test_data/barscatter_1.png", size=None, chart_type="bar+marker_dual_axis",
    title="Anode COD per sample (bars, 5 cycles, left axis) and anode / cathode pH (markers, right axis)",
    style={"font": "sans", "weight": "bold", "tick_px": 20, "title_px": 20, "tick_dir": "in", "tick_len": 5, "frame_lw": 1.5},
    panels=[dict(id="main", role="main", frame=(94.0, 566.5, 13.0, 398.0), categories=[str(i + 1) for i in range(33)],
                 x=(108.0, 0, 553.0, 32), y=(396.45, 0, 47.88, 2000), y_right=(396.87, 0, 12.38, 14),
                 y_ticks=list(range(0, 2001, 400)), y_right_ticks=list(range(0, 15, 2)),
                 x_ticks=[_px2i(v) for v in (142, 226, 316, 413, 511)], x_ticklabels=[f"Cycle {i}" for i in range(1, 6)],
                 xtick_px=21, y_title="Anode COD (mg L$^{-1}$)", y_right_title="pH",
                 label={"text": "B", "xy": (104, 30), "size_px": 24}, exclude=[(100, 520, 15, 45)],
                 legend={"xy": (180, 18), "ncol": 2, "size_px": 20, "order": ["Anode pH", "Cathode pH"]},
                 series={**{f"COD cycle {k + 1}": dict(kind="bar", match={"rgb": c, "tol": 35}, at=list(AT[k]), width_px=11,
                                                      err=True, max_row_gap=40, color="#%02x%02x%02x" % c, fill="#%02x%02x%02x" % c, cap_px=8,
                                                      label=f"Cycle {k + 1}")
                            for k, c in enumerate(CYC)},
                         "Anode pH": dict(kind="marker", axis="right", match={"rgb": (220, 40, 40), "tol": 80, "min_sat": 80},
                                          marker="o", ms_px=15, hollow=True, color="#DC2828", mew=2.2, expected_x=list(range(33))),
                         "Cathode pH": dict(kind="marker", axis="right", match={"dark": True, "max_gray": 110, "max_sat": 60}, marker="s",
                                            ms_px=14, hollow=True, color="#000000", mew=1.6, expected_x=list(range(33)))})])

def _ie(n=5):
    return [f"Cycle {c} {k}" for c in range(1, n + 1) for k in ("I", "E")]

BLUE = dict(match={"rgb": (62, 62, 192), "tol": 45}, color="#3E3EC0", fill="#3E3EC0")
CRIM = dict(match={"rgb": (206, 56, 85), "tol": 45}, color="#CE3855", fill="#CE3855")
RED_RING = dict(match={"rgb": (220, 50, 50), "tol": 90, "min_sat": 90}, marker="o", hollow=True, color="#DC3232", mew=2.0)
BLK_SQ = dict(match={"dark": True, "max_gray": 100, "max_sat": 60}, marker="s", hollow=True, color="#000000", mew=1.6)

cp2 = [103.5, 141.5, 205.0, 242.5, 306.0, 343.5, 406.5, 444.5, 508.0, 545.5]
add("barscatter_2", file="test_data/barscatter_2.png", size=None, chart_type="stacked_bar+marker_dual_axis",
    title="NH4-N amount in anode / cathode (stacked bars, left axis) with removal and recovery (markers, right axis)",
    style={"font": "sans", "weight": "bold", "tick_px": 21, "title_px": 22, "tick_dir": "out", "tick_len": 5, "frame_lw": 1.5},
    panels=[dict(id="main", role="main", frame=(72.0, 577.0, 18.0, 429.0), categories=_ie(), cat_pos=cp2,
                 x=(103.5, 0, 141.5, 1), y=(428.38, 0, 17.84, 32), y_right=(428.16, 0, 17.7, 18),
                 y_ticks=list(range(0, 33, 4)), y_right_ticks=list(range(0, 19, 3)),
                 x_ticks=[2 * c + 0.5 for c in range(5)],
                 x_ticklabels=[f"Cycle {c}" for c in range(1, 6)], xtick_px=22,
                 y_title="NH$_4$-N Amount (mmol)", y_right_title="Removal/Recovery (mmol)",
                 label={"text": "A", "xy": (86, 46), "size_px": 26}, exclude=[(80, 577, 20, 72)],
                 legend={"xy": (160, 22), "ncol": 2, "size_px": 18,
                         "order": ["Cathode Amount", "Cathode Recovery", "Anode Amount", "Anode Removal"]},
                 series={"Anode Amount": dict(kind="stacked", width_px=20, **BLUE),
                         "Cathode Amount": dict(kind="stacked", width_px=20, **CRIM),
                         "Anode Removal": dict(kind="marker", axis="right", ms_px=16, err=True, expected_x=[0, 2, 4, 6, 8], **RED_RING),
                         "Cathode Recovery": dict(kind="marker", axis="right", ms_px=15, err=True, expected_x=[1, 3, 5, 7, 9], **BLK_SQ)})])

cp3 = [119.0, 156.5, 220.0, 258.0, 321.5, 359.0, 422.0, 460.0, 523.5, 561.0]
add("barscatter_3", file="test_data/barscatter_3.png", size=None, chart_type="stacked_bar+marker_dual_axis",
    title="NH4-N amount in feed / draw (stacked bars, left axis) with draw NH4-N loss (markers, right axis)",
    style={"font": "sans", "weight": "bold", "tick_px": 21, "title_px": 22, "tick_dir": "out", "tick_len": 5, "frame_lw": 1.5},
    panels=[dict(id="main", role="main", frame=(87.0, 593.0, 15.5, 425.5), categories=_ie(), cat_pos=cp3,
                 x=(119.0, 0, 156.5, 1), y=(424.7, 0, 14.7, 200), y_right=(425.08, 0, 14.92, 21),
                 y_ticks=list(range(0, 201, 50)), y_right_ticks=list(range(0, 22, 3)),
                 x_ticks=[2 * c + 0.5 for c in range(5)],
                 x_ticklabels=[f"Cycle {c}" for c in range(1, 6)], xtick_px=22,
                 y_title="NH$_4$-N Amount (mmol)", y_right_title="Draw NH$_4$-N Loss (mmol)",
                 label={"text": "B", "xy": (100, 40), "size_px": 26}, exclude=[(95, 570, 18, 75)],
                 legend={"xy": (145, 20), "ncol": 2, "size_px": 18,
                         "order": ["NH4-N @ Feed", "NH4-N @ Draw", "Draw NH4-N Loss"]},
                 series={"NH4-N @ Draw": dict(kind="stacked", width_px=21, **BLUE),
                         "NH4-N @ Feed": dict(kind="stacked", width_px=21, **CRIM),
                         "Draw NH4-N Loss": dict(kind="marker", axis="right", ms_px=16, err=True, split=False, exclude=[(88, 593, 214, 426)], **RED_RING)})])

cp4 = [106.5, 139.0, 192.0, 224.5, 278.0, 310.0, 363.5, 396.0, 449.5, 481.5]
add("barscatter_4", file="test_data/barscatter_4.png", size=None, chart_type="stacked_bar+marker_dual_axis",
    title="PO4-P amount at feed / draw (stacked bars, left axis) with initial and end P concentration (markers, right axis)",
    style={"font": "sans", "weight": "bold", "tick_px": 17, "title_px": 18, "tick_dir": "out", "tick_len": 4, "frame_lw": 1.3},
    panels=[dict(id="main", role="main", frame=(80.0, 508.5, 13.5, 361.0), categories=_ie(), cat_pos=cp4,
                 x=(106.5, 0, 139.0, 1), y=(360.36, 0, 12.76, 0.4), y_right=(360.02, 0, 12.81, 200),
                 y_ticks=[round(0.05 * k, 2) for k in range(9)], y_ticklabels=[f"{0.05 * k:.2f}" for k in range(9)],
                 y_right_ticks=list(range(0, 201, 40)),
                 x_ticks=[2 * c + 0.5 for c in range(5)],
                 x_ticklabels=[f"Cycle {c}" for c in range(1, 6)], xtick_px=18,
                 y_title="PO$_4$-P Amount (mmol)", y_right_title="Feed PO$_4$-P Conc. (mg L$^{-1}$)",
                 label={"text": "B", "xy": (92, 34), "size_px": 20}, exclude=[(88, 505, 15, 62)],
                 legend={"xy": (135, 16), "ncol": 2, "size_px": 16,
                         "order": ["PO4-P @ Feed", "Initial P Conc.", "PO4-P @ Draw", "End P Conc."]},
                 series={"PO4-P @ Draw": dict(kind="stacked", width_px=19, max_row_gap=26, match={"rgb": (221, 141, 140), "tol": 40},
                                              color="#DD8D8C", fill="#DD8D8C"),
                         "PO4-P @ Feed": dict(kind="stacked", width_px=19, max_row_gap=26, match={"rgb": (126, 155, 136), "tol": 40},
                                              color="#7E9B88", fill="#7E9B88"),
                         "Initial P Conc.": dict(kind="marker", axis="right", ms_px=14, err=True, expected_x=[0, 2, 4, 6, 8], **RED_RING),
                         "End P Conc.": dict(kind="marker", axis="right", ms_px=13, err=True, expected_x=[1, 3, 5, 7, 9], exclude=[(95, 495, 296, 318)], **BLK_SQ)})])
