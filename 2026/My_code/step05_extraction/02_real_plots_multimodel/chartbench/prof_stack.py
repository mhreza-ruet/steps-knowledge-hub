"""stacked bar charts"""
from .profiles import add


def _m(c, t):
    return c if isinstance(c, dict) else {"rgb": c, "tol": t}


def _hex(c, m):
    return "#%02x%02x%02x" % (m if isinstance(c, dict) else c)


def stack(fid, fr, y, yt, cat_pos, cats, width, segs, title, style, ytitle, label=None, legend=None, exclude=(), xtl=None,
          xrot=0, groups=(), err=False, ylabels=None, gap_tol=None, vlines=None, **kw):
    """segs: [(name, rgb, tol)] bottom -> top"""
    add(fid, file=f"test_data/{fid}.png", size=None, chart_type="bar_stacked", title=title, style=style,
        panels=[dict(id="main", role="main", frame=fr, categories=cats, cat_pos=cat_pos, x=(cat_pos[0], 0, cat_pos[1], 1), y=y,
                     y_ticks=yt, y_ticklabels=ylabels, x_ticks=list(range(len(cats))), x_ticklabels=xtl if xtl is not None else cats,
                     x_rotation=xrot, y_title=ytitle, label=label, legend=legend, exclude=list(exclude), vlines=vlines or [],
                     series={n: dict(kind="stacked", match=_m(c, t), width_px=width, color=_hex(c, t),
                                     fill=_hex(c, t), edge="#000000", edge_lw=0.8,
                                     **({"max_row_gap": gap_tol} if gap_tol else {}))
                             for n, c, t in segs}, **kw)],
        annotations=[{"text": t, "xy": xy, "size_px": style.get("title_px", 18), "ha": "center", "weight": style.get("weight", "normal")}
                     for t, xy in groups])


SERIF_B = {"font": "serif", "weight": "bold", "tick_px": 20, "title_px": 22, "tick_dir": "in", "tick_len": 5, "frame_lw": 1.2}

p1 = [c + d for c in (155.5, 268.0, 380.5, 493.0, 605.5) for d in (-28, 0, 28)]
stack("bar_stacked_1", (94.0, 668.0, 10.5, 473.0), (472.55, 0, 52.62, 1000), [0, 200, 400, 600, 800, 1000], p1,
      [f"{r} {d}" for d in ("Day 1", "Day 3", "Day 7", "Day 15", "Day 30") for r in ("CT", "RM", "BM")], 28,
      [("acetate", {"hue": 359, "hue_tol": 12, "min_sat": 0.3}, (241, 106, 109)),
       ("propionate", {"hue": 228, "hue_tol": 15, "min_sat": 0.25}, (118, 136, 206)),
       ("iso-butyrate", {"hue": 43, "hue_tol": 10, "min_sat": 0.35}, (254, 209, 93)),
       ("n-butyrate", {"hue": 143, "hue_tol": 15, "min_sat": 0.25}, (119, 211, 154)),
       ("iso-valerate", {"hue": 22, "hue_tol": 12, "min_sat": 0.07, "max_sat_frac": 0.25, "min_val": 0.35}, (178, 164, 156)),
       ("n-valerate", {"gray": True, "gray_range": (90, 215), "max_sat": 6}, (173, 173, 173))],
      "Short-chain fatty acids per reactor and day (stacked bars, 6 acids)", SERIF_B, "Short-chain fatty acids (mg COD/L)",
      label={"text": "(c)", "xy": (104, 30), "size_px": 22}, exclude=[(470, 665, 15, 175), (100, 140, 15, 45)],
      legend={"xy": (488, 15), "size_px": 19, "order": ["n-valerate", "iso-valerate", "n-butyrate", "iso-butyrate", "propionate", "acetate"]},
      xtl=["CT", "RM", "BM"] * 5, xrot=60,
      groups=[(f"Day {d}", (c, 538)) for d, c in zip((1, 3, 7, 15, 30), (155.5, 268.0, 380.5, 493.0, 605.5))])

p2 = [130.0, 186.5, 298.0, 354.0, 466.0, 522.0]
stack("bar_stacked_2", (85.5, 567.0, 13.0, 401.0), (400.95, 0, 12.88, 150), [0, 30, 60, 90, 120, 150], p2,
      [f"{g} {d}" for g in ("MBC", "FBM", "SBM") for d in ("Day 1", "Day 30")], 43,
      [("DOM", {"hue": 226, "hue_tol": 15, "min_sat": 0.3}, (104, 122, 192)),
       ("LB-EPSs", {"hue": 358, "hue_tol": 12, "min_sat": 0.4}, (240, 85, 88)),
       ("TB-EPSs", {"gray": True, "gray_range": (55, 200), "max_sat": 6}, (135, 135, 135))],
      "Polysaccharides in three sludge fractions (stacked bars)", SERIF_B, "Polysaccharides (mg COD/L)",
      label={"text": "(a)", "xy": (98, 50), "size_px": 24}, exclude=[(95, 470, 15, 50), (95, 140, 35, 65)],
      legend={"xy": (180, 18), "ncol": 3, "size_px": 18},
      xtl=["Day 1", "Day 30"] * 3, groups=[(g, ((p2[2 * i] + p2[2 * i + 1]) / 2, 445)) for i, g in enumerate(("MBC", "FBM", "SBM"))])

p3 = [141.0, 197.5, 309.0, 365.5, 477.0, 533.5]
stack("bar_stacked_3", (96.5, 578.0, 19.0, 407.0), (406.95, 0, 18.88, 1500), [0, 300, 600, 900, 1200, 1500], p3,
      [f"{g} {d}" for g in ("MBC", "FBM", "SBM") for d in ("Day 1", "Day 30")], 43,
      [("DOM", {"hue": 226, "hue_tol": 15, "min_sat": 0.3}, (104, 122, 192)),
       ("LB-EPSs", {"hue": 358, "hue_tol": 12, "min_sat": 0.4}, (240, 85, 88)),
       ("TB-EPSs", {"gray": True, "gray_range": (55, 200), "max_sat": 6}, (135, 135, 135))],
      "Proteins in three sludge fractions (stacked bars)", SERIF_B, "Proteins (mg COD/L)",
      label={"text": "(b)", "xy": (108, 52), "size_px": 24}, exclude=[(100, 480, 20, 70)],
      legend={"xy": (190, 22), "ncol": 3, "size_px": 18},
      xtl=["Day 1", "Day 30"] * 3, groups=[(g, ((p3[2 * i] + p3[2 * i + 1]) / 2, 452)) for i, g in enumerate(("MBC", "FBM", "SBM"))])

# ------------------------------------------------------------- microbial community: relative abundance (100 % stacks)
SANS_B = {"font": "serif", "weight": "bold", "tick_px": 17, "title_px": 18, "tick_dir": "in", "tick_len": 4, "frame_lw": 1.2}
COMM = ["MBC", "FBM", "SBM"]
P45 = [122.5, 180.0, 237.5, 295.0, 352.0, 410.0]
TAXA4 = [("Firmicutes", (56, 126, 184), 22), ("Bacteroidota", (122, 170, 213), 22), ("Chloroflexi", (253, 218, 170), 22),
         ("Proteobacteria", (146, 211, 92), 25), ("Actinobacteriota", (20, 158, 44), 28), ("Patescibacteria", (240, 104, 110), 25),
         ("Acidobacteriota", (253, 193, 2), 30), ("Spirochaetota", (175, 175, 175), 12), ("Desulfobacterota", (199, 2, 2), 30),
         ("Synergistota", (180, 152, 212), 22), ("Sumerlaeota", (254, 254, 182), 18), ("others", (92, 92, 92), 18)]
stack("bar_stacked_4", (82.5, 450.0, 26.5, 319.0), (318.67, 0, 22.17, 100), [0, 20, 40, 60, 80, 100], P45,
      [f"{c} ({g})" for g in ("Suspension", "Biofilm") for c in COMM], 35, TAXA4,
      "Relative abundance of bacterial phyla (100 % stacked bars, 12 phyla)", SANS_B, "Relative abundance (%)",
      label={"text": "(a)", "xy": (2, 12), "size_px": 20}, gap_tol=2,
      legend={"xy": (458, 110), "size_px": 13, "order": [t[0] for t in TAXA4[::-1]], "labelspacing": 0.25},
      xtl=["", "", "", "", "", ""], vlines=[{"x": 2.5, "ls": "--", "lw": 1.5}])

TAXA5 = [("Methanosarcina", (56, 126, 184), 22), ("Methanobacterium", (122, 170, 213), 22), ("Methanosaeta", (253, 218, 170), 22),
         ("Methanospirillum", (146, 211, 92), 25), ("Methanobrevibacter", (20, 158, 44), 28), ("others", (236, 80, 86), 30)]
stack("bar_stacked_5", (82.0, 449.0, 16.5, 312.5), (312.21, 0, 16.29, 100), [0, 20, 40, 60, 80, 100], P45,
      [f"{c} ({g})" for g in ("Suspension", "Biofilm") for c in COMM], 34, TAXA5,
      "Relative abundance of methanogen genera (100 % stacked bars, 6 genera)", SANS_B, "Relative abundance (%)",
      label={"text": "(c)", "xy": (2, 12), "size_px": 20}, gap_tol=2,
      legend={"xy": (458, 195), "size_px": 13, "order": [t[0] for t in TAXA5[::-1]], "labelspacing": 0.25},
      xtl=COMM * 2, vlines=[{"x": 2.5, "ls": "--", "lw": 1.5}],
      groups=[("Suspension", (180, 338)), ("Biofilm", (352, 338))])

# ------------------------------------------------------------- GWP broken down by cause: stacks of fill PATTERNS, above and below 0
P6 = [117.5, 178.0, 298.0, 358.0, 418.5, 538.5, 598.5, 658.5, 718.5, 838.5, 898.5, 1019.0, 1139.5, 1259.5, 1379.5]
L6 = ["Lime (agri.)", "Lime (lf.)", "AD + SCWO (land)", "Comp. (low)", "Comp. (high)", "AD (no lime)", "AD (lime)", "Dry. + comp.",
      "Lime", "AD (fertilizer)", "AD (no fertilizer)", "AD", "Comp.", "Co AD + comp.", "Comp."]
S6 = ["Houillon and Jolliet, 2005"] * 2 + ["Johansson et al., 2008"] * 3 + ["Murray et al., 2008"] * 4 + ["Brown et al., 2010"] * 2 + \
     ["Carballa et al., 2011", "Liu et al., 2013", "Righi et al., 2013", "Usapein and Chavalparit, 2017"]
add("bar_stacked_6", file="test_data/bar_stacked_6.png", size=None, chart_type="bar_stacked_patterns_signed",
    title="GWP of 15 sludge-treatment scenarios split into four causes (stacked bars told apart by fill pattern, "
          "segments above and below zero)",
    style={"font": "sans", "weight": "normal", "tick_px": 16, "title_px": 16, "tick_dir": "out", "tick_len": 0, "frame_lw": 0.8},
    panels=[dict(id="main", role="main", frame=(88.0, 1409.5, 13.5, 514.5), categories=[f"{a} [{b}]" for a, b in zip(L6, S6)],
                 cat_pos=P6, x=(117.5, 0, 178.0, 1), y=(348.0, 0, 13.5, 4000), y_ticks=list(range(-2000, 4001, 1000)), x_ticks=[],
                 bar_width_px=15, grid=True, frame_color="#BBBBBB", y_title="GWP (kg-CO$_2$e/t-DS)",
                 vlines=[{"x": (v - 117.5) / 60.5, "ls": "--", "lw": 0.8} for v in (247, 482.5, 770, 954.5, 1081, 1201, 1315)],
                 series={"GWP due to other factors": dict(kind="pattern_stack", pattern="solid", color="#111111", fill="#111111"),
                         "GWP due to release of methane and nitrous oxide following land application":
                             dict(kind="pattern_stack", pattern="hatched", color="#555555", fill="#FFFFFF", hatch="////",
                                  hatch_color="#555555", edge="#333333"),
                         "GWP due to fertiliser avoidance": dict(kind="pattern_stack", pattern="dotted", color="#555555", fill="#FFFFFF",
                                                                 hatch="..", hatch_color="#555555", edge="#333333"),
                         "GWP due to carbon sequestration": dict(kind="pattern_stack", pattern="empty", color="#333333", fill="#FFFFFF",
                                                                 edge="#333333")})],
    fig_legend={"xy": (445, 630), "ncol": 1, "size_px": 16,
                "entries": [("GWP due to carbon sequestration", "line", "#BBBBBB"), ("GWP due to fertiliser avoidance", "dashed", "#555555"),
                            ("GWP due to release of methane and nitrous oxide following land application", "line", "#777777"),
                            ("GWP due to other factors", "line", "#111111")]},
    annotations=[{"text": t, "xy": (x0, 553), "size_px": 16, "ha": "center"} for t, x0 in
                 (("Houillon and\nJolliet, 2005", 148), ("Johansson et al.,\n2008", 351), ("Murray et al., 2008", 627),
                  ("Brown et al., 2010", 862), ("Carballa et al.,\n2011", 1006), ("Liu et al.,\n2013", 1133),
                  ("Righi et al.,\n2013", 1253), ("Usapein and\nChavalparit,\n2017", 1370))])

add("bar_stacked_7", file="test_data/bar_stacked_7.png", size=None, chart_type="bar_stacked",
    title="NH4-N RSF and NH4-N concentration share per cycle (hatched 100 % stacked bars)",
    style={"font": "sans", "weight": "bold", "tick_px": 20, "title_px": 20, "tick_dir": "out", "tick_len": 5, "frame_lw": 1.5},
    panels=[dict(id="main", role="main", frame=(74.5, 581.0, 12.5, 220.5), categories=[f"Cycle {i}" for i in range(1, 6)],
                 x=(125.0, 0, 531.0, 4), y=(219.5, 0, 11.5, 125), y_ticks=[0, 25, 50, 75, 100, 125], x_ticks=[0, 1, 2, 3, 4],
                 x_ticklabels=[f"Cycle {i}" for i in range(1, 6)], y_title="Percentage (%)", exclude=[(150, 545, 14, 42)],
                 legend={"xy": (150, 15), "ncol": 2, "size_px": 17},
                 series={"NH4-N Concentration": dict(kind="stacked", match={"hue": 235, "hue_tol": 18, "min_sat": 0.3, "close": 5},
                                                     width_px=50, color="#3030D0", fill="#FFFFFF", hatch="////", hatch_color="#3030D0",
                                                     max_row_gap=4),
                         "NH4-N RSF": dict(kind="stacked", match={"hue": 352, "hue_tol": 14, "min_sat": 0.3, "close": 5}, width_px=50,
                                           color="#D02040", fill="#FFFFFF", hatch="////", hatch_color="#D02040", max_row_gap=4)})])
