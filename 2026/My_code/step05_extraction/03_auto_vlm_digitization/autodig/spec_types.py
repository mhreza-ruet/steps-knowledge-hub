"""VLM questions and spec parsing for the chart types added in step 5:
stacked / horizontal bars, histograms, box plots, heatmaps, contour / colour maps,
polar plots and 3-D bars, scatter and surfaces - plus the chart-type vocabulary.

canonical_type() maps whatever the VLM calls a chart to one supported type or
'other' (unsupported -> rejected later, never forced into the wrong extractor).
"""
from __future__ import annotations

from collections import Counter

NEW_TYPES = ("stacked_bar", "histogram", "box", "heatmap", "contour", "polar", "bar3d", "scatter3d", "surface3d")
SUPPORTED = ("scatter", "line", "scatter_line", "bar", "pie") + NEW_TYPES

CHART_TYPE_GUIDE = """- chart_type guide: "histogram" = touching bars over a numeric x-axis (bins); "box" = box-and-whisker plots;
  "heatmap" = a grid of coloured cells with row / column names and a colour bar; "contour" = a colour map or
  filled contours over numeric x and y axes with a colour bar; "polar" = circular axes with angle labels around
  the circle; "stacked_bar" = bars made of stacked segments; "bar3d" / "scatter3d" / "surface3d" = 3-D axes
  (three axes drawn in perspective). Anything else (violin, radar / spider, network, Venn, ternary, table,
  map, photo, schematic, several panels in one image) -> "other".
- Horizontal bars: put the numbers under the bars in x_axis and the category names in y_axis "tick_labels" (top to bottom)."""

HIST_PROMPT = """You are reading a histogram. List its data series as JSON.
Return ONLY this JSON object:
{"legend": true or false,
 "series": [{"name": "<name in the legend; without a legend use the y-axis title>",
             "color": "<bar colour, e.g. blue, light orange, green, red>",
             "style": "filled" (solid bars) or "step" (only an outline)}]}
Rules: one entry per histogram drawn (overlapping semi-transparent histograms are separate series);
ignore fitted curves drawn over the bars."""

BOX_PROMPT = """You are reading a box plot. Describe it as JSON.
Return ONLY this JSON object:
{"orientation": "vertical" (boxes stand upright) or "horizontal",
 "categories": ["<category label of each box group, exactly as printed: left to right for vertical boxes, top to bottom for horizontal boxes>"],
 "legend": true or false,
 "series": [{"name": "<legend name; without a legend use the value-axis title>", "color": "<box fill colour, or black for unfilled boxes>"}]}
Rules: several boxes side by side in one category = several series (legend order). Do not report values."""

STACK_PROMPT = """You are reading a bar chart. Describe its layout as JSON.
Return ONLY this JSON object:
{"orientation": "vertical" (bars grow upwards) or "horizontal" (bars grow to the right),
 "stacked": true if each bar is made of several coloured segments on top of each other, else false,
 "categories": ["<category labels exactly as printed: left to right for vertical bars, top to bottom for horizontal bars>"],
 "legend": true or false,
 "series": [{"name": "<legend name>", "color": "<fill colour, e.g. light blue, navy, orange, gray>"}]}
Rules: stacked bars: series in stack order (the segment at the base first). Do not report values."""

HEATMAP_PROMPT = """You are reading a heatmap or colour map. Describe it as JSON.
Return ONLY this JSON object:
{"rows": ["<row labels on the left, top to bottom, exactly as printed; [] if that axis is numeric>"],
 "columns": ["<column labels under (or above) the grid, left to right; [] if that axis is numeric>"],
 "annotated": true if numbers are printed inside the cells, else false,
 "cell_values": if numbers are printed inside the cells: [["<the number printed in each cell of the first (top) row, left to right, exactly as printed>"], ... one list per row, top to bottom]; else [],
 "colorbar": {"title": "<colour bar title or empty>", "tick_labels": ["<every number printed on the colour bar, from the low end to the high end>"],
              "orientation": "vertical" or "horizontal"}}
Copy every label exactly as printed."""

POLAR_PROMPT = """You are reading a polar plot (circular axes). Describe it as JSON.
Return ONLY this JSON object:
{"kind": "line", "scatter" or "bar",
 "angle_labels": ["<every label printed around the circle, as met going COUNTER-CLOCKWISE starting from the label at the 3 o'clock (right-hand) position>"],
 "radial_labels": ["<every number printed along the radius, from the centre outwards>"],
 "legend": true or false,
 "series": [{"name": "<legend name, or the radial axis title>", "color": "<colour>",
             "marker": "circle", "square", "triangle_up", ... or "none", "line": "solid", "dashed" or "none"}]}
Copy the labels exactly as printed (e.g. "45°", "N", "NE")."""

THREED_PROMPT = """You are reading a 3-D chart (three axes drawn in perspective). Describe it as JSON.
Return ONLY this JSON object:
{"x_axis": {"title": "<title>", "tick_labels": ["<every tick label of this axis, in the order met going LEFT to RIGHT across the image>"]},
 "y_axis": {"title": "<title>", "tick_labels": ["<every tick label, LEFT to RIGHT across the image>"]},
 "z_axis": {"title": "<title of the vertical axis>", "tick_labels": ["<every tick label, BOTTOM to TOP>"]},
 "kind": "surface" or "wireframe" (3-D surfaces only, else null),
 "colorbar": null, or {"title": "<colour bar title>", "tick_labels": ["<numbers from low to high>"], "orientation": "vertical" or "horizontal",
                      "same_as": "z" if the colours show the vertical (z) value, else null},
 "stems": true if every marker has a line dropping to the floor, else false,
 "legend": true or false,
 "series": [{"name": "<legend name or the z-axis title>", "color": "<colour, or colormap>", "marker": "circle", "triangle_up", ... or "none"}]}
Copy every label exactly as printed."""

SECOND_PROMPT = {"bar": STACK_PROMPT, "stacked_bar": STACK_PROMPT, "histogram": HIST_PROMPT, "box": BOX_PROMPT,
                 "heatmap": HEATMAP_PROMPT, "contour": HEATMAP_PROMPT, "polar": POLAR_PROMPT,
                 "bar3d": THREED_PROMPT, "scatter3d": THREED_PROMPT, "surface3d": THREED_PROMPT}


def canonical_type(t):
    """VLM wording -> one of the supported chart types, or 'other'"""
    t = str(t or "").lower().strip().replace("-", "_").replace(" ", "_")
    if not t:
        return "scatter_line"
    if any(w in t for w in ("violin", "radar", "spider", "venn", "ternary", "network", "dendrogram", "sankey", "table")):
        return "other"
    if "3d" in t or "3_d" in t or "three" in t:
        if "bar" in t or "column" in t:
            return "bar3d"
        if "surf" in t or "mesh" in t or "wire" in t:
            return "surface3d"
        return "scatter3d"
    for key, out in (("stack", "stacked_bar"), ("hist", "histogram"), ("box", "box"), ("whisker", "box"),
                     ("heat", "heatmap"), ("contour", "contour"), ("colormap", "contour"), ("color_map", "contour"),
                     ("polar", "polar"), ("rose", "polar"), ("pie", "pie"), ("donut", "pie"), ("doughnut", "pie")):
        if key in t:
            return out
    if t.startswith("bar") or "column" in t:
        return "bar"
    if t in ("scatter", "line", "scatter_line"):
        return t
    if t in ("line_scatter", "xy", "scatter_and_line"):
        return "scatter_line"
    if "scatter" in t:
        return "scatter"
    if "line" in t or "curve" in t or "spectrum" in t:
        return "line"
    return "other"


def _axis(d, allow_broken=False):
    if not isinstance(d, dict):
        return None
    labs = []
    for x in d.get("tick_labels") or []:
        x = str(x).strip()
        if x and x not in labs:
            labs.append(x)
    out = {"title": str(d.get("title") or ""), "tick_labels": labs,
           "scale": "log" if str(d.get("scale", "linear")).lower().startswith("log") else "linear"}
    if allow_broken:
        out["broken"] = bool(d.get("broken", False))
    return out


def _series_list(lst, default_name, norm_marker):
    out = []
    for i, q in enumerate(lst or []):
        if not isinstance(q, dict):
            continue
        out.append({"name": str(q.get("name") or (default_name if i == 0 else f"series {i + 1}")),
                    "color": str(q.get("color") or "").lower(), "marker": norm_marker(q.get("marker")),
                    "marker_fill": "filled", "line": str(q.get("line") or "none").lower(), "line_color": None,
                    "style": str(q.get("style") or "filled").lower(), "kind": q.get("kind") or "",
                    "y_axis": "left", "position": None, "n_points": None})
    cnt = Counter(q["name"] for q in out)
    for k, q in enumerate(out):
        if cnt[q["name"]] > 1:
            q["name"] = f"{q['name']} #{k + 1}"
    return out


def normalize_new(ct, axes, ans, norm_marker):
    """spec for the step-5 chart types (fields of the second answer win over the first)"""
    a = {**(axes or {}), **{k: v for k, v in (ans or {}).items() if v is not None}}
    spec = {"chart_type": ct, "x_axis": _axis(a.get("x_axis")) or {"title": "", "tick_labels": [], "scale": "linear"},
            "y_axis": _axis(a.get("y_axis"), True) or {"title": "", "tick_labels": [], "scale": "linear", "broken": False},
            "y_axis_right": None, "legend": bool(a.get("legend", False))}
    vt = spec["y_axis"]["title"] or spec["x_axis"]["title"] or "values"
    if ct in ("bar3d", "scatter3d", "surface3d"):
        spec["z_axis"] = _axis(a.get("z_axis")) or {"title": "", "tick_labels": [], "scale": "linear"}
        vt = spec["z_axis"]["title"] or "values"
        spec["stems"] = str(a.get("stems", False)).lower() in ("true", "1", "yes")
        spec["kind"] = a.get("kind")
    if ct in ("heatmap", "contour", "scatter3d", "surface3d"):
        cb = a.get("colorbar")
        spec["colorbar"] = None if not isinstance(cb, dict) else {
            "title": str(cb.get("title") or ""), "tick_labels": [str(x) for x in cb.get("tick_labels") or []],
            "orientation": str(cb.get("orientation") or ""), "same_as": cb.get("same_as")}
    if ct == "heatmap":
        spec["rows"] = [str(x) for x in a.get("rows") or []]
        spec["columns"] = [str(x) for x in a.get("columns") or []]
        spec["annotated"] = bool(a.get("annotated", False))
        cv = a.get("cell_values") or []
        spec["cell_values"] = [[str(x) for x in r] for r in cv if isinstance(r, (list, tuple))] if isinstance(cv, list) else []
        if not spec["rows"] and not spec["columns"] and spec["x_axis"]["tick_labels"]:
            spec["chart_type"] = "contour"          # numeric axes: a colour map, not a labelled grid
    if ct in ("box", "stacked_bar", "bar"):
        spec["orientation"] = "horizontal" if str(a.get("orientation", "")).lower().startswith("h") else "vertical"
        spec["categories"] = [str(c.get("name") if isinstance(c, dict) else c) for c in a.get("categories") or []]
        if not spec["categories"]:
            spec["categories"] = list((spec["y_axis"] if spec["orientation"] == "horizontal" else spec["x_axis"])["tick_labels"])
    if ct == "polar":
        spec["kind"] = str(a.get("kind") or "line").lower()
        spec["angle_labels"] = [str(x) for x in a.get("angle_labels") or []]
        spec["radial_labels"] = [str(x) for x in a.get("radial_labels") or []]
    spec["series"] = _series_list(a.get("series"), vt, norm_marker)
    return spec


def numeric_share(labels):
    from .calibrate import parse_number
    t = [x for x in labels or [] if str(x).strip()]
    return (sum(parse_number(x) is not None for x in t) / len(t)) if t else 0.0


def value_axis(spec):
    """which axis carries the numbers (bars / boxes): the tick labels decide - numeric labels on
    one axis and names on the other; the VLM's 'orientation' word only breaks a tie"""
    sx = numeric_share((spec.get("x_axis") or {}).get("tick_labels"))
    sy = numeric_share((spec.get("y_axis") or {}).get("tick_labels"))
    if sx >= 0.8 and sy < 0.5:
        return "x_axis"
    if sy >= 0.8 and sx < 0.5:
        return "y_axis"
    return "x_axis" if str(spec.get("orientation") or "vertical").lower().startswith("h") else "y_axis"


def settle_layout(ct, spec):
    """cross-checks of the VLM's description against its own tick labels (no pixels needed):
    - bars / boxes: orientation from which axis is numeric
    - a 'heatmap' whose x and y tick labels are both numbers and which has no real row /
      column names is a continuous colour map (field)"""
    spec = dict(spec)
    notes = []
    if ct in ("bar", "stacked_bar", "box"):
        o = "horizontal" if value_axis(spec) == "x_axis" else "vertical"
        if spec.get("orientation") and str(spec.get("orientation")) != o:
            notes.append(f"orientation set to {o}: the numeric tick labels are on the {'x' if o == 'horizontal' else 'y'} axis")
        spec["orientation"] = o
        if not (ct == "bar" and o == "vertical"):      # vertical plain bars keep the xy-reader's category records
            cats = [str(c.get("name") if isinstance(c, dict) else c) for c in spec.get("categories") or []]
            if not cats:
                cats = list(((spec.get("y_axis") if o == "horizontal" else spec.get("x_axis")) or {}).get("tick_labels") or [])
            spec["categories"] = cats
    if ct == "heatmap":
        titles = {str((spec.get(k) or {}).get("title") or "").strip().lower() for k in ("x_axis", "y_axis")}
        names = [r for r in list(spec.get("rows") or []) + list(spec.get("columns") or []) if str(r).strip().lower() not in titles]
        nx = numeric_share((spec.get("x_axis") or {}).get("tick_labels"))
        ny = numeric_share((spec.get("y_axis") or {}).get("tick_labels"))
        if nx >= 0.8 and ny >= 0.8 and (len(spec.get("rows") or []) < 2 or len(spec.get("columns") or []) < 2 or len(names) < 2):
            notes.append("read as a continuous colour map: x and y tick labels are numbers and there are no row / column names")
            ct = "contour"
            spec["chart_type"] = "contour"
    return ct, spec, notes
