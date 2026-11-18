"""What the VLM is asked, how its answer is parsed, and the 'spec' it produces.

The spec holds everything we used to type by hand for each figure - but only
the parts a language model is good at (reading text, naming things):

    {"chart_type": "scatter_line",
     "x_axis":  {"title": "Time (days)", "tick_labels": ["0", "5", ...], "scale": "linear"},
     "y_axis":  {"title": "...", "tick_labels": [...], "scale": "linear", "broken": false},
     "y_axis_right": null | {"title": ..., "tick_labels": [...], "scale": ...},
     "legend": true,
     "series": [{"name": "CT", "color": "dark gray", "marker": "square",
                 "marker_fill": "filled", "line": "solid", "line_color": null,
                 "y_axis": "left", "position": null}, ...]}

Pixel positions are NEVER asked from the VLM - those come from detect.py.
"""
from __future__ import annotations

import json
import re

from .spec_types import (NEW_TYPES, SUPPORTED, CHART_TYPE_GUIDE, HIST_PROMPT, BOX_PROMPT, STACK_PROMPT, HEATMAP_PROMPT,
                         POLAR_PROMPT, THREED_PROMPT, SECOND_PROMPT, canonical_type, normalize_new)

PROMPT_VERSION = 6      # bump when the prompts change: cached answers from older prompts are then re-asked

AXES_PROMPT = """You are reading a scientific chart. Report its axes as JSON, copying text exactly as printed.
Return ONLY this JSON object:
{
 "chart_type": one of "scatter", "line", "scatter_line", "bar", "stacked_bar", "histogram", "box", "heatmap", "contour", "polar", "pie", "bar3d", "scatter3d", "surface3d", "other",
 "x_axis": {"title": "<x-axis title>", "tick_labels": ["<every numeric tick label under the x-axis, left to right>"], "scale": "linear" or "log"},
 "y_axis": {"title": "<left y-axis title>", "tick_labels": ["<every numeric tick label on the LEFT y-axis, bottom to top>"], "scale": "linear" or "log", "broken": true or false},
 "y_axis_right": null or {"title": "...", "tick_labels": ["<every numeric tick label on the RIGHT side, bottom to top>"], "scale": "linear" or "log"}
}
Rules:
- Copy every tick label exactly as printed (e.g. "0.0003", "1.0", "10"). Count them: do not skip any, do not add values that are not printed.
- "broken": true only if the y-axis is interrupted (a // mark or a gap) and shows two different ranges. Then list the labels of the lower range followed by the labels of the upper range, each printed label exactly once, bottom to top.
- "y_axis_right": only if numbers are printed along the right edge of the plot. A plain frame line on the right is NOT a second axis -> null.
- If an axis has no numbers printed (e.g. a spectrum with offset curves), use [].
- A bar chart has category names (text) under the x-axis: put them in x_axis "tick_labels", left to right.
- A pie chart has no axes: return "chart_type": "pie" and null for x_axis, y_axis and y_axis_right.
""" + CHART_TYPE_GUIDE

SERIES_PROMPT = """You are reading a scientific chart. List every data series as JSON.
Return ONLY this JSON object:
{
 "legend": true or false,
 "series": [
  {"name": "<series name exactly as written in the legend, or in the text label next to the curve>",
   "color": "<main colour of its markers or line: red, blue, green, magenta, orange, purple, cyan, black, dark gray, ...>",
   "marker": one of "circle", "square", "triangle_up", "triangle_down", "diamond", "star", "cross", "none",
   "marker_fill": "filled" or "hollow",
   "line": "solid", "dashed" or "none",
   "line_color": null if the line has the same colour as the markers, else its colour,
   "y_axis": "left" or "right",
   "position": null, or "upper" / "lower" (where this series lies relative to the others),
   "n_points": <how many markers this series has; 0 for a line without markers>}
 ]
}
Rules:
- "legend": true only if there is a legend box (a sample marker or line next to each name). Names written next to the curves are not a legend.
- One entry per curve. Two curves drawn in the SAME colour and marker are still TWO series. For example an adsorption-desorption isotherm is two branches forming a loop: list the lower branch and the upper branch separately (names from the plot, e.g. "adsorption" and "desorption"), with "position" "lower" and "upper".
- A fitted model curve is its own series (marker "none", line "solid"), separate from the measured points.
- Look closely at triangles: pointing up = "triangle_up", pointing down = "triangle_down".
- Order: as in the legend (top to bottom, then left to right); without a legend, from top to bottom in the plot.
- Do not list axis lines, grid lines, error bars, guide / reference lines or text annotations as series."""

BAR_PROMPT = """You are reading a bar chart. Describe its categories and bar series as JSON.
Return ONLY this JSON object:
{
 "legend": true or false,
 "categories": [{"name": "<category label under the x-axis, exactly as printed>", "y_axis": "left" or "right"}],
 "series": [
  {"name": "<series name exactly as in the legend; for a single series without a legend use the y-axis title>",
   "color": "<fill colour of its bars, e.g. light orange, light green, light purple, dark green, red, blue, gray>",
   "hatched": true or false,
   "error_bars": true or false}
 ]
}
Rules:
- Categories from left to right, every category once.
- "y_axis" of a category: "right" only if the chart has a second y-axis on the right AND the bars of that category are read against it (look for boxes, arrows or colours that link the category to an axis); otherwise "left".
- Series in legend order, which is also their left-to-right order inside each group of bars.
- Do not report bar heights."""

PIE_PROMPT = """You are reading a pie chart. List its slices as JSON.
Return ONLY this JSON object:
{
 "slices": [{"name": "<slice label exactly as printed>", "color": "<colour of the slice, e.g. blue, red, yellow, light green, purple>",
             "printed_value": <the percentage printed for this slice as a number, or null if none is printed>}]
}
Rules: every slice once, starting at the 12 o'clock position and going clockwise; copy the printed numbers exactly (e.g. 1.408)."""

TITLE_PROMPT = """This chart image has been turned by 90 degrees. Read the axis title that is now written horizontally along the TOP edge of the image (copy it exactly, including units in brackets and small letters such as the p in pH).
Return ONLY a JSON object: {"title": "<the title>"}"""

BOX_TICKS_PROMPT = """A red rectangle has been drawn on this chart around some of the numeric tick labels of the {where}.
Read ONLY the numbers inside the red rectangle, exactly as printed (keep every digit and decimal place), in order from {order}.
Image analysis found {n} labels inside it. Ignore every label outside the rectangle.
Return ONLY a JSON list of strings, e.g. ["0.0", "0.5", "1.0"]."""

TICKS_PROMPT = """Look only at the {where} of this chart. Image analysis found {n} numeric tick labels there{extra}.
Read all {n} of them exactly as printed, in order from {order}.
Return ONLY a JSON list of strings, e.g. ["0", "20", "40"]."""


def _close_truncated(t):
    """A VLM answer cut off by the token limit (or stuck repeating a list):
    drop the unfinished last item and close the open brackets."""
    t = t.rstrip()
    stack, in_str, esc, last_ok = [], False, False, 0
    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
        if ch in ",{[" and not in_str:
            last_ok = (i, list(stack))
    if not stack:
        return t
    if last_ok:
        i, st = last_ok
        t = t[:i + 1] if t[i] in "{[" else t[:i]      # cut after the last complete item
        stack = st
    return t + "".join("}" if c == "{" else "]" for c in reversed(stack))


def _json_blob(text):
    t = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)(```|$)", t, re.DOTALL)
    if fence:
        t = fence.group(1)
    s = t.find("{")
    if s < 0:
        return None
    t = t[s:]
    e = t.rfind("}")
    cands = ([t[:e + 1]] if e >= 0 else []) + [_close_truncated(t)]
    for c in cands:
        for u in (c, re.sub(r",\s*([}\]])", r"\1", c).replace("None", "null").replace("True", "true").replace("False", "false")):
            try:
                return json.loads(u)
            except Exception:
                pass
    return None


MARKERS = {"circle", "square", "triangle_up", "triangle_down", "diamond", "star", "cross", "none"}


def _norm_marker(m):
    m = (m or "none").lower().replace("-", "_").replace(" ", "_")
    if m in MARKERS:
        return m
    if "tri" in m:
        return "triangle_down" if "down" in m or "inverted" in m else "triangle_up"
    for k in ("circle", "square", "diamond", "star", "cross"):
        if k in m or (k == "circle" and ("dot" in m or "round" in m)):
            return k
    if m in ("x", "plus", "+"):
        return "cross"
    return "none"


def _int_or_none(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _num(v):
    try:
        return float(re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)", str(v))[0])
    except Exception:
        return None


def normalize_bar(spec, ans):
    """Bar-chart answer -> spec["categories"], spec["series"] (kind 'bar')."""
    ans = ans or {}
    cats = []
    for i, c in enumerate(ans.get("categories") or []):
        if isinstance(c, dict):
            name, ax = str(c.get("name") or f"category {i + 1}"), str(c.get("y_axis", "left")).lower()
        else:
            name, ax = str(c), "left"
        cats.append({"name": name, "y_axis": "right" if ax.startswith("r") and spec.get("y_axis_right") else "left"})
    if not cats:   # fall back to the x tick labels of the axes answer
        cats = [{"name": t, "y_axis": "left"} for t in spec["x_axis"]["tick_labels"]]
    spec["categories"] = cats
    spec["legend"] = bool(ans.get("legend", False))
    spec["series"] = []
    for i, q in enumerate(ans.get("series") or []):
        if not isinstance(q, dict):
            continue
        spec["series"].append({"name": str(q.get("name") or f"series {i + 1}"), "kind": "bar",
                               "color": str(q.get("color") or "gray").lower(),
                               "hatched": str(q.get("hatched", False)).lower() in ("true", "yes", "1"),
                               "error_bars": str(q.get("error_bars", False)).lower() in ("true", "yes", "1"),
                               "marker": "none", "marker_fill": "filled", "line": "none", "line_color": None,
                               "y_axis": "left", "position": None, "n_points": len(cats)})
    from collections import Counter
    cnt = Counter(q["name"] for q in spec["series"])
    for k, q in enumerate(spec["series"]):
        if cnt[q["name"]] > 1:
            q["name"] = f"{q['name']} #{k + 1}"
    return spec


def normalize_pie(spec, ans):
    ans = ans or {}
    sl = []
    for i, q in enumerate(ans.get("slices") or []):
        if isinstance(q, dict):
            sl.append({"name": str(q.get("name") or f"slice {i + 1}"), "color": str(q.get("color") or "gray").lower(),
                       "printed_value": _num(q.get("printed_value")) if q.get("printed_value") not in (None, "", "null") else None})
    spec["slices"] = sl
    spec["series"] = [{"name": "share (%)", "kind": "pie", "color": "", "marker": "none", "marker_fill": "filled",
                       "line": "none", "line_color": None, "y_axis": "left", "position": None, "n_points": len(sl)}]
    spec["legend"] = False
    return spec


def renormalize(spec):
    """Clean an already-built spec again (cached / hand-edited files)."""
    second = {"legend": spec.get("legend"), "series": spec.get("series")}
    ct = canonical_type(spec.get("chart_type"))
    if ct in NEW_TYPES or ct == "other" or (ct == "bar" and str(spec.get("orientation", "")).startswith("h")):
        return normalize(spec, spec)
    if spec.get("chart_type") == "pie":
        second = {"slices": spec.get("slices")}
    elif spec.get("chart_type") == "bar":
        second = {"legend": spec.get("legend"), "categories": spec.get("categories"), "series": spec.get("series")}
    return normalize(spec, second)


def normalize(axes, series):
    """Merge the two VLM answers into one clean spec (fills defaults, fixes types)."""
    axes = axes or {}
    series = series or {}
    ct = canonical_type(axes.get("chart_type", ""))
    horiz = str(series.get("orientation", axes.get("orientation", ""))).lower().startswith("h")
    stacked = str(series.get("stacked", "")).lower() == "true"
    if ct == "bar" and stacked:
        ct = "stacked_bar"
    if ct == "bar" and horiz:
        return normalize_new("bar", axes, series, _norm_marker)          # horizontal bars -> stacked.py (not stacked)
    if ct in NEW_TYPES:
        return normalize_new(ct, axes, series, _norm_marker)
    if ct == "other":
        return {"chart_type": "other", "vlm_chart_type": str(axes.get("chart_type", "")), "x_axis": None, "y_axis": None,
                "y_axis_right": None, "legend": False, "series": []}
    if "slices" in series or ct == "pie":
        base = normalize({k: v for k, v in axes.items() if k != "chart_type"}, {})
        base["chart_type"] = "pie"
        return normalize_pie(base, series)
    if "categories" in series or ct.startswith("bar"):
        base = normalize({k: v for k, v in axes.items() if k != "chart_type"}, {})
        base["chart_type"] = "bar"
        return normalize_bar(base, series)

    def ax(d, allow_broken=False):
        if not isinstance(d, dict):
            return None
        labs = []
        for x in d.get("tick_labels") or []:
            x = str(x).strip()
            if x and x not in labs:      # each printed label appears once on an axis (VLMs sometimes repeat a block)
                labs.append(x)
        out = {"title": str(d.get("title") or ""),
               "tick_labels": labs,
               "scale": "log" if str(d.get("scale", "linear")).lower().startswith("log") else "linear"}
        if allow_broken:
            out["broken"] = bool(d.get("broken", False))
        return out
    spec = {"chart_type": str(axes.get("chart_type", "scatter_line")),
            "x_axis": ax(axes.get("x_axis")) or {"title": "", "tick_labels": [], "scale": "linear"},
            "y_axis": ax(axes.get("y_axis"), True) or {"title": "", "tick_labels": [], "scale": "linear", "broken": False},
            "y_axis_right": ax(axes.get("y_axis_right")),
            "legend": bool(series.get("legend", False)),
            "series": []}
    r = spec["y_axis_right"]
    if r is not None and (not r["tick_labels"] or r["tick_labels"] == spec["y_axis"]["tick_labels"]):
        spec["y_axis_right"] = None        # no labels, or a copy of the left axis = not a second axis
    for i, s in enumerate(series.get("series") or []):
        if not isinstance(s, dict):
            continue
        line = str(s.get("line", "none")).lower()
        name = str(s.get("name") or f"series {i + 1}")
        spec["series"].append({
            "name": name,
            "color": str(s.get("color") or "black").lower(),
            "marker": _norm_marker(s.get("marker")),
            "marker_fill": "hollow" if "hollow" in str(s.get("marker_fill", "")).lower() or "open" in str(s.get("marker_fill", "")).lower() else "filled",
            "line": "dashed" if "dash" in line else ("solid" if line in ("solid", "true", "yes", "line") else "none"),
            "line_color": (str(s["line_color"]).lower() if s.get("line_color") not in (None, "", "null", "same") else None),
            "y_axis": "right" if str(s.get("y_axis", "left")).lower().startswith("r") and spec["y_axis_right"] else "left",
            "position": (str(s["position"]).lower() if s.get("position") else None),
            "n_points": _int_or_none(s.get("n_points")),
        })
    # two series with one name (e.g. both isotherm branches named after the axis): make names unique
    from collections import Counter
    cnt = Counter(q["name"] for q in spec["series"])
    for k, q in enumerate(spec["series"]):
        if cnt[q["name"]] > 1:
            q["name"] = f"{q['name']} ({q['position'] or k + 1})"
    cnt = Counter(q["name"] for q in spec["series"])
    for k, q in enumerate(spec["series"]):
        if cnt[q["name"]] > 1:
            q["name"] = f"{q['name']} #{k + 1}"
    return spec


# ----------------------------------------------------------------------- readers
class _VLMReader:
    """The questions are the same for every VLM; a reader only has to implement ask(image, prompt) -> text."""
    upscale_below = 1000

    def _image(self, image_path, rotate=0):
        from PIL import Image
        im = Image.open(image_path).convert("RGB")
        box = getattr(self, "_box", None)
        if box is not None:                 # mark (not crop) a region the question is about
            from PIL import ImageDraw
            w = max(2, round(max(im.size) / 300))
            ImageDraw.Draw(im).rectangle([round(v) for v in box], outline=(255, 0, 0), width=w)
        if rotate:
            im = im.rotate(rotate, expand=True, fillcolor="white")
        if self.upscale_below and max(im.size) < self.upscale_below:
            f = self.upscale_below / max(im.size)
            im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
        return im

    def read(self, image_path):
        raw_axes = self.ask(image_path, AXES_PROMPT)
        axes = _json_blob(raw_axes) or {}
        ct = canonical_type(axes.get("chart_type", ""))
        if ct == "other":                       # unsupported: nothing more to ask, it will be rejected
            return {"raw_axes": raw_axes, "raw_series": "", "second_question": "(not asked: unsupported chart type)",
                    "spec": normalize(axes, {})}
        # the second question depends on the kind of chart the VLM says it is
        second = PIE_PROMPT if ct == "pie" else SECOND_PROMPT.get(ct, SERIES_PROMPT)
        if ct == "bar":
            # first the layout (vertical / horizontal, stacked or not); upright grouped bars then get
            # the original bar question (categories + which y-axis each category uses)
            raw_l = self.ask(image_path, STACK_PROMPT)
            lay = _json_blob(raw_l) or {}
            if str(lay.get("orientation", "")).lower().startswith("h") or str(lay.get("stacked", "")).lower() == "true":
                return {"raw_axes": raw_axes, "raw_series": raw_l, "second_question": STACK_PROMPT.split(".")[0],
                        "spec": normalize(axes, lay)}
            second = BAR_PROMPT
        raw_series = self.ask(image_path, second)
        return {"raw_axes": raw_axes, "raw_series": raw_series, "second_question": second.split(".")[0],
                "spec": normalize(axes, _json_blob(raw_series))}

    def reread_title(self, image_path, side="left"):
        """Vertical axis titles are hard to read: show the image turned so that the
        title of the left (right) y-axis lies horizontally along the top edge."""
        self._rotate = -90 if side == "left" else 90
        try:
            raw = self.ask(image_path, TITLE_PROMPT)
        finally:
            self._rotate = 0
        d = _json_blob(raw) or {}
        return raw, (str(d.get("title")).strip() if d.get("title") else None)

    def reread_ticks_box(self, image_path, axis, box, n):
        """Read only the tick labels inside a red rectangle drawn on (a copy of) the
        image - used for each piece of a broken axis separately."""
        where, order = {"x_axis": ("x-axis", "left to right"), "y_axis": ("left y-axis", "bottom to top"),
                        "y_axis_right": ("right y-axis", "bottom to top")}[axis]
        self._box = box
        try:
            raw = self.ask(image_path, BOX_TICKS_PROMPT.format(where=where, n=n, order=order))
        finally:
            self._box = None
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        try:
            vals = [str(v) for v in json.loads(m.group(0))] if m else None
        except Exception:
            vals = None
        return raw, vals

    def reread_ticks(self, image_path, axis, n, broken=False):
        """Second, focused question when the pixels disagree with the first answer."""
        where, order = {"x_axis": ("x-axis (bottom edge)", "left to right"),
                        "y_axis": ("left y-axis", "bottom to top"),
                        "y_axis_right": ("right y-axis", "bottom to top")}[axis]
        extra = " (the axis is broken into two ranges - include the labels of both, each once)" if broken else ""
        raw = self.ask(image_path, TICKS_PROMPT.format(where=where, n=n, extra=extra, order=order))
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        try:
            vals = [str(v) for v in json.loads(m.group(0))] if m else None
        except Exception:
            vals = None
        return raw, vals

    def unload(self):
        pass


class QwenSpecReader(_VLMReader):
    """Local model: Qwen3-VL (default 4B) on the Mac (MPS) / GPU. Loads once, reuse for many images."""

    def __init__(self, model_id="Qwen/Qwen3-VL-4B-Instruct", max_new_tokens=2048, upscale_below=1000):
        """upscale_below: images whose longer side is smaller than this are enlarged
        (LANCZOS) before the VLM sees them - small tick labels read better. The spec
        holds no pixel positions, so resizing for the VLM changes nothing downstream."""
        self.model_id, self.max_new_tokens, self.upscale_below = model_id, max_new_tokens, upscale_below
        self.model = self.processor = None

    def load(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = dev
        dtype = torch.bfloat16 if dev != "cpu" else torch.float32
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_id, dtype=dtype).to(dev).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        return self

    def ask(self, image_path, prompt):
        import torch
        from PIL import Image
        if self.model is None:
            self.load()
        im = self._image(image_path, getattr(self, "_rotate", 0))
        msgs = [{"role": "user", "content": [{"type": "image", "image": im},
                                             {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def unload(self):
        self.model = self.processor = None
        try:
            import gc, torch
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class OpenAISpecReader(_VLMReader):
    """OpenAI API (Responses API, image as base64). Same questions as for Qwen.
    The key is read from the OPENAI_API_KEY environment variable (never written to any file)."""

    def __init__(self, model_id="gpt-6-astra", detail="high", reasoning_effort="medium",
                 upscale_below=1000, max_retries=3, api_key=None):
        self.model_id, self.detail, self.reasoning_effort = model_id, detail, reasoning_effort
        self.upscale_below, self.max_retries = upscale_below, max_retries
        self._api_key = api_key
        self.client = None
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def load(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=self._api_key) if self._api_key else OpenAI()
        return self

    def ask(self, image_path, prompt):
        import base64, io, time as _t
        if self.client is None:
            self.load()
        buf = io.BytesIO()
        self._image(image_path, getattr(self, "_rotate", 0)).save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        kw = dict(model=self.model_id,
                  input=[{"role": "user", "content": [{"type": "input_text", "text": prompt},
                                                      {"type": "input_image", "image_url": url, "detail": self.detail}]}])
        if self.reasoning_effort:
            kw["reasoning"] = {"effort": self.reasoning_effort}
        for k in range(self.max_retries):
            try:
                r = self.client.responses.create(**kw)
                u = getattr(r, "usage", None)
                if u is not None:
                    self.usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
                    self.usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
                self.usage["calls"] += 1
                return r.output_text
            except Exception as e:
                msg = str(e)
                if "reasoning" in msg and "reasoning" in kw:      # model without reasoning settings
                    kw.pop("reasoning"); continue
                if k == self.max_retries - 1:
                    raise
                _t.sleep(2 ** k * 2)


class FileSpecReader:
    """Reads specs from JSON files instead of a model: used to test the pipeline
    without a GPU, and to re-run with a corrected spec. Looks up
    <spec_dir>/<image stem>.json, else a sidecar <image>.spec.json."""

    def __init__(self, spec_dir=None):
        self.spec_dir = spec_dir

    def read(self, image_path):
        from pathlib import Path
        p = Path(image_path)
        cands = ([Path(self.spec_dir) / f"{p.stem}.json"] if self.spec_dir else []) + [p.with_suffix(".spec.json")]
        for c in cands:
            if c.exists():
                d = json.loads(c.read_text())
                spec = d.get("spec", d)
                return {"raw_axes": "(from file)", "raw_series": "(from file)", "spec": renormalize(spec)}
        raise FileNotFoundError(f"no spec file for {image_path}")
