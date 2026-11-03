"""Chart-data extractors compared in the step-4b benchmark.

Every extractor exposes the same tiny interface::

    runner.load()                         # download / move to device
    text = runner.generate(image_path, fig_id)   # raw model output
    data = runner.parse(text, fig_id)     # -> {"series": {name: {"x": [...], "y": [...]}}}
    runner.unload()                       # free GPU/MPS memory before the next model

so the notebook can give each model its own cell while the bookkeeping
(caching, scoring, plotting) stays shared.

Models
------
ClassicalRunner   - axis calibration + color tracing (no ML, no download).
QwenVLRunner      - Qwen2.5-VL family (3B by default; 7B optional).
Qwen3VLRunner     - Qwen3-VL family (4B by default; 8B optional) - newer, stronger
                    at charts/OCR than Qwen2.5-VL at similar size.
DePlotRunner      - google/deplot (Pix2Struct): image -> linearized data table.
OlmOCRRunner      - allenai/olmOCR-2-7B-1025 (Qwen2.5-VL-7B fine-tuned for PDF-page
                    OCR). It was trained to transcribe *text*, not to read chart
                    geometry, so it is run twice: with its own native page-OCR
                    prompt and with the same JSON data prompt the other VLMs get.
"""
from __future__ import annotations

import difflib
import gc
import json
import re
import time
from pathlib import Path

from . import config as cfgmod
from .figures import FIGURES

# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------
def build_prompt(fig_id: str, only: str | None = None) -> str:
    """The JSON data-extraction prompt. ``only`` = ask for a single series
    (prompt mode "per_series")."""
    F = FIGURES[fig_id]
    names = [only] if only else list(F["series"])
    if "categories" in F:   # bar / pie charts: x is a category name
        cats = F["categories"]
        example = ", ".join(f'"{n}": [["{cats[0]}", v1], ["{cats[1]}", v2], ...]' for n in names[:2])
        task = (f"Task: extract the underlying data of ONLY the series '{only}' in this chart. {F['sample_hint']}\n"
                if only else f"Task: extract the underlying data of this chart. {F['sample_hint']}\n")
        return (
            f"{F['prompt']}\n\n" + task +
            f"Respond with ONLY a JSON object that maps each series name to a list of [category, value] pairs, "
            f"for example {{{example}}}.\n"
            f"Use exactly these series names: {json.dumps(names)} and these categories: {json.dumps(cats)}. "
            f"Values are numbers only (no units), no explanation, no markdown."
        )
    example = ", ".join(f'"{n}": [[x1, y1], [x2, y2], ...]' for n in names[:2])
    task = (f"Task: extract the underlying data of ONLY the series '{only}' in this chart. {F['sample_hint']}\n"
            if only else f"Task: extract the underlying data of this chart. {F['sample_hint']}\n")
    return (
        f"{F['prompt']}\n\n" + task +
        f"Respond with ONLY a JSON object that maps each series name to a list of [x, y] number pairs, "
        f"for example {{{example}}}.\n"
        f"Use exactly these series names: {json.dumps(names)}. Numbers only (no units, no strings), "
        f"no explanation, no markdown."
    )


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("po43", "po4"))


def match_series(key: str, names: list[str], used: set | None = None) -> str | None:
    """Map a series label the model produced onto one of the expected names."""
    used = used or set()
    k = _norm(key)
    cands = [n for n in names if n not in used]
    for n in cands:  # exact first ('OPT-HC-N/P' must not be taken by 'OPT-HC')
        if k == _norm(n):
            return n
    cont = [n for n in cands if k and (k in _norm(n) or _norm(n) in k)]
    if cont:          # containment: the closest in length
        return min(cont, key=lambda n: abs(len(_norm(n)) - len(k)))
    best = difflib.get_close_matches(k, [_norm(n) for n in cands], n=1, cutoff=0.55)
    if best:
        return next(n for n in cands if _norm(n) == best[0])
    return None


def _repair_json(s: str) -> str:
    """Close a JSON object that was cut off by max_new_tokens."""
    s = s.strip()
    # drop a trailing partial number / pair
    s = re.sub(r",?\s*\[?\s*" + _NUM + r"?\s*,?\s*$", "", s)
    s = re.sub(r",\s*$", "", s)
    opens = []
    in_str = False
    for ch in s:
        if ch == '"':
            in_str = not in_str
        elif not in_str and ch in "[{":
            opens.append(ch)
        elif not in_str and ch in "]}":
            if opens:
                opens.pop()
    closing = "".join("]" if c == "[" else "}" for c in reversed(opens))
    return s + closing


def _load_json_blob(text: str):
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)(```|$)", t, re.DOTALL)
    if fence:
        t = fence.group(1)
    start = t.find("{")
    if start < 0:
        return None
    blob = t[start:]
    end = blob.rfind("}")
    for cand in ([blob[: end + 1]] if end > 0 else []) + [_repair_json(blob)]:
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def category_index(fig_id, x):
    """Category name (or index) a model returned -> category index; None if unknown."""
    cats = FIGURES[fig_id].get("categories")
    if cats is None:
        return None
    if isinstance(x, (int, float)) and float(x).is_integer() and 0 <= x < len(cats):
        return float(x)
    k = _norm(str(x))
    for i, c in enumerate(cats):
        if k and (_norm(c) == k):
            return float(i)
    for i, c in enumerate(cats):
        if k and (k in _norm(c) or _norm(c) in k):
            return float(i)
    best = difflib.get_close_matches(k, [_norm(c) for c in cats], n=1, cutoff=0.6)
    if best:
        return float([_norm(c) for c in cats].index(best[0]))
    m = re.findall(_NUM, str(x))
    if m and float(m[0]).is_integer() and 0 <= float(m[0]) < len(cats):
        return float(m[0])
    return None


def _points_from_value(v, fig_id=None):
    """Accept [[x,y],...], {"x":[...],"y":[...]}, [{"x":..,"y":..},...]."""
    xs, ys = [], []
    if isinstance(v, dict):
        if "x" in v and "y" in v and isinstance(v["x"], list):
            for a, b in zip(v["x"], v["y"]):
                xs.append(a); ys.append(b)
        elif "points" in v or "data" in v:
            return _points_from_value(v.get("points", v.get("data")), fig_id)
        else:  # {"x_value": y_value}
            for a, b in v.items():
                xs.append(a); ys.append(b)
    elif isinstance(v, list):
        for p in v:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                xs.append(p[0]); ys.append(p[1])
            elif isinstance(p, dict) and "x" in p and "y" in p:
                xs.append(p["x"]); ys.append(p["y"])
    out_x, out_y = [], []
    cat = fig_id is not None and "categories" in FIGURES[fig_id]
    for a, b in zip(xs, ys):
        try:
            xv = category_index(fig_id, a) if cat else float(re.findall(_NUM, str(a))[0])
            if xv is None:
                continue
            out_x.append(xv); out_y.append(float(re.findall(_NUM, str(b))[0]))
        except Exception:
            continue
    return out_x, out_y


def parse_json_points(text: str, fig_id: str) -> dict:
    names = list(FIGURES[fig_id]["series"])
    obj = _load_json_blob(text)
    out = {"series": {}}
    if obj is None:
        return out
    if isinstance(obj, dict) and "series" in obj and isinstance(obj["series"], (dict, list)):
        obj = obj["series"]
    items = obj.items() if isinstance(obj, dict) else [(d.get("name", d.get("label", "")), d) for d in obj if isinstance(d, dict)]
    used = set()
    F = FIGURES[fig_id]
    if "categories" in F and len(names) == 1 and isinstance(obj, dict) and \
            sum(category_index(fig_id, k) is not None for k in obj) >= max(2, len(obj) // 2):
        # single-series chart (pie) answered as {"Separator": 62.5, ...}
        xs, ys = _points_from_value(obj, fig_id)
        if xs:
            out["series"][names[0]] = {"x": xs, "y": ys}
        return out
    for key, val in items:
        name = match_series(str(key), names, used)
        if name is None:
            continue
        xs, ys = _points_from_value(val, fig_id)
        if xs:
            out["series"][name] = {"x": xs, "y": ys}
            used.add(name)
    return out


def parse_table(text: str, fig_id: str) -> dict:
    """DePlot linearized tables ('a | b <0x0A> 1 | 2'), markdown or HTML tables."""
    names = list(FIGURES[fig_id]["series"])
    t = text.replace("<0x0A>", "\n")
    if "<tr" in t.lower():  # HTML -> pipes
        t = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", " | ", t, flags=re.I)
        t = re.sub(r"</tr>", "\n", t, flags=re.I)
        t = re.sub(r"<[^>]+>", "", t)
    rows = []
    for line in t.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue  # markdown separator
        rows.append(cells)
    rows = [r for r in rows if not (r and r[0].upper().startswith("TITLE"))]
    if len(rows) < 2:
        return {"series": {}}
    header, body = rows[0], rows[1:]
    out = {"series": {}}
    used = set()
    numeric_cols = header[1:]
    for j, col in enumerate(numeric_cols, start=1):
        name = match_series(col, names, used)
        if name is None and len(numeric_cols) <= len(names):  # fall back to column order
            free = [n for n in names if n not in used]
            name = free[0] if free else None
        if name is None:
            continue
        xs, ys = [], []
        cat = "categories" in FIGURES[fig_id]
        for r in body:
            if len(r) <= j:
                continue
            ym = re.findall(_NUM, r[j])
            if cat:
                xv = category_index(fig_id, r[0])
                if xv is not None and ym:
                    xs.append(xv); ys.append(float(ym[0]))
                continue
            xm = re.findall(_NUM, r[0])
            if xm and ym:
                xs.append(float(xm[0])); ys.append(float(ym[0]))
        if xs:
            out["series"][name] = {"x": xs, "y": ys}
            used.add(name)
    return out


def parse_any(text: str, fig_id: str) -> dict:
    d = parse_json_points(text, fig_id)
    if not d["series"]:
        d = parse_table(text, fig_id)
    return d


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def free_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


class BaseRunner:
    key = "base"
    label = "base"
    model_id = None
    needs_download = True
    uses_json_prompt = True

    def load(self):
        pass

    def unload(self):
        for attr in ("model", "processor"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        free_memory()

    def parse(self, text, fig_id):
        if text.startswith('{"__per_series__"'):
            pieces = json.loads(text)["__per_series__"]
            out = {"series": {}}
            for name, raw in pieces.items():
                d = parse_any(raw, fig_id)
                if name in d["series"]:
                    out["series"][name] = d["series"][name]
                elif len(d["series"]) == 1:  # model used a slightly different label
                    out["series"][name] = next(iter(d["series"].values()))
                else:  # a bare [[x, y], ...] list without the series name
                    m = re.search(r"\[\s*\[.*\]\s*\]", raw, re.DOTALL)
                    if m:
                        try:
                            xs, ys = _points_from_value(json.loads(m.group(0)), fig_id)
                            if xs:
                                out["series"][name] = {"x": xs, "y": ys}
                        except Exception:
                            pass
            return out
        return parse_any(text, fig_id)

    # ---- bookkeeping: what exactly did the model get?
    def describe_input(self, image_path, fig_id):
        from PIL import Image
        w, h = Image.open(image_path).size
        return {"model_id": self.model_id, "image_size_on_disk": [w, h], "cropped": False}


def prepare_image(image_path, scale=1.0, long_side=None):
    """Return (path, (w, h)) of the image the model will actually receive.
    Upscaled copies are written next to the image and deleted after use."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if long_side:
        scale = long_side / max(w, h)
    if abs(scale - 1.0) < 1e-6:
        return Path(image_path), (w, h), False
    nw, nh = round(w * scale), round(h * scale)
    tmp = Path(image_path).with_name(Path(image_path).stem + f".model_input_{nw}x{nh}.png")
    img.resize((nw, nh), Image.LANCZOS).save(tmp)
    return tmp, (nw, nh), True


def visual_tokens(w, h, factor, min_pixels=None, max_pixels=None):
    """Qwen-style 'smart resize': both sides rounded to a multiple of `factor`
    (28 px for Qwen2.5-VL / olmOCR, 32 px for Qwen3-VL), area clamped to
    [min_pixels, max_pixels]. Each visual token then covers factor x factor px."""
    import math
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if max_pixels and hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb, wb = math.floor(h / beta / factor) * factor, math.floor(w / beta / factor) * factor
    if min_pixels and hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return (wb, hb), (wb // factor) * (hb // factor)


# ---------------------------------------------------------------------------
# 0. classical digitizer
# ---------------------------------------------------------------------------
class ClassicalRunner(BaseRunner):
    key = "classical"
    label = "Classical digitizer (axis calibration + colour tracing)"
    model_id = "chartbench.digitize (no ML)"
    needs_download = False
    uses_json_prompt = False

    def generate(self, image_path, fig_id):
        from .digitize import extract_figure
        return json.dumps(extract_figure(fig_id, image_path))

    def parse(self, text, fig_id):
        d = json.loads(text)
        return {"series": {n: {"x": s["x"], "y": s["y"], **({"yerr": s["yerr"]} if "yerr" in s else {})}
                           for n, s in d["series"].items()}}

    def describe_input(self, image_path, fig_id):
        d = super().describe_input(image_path, fig_id)
        d.update({"note": "uses the per-figure axis calibration and series colours in figures.py"})
        return d


# ---------------------------------------------------------------------------
# 1. Qwen2.5-VL (also the backbone of olmOCR)
# ---------------------------------------------------------------------------
class QwenVLRunner(BaseRunner):
    token_factor = 28

    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", processor_id=None, key=None, label=None):
        C = cfgmod.CONFIG
        self.model_id = model_id
        self.processor_id = processor_id or model_id
        self.key = (key or model_id.split("/")[-1]) + cfgmod.run_tag()
        self.label = label or model_id
        self.max_new_tokens = C["generation"]["max_new_tokens"]
        self.max_pixels = C["vlm_input"]["qwen25_max_pixels"]
        self.upscale = C["vlm_input"]["upscale"]
        self.prompt_mode = C["prompt"]["mode"]

    def load(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.device = pick_device()
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, dtype=dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.processor_id, max_pixels=self.max_pixels)

    def prompt_for(self, fig_id, only=None):
        return build_prompt(fig_id, only)

    def _input_image(self, image_path):
        return prepare_image(image_path, scale=self.upscale)

    def _ask(self, image_path, prompt):
        import torch
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [{"type": "image", "image": str(image_path)},
                                                  {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def generate(self, image_path, fig_id):
        path, _, is_tmp = self._input_image(image_path)
        try:
            if self.prompt_mode == "per_series" and self.uses_json_prompt:
                pieces = {n: self._ask(path, self.prompt_for(fig_id, n)) for n in FIGURES[fig_id]["series"]}
                return json.dumps({"__per_series__": pieces})
            return self._ask(path, self.prompt_for(fig_id))
        finally:
            if is_tmp:
                Path(path).unlink(missing_ok=True)

    def describe_input(self, image_path, fig_id):
        d = super().describe_input(image_path, fig_id)
        path, (w, h), is_tmp = self._input_image(image_path)
        if is_tmp:
            Path(path).unlink(missing_ok=True)
        (rw, rh), ntok = visual_tokens(w, h, self.token_factor, max_pixels=self.max_pixels)
        d.update({"image_size_fed": [w, h], "upscale": self.upscale, "model_internal_size": [rw, rh],
                  "visual_tokens": ntok, "px_per_visual_token": self.token_factor,
                  "prompt_mode": self.prompt_mode, "max_new_tokens": self.max_new_tokens, "do_sample": False,
                  "prompt": self.prompt_for(fig_id)})
        return d


# ---------------------------------------------------------------------------
# 2. Qwen3-VL
# ---------------------------------------------------------------------------
class Qwen3VLRunner(QwenVLRunner):
    token_factor = 32

    def __init__(self, model_id="Qwen/Qwen3-VL-4B-Instruct", **kw):
        super().__init__(model_id=model_id, **kw)
        self.max_pixels = None  # Qwen3-VL processor default

    def load(self):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen3VLForConditionalGeneration as Cls
        except ImportError as e:  # transformers < 4.57
            raise ImportError("Qwen3-VL needs transformers >= 4.57 (pip install -U transformers)") from e
        self.device = pick_device()
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        self.model = Cls.from_pretrained(self.model_id, dtype=dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.processor_id)

    def _ask(self, image_path, prompt):
        import torch
        from PIL import Image
        messages = [{"role": "user", "content": [{"type": "image", "image": Image.open(image_path).convert("RGB")},
                                                  {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# 3. DePlot
# ---------------------------------------------------------------------------
class DePlotRunner(BaseRunner):
    label = "google/deplot (Pix2Struct, chart -> table)"
    uses_json_prompt = False

    def __init__(self, model_id="google/deplot"):
        C = cfgmod.CONFIG
        self.model_id = model_id
        self.key = "deplot" + (f"__x{C['vlm_input']['upscale']:g}" if C["vlm_input"]["upscale"] != 1.0 else "")
        self.max_new_tokens = C["generation"]["deplot_max_new_tokens"]
        self.max_patches = C["vlm_input"]["deplot_max_patches"]
        self.upscale = C["vlm_input"]["upscale"]
        self.prompt = "Generate underlying data table of the figure below:"

    def load(self):
        from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
        self.device = pick_device()
        self.processor = Pix2StructProcessor.from_pretrained(self.model_id)
        self.model = Pix2StructForConditionalGeneration.from_pretrained(self.model_id).to(self.device).eval()

    def generate(self, image_path, fig_id):
        import torch
        from PIL import Image
        path, _, is_tmp = prepare_image(image_path, scale=self.upscale)
        try:
            img = Image.open(path).convert("RGB")
            kw = {"max_patches": self.max_patches} if self.max_patches else {}
            inputs = self.processor(images=img, text=self.prompt, return_tensors="pt", **kw).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            return self.processor.decode(out[0], skip_special_tokens=True)
        finally:
            if is_tmp:
                Path(path).unlink(missing_ok=True)

    def parse(self, text, fig_id):
        return parse_table(text, fig_id)

    def describe_input(self, image_path, fig_id):
        d = super().describe_input(image_path, fig_id)
        path, (w, h), is_tmp = prepare_image(image_path, scale=self.upscale)
        if is_tmp:
            Path(path).unlink(missing_ok=True)
        d.update({"image_size_fed": [w, h], "upscale": self.upscale,
                  "max_patches": self.max_patches or "processor default",
                  "prompt": self.prompt + " (fixed - DePlot takes no instructions)",
                  "max_new_tokens": self.max_new_tokens, "do_sample": False})
        return d


# ---------------------------------------------------------------------------
# 4. olmOCR-2
# ---------------------------------------------------------------------------
OLMOCR_NATIVE_PROMPT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax "
    "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top specifying values for the "
    "primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
)


def olmocr_native_prompt():
    try:  # use the exact prompt of the installed olmocr package when available
        from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
        return build_no_anchoring_v4_yaml_prompt()
    except Exception:
        return OLMOCR_NATIVE_PROMPT


class OlmOCRRunner(QwenVLRunner):
    """olmOCR-2 = Qwen2.5-VL-7B fine-tune; loaded with the Qwen2.5-VL class and
    the base model's processor, as in the olmOCR README. Its input is resized to
    1288 px on the long side (its training render size)."""

    def __init__(self, mode="json", model_id="allenai/olmOCR-2-7B-1025"):
        super().__init__(model_id=model_id, processor_id="Qwen/Qwen2.5-VL-7B-Instruct",
                         key=f"olmocr2-{mode}", label=f"olmOCR-2-7B ({mode} prompt)")
        self.mode = mode
        self.uses_json_prompt = mode != "native"
        if mode == "native":
            self.key = "olmocr2-native"  # native prompt ignores prompt mode / upscale knobs
        self.long_side = cfgmod.CONFIG["vlm_input"]["olmocr_long_side"]

    def prompt_for(self, fig_id, only=None):
        return olmocr_native_prompt() if self.mode == "native" else build_prompt(fig_id, only)

    def _input_image(self, image_path):
        return prepare_image(image_path, long_side=self.long_side)

    def describe_input(self, image_path, fig_id):
        d = super().describe_input(image_path, fig_id)
        d["upscale"] = f"resized to {self.long_side} px long side"
        return d


# ---------------------------------------------------------------------------
# run + cache
# ---------------------------------------------------------------------------
def cached_count(runner, jobs, results_dir):
    return sum((Path(results_dir) / runner.key / s / f"{f}.json").exists() for s, f, _ in jobs)


def run_on_images(runner, jobs, results_dir, force=False, log=print):
    """jobs: list of (image_set, fig_id, image_path). Caches every answer as
    results_dir/<runner.key>/<image_set>/<fig_id>.json together with the exact
    input settings (run_config), so a cell can be re-run (or the kernel
    restarted) without paying for generation again."""
    results_dir = Path(results_dir)
    todo = []
    for image_set, fig_id, path in jobs:
        out = results_dir / runner.key / image_set / f"{fig_id}.json"
        if out.exists() and not force:
            continue
        todo.append((image_set, fig_id, path, out))
    if not todo:
        log(f"[{runner.key}] all {len(jobs)} answers already cached - nothing to run")
        return
    t0 = time.time()
    runner.load()
    log(f"[{runner.key}] loaded in {time.time() - t0:.0f}s on {getattr(runner, 'device', 'cpu')}")
    try:
        for image_set, fig_id, path, out in todo:
            t1 = time.time()
            try:
                raw = runner.generate(path, fig_id)
                err = None
            except Exception as e:  # keep going: one failure must not kill the whole model run
                raw, err = "", f"{type(e).__name__}: {e}"
            parsed = runner.parse(raw, fig_id) if raw else {"series": {}}
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"model": runner.key, "image_set": image_set, "figure": fig_id,
                                       "image": str(path), "seconds": round(time.time() - t1, 1),
                                       "run_config": runner.describe_input(path, fig_id),
                                       "error": err, "raw": raw, "parsed": parsed}, indent=1))
            n = sum(len(s["x"]) for s in parsed["series"].values())
            log(f"[{runner.key}] {image_set:9s} {fig_id:9s} {time.time() - t1:6.1f}s  "
                f"{len(parsed['series'])} series / {n} points" + (f"  ERROR {err}" if err else ""))
    finally:
        runner.unload()
