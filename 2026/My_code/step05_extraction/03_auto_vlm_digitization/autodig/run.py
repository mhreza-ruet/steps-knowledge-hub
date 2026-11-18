"""One call per image: VLM spec (+ pixel check) -> digitize -> CSV -> recreated plot.

    reader = QwenSpecReader()                    # or FileSpecReader(dir) for offline tests
    r = process_image("fig.png", reader, "outputs", name="kinetics")

Writes into outputs/<name>/:
    vlm_spec.json       raw VLM answers + parsed spec (edit 'spec' and set use_cache to re-run with your fix)
    extracted.csv       series, x, y, y_axis, kind
    extraction.json     everything the digitizer found (frame, calibration, legend, pixels, warnings)
    recreated.png       the plot redrawn from extracted.csv  (+ recreated.spec.json sidecar)
    side_by_side.png    input | recreated
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import sys as _sys
import matplotlib
if "ipykernel" not in _sys.modules:      # scripts: no screen needed. In a notebook keep its inline backend,
    matplotlib.use("Agg")                # otherwise plt.show() there would display nothing
import matplotlib.pyplot as plt
from PIL import Image

from . import evaluate as E
from .extract import pipeline
from .render import recreate
from .spec import normalize, renormalize


def _jsonable(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def read_spec(image, reader, cache_path, use_cache=True, check=True):
    """Spec for one image, cached as JSON (so the model runs once per image).

    1. the VLM answers the two questions (spec.AXES_PROMPT, spec.SERIES_PROMPT)
    2. check.check_spec compares it with what the pixels show (tick-label counts,
       right axis) and re-asks the VLM a focused question where they disagree
    The cache keeps the first answer, every re-ask and the final spec. A cache
    written with older prompts (different PROMPT_VERSION) is ignored; one written
    with older checks (CHECK_VERSION) keeps the VLM answers and is re-checked."""
    from .spec import PROMPT_VERSION
    from .check import check_spec, CHECK_VERSION
    cache_path = Path(cache_path)
    is_vlm = hasattr(reader, "model_id")
    if use_cache and cache_path.exists():
        d = json.loads(cache_path.read_text())
        if not is_vlm or d.get("prompt_version") == PROMPT_VERSION:
            if is_vlm and check and d.get("check_version") != CHECK_VERSION and d.get("spec_first_answer"):
                # same VLM answers, newer pixel checks: re-check the first answer (asks only the follow-ups)
                d["spec"], d["checks"] = check_spec(image, d["spec_first_answer"], reader)
                d["check_version"] = CHECK_VERSION
                cache_path.write_text(json.dumps(_jsonable(d), indent=1))
            d["spec"] = renormalize(d["spec"])
            d["from_cache"] = True
            return d
    t = time.time()
    d = reader.read(image)
    d["spec_first_answer"] = d["spec"]
    d["checks"] = []
    if check:
        d["spec"], d["checks"] = check_spec(image, d["spec"], reader)
    d["seconds"] = round(time.time() - t, 1)
    d["reader"] = type(reader).__name__ + (f" ({reader.model_id})" if is_vlm else "")
    d["prompt_version"] = PROMPT_VERSION
    d["check_version"] = CHECK_VERSION
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(_jsonable(d), indent=1))
    d["from_cache"] = False
    return d


def side_by_side(orig, rec, out, title="", height=900, gap=24, pad=12):
    """original | recreated, composed at pixel level (no figure margins): both images scaled to the
    same height (default 900 px, never shrunk below their own size), a thin gap between them, a
    title line above each and an optional caption on top. The two plots fill the canvas."""
    from PIL import ImageDraw, ImageFont
    a, b = Image.open(orig).convert("RGB"), Image.open(rec).convert("RGB")
    h = max(height, a.height, b.height)
    a = a.resize((round(a.width * h / a.height), h), Image.LANCZOS)
    b = b.resize((round(b.width * h / b.height), h), Image.LANCZOS)

    def font(size, bold=False):
        try:
            from matplotlib import font_manager
            return ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(weight="bold" if bold else "normal")), size)
        except Exception:
            return ImageFont.load_default()
    f_head, f_cap = font(max(20, h // 32), True), font(max(18, h // 38))
    head = f_head.size + 2 * pad
    cap = (f_cap.size + 2 * pad) if title else 0
    W = a.width + gap + b.width + 2 * pad
    H = cap + head + h + pad
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    if title:
        d.text((W / 2, pad + f_cap.size / 2), title, fill=(0, 0, 0), font=f_cap, anchor="mm")
    xa, xb = pad, pad + a.width + gap
    for x0, im, t in ((xa, a, "Original"), (xb, b, "Recreated from extracted data")):
        d.text((x0 + im.width / 2, cap + pad + f_head.size / 2), t, fill=(0, 0, 0), font=f_head, anchor="mm")
        canvas.paste(im, (x0, cap + head))
    d.line([(xb - gap / 2, cap + head), (xb - gap / 2, H - pad)], fill=(225, 225, 225), width=2)
    canvas.save(out)
    return out


def process_image(image, reader, out_root, name=None, use_cache=True, check=True):
    """VLM spec (+ pixel check) -> digitize -> extracted.csv -> recreated.png.
    Writes into out_root/<name>/ and returns everything."""
    image = Path(image)
    name = name or image.stem
    od = Path(out_root) / name
    od.mkdir(parents=True, exist_ok=True)
    sp = read_spec(image, reader, od / "vlm_spec.json", use_cache, check)
    t = time.time()
    res = pipeline(image, sp["spec"])
    secs = time.time() - t
    (od / "extraction.json").write_text(json.dumps(_jsonable(res), indent=1))
    E.write_csv(res, od / "extracted.csv")
    rec = recreate(res, od / "recreated.png")
    side_by_side(image, rec, od / "side_by_side.png", name)
    return {"name": name, "dir": str(od), "spec_info": sp, "result": res, "recreated": rec, "digitize_seconds": secs}
