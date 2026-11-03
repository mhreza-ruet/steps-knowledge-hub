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

import matplotlib
matplotlib.use("Agg")
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


def side_by_side(orig, rec, out, title=""):
    a, b = Image.open(orig).convert("RGB"), Image.open(rec).convert("RGB")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2 * max(a.height / a.width, b.height / b.width)))
    for x, im, t in zip(ax, (a, b), ("Original", "Recreated from extracted data")):
        x.imshow(im); x.set_title(t, fontsize=12, fontweight="bold"); x.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
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
