"""Every configurable parameter of the benchmark, in one place.

The notebook shows this whole dictionary as a table in its first cell and lets
you override any value there (``config.update({...})``) before anything runs.
Each answer a model gives is saved together with the exact settings that
produced it (``run_config`` inside outputs/model_results/.../*.json), so a
result can always be traced back to its parameters.

Values marked "as used in the first run" are the defaults the cached results
in outputs/model_results/ were produced with; changing a value that affects
the model input (image scaling, prompt mode, ...) automatically gives the
model a new result folder (see ``run_tag``), so old results are never
overwritten or mixed with new ones.
"""
from __future__ import annotations

import copy

DEFAULTS = {
    # ------------------------------------------------------------ images
    "images": {
        "crop": False,                 # figures are NEVER cropped: the full image file is used
        "original_sizes_px": {         # native size of each original (W x H), fed as-is
            "ftir": (493, 437), "pore": (554, 431), "methane": (603, 470),
            "dosage": (687, 526), "n2": (469, 367), "kinetics": (701, 554),
        },
        "rebuilt_size": "identical to the original, pixel for pixel",
        "rebuilt_dpi": 100,            # matplotlib DPI of the rebuilt figures (1 inch = 100 px)
        "rebuilt_fonts": "Times New Roman (serif figures) / Arial (sans figures); "
                         "Liberation Serif / Sans if those are not installed",
    },
    # --------------------------------------------- input given to the VLMs
    "vlm_input": {
        # 1.0 = feed the image at its native resolution (first run). 2.0 = upscale
        # 2x (LANCZOS) before the model sees it. Small figures give the model very
        # few visual tokens (see the diagnostics section), so this is the first
        # knob to try.
        "upscale": 1.0,
        # Qwen2.5-VL processor cap on image area (pixels). Images larger than this
        # are shrunk back down, so raise it together with "upscale".
        "qwen25_max_pixels": 1280 * 28 * 28,
        # olmOCR was trained on pages rendered at 1288 px on the long side; its
        # input is always resized to this.
        "olmocr_long_side": 1288,
        # DePlot / Pix2Struct: None = processor default patch budget.
        "deplot_max_patches": None,
    },
    # ------------------------------------------------------------ prompting
    "prompt": {
        # "all_series": one request per figure asking for every series at once
        #               (first run).
        # "per_series": one request per series - shorter answers, less chance
        #               the model falls back to an invented regular grid.
        "mode": "all_series",
        "output_format": "JSON object {series name: [[x, y], ...]}",
        "include_axis_ranges": True,   # the axis ranges are given in the text (never data values)
        "olmocr_native_prompt": "olmOCR's own page-to-markdown prompt (olmocr build_no_anchoring_v4_yaml_prompt)",
    },
    # ------------------------------------------------------------ generation
    "generation": {
        "max_new_tokens": 2048,        # Qwen / olmOCR
        "deplot_max_new_tokens": 1024,
        "do_sample": False,            # greedy decoding -> deterministic answers
        "dtype": "bfloat16 on GPU/MPS, float32 on CPU",
        "device": "auto: cuda > mps > cpu",
    },
    # ------------------------------------------------------------ models
    "models": {
        # key: (Hugging Face id, run this cell when its answers are not cached yet?)
        "classical":      ("classical digitizer (no download)", True),
        "qwen2.5-vl-3b":  ("Qwen/Qwen2.5-VL-3B-Instruct", True),
        "qwen3-vl-4b":    ("Qwen/Qwen3-VL-4B-Instruct", True),
        "deplot":         ("google/deplot", True),
        "olmocr2-native": ("allenai/olmOCR-2-7B-1025", True),
        "olmocr2-json":   ("allenai/olmOCR-2-7B-1025", False),   # not run yet
        "qwen2.5-vl-7b":  ("Qwen/Qwen2.5-VL-7B-Instruct", True),
        "qwen3-vl-8b":    ("Qwen/Qwen3-VL-8B-Instruct", False),  # not run yet
    },
    # ------------------------------------------------------------ scoring
    "scoring": {
        # error of one point = |y_model - y_truth| / (span of that y-axis) x 100
        "error_unit": "% of the y-axis span",
        "error_cap": 100.0,            # one point can't cost more than 100 %
        "hit_threshold": 5.0,          # a point is "correct" if its error <= 5 % of the axis span
        "missing_point_error": 100.0,  # a point the model did not return counts as 100 % error
        "line_points": 25,             # traced curves (FTIR, kinetic fits) are compared at 25 evenly spaced x
        "x_tolerance": 0.5,            # model points may sit up to 0.5 x the reference spacing beyond its own x-range
        "image_set_for_tables": "rebuilt",   # tables under each model cell: "rebuilt" (exact truth) or "original"
        "show_original_tables_too": False,
    },
}

CONFIG = copy.deepcopy(DEFAULTS)


def update(overrides: dict):
    """config.update({"vlm_input": {"upscale": 2.0}}) - nested, partial updates."""
    for section, values in overrides.items():
        if isinstance(values, dict) and section in CONFIG:
            CONFIG[section].update(values)
        else:
            CONFIG[section] = values
    return CONFIG


def reset():
    CONFIG.clear(); CONFIG.update(copy.deepcopy(DEFAULTS))


def run_tag():
    """Suffix added to model result folders when the model input differs from the
    first run, e.g. 'qwen3-vl-4b' -> 'qwen3-vl-4b__x2__perseries'."""
    tag = ""
    if CONFIG["vlm_input"]["upscale"] != 1.0:
        tag += f"__x{CONFIG['vlm_input']['upscale']:g}"
    if CONFIG["prompt"]["mode"] != "all_series":
        tag += "__perseries"
    return tag


def as_table():
    """Flat (section, parameter, value) rows for display."""
    rows = []
    for section, values in CONFIG.items():
        for k, v in values.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    rows.append((section, f"{k}.{kk}", vv))
            else:
                rows.append((section, k, v))
    return rows
