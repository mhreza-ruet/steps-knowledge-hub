"""Hand-calibrated profiles of the extended figure set (``../data`` and ``../data/test_data``).

Same principle as ``figures.py``: every number here was measured on the original image's own
pixels (frame lines, tick marks -> pixel/value pairs, bar centres, series colours) and checked
on an overlay. A profile never contains data values - those are measured by ``engine.py``.

One figure = one or more PANELS. A panel has its own frame, axes and series; role "inset" marks
a small plot drawn inside another one. Axis maps are (pixel0, value0, pixel1, value1).
"""
from __future__ import annotations

PROFILES = {}


def add(fid, **F):
    F["id"] = fid
    if not F.get("size"):
        from pathlib import Path
        from PIL import Image
        F["size"] = Image.open(Path(__file__).resolve().parents[2] / "data" / F["file"]).size
    PROFILES[fid] = F
    return F


# Images in the folders that get NO ground truth from this set, with the reason (shown in the notebook).
NOT_PROFILED = {
    "test_data/surface3d_1.png": "3-D response surface (Design-Expert style): no colour bar, z cannot be read from colour; "
                                 "the camera is not a matplotlib view - needs a hand-fitted 3-D camera (not done yet)",
    "test_data/surface3d_2.png": "3-D response surface: same as surface3d_1",
    "test_data/surface3d_3.png": "3-D response surface: same as surface3d_1",
    "test_data/surface3d_4.png": "3-D response surface: same as surface3d_1",
    "test_data/surface3d_5.png": "3-D response surface: same as surface3d_1",
    "test_data/surface3d_6.png": "3-D response surface: same as surface3d_1",
    "test_data/contour2d_1.png": "EEM fluorescence map: the colour bar has no numbers, so intensities have no values to recover",
    "test_data/contour2d_2.png": "EEM fluorescence map: the colour bar has no numbers, so intensities have no values to recover",
    "test_data/bar_grouped_2.png": "same figure as concentration.png (already in the original 12 with ground truth)",
    "test_data/bar_grouped_4.png": "same figure as concentration2.png (already in the original 12 with ground truth)",
}
