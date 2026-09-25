#!/usr/bin/env python3
"""
step04a_measure_dolphin_recall.py
----------------------------------
Answers one question before we decide whether Dolphin is worth keeping in
the pipeline at all: of every box we actually labelled by hand (Table,
Plot, Diagram, Figure_Mixed -- the same "detector dataset" definition
step03/step04a_prepare_yolo_dataset use), how many did Dolphin's zero-shot
layout detector *find at all*, regardless of what label it gave them or
how many extra/nested boxes it also drew?

Dolphin over-detects on purpose (it happily draws a box for a sub-panel
INSIDE a figure, a sub-table, etc.), so a plain "same box" comparison is
the wrong test -- a Dolphin box that's much bigger (it grabbed a whole
composite figure our own box only partially covers) or one of several
smaller Dolphin boxes tiling our one box should still count as "found it".
So a ground-truth box counts as detected if the SINGLE best-matching
Dolphin box on that page clears either:

  - IoU >= --iou-threshold (well-aligned boxes), or
  - containment(gt, det) >= --containment-threshold (most of our box's
    area is covered by one Dolphin box, even if that Dolphin box is much
    bigger because it grabbed a whole composite region)

If recall comes out ~100%, the plan is: keep Dolphin purely as a region
proposer, and train a small classifier head on top of its detections to
sort "figure/table we want" from everything else it also happens to find
(sub-panels, headers, equations, ...) -- cheaper than fine-tuning a whole
detector from scratch, IF this recall check holds up.

Reuses step04a_prepare_yolo_dataset.collect_pages() for ground truth, so
"what counts as a labelled box" never drifts between the YOLO/Docling
fine-tuning dataset and this comparison -- same Figure_Mixed/nested-child
resolution logic throughout.

Usage:
    python step04a_measure_dolphin_recall.py \\
        export.ndjson \\
        --images-root ../../../Pre-Pages-2Annotate \\
        --dolphin-csv ./dolphin_results/dolphin_flat.csv \\
        --output-csv ./dolphin_recall.csv

Requirements:
    pip install pillow pandas
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("[Error] pandas not installed. Run: pip install pandas")

# Reuse step04a's page/box collection (which itself reuses step03's
# effective_class / resolve_composites / select_labels) instead of
# re-deriving "what's a ground-truth box" a third time.
# Folder layout: My_code/step04_detection/01_dolphin_baseline/<this file>
#                My_code/step04_detection/03_doclayout_yolo_finetune/step04a_prepare_yolo_dataset.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_doclayout_yolo_finetune"))
from step04a_prepare_yolo_dataset import collect_pages  # noqa: E402


# ── box geometry ────────────────────────────────────────────────────────────

def _corners(box):
    """box = (left, top, width, height) -> (x1, y1, x2, y2)."""
    left, top, width, height = box
    return left, top, left + width, top + height


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = _corners(box_a)
    bx1, by1, bx2, by2 = _corners(box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1e-9, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-9, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def containment(inner_box, outer_box) -> float:
    """Fraction of inner_box's area that lies inside outer_box."""
    ix1, iy1, ix2, iy2 = _corners(inner_box)
    ox1, oy1, ox2, oy2 = _corners(outer_box)
    cx1, cy1 = max(ix1, ox1), max(iy1, oy1)
    cx2, cy2 = min(ix2, ox2), min(iy2, oy2)
    inter = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    inner_area = max(1e-9, (ix2 - ix1) * (iy2 - iy1))
    return inter / inner_area


# ── load Dolphin's output ────────────────────────────────────────────────────

def load_dolphin_detections(csv_path: Path) -> dict:
    """dolphin_flat.csv (written by Dolphin_Inference.ipynb) -> {image: [detections]}."""
    by_image = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            by_image.setdefault(row["image"], []).append({
                "label": row["dolphin_label"],
                "box": (float(row["left"]), float(row["top"]),
                        float(row["width"]), float(row["height"])),
            })
    return by_image


# ── matching ─────────────────────────────────────────────────────────────────

def match_ground_truth(pages: list, detections_by_image: dict,
                        iou_threshold: float = 0.5,
                        containment_threshold: float = 0.8) -> list:
    """
    For every ground-truth box on a page Dolphin actually ran on, finds the
    single best-matching Dolphin detection (by max(IoU, containment)) and
    records whether that clears either threshold.

    Pages with no entry in detections_by_image are skipped (Dolphin wasn't
    run on that page yet -- e.g. you pointed it at one topic folder only),
    not counted as "0 boxes found".
    """
    records = []
    for page in pages:
        external_id = page["external_id"]
        if external_id not in detections_by_image:
            continue
        dets = detections_by_image[external_id]

        for class_name, top, left, width, height in page["boxes"]:
            gt_box = (left, top, width, height)

            best = {"iou": 0.0, "containment": 0.0, "label": None, "det_box": None}
            for d in dets:
                d_iou = iou(gt_box, d["box"])
                d_contain = containment(gt_box, d["box"])
                score = max(d_iou, d_contain)
                best_score = max(best["iou"], best["containment"])
                if score > best_score:
                    best = {"iou": d_iou, "containment": d_contain,
                            "label": d["label"], "det_box": d["box"]}

            matched = (best["iou"] >= iou_threshold or
                       best["containment"] >= containment_threshold)

            records.append({
                "image": external_id, "class_name": class_name,
                "gt_box": gt_box, "matched": matched,
                "best_iou": round(best["iou"], 3),
                "best_containment": round(best["containment"], 3),
                "best_dolphin_label": best["label"],
                "best_dolphin_box": best["det_box"],
            })
    return records


def compute_recall(records: list) -> pd.DataFrame:
    """Per-class + overall recall table from match_ground_truth()'s output."""
    if not records:
        return pd.DataFrame(columns=["class_name", "n_boxes", "n_matched", "recall"])

    df = pd.DataFrame(records)
    per_class = (
        df.groupby("class_name")
        .agg(n_boxes=("matched", "size"), n_matched=("matched", "sum"))
        .reset_index()
    )
    per_class["recall"] = per_class["n_matched"] / per_class["n_boxes"]

    overall = pd.DataFrame([{
        "class_name": "ALL",
        "n_boxes": len(df),
        "n_matched": int(df["matched"].sum()),
        "recall": df["matched"].mean(),
    }])
    return pd.concat([per_class, overall], ignore_index=True)


# ── end-to-end convenience function (for notebook use) ──────────────────────

def measure_recall(ndjson_path, images_root, dolphin_csv_path,
                    label_select: str = "first", containment_threshold_gt: float = 0.85,
                    iou_threshold: float = 0.5, containment_threshold: float = 0.8,
                    verbose: bool = True) -> dict:
    """
    Does the full job: collects ground-truth detector-dataset boxes, loads
    Dolphin's detections, matches them, and returns
    {"records": [...], "recall_table": DataFrame, "missed": [...]}.

    NOTE: `containment_threshold_gt` is step03/step04a's threshold for
    resolving Figure_Mixed composites (unrelated to `containment_threshold`,
    which is this script's own detected/not-detected cutoff) -- named
    differently on purpose so the two are never confused.
    """
    ndjson_path = Path(ndjson_path).expanduser().resolve()
    images_root = Path(images_root).expanduser().resolve()
    dolphin_csv_path = Path(dolphin_csv_path).expanduser().resolve()

    if not ndjson_path.exists():
        raise FileNotFoundError(f"NDJSON file not found: {ndjson_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")
    if not dolphin_csv_path.exists():
        raise FileNotFoundError(
            f"Dolphin detections CSV not found: {dolphin_csv_path} "
            f"-- run Dolphin_Inference.ipynb first."
        )

    pages, _categories = collect_pages(ndjson_path, images_root, label_select,
                                        split_figure_subtype=True,
                                        containment_threshold=containment_threshold_gt)
    detections_by_image = load_dolphin_detections(dolphin_csv_path)

    n_pages_with_dets = sum(1 for p in pages if p["external_id"] in detections_by_image)
    records = match_ground_truth(pages, detections_by_image, iou_threshold, containment_threshold)
    recall_table = compute_recall(records)
    missed = [r for r in records if not r["matched"]]

    if verbose:
        print("─" * 54)
        print(f"Pages with ground truth: {len(pages)}")
        print(f"Pages Dolphin was run on (found in {dolphin_csv_path.name}): {n_pages_with_dets}")
        if n_pages_with_dets < len(pages):
            print(f"  (skipping {len(pages) - n_pages_with_dets} page(s) Dolphin hasn't seen yet)")
        print(f"\nMatch thresholds: IoU >= {iou_threshold}  OR  containment(gt in det) >= {containment_threshold}\n")
        print(recall_table.to_string(index=False))
        if missed:
            print(f"\n{len(missed)} ground-truth box(es) Dolphin did NOT find:")
            for m in missed:
                print(f"    - {m['image']}  [{m['class_name']}]  "
                      f"best_iou={m['best_iou']}  best_containment={m['best_containment']}  "
                      f"best_dolphin_label={m['best_dolphin_label']}")
        else:
            print("\nEvery ground-truth box was found by Dolphin -- 100% recall at these thresholds.")

    return {"records": records, "recall_table": recall_table, "missed": missed,
            "pages": pages, "detections_by_image": detections_by_image}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ndjson", help="Path to the Labelbox NDJSON export file.")
    parser.add_argument("--images-root", required=True,
                         help="Root directory containing the local page PNGs (searched recursively).")
    parser.add_argument("--dolphin-csv", required=True,
                         help="Path to dolphin_flat.csv, written by Dolphin_Inference.ipynb.")
    parser.add_argument("--label-select", choices=["first", "all"], default="first")
    parser.add_argument("--containment-threshold-gt", type=float, default=0.85,
                         help="step03/step04a's Figure_Mixed composite-resolution threshold "
                              "(default 0.85) -- NOT the detected/missed cutoff below.")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                         help="A ground-truth box counts as found if the best-matching "
                              "Dolphin box's IoU is at least this (default 0.5).")
    parser.add_argument("--containment-threshold", type=float, default=0.8,
                         help="...OR if at least this fraction of the ground-truth box's "
                              "area is covered by the best-matching Dolphin box (default 0.8) "
                              "-- handles Dolphin drawing a bigger box around a whole "
                              "composite region.")
    parser.add_argument("--output-csv", default=None,
                         help="If set, writes the full per-box match table here.")
    args = parser.parse_args()

    try:
        result = measure_recall(
            args.ndjson, args.images_root, args.dolphin_csv, args.label_select,
            args.containment_threshold_gt, args.iou_threshold, args.containment_threshold,
        )
    except FileNotFoundError as e:
        sys.exit(f"[Error] {e}")

    if args.output_csv:
        pd.DataFrame(result["records"]).to_csv(args.output_csv, index=False)
        print(f"\nFull match table written to: {args.output_csv}")


if __name__ == "__main__":
    main()
