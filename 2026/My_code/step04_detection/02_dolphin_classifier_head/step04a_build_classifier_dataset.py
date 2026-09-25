#!/usr/bin/env python3
"""
step04a_build_classifier_dataset.py
-------------------------------------
Turns Dolphin's (over-)detections into a labelled image-classification
dataset: one crop per Dolphin-detected box, labelled with whichever of our
4 classes (Table/Plot/Diagram/Figure_Mixed) it actually corresponds to, or
"Background" if it doesn't correspond to anything we labelled (a heading,
an equation, a sub-panel duplicate, running text Dolphin also boxed, etc.).

This is the training data for the classifier HEAD described in the recall
analysis: since Dolphin's recall against our ground truth is high, the
remaining problem isn't finding the regions -- it's that Dolphin also
happily finds a lot of things we don't want. A small classifier trained on
Dolphin's *own* box shapes (not our tighter/looser hand-drawn boxes) can
sit on top of it at inference time and keep only the boxes worth handing
downstream.

Matching a Dolphin box to a class (see `label_detections`): a Dolphin box
is assigned the class of whichever ground-truth box best overlaps it, if
either
  - IoU(dolphin_box, gt_box) >= --iou-threshold, or
  - containment(dolphin_box, gt_box) >= --containment-threshold
    (most of the DOLPHIN box's own area lies inside that gt_box -- this is
    what catches the "Dolphin only found part of a Plot" case: a small
    sub-panel Dolphin boxed is still worth labelling Plot, since it's part
    of a plot we care about, even though it isn't the box's full extent)
Otherwise the crop is labelled "Background".

Note the direction is reversed from step04a_measure_dolphin_recall.py's
containment check (there it was "how much of our GT box did Dolphin
cover", for a recall count; here it's "how much of Dolphin's own box
overlaps our GT region", for labelling that specific crop) -- both use the
same containment() function, just with inner/outer swapped.

Splits by PAPER (not page), same convention as step04a_prepare_yolo_dataset,
so a classifier's validation accuracy actually reflects generalization to
unseen papers.

Usage:
    python step04a_build_classifier_dataset.py \\
        export.ndjson \\
        --images-root ../../../Pre-Pages-2Annotate \\
        --dolphin-csv ../01_dolphin_baseline/dolphin_results/dolphin_flat.csv \\
        --output ./classifier_dataset

Requirements:
    pip install pillow
"""

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("[Error] Pillow not installed. Run: pip install pillow")

# Reuse step04a_prepare_yolo_dataset's ground-truth collection (itself reusing
# step03's effective_class / resolve_composites / select_labels) and
# step04a_measure_dolphin_recall's box geometry + Dolphin-CSV loader, instead
# of re-deriving either a fourth time.
# Folder layout: My_code/step04_detection/03_dolphin_classifier_head/<this file>
#                My_code/step04_detection/03_doclayout_yolo_finetune/step04a_prepare_yolo_dataset.py
#                My_code/step04_detection/01_dolphin_baseline/step04a_measure_dolphin_recall.py
_STEP04_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_STEP04_DIR / "03_doclayout_yolo_finetune"))
sys.path.insert(0, str(_STEP04_DIR / "01_dolphin_baseline"))
from step04a_prepare_yolo_dataset import collect_pages, paper_stem  # noqa: E402
from step04a_measure_dolphin_recall import containment, iou, load_dolphin_detections  # noqa: E402

BACKGROUND = "Background"


def label_detections(pages: list, detections_by_image: dict,
                      iou_threshold: float = 0.3,
                      containment_threshold: float = 0.5) -> list:
    """
    For every Dolphin detection on a page we have ground truth for, finds
    the best-overlapping ground-truth box and assigns its class if the
    match clears either threshold, else BACKGROUND.

    Returns a flat list of dicts: image, dolphin_label, box (left, top,
    width, height), target_class, best_iou, best_containment.
    """
    records = []
    for page in pages:
        external_id = page["external_id"]
        if external_id not in detections_by_image:
            continue

        for d in detections_by_image[external_id]:
            best = {"class_name": BACKGROUND, "iou": 0.0, "containment": 0.0}
            for class_name, top, left, width, height in page["boxes"]:
                gt_box = (left, top, width, height)
                d_iou = iou(d["box"], gt_box)
                # containment(inner=dolphin box, outer=gt box): how much of
                # THIS Dolphin box's own area falls inside the labelled region.
                d_contain = containment(d["box"], gt_box)
                score = max(d_iou, d_contain)
                best_score = max(best["iou"], best["containment"])
                if score > best_score:
                    best = {"class_name": class_name, "iou": d_iou, "containment": d_contain}

            matched = (best["iou"] >= iou_threshold or
                       best["containment"] >= containment_threshold)
            target_class = best["class_name"] if matched else BACKGROUND

            records.append({
                "image": external_id, "local_path": page["local_path"],
                "dolphin_label": d["label"], "box": d["box"],
                "target_class": target_class,
                "best_iou": round(best["iou"], 3),
                "best_containment": round(best["containment"], 3),
            })
    return records


def split_by_paper(records: list, val_fraction: float, seed: int = 42) -> tuple:
    papers = sorted({paper_stem(r["image"]) for r in records})
    rng = random.Random(seed)
    rng.shuffle(papers)
    n_val = max(1, round(len(papers) * val_fraction))
    val_papers = set(papers[:n_val])

    train, val = [], []
    for r in records:
        (val if paper_stem(r["image"]) in val_papers else train).append(r)
    return train, val, sorted(val_papers)


def write_crops(records: list, split_name: str, output_root: Path, all_classes: set,
                 pad: int = 8) -> dict:
    """Crops every record's Dolphin box out of its page image into
    output_root/<split_name>/<target_class>/*.png (ImageFolder-ready).

    Pre-creates an (empty, if need be) folder for every class in
    `all_classes` -- torchvision's ImageFolder infers its class list from
    whichever subfolders exist, so a rare class with zero examples in this
    split (Figure_Mixed only has a handful total) would otherwise silently
    vanish from this split's class list and break train/val alignment.
    """
    for class_name in all_classes:
        (output_root / split_name / class_name).mkdir(parents=True, exist_ok=True)

    counts = {}
    open_images = {}
    for i, r in enumerate(records):
        local_path = r["local_path"]
        if local_path not in open_images:
            open_images[local_path] = Image.open(local_path).convert("RGB")
        im = open_images[local_path]
        img_w, img_h = im.size

        left, top, width, height = r["box"]
        l = max(0, int(left) - pad)
        t = max(0, int(top) - pad)
        rgt = min(img_w, int(left + width) + pad)
        b = min(img_h, int(top + height) + pad)
        if rgt <= l or b <= t:
            continue  # degenerate box, skip

        class_dir = output_root / split_name / r["target_class"]
        class_dir.mkdir(parents=True, exist_ok=True)
        crop = im.crop((l, t, rgt, b))
        out_name = f"{Path(r['image']).stem}__{i}.png"
        crop.save(class_dir / out_name)
        counts[r["target_class"]] = counts.get(r["target_class"], 0) + 1

    for im in open_images.values():
        im.close()
    return counts


def build_classifier_dataset(ndjson_path, images_root, dolphin_csv_path, output_root,
                              label_select: str = "first", containment_threshold_gt: float = 0.85,
                              iou_threshold: float = 0.3, containment_threshold: float = 0.5,
                              val_fraction: float = 0.15, seed: int = 42, pad: int = 8,
                              verbose: bool = True) -> dict:
    """
    Full job (usable from a notebook): builds the ground truth, loads
    Dolphin's detections, labels every detection, splits by paper, writes
    an ImageFolder-style crop dataset, and returns a stats dict.
    """
    ndjson_path = Path(ndjson_path).expanduser().resolve()
    images_root = Path(images_root).expanduser().resolve()
    dolphin_csv_path = Path(dolphin_csv_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()

    if not ndjson_path.exists():
        raise FileNotFoundError(f"NDJSON file not found: {ndjson_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")
    if not dolphin_csv_path.exists():
        raise FileNotFoundError(f"Dolphin detections CSV not found: {dolphin_csv_path}")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    pages, _categories = collect_pages(ndjson_path, images_root, label_select,
                                        split_figure_subtype=True,
                                        containment_threshold=containment_threshold_gt)
    detections_by_image = load_dolphin_detections(dolphin_csv_path)
    missing_detections = [p["external_id"] for p in pages
                          if p["external_id"] not in detections_by_image]
    if missing_detections:
        print(f"[Warning] {len(missing_detections)} labeled pages have no detections in the CSV. "
              "Rerun Dolphin_Inference.ipynb after adding annotations to include new pages. "
              "Pages with no detected elements also have no CSV rows.")

    records = label_detections(pages, detections_by_image, iou_threshold, containment_threshold)
    train_records, val_records, val_papers = split_by_paper(records, val_fraction, seed)
    all_classes = {r["target_class"] for r in records}

    train_counts = write_crops(train_records, "train", output_root, all_classes, pad)
    val_counts = write_crops(val_records, "val", output_root, all_classes, pad)

    manifest_path = output_root / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "dolphin_label", "target_class",
                                                "best_iou", "best_containment"])
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    classes = sorted({r["target_class"] for r in records})
    stats = {
        "output_root": output_root, "manifest_path": manifest_path, "classes": classes,
        "n_pages_with_detections": len({r["image"] for r in records}),
        "val_papers": val_papers,
        "train_counts": train_counts, "val_counts": val_counts,
    }

    if verbose:
        print("─" * 54)
        print(f"Pages with Dolphin detections matched against ground truth: "
              f"{stats['n_pages_with_detections']}")
        print(f"Classes: {classes}")
        print(f"\nTrain crops:")
        for c in classes:
            print(f"  {c:<14} {train_counts.get(c, 0)}")
        print(f"\nVal crops:")
        for c in classes:
            print(f"  {c:<14} {val_counts.get(c, 0)}")

        small_classes = [c for c in classes if c != BACKGROUND
                         and train_counts.get(c, 0) + val_counts.get(c, 0) < 15]
        if small_classes:
            print(f"\n[Warning] Very few examples for: {small_classes} -- expect noisy "
                  f"per-class metrics for these until more annotated pages come in. "
                  f"Consider merging rare classes or watching recall/precision on them "
                  f"specifically rather than overall accuracy.")

        n_background = train_counts.get(BACKGROUND, 0) + val_counts.get(BACKGROUND, 0)
        n_wanted = sum(train_counts.get(c, 0) + val_counts.get(c, 0) for c in classes if c != BACKGROUND)
        if n_background > 3 * max(n_wanted, 1):
            print(f"\n[Note] Background crops ({n_background}) heavily outnumber wanted-class "
                  f"crops ({n_wanted}), as expected since Dolphin also boxes headings/equations/etc. "
                  f"The training script class-weights the loss for this by default.")

        print(f"\nWritten to: {output_root}/")
        print("  train/<class>/*.png, val/<class>/*.png   (ImageFolder-ready)")
        print("  manifest.csv                              every detection + its assigned label")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ndjson", help="Path to the Labelbox NDJSON export file.")
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--dolphin-csv", required=True,
                         help="Path to dolphin_flat.csv, written by Dolphin_Inference.ipynb.")
    parser.add_argument("--output", default="./classifier_dataset")
    parser.add_argument("--label-select", choices=["first", "all"], default="first")
    parser.add_argument("--containment-threshold-gt", type=float, default=0.85,
                         help="step03/step04a's Figure_Mixed composite-resolution threshold.")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--containment-threshold", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.15,
                         help="Fraction of PAPERS held out for validation (default 0.15).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pad", type=int, default=8, help="Padding in pixels around each crop.")
    args = parser.parse_args()

    try:
        build_classifier_dataset(
            args.ndjson, args.images_root, args.dolphin_csv, args.output, args.label_select,
            args.containment_threshold_gt, args.iou_threshold, args.containment_threshold,
            args.val_fraction, args.seed, args.pad,
        )
    except FileNotFoundError as e:
        sys.exit(f"[Error] {e}")


if __name__ == "__main__":
    main()
