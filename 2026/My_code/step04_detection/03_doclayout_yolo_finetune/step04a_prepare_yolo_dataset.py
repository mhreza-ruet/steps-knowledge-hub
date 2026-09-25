#!/usr/bin/env python3
"""
step04a_prepare_yolo_dataset.py
--------------------------------
Turns the Labelbox NDJSON export into a standard Ultralytics-style YOLO
training layout, ready to fine-tune DocLayout-YOLO (or plain YOLO) on:

    yolo_dataset/
      images/train/*.png
      images/val/*.png
      labels/train/*.txt      (YOLO format: class cx cy w h, normalized)
      labels/val/*.txt
      data.yaml                # train/val paths + class names, in class-id order
      classes.txt               # same class names, one per line

Reuses the exact same box-resolution logic as step03
(effective_class / resolve_composites / select_labels) by importing it
directly, so "what counts as a detector-dataset box" never drifts between
the two scripts: a Figure_Mixed composite's outer box is kept, its nested
Plot/Diagram sub-boxes are excluded here too (they go to step03's
extraction_crops instead, not into detector training).

Two things this script adds on top of step03's output:

  1. Background (no-graphics) pages are included as negative examples --
     an image with an EMPTY label file. Object detectors trained only on
     positive examples tend to over-predict; giving the model pages that
     truly have nothing to detect helps it learn what "no figure/table
     here" looks like, which matters once it runs on the other ~90
     unannotated papers.

  2. The train/val split is done AT THE PAPER LEVEL, not the page level.
     Pages from the same paper share fonts, column layout, and figure
     style, so a random page-level split would leak that paper's specific
     style into validation and overstate how well the model generalizes
     to a paper it's never seen -- which is exactly the situation for the
     other ~90 papers this model needs to run on next.

Usage:
    python step04a_prepare_yolo_dataset.py \\
        export.ndjson \\
        --images-root ../Pre-Pages-2Annotate \\
        --output ./yolo_dataset \\
        --val-fraction 0.15

Requirements:
    pip install pillow
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("[Error] Pillow not installed. Run: pip install pillow")

# Reuse step03's box-resolution logic instead of re-deriving it.
# Folder layout: My_code/step03_labelbox_export/step03_labelbox_export_to_dataset.py
#                My_code/step04_detection/03_doclayout_yolo_finetune/<this file>
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "step03_labelbox_export"))
from step03_labelbox_export_to_dataset import (  # noqa: E402
    build_image_index,
    effective_class,
    flatten_classifications,
    resolve_composites,
    select_labels,
)


def paper_stem(external_id: str) -> str:
    """'<paper>_page_0007.png' -> '<paper>' -- same convention as Annotations.ipynb."""
    return external_id.rsplit("_page_", 1)[0]


def collect_pages(ndjson_path: Path, images_root: Path, label_select: str,
                   split_figure_subtype: bool, containment_threshold: float) -> tuple:
    """
    Walks the NDJSON once and returns (pages, categories):
      pages: list of dicts {external_id, local_path, img_w, img_h, boxes: [...]}.
             boxes is a list of (class_name, top, left, width, height) for the
             DETECTOR dataset only (nested composite sub-boxes excluded).
             A page with boxes == [] is a legitimate background/negative page.
      categories: {class_name: class_id}, ids assigned by descending frequency
                  (ties broken alphabetically) so the mapping is deterministic.
    """
    image_index = build_image_index(images_root)
    pages = []
    class_counter = {}

    with open(ndjson_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            external_id = row.get("data_row", {}).get("external_id", "")
            media = row.get("media_attributes", {})
            declared_w, declared_h = media.get("width"), media.get("height")

            local_path = image_index.get(external_id)
            if local_path is not None:
                with Image.open(local_path) as im:
                    img_w, img_h = im.size
            elif declared_w and declared_h:
                img_w, img_h = declared_w, declared_h
            else:
                img_w, img_h = None, None

            any_label_on_row = False
            page_boxes = []
            explicit_no_graphics = False

            for project_id, project in row.get("projects", {}).items():
                labels = select_labels(project.get("labels", []), label_select)
                for label in labels:
                    any_label_on_row = True
                    objects = label.get("annotations", {}).get("objects", [])
                    top_classifications = label.get("annotations", {}).get("classifications", [])
                    if not objects and any(
                        "no_graphics" in str(c).lower() for c in top_classifications
                    ):
                        explicit_no_graphics = True

                    page_objects = []
                    for obj in objects:
                        bbox = obj.get("bounding_box")
                        if not bbox:
                            continue
                        raw_class = obj.get("name", "unlabeled")
                        subtypes = flatten_classifications(obj.get("classifications", []))
                        class_name = effective_class(raw_class, subtypes, split_figure_subtype)
                        page_objects.append({
                            "obj": obj, "bbox": bbox, "raw_class": raw_class,
                            "subtypes": subtypes, "class_name": class_name,
                        })

                    resolve_composites(page_objects, containment_threshold)

                    for po in page_objects:
                        if po["nested_child_of_mixed"]:
                            continue  # excluded from the detector dataset, same as step03
                        bbox = po["bbox"]
                        page_boxes.append((po["class_name"], bbox["top"], bbox["left"],
                                            bbox["width"], bbox["height"]))
                        class_counter[po["class_name"]] = class_counter.get(po["class_name"], 0) + 1

            if not any_label_on_row:
                continue  # page was never labeled at all -- not part of this dataset

            if local_path is None:
                continue  # can't train on a box/page whose image we don't have locally

            pages.append({
                "external_id": external_id, "local_path": local_path,
                "img_w": img_w, "img_h": img_h, "boxes": page_boxes,
                "blank_submission": (not page_boxes) and not explicit_no_graphics,
            })

    categories = {
        name: i for i, (name, _) in enumerate(
            sorted(class_counter.items(), key=lambda kv: (-kv[1], kv[0]))
        )
    }
    return pages, categories


def split_by_paper(pages: list, val_fraction: float, seed: int = 42) -> tuple:
    """Groups pages by paper_stem, then splits whole papers into train/val."""
    papers = {}
    for p in pages:
        papers.setdefault(paper_stem(p["external_id"]), []).append(p)

    paper_names = sorted(papers.keys())
    rng = random.Random(seed)
    rng.shuffle(paper_names)

    n_val_papers = max(1, round(len(paper_names) * val_fraction))
    val_papers = set(paper_names[:n_val_papers])

    train_pages, val_pages = [], []
    for name, plist in papers.items():
        (val_pages if name in val_papers else train_pages).extend(plist)
    return train_pages, val_pages, sorted(val_papers)


def write_split(pages: list, split_name: str, output_root: Path, categories: dict) -> None:
    img_dir = output_root / "images" / split_name
    lbl_dir = output_root / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for p in pages:
        dst_img = img_dir / p["local_path"].name
        shutil.copy2(p["local_path"], dst_img)

        lbl_path = lbl_dir / (Path(p["external_id"]).stem + ".txt")
        img_w, img_h = p["img_w"], p["img_h"]
        lines = []
        for class_name, top, left, width, height in p["boxes"]:
            cx, cy = (left + width / 2) / img_w, (top + height / 2) / img_h
            nw, nh = width / img_w, height / img_h
            lines.append(f"{categories[class_name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def prepare_dataset(ndjson_path, images_root, output_root, label_select: str = "first",
                     split_figure_subtype: bool = True, containment_threshold: float = 0.85,
                     val_fraction: float = 0.15, seed: int = 42, verbose: bool = True) -> dict:
    """
    Does the full job (usable directly from a notebook, not just the CLI below):
    reads the NDJSON, resolves detector-dataset boxes the same way step03 does,
    splits by paper into train/val, writes the YOLO folder structure + data.yaml,
    and returns a stats dict. Prints a human-readable report when verbose=True.
    """
    ndjson_path = Path(ndjson_path).expanduser().resolve()
    images_root = Path(images_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()

    if not ndjson_path.exists():
        raise FileNotFoundError(f"NDJSON file not found: {ndjson_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    pages, categories = collect_pages(ndjson_path, images_root, label_select,
                                       split_figure_subtype, containment_threshold)
    train_pages, val_pages, val_paper_names = split_by_paper(pages, val_fraction, seed)

    write_split(train_pages, "train", output_root, categories)
    write_split(val_pages, "val", output_root, categories)

    class_names_in_order = [name for name, _ in sorted(categories.items(), key=lambda kv: kv[1])]
    with open(output_root / "classes.txt", "w") as f:
        f.write("\n".join(class_names_in_order) + "\n")

    data_yaml = output_root / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {output_root}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(class_names_in_order)}\n")
        f.write("names:\n")
        for i, name in enumerate(class_names_in_order):
            f.write(f"  {i}: {name}\n")

    n_train_boxes = sum(len(p["boxes"]) for p in train_pages)
    n_val_boxes = sum(len(p["boxes"]) for p in val_pages)
    n_train_bg = sum(1 for p in train_pages if not p["boxes"])
    n_val_bg = sum(1 for p in val_pages if not p["boxes"])
    blank_pages = [p["external_id"] for p in pages if p["blank_submission"]]

    stats = {
        "output_root": output_root,
        "data_yaml": data_yaml,
        "classes": class_names_in_order,
        "n_papers": len(set(paper_stem(p["external_id"]) for p in pages)),
        "val_papers": val_paper_names,
        "n_train_pages": len(train_pages), "n_train_boxes": n_train_boxes, "n_train_bg": n_train_bg,
        "n_val_pages": len(val_pages), "n_val_boxes": n_val_boxes, "n_val_bg": n_val_bg,
        "blank_submission_pages": blank_pages,
    }

    if verbose:
        print("─" * 54)
        print(f"Papers total: {stats['n_papers']}  (val: {len(val_paper_names)})")
        print(f"Train: {stats['n_train_pages']} pages, {n_train_boxes} boxes, {n_train_bg} background")
        print(f"Val:   {stats['n_val_pages']} pages, {n_val_boxes} boxes, {n_val_bg} background")
        print(f"\nClasses ({len(class_names_in_order)}): {class_names_in_order}")
        print(f"\nVal papers (held out entirely, to test generalization to unseen papers):")
        for name in val_paper_names:
            print(f"    - {name}")

        if blank_pages:
            print(f"\n[Warning] {len(blank_pages)} background page(s) are BLANK SUBMISSIONS -- "
                  f"no objects were drawn AND the 'No Graphics' checkbox wasn't checked, so it's "
                  f"unconfirmed whether these pages truly have nothing to detect (vs. an annotator "
                  f"who skipped them). They're currently included as negative training examples. "
                  f"Worth a quick look before trusting them as ground truth:")
            for eid in blank_pages:
                print(f"    - {eid}")

        print(f"\nWritten to: {output_root}/")
        print("  data.yaml   <- pass this to the training script")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ndjson", help="Path to the Labelbox NDJSON export file.")
    parser.add_argument("--images-root", required=True,
                         help="Root directory containing the local page PNGs (searched recursively).")
    parser.add_argument("--output", default="./yolo_dataset", help="Output directory.")
    parser.add_argument("--label-select", choices=["first", "all"], default="first")
    parser.add_argument("--no-split-figure-subtype", dest="split_figure_subtype",
                         action="store_false")
    parser.add_argument("--containment-threshold", type=float, default=0.85)
    parser.add_argument("--val-fraction", type=float, default=0.15,
                         help="Fraction of PAPERS (not pages) held out for validation (default 0.15).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        prepare_dataset(args.ndjson, args.images_root, args.output, args.label_select,
                         args.split_figure_subtype, args.containment_threshold,
                         args.val_fraction, args.seed)
    except FileNotFoundError as e:
        sys.exit(f"[Error] {e}")


if __name__ == "__main__":
    main()
