#!/usr/bin/env python3
"""
step03_labelbox_export_to_dataset.py
-------------------------------------
Bridge script: turns a Labelbox NDJSON export (annotation metadata only)
into an actually trainable dataset, by fusing it with the local PNG images
produced in Stage 00 (step01_pdf_to_images.py).

Labelbox never exports rendered/annotated images — only per-object bounding
box coordinates referencing the *original* uploaded image. That's true of
every annotation tool (COCO, CVAT, Label Studio work the same way). This
script is the missing bridge step:

  1. Matches each exported row to its local PNG via `data_row.external_id`
     — the filename the image had when uploaded to Labelbox. If you upload
     the files produced by step01 as-is, this is e.g. "GenSinc_p03.png".
  2. Crops every DETECTOR-dataset region out of the full-page image and
     saves it, organized by class folder under crops/ — one box per
     detector target, meant for training a page-layout model (Dolphin/
     Docling/etc.) to *locate* figures and tables. A "Figure" box carrying
     a Diagram/Plot subtype classification is split into its own Plot/ or
     Diagram/ folder (see --no-split-figure-subtype to disable).
     A composite figure originally tagged as both Diagram and Plot (a
     multi-panel figure mixing e.g. micrographs with a data plot) stays in
     this dataset as its full outer box — panels + caption together —
     since that's the whole region a layout detector should learn to find.
     Once you've added tighter Plot/Diagram sub-boxes for the individual
     panels (child box ≥ --containment-threshold contained in the outer
     box), those sub-boxes are excluded from THIS dataset only (keeping
     both would give the detector two overlapping ground-truth boxes with
     different classes for the same pixels). A "Figure_Mixed" box with no
     such children yet is flagged in the report as still needing sub-boxes.
  3. Separately crops every EXTRACTION-classes region (default: "Plot",
     see --extraction-classes) into extraction_crops/<class>/ — tight,
     caption-free crops meant to feed a downstream model (e.g. DePlot).
     This pulls from every matching box regardless of nesting, so a
     resolved composite's inner Plot sub-box lands here too, alongside
     every standalone Plot box. No COCO/YOLO format needed — just crops.
  4. Emits a COCO-format annotations.json (full pages + bboxes) for the
     detector dataset — directly usable by Detectron2, HuggingFace, or
     torchvision detection trainers.
  5. Emits YOLO-format .txt labels + classes.txt for the detector dataset —
     directly usable by ultralytics / DocLayout-YOLO training.
  6. Emits a flat CSV of every annotated object (both datasets combined,
     tagged with nested_child_of_mixed) for quick inspection in Excel/pandas.

Handles Labelbox consensus workflows (multiple independent labels per data
row) via --label-select.

Usage:
    python step03_labelbox_export_to_dataset.py \\
        export.ndjson \\
        --images-root ./images \\
        --output ./dataset

Requirements:
    pip install pillow
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("[Error] Pillow not installed. Run: pip install pillow")


# ── classification flattening ────────────────────────────────────────────────

def flatten_classifications(classifications: list) -> list:
    """
    Turn Labelbox's nested classification structure into flat strings like
    'Type:Diagram'. Handles checklist (multi-select), radio (single-select),
    and free text answers, recursively (a classification can nest more
    classifications under each answer).
    """
    out = []
    for c in classifications or []:
        cname = c.get("name", "")
        if "checklist_answers" in c:
            for ans in c["checklist_answers"]:
                out.append(f"{cname}:{ans.get('name', '')}")
                out.extend(flatten_classifications(ans.get("classifications", [])))
        elif "answer" in c and isinstance(c["answer"], dict):
            ans = c["answer"]
            out.append(f"{cname}:{ans.get('name', '')}")
            out.extend(flatten_classifications(ans.get("classifications", [])))
        elif "text_answer" in c:
            out.append(f"{cname}:{c['text_answer'].get('content', '')}")
    return out


# ── local image index ─────────────────────────────────────────────────────────

# Shared with inference and all step-4 dataset consumers.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from annotation_inputs import build_image_index


# ── figure subtype splitting ──────────────────────────────────────────────────

def effective_class(class_name: str, subtypes: list, split_figure_subtype: bool) -> str:
    """
    Resolve the class actually used for crop folders / COCO / YOLO categories.

    A Labelbox "Figure" box can carry a nested Diagram/Plot subtype answer
    (or both, or neither). When split_figure_subtype is True (default):
      - Figure + only "plot"            -> "Plot"
      - Figure + only "diagram"         -> "Diagram"
      - Figure + both plot and diagram  -> "Figure_Mixed"
      - Figure + no subtype answer      -> "Figure_Unlabeled"
      - anything else (e.g. "Table")    -> unchanged

    This matters beyond folder tidiness: Plot crops are the direct input to
    the plot-digitization stage (e.g. DePlot), so lumping them in with plain
    diagrams means every downstream step has to re-derive this split from
    the subtype tag instead of just reading the folder/category name.
    """
    if not split_figure_subtype or class_name != "Figure":
        return class_name

    tags = {t.split(":", 1)[-1].strip().lower() for t in subtypes if t}
    has_plot = "plot" in tags
    has_diagram = "diagram" in tags

    if has_plot and has_diagram:
        return "Figure_Mixed"
    if has_plot:
        return "Plot"
    if has_diagram:
        return "Diagram"
    return "Figure_Unlabeled"


# ── composite (mixed-figure) resolution ───────────────────────────────────────

def containment_ratio(inner: dict, outer: dict) -> float:
    """
    Fraction of `inner`'s box area that lies inside `outer`'s box.
    Both are Labelbox bounding_box dicts: {top, left, width, height}.
    1.0 = inner is fully inside outer; 0.0 = no overlap at all.
    """
    ix1, iy1 = inner["left"], inner["top"]
    ix2, iy2 = ix1 + inner["width"], iy1 + inner["height"]
    ox1, oy1 = outer["left"], outer["top"]
    ox2, oy2 = ox1 + outer["width"], oy1 + outer["height"]

    cx1, cy1 = max(ix1, ox1), max(iy1, oy1)
    cx2, cy2 = min(ix2, ox2), min(iy2, oy2)
    inter_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    inner_area = max(1e-9, inner["width"] * inner["height"])
    return inter_area / inner_area


def resolve_composites(page_objects: list, containment_threshold: float = 0.85) -> None:
    """
    Mutates `page_objects` in place, adding a "nested_child_of_mixed" bool
    (and a "nested_parent_feature_id") to each.

    A "Figure_Mixed" box is the whole figure region as originally drawn —
    panels *and* caption together. That's exactly what a page-layout
    detector (Dolphin/Docling) needs, so it's kept as-is and counted
    normally: it is NOT dropped or relabeled.

    What gets excluded from the *detector* dataset instead is the tighter
    Plot/Diagram sub-boxes you added afterward, when they sit almost
    entirely inside a "Figure_Mixed" box on the same page — those are
    marked nested_child_of_mixed=True. Keeping both the outer box and its
    children in the same detector dataset would hand it two overlapping
    ground-truth boxes for the same pixels with different classes, which
    is the label noise object detectors are bad at learning from. But the
    nested children are not thrown away: they still feed a separate
    extraction-crops output (see EXTRACTION_CLASSES / --extraction-classes)
    used for plot digitization, where the tight, caption-free crop is
    exactly what's wanted and the outer box would be the wrong input.

    A "Figure_Mixed" box with no such children is left alone — it still
    needs sub-boxes added, and is flagged in the report as unresolved.
    """
    for obj in page_objects:
        obj["nested_child_of_mixed"] = False
        obj["nested_parent_feature_id"] = ""
        obj["mixed_resolved"] = False   # only meaningful for Figure_Mixed objects

    composites = [o for o in page_objects if o["class_name"] == "Figure_Mixed"]
    for comp in composites:
        children = [
            o for o in page_objects
            if o is not comp
            and o["class_name"] in ("Plot", "Diagram")
            and containment_ratio(o["bbox"], comp["bbox"]) >= containment_threshold
        ]
        if children:
            comp["mixed_resolved"] = True
        for child in children:
            child["nested_child_of_mixed"] = True
            child["nested_parent_feature_id"] = comp["obj"].get("feature_id", "")


# ── label selection (consensus handling) ─────────────────────────────────────

def select_labels(labels: list, mode: str) -> list:
    """
    Labelbox consensus projects can attach multiple independent labels to one
    data row (see project_details.consensus_expected_label_count in the raw
    export). `mode` controls which to use:
      "first" -> labels[0] only   (default; matches single-annotator projects)
      "all"   -> every label      (use for consensus / inter-annotator QA —
                                    NOTE: this will emit overlapping/duplicate
                                    boxes per annotator, by design)
    """
    if not labels:
        return []
    if mode == "all":
        return labels
    return [labels[0]]


# ── main extraction ───────────────────────────────────────────────────────────

def process_export(ndjson_path: Path, images_root: Path, output_root: Path,
                    label_select: str, pad_crop: int, split_figure_subtype: bool = True,
                    containment_threshold: float = 0.85,
                    extraction_classes: tuple = ("Plot",)) -> None:
    """
    Emits two separate outputs from the same annotations:

    1. Detector dataset (coco_annotations.json, yolo/, crops/<class>/) — one
       box per detector target, meant for training a page-layout model
       (Dolphin/Docling/etc.) to *locate* figures and tables. Every box is
       kept at whatever level it was drawn: a "Figure_Mixed" composite goes
       in as its full outer box (panels + caption together), since that's
       the whole region a layout detector should learn to find. The only
       boxes excluded here are the tighter Plot/Diagram sub-boxes nested
       inside a "Figure_Mixed" box — keeping both would give the detector
       two overlapping ground-truth boxes with different classes for the
       same pixels.

    2. Extraction crops (extraction_crops/<class>/) — tight, caption-free
       crops for classes in `extraction_classes` (default: just "Plot"),
       meant to feed a downstream extraction model (e.g. DePlot). This
       pulls from *every* matching box regardless of nesting: a resolved
       composite's inner Plot sub-box, and every standalone Plot box
       elsewhere. Nothing here needs COCO/YOLO format — it's just crops.
    """

    image_index = build_image_index(images_root)
    print(f"Indexed {len(image_index)} local image(s) under {images_root}/\n")

    crops_dir = output_root / "crops"
    yolo_dir = output_root / "yolo"
    extraction_dir = output_root / "extraction_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir.mkdir(parents=True, exist_ok=True)
    extraction_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    coco_images, coco_annotations = [], []
    categories = {}                       # class name -> category id
    class_counts = defaultdict(int)
    extraction_counts = defaultdict(int)
    missing_images = set()
    no_graphics_rows = 0
    nested_excluded_count = 0              # nested sub-boxes excluded from the detector dataset
    unresolved_mixed = []                  # external_ids still needing sub-boxes
    ann_id = 0
    img_id = 0

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
            if local_path is None:
                missing_images.add(external_id)

            if local_path is not None:
                with Image.open(local_path) as im:
                    img_w, img_h = im.size
            elif declared_w and declared_h:
                img_w, img_h = declared_w, declared_h
            else:
                img_w, img_h = None, None

            coco_img_id = None   # assigned lazily — only if this row yields a real object

            for project_id, project in row.get("projects", {}).items():
                labels = select_labels(project.get("labels", []), label_select)

                for label in labels:
                    objects = label.get("annotations", {}).get("objects", [])
                    top_classifications = label.get("annotations", {}).get("classifications", [])

                    if not objects and any(
                        "no_graphics" in str(c).lower() for c in top_classifications
                    ):
                        no_graphics_rows += 1

                    # ── pass 1: build the page's object list with resolved classes ──
                    page_objects = []
                    for obj in objects:
                        bbox = obj.get("bounding_box")
                        if not bbox:
                            continue  # classification-only annotation, no box to crop

                        raw_class = obj.get("name", "unlabeled")
                        subtypes = flatten_classifications(obj.get("classifications", []))
                        class_name = effective_class(raw_class, subtypes, split_figure_subtype)

                        page_objects.append({
                            "obj": obj, "bbox": bbox, "raw_class": raw_class,
                            "subtypes": subtypes, "class_name": class_name,
                        })

                    # ── pass 2: resolve composites (mark nested Plot/Diagram sub-boxes) ──
                    resolve_composites(page_objects, containment_threshold)
                    if any(o["class_name"] == "Figure_Mixed" and not o["mixed_resolved"]
                           for o in page_objects):
                        unresolved_mixed.append(external_id)

                    # ── pass 3: emit CSV / detector dataset (COCO/YOLO/crops) / extraction crops ──
                    for po in page_objects:
                        obj, bbox = po["obj"], po["bbox"]
                        class_name, raw_class, subtypes = po["class_name"], po["raw_class"], po["subtypes"]
                        nested = po["nested_child_of_mixed"]

                        top, left = bbox["top"], bbox["left"]
                        width, height = bbox["width"], bbox["height"]

                        if nested:
                            nested_excluded_count += 1
                        else:
                            class_counts[class_name] += 1

                        csv_rows.append({
                            "image": external_id,
                            "class": class_name,
                            "raw_class": raw_class,
                            "subtypes": ";".join(subtypes),
                            "nested_child_of_mixed": nested,
                            "nested_parent_feature_id": po["nested_parent_feature_id"],
                            "top": top, "left": left,
                            "width": width, "height": height,
                            "img_width": img_w, "img_height": img_h,
                            "feature_id": obj.get("feature_id", ""),
                            "project": project.get("name", project_id),
                        })

                        # ── extraction crops: every matching class, nested or not ──
                        if class_name in extraction_classes and local_path is not None:
                            extraction_counts[class_name] += 1
                            l = max(0, int(left) - pad_crop)
                            t = max(0, int(top) - pad_crop)
                            r = min(img_w, int(left + width) + pad_crop)
                            b = min(img_h, int(top + height) + pad_crop)
                            with Image.open(local_path) as im:
                                crop = im.crop((l, t, r, b))
                                ex_dir = extraction_dir / class_name
                                ex_dir.mkdir(exist_ok=True)
                                out_name = f"{Path(external_id).stem}__{obj.get('feature_id', ann_id)}.png"
                                crop.save(ex_dir / out_name)

                        if nested:
                            continue  # excluded from the detector dataset — see docstring above

                        if coco_img_id is None and img_w and img_h:
                            coco_img_id = img_id
                            coco_images.append({
                                "id": coco_img_id,
                                "file_name": external_id,
                                "width": img_w,
                                "height": img_h,
                            })
                            img_id += 1

                        if class_name not in categories:
                            categories[class_name] = len(categories)

                        if coco_img_id is not None:
                            ann_id += 1
                            coco_annotations.append({
                                "id": ann_id,
                                "image_id": coco_img_id,
                                "category_id": categories[class_name],
                                "bbox": [left, top, width, height],   # COCO: x, y, w, h
                                "area": width * height,
                                "iscrowd": 0,
                            })

                            # YOLO format: class cx cy w h, all normalized 0-1
                            cx, cy = (left + width / 2) / img_w, (top + height / 2) / img_h
                            nw, nh = width / img_w, height / img_h
                            yolo_txt = yolo_dir / (Path(external_id).stem + ".txt")
                            with open(yolo_txt, "a") as yf:
                                yf.write(f"{categories[class_name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

                        if local_path is not None:
                            l = max(0, int(left) - pad_crop)
                            t = max(0, int(top) - pad_crop)
                            r = min(img_w, int(left + width) + pad_crop)
                            b = min(img_h, int(top + height) + pad_crop)

                            with Image.open(local_path) as im:
                                crop = im.crop((l, t, r, b))
                                class_dir = crops_dir / class_name
                                class_dir.mkdir(exist_ok=True)
                                out_name = f"{Path(external_id).stem}__{obj.get('feature_id', ann_id)}.png"
                                crop.save(class_dir / out_name)

    # ── write flat CSV ──
    csv_path = output_root / "annotations_flat.csv"
    fieldnames = list(csv_rows[0].keys()) if csv_rows else [
        "image", "class", "raw_class", "subtypes", "nested_child_of_mixed",
        "nested_parent_feature_id", "top", "left", "width", "height",
        "img_width", "img_height", "feature_id", "project",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # ── write COCO ──
    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": i, "name": n} for n, i in categories.items()],
    }
    with open(output_root / "coco_annotations.json", "w") as f:
        json.dump(coco, f, indent=2)

    # ── write YOLO class list ──
    with open(yolo_dir / "classes.txt", "w") as f:
        for name, _ in sorted(categories.items(), key=lambda kv: kv[1]):
            f.write(name + "\n")

    # ── report ──
    print("─" * 54)
    print("DETECTOR DATASET  (coco_annotations.json / yolo / crops/<class>)")
    print(f"Objects:                  {sum(class_counts.values())}")
    for name, count in sorted(class_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<22} {count}")
    print(f"\nPages with 'No Graphics': {no_graphics_rows}")
    print(f"Images referenced:        {img_id}")
    print(f"Local images not found:   {len(missing_images)}")
    if missing_images and len(missing_images) <= 5:
        for m in missing_images:
            print(f"    - {m}")
    elif missing_images:
        print(f"    (crops skipped for these — COCO/YOLO still written "
              f"using Labelbox's recorded width/height)")

    print(f"\nNested Plot/Diagram sub-boxes excluded from the detector dataset "
          f"(kept only in the CSV + extraction crops): {nested_excluded_count}")
    if unresolved_mixed:
        print(f"Still-ambiguous 'Figure_Mixed' boxes needing sub-boxes "
              f"({len(unresolved_mixed)} page(s)):")
        for eid in unresolved_mixed:
            print(f"    - {eid}")
    else:
        print("Still-ambiguous 'Figure_Mixed' boxes needing sub-boxes: none")

    print(f"\nEXTRACTION CROPS  (extraction_crops/<class>) — classes: "
          f"{', '.join(extraction_classes)}")
    if extraction_counts:
        for name, count in sorted(extraction_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<22} {count}")
    else:
        print("  (none — no boxes matched --extraction-classes)")

    print(f"\nOutput written to: {output_root}/")
    print("  annotations_flat.csv             one row per annotated object")
    print("  coco_annotations.json            detector dataset, full-page COCO format")
    print("  yolo/*.txt + classes.txt         detector dataset, YOLO format")
    print("  crops/<class>/*.png              detector dataset, cropped regions by class")
    print("  extraction_crops/<class>/*.png   tight crops for plot digitization, "
          "incl. resolved composites' inner boxes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse a Labelbox NDJSON export with local page images into a trainable dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("ndjson", help="Path to the Labelbox NDJSON export file.")
    parser.add_argument("--images-root", required=True,
                         help="Root directory containing the local page PNGs (searched recursively).")
    parser.add_argument("--output", default="./dataset", help="Output directory (default: ./dataset).")
    parser.add_argument("--label-select", choices=["first", "all"], default="first",
                         help="'first': one label per row (default, matches single-annotator projects). "
                              "'all': every independent label per row (consensus / inter-annotator QA).")
    parser.add_argument("--pad", type=int, default=0,
                         help="Pixels of padding added around each crop (default: 0).")
    parser.add_argument("--no-split-figure-subtype", dest="split_figure_subtype",
                         action="store_false",
                         help="Disable splitting 'Figure' into Plot/Diagram/Figure_Mixed/"
                              "Figure_Unlabeled by subtype (splitting is on by default).")
    parser.add_argument("--containment-threshold", type=float, default=0.85,
                         help="Minimum fraction (0-1) of a Plot/Diagram box's area that must "
                              "lie inside a 'Figure_Mixed' box for it to count as a resolving "
                              "sub-box (default: 0.85).")
    parser.add_argument("--extraction-classes", default="Plot",
                         help="Comma-separated class name(s) to also crop into "
                              "extraction_crops/<class>/, pulled from every matching box "
                              "regardless of nesting (default: 'Plot'). Use e.g. 'Plot,Diagram' "
                              "to also get Diagram crops for extraction.")
    args = parser.parse_args()

    ndjson_path = Path(args.ndjson).expanduser().resolve()
    images_root = Path(args.images_root).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not ndjson_path.exists():
        sys.exit(f"[Error] NDJSON file not found: {ndjson_path}")
    if not images_root.exists():
        sys.exit(f"[Error] Images root not found: {images_root}")

    extraction_classes = tuple(c.strip() for c in args.extraction_classes.split(",") if c.strip())

    process_export(ndjson_path, images_root, output_root, args.label_select, args.pad,
                    split_figure_subtype=args.split_figure_subtype,
                    containment_threshold=args.containment_threshold,
                    extraction_classes=extraction_classes)


if __name__ == "__main__":
    main()
