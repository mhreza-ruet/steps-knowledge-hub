#!/usr/bin/env python3
"""
step01_pdf_to_images.py
-----------------------
Convert PDF files to per-page PNG images for Labelbox annotation.

Naming convention:
    <PDF stem>_p<zero-padded page number>.png

    Example — GenSinc.pdf (11 pages):
        GenSinc_p01.png
        GenSinc_p02.png
        ...
        GenSinc_p11.png

    Example — LongReport.pdf (120 pages):
        LongReport_p001.png  ...  LongReport_p120.png

Usage:
    # Single PDF
    python step01_pdf_to_images.py path/to/paper.pdf

    # All PDFs in a directory
    python step01_pdf_to_images.py path/to/pdf_folder/

    # Custom output directory and DPI
    python step01_pdf_to_images.py path/to/paper.pdf --output ./images --dpi 300

Requirements:
    pip install pymupdf
"""

import argparse
import sys
from pathlib import Path

# ── dependency check ──────────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit(
        "\n[Error] PyMuPDF is not installed.\n"
        "Run:  pip install pymupdf\n"
    )


# ── core conversion ───────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: Path, output_root: Path, dpi: int = 300) -> list:
    """
    Convert a single PDF to per-page PNG images.

    Args:
        pdf_path    : Path to the source PDF file.
        output_root : Parent directory. Images land in output_root/<stem>/.
        dpi         : Render resolution (300 Dpi recommended for annotation).

    Returns:
        List of Path objects pointing to the saved PNG files.
    """
    stem    = pdf_path.stem                 # e.g. "GenSinc"
    out_dir = output_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    n   = len(doc)
    pad = len(str(n))                       # digit width: 11 pages → 2, 120 → 3

    scale  = dpi / 72.0                     # PDF native unit is 72 pt/inch
    matrix = fitz.Matrix(scale, scale)

    saved = []
    for i, page in enumerate(doc):
        filename = f"{stem}_p{str(i + 1).zfill(pad)}.png"
        dest     = out_dir / filename

        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pix.save(str(dest))
        saved.append(dest)

        # Inline progress bar
        filled  = "█" * (i + 1)
        empty   = "░" * (n - i - 1)
        print(f"  [{filled}{empty}] {i + 1}/{n}  {filename}", end="\r", flush=True)

    print()  # newline after progress bar clears
    doc.close()
    return saved


# ── directory / single-file routing ──────────────────────────────────────────

def process(input_path: Path, output_root: Path, dpi: int) -> None:
    """Route a single PDF or every PDF found in a directory."""

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            sys.exit(f"[Error] Not a PDF file: {input_path}")
        pdfs = [input_path]

    elif input_path.is_dir():
        pdfs = sorted(input_path.glob("*.pdf")) + sorted(input_path.glob("*.PDF"))
        if not pdfs:
            sys.exit(f"[Error] No PDF files found in: {input_path}")

    else:
        sys.exit(f"[Error] Path does not exist: {input_path}")

    print(f"\nFound {len(pdfs)} PDF(s) — rendering at {dpi} DPI\n")

    total_pages = 0
    for pdf in pdfs:
        print(f"▸ {pdf.name}")
        saved = pdf_to_images(pdf, output_root, dpi)
        n = len(saved)
        total_pages += n
        print(f"  → {n} page(s) saved to: {saved[0].parent}/\n")

    print(f"{'─' * 50}")
    print(f"Done. {total_pages} image(s) across {len(pdfs)} PDF(s).")
    print(f"Output root: {output_root}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDF pages to named PNG images for Labelbox annotation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        help="Path to a PDF file, or a directory containing PDF files.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="DIR",
        help=(
            "Root directory where output folders are created. "
            "Defaults to an 'images/' subfolder next to the input."
        ),
    )
    parser.add_argument(
        "--dpi", "-d",
        type=int,
        default=300,
        metavar="N",
        help="Render resolution in DPI (default: 300).",
    )

    args        = parser.parse_args()
    input_path  = Path(args.input).expanduser().resolve()

    if args.output:
        output_root = Path(args.output).expanduser().resolve()
    else:
        # Place images/ next to the PDF file, or inside the input directory
        base        = input_path.parent if input_path.is_file() else input_path
        output_root = base / "images"

    output_root.mkdir(parents=True, exist_ok=True)
    process(input_path, output_root, args.dpi)


if __name__ == "__main__":
    main()
