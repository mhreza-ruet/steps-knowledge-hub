import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from config import IMAGE_EXTS, MAX_RETRIES
from vlm_client import extract_page_tables


def list_page_files(input_dir: Path) -> list[Path]:
    return sorted([p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def process_pages(input_dir: Path, output_dir: Path) -> None:
    client = OpenAI()
    output_dir.mkdir(parents=True, exist_ok=True)

    page_files = list_page_files(input_dir)
    merged_results = []

    for idx, page_path in enumerate(page_files, start=1):
        print(f"[{idx}/{len(page_files)}] Processing {page_path.name} ...")

        result = None
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = extract_page_tables(client, page_path)
                break
            except Exception as exc:
                last_error = exc
                print(f"  Attempt {attempt} failed: {exc}")

        if result is None:
            result = {
                "page_summary": f"Failed to process {page_path.name}",
                "relevant_tables": [],
                "ignored_tables": [],
                "error": str(last_error),
            }

        page_record = {
            "page_file": page_path.name,
            **result,
        }

        merged_results.append(page_record)

        page_json_path = output_dir / f"{page_path.stem}.json"
        save_json(page_json_path, page_record)

    merged_json_path = output_dir / "merged_extraction.json"
    save_json(merged_json_path, merged_results)

    print(f"\nDone. Results saved to: {output_dir}")