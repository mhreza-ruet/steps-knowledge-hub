"""Shared annotation discovery for the notebooks through step 4."""
import hashlib
import json
import re
import warnings
from datetime import datetime
from pathlib import Path


def latest_annotation_export(images_root):
    """Search nested export folders and select by the date in the filename."""
    candidates = []
    for path in Path(images_root).rglob('Export*STEPS-Paper-Annotation*.ndjson'):
        match = re.search(r'(\d{1,2}_\d{1,2}_\d{4})\.ndjson$', path.name)
        if match:
            candidates.append((datetime.strptime(match[1], '%m_%d_%Y'),
                               path.stat().st_mtime_ns, path.name, path))
    if not candidates:
        raise FileNotFoundError(f'No dated annotation exports found in {images_root}')
    return max(candidates)[-1]


def build_image_index(images_root):
    """Search every topic folder; reject ambiguous names instead of overwriting."""
    index = {}
    for path in sorted(Path(images_root).rglob('*')):
        if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            if path.name in index:
                if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(index[path.name].read_bytes()).digest():
                    raise ValueError(f'Ambiguous image filename {path.name}: {index[path.name]} and {path}')
                continue  # Identical pages may belong to several topic folders.
            index[path.name] = path
    return index


def annotated_image_paths(export_path, images_root):
    """Return locally available labeled pages, including labeled blank pages."""
    index = build_image_index(images_root)
    selected, missing = {}, set()
    with Path(export_path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not any(p.get('labels') for p in row.get('projects', {}).values()):
                continue
            name = row.get('data_row', {}).get('external_id', '')
            if name in index:
                selected[name] = index[name]
            else:
                missing.add(name)
    if missing:
        warnings.warn(f'{len(missing)} labeled images missing locally; skipped: ' + ', '.join(sorted(missing)))
    return [selected[name] for name in sorted(selected)]
