import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from annotation_inputs import latest_annotation_export, annotated_image_paths, build_image_index


class AnnotationInputsTests(unittest.TestCase):
    def test_new_export_and_new_folder_are_picked_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / 'Export project - STEPS-Paper-Annotation - 9_12_2026.ndjson'
            old.write_text('')
            export_dir = root / 'Exported_annotated_data'
            export_dir.mkdir()
            new = export_dir / 'Export project - STEPS-Paper-Annotation - 10_1_2026.ndjson'
            rows = []
            for topic, name, labeled in [('Recovery', 'old.png', True),
                                          ('Recovery - Fertilizer', 'new.JPEG', True),
                                          ('Future topic', 'unlabeled.png', False)]:
                folder = root / topic
                folder.mkdir()
                (folder / name).write_bytes(b'image')
                rows.append({'data_row': {'external_id': name}, 'projects': {
                    'project': {'labels': [{'annotations': {'objects': []}}] if labeled else []}}})
            new.write_text('\n'.join(json.dumps(row) for row in rows))
            self.assertEqual(latest_annotation_export(root), new)
            self.assertEqual({p.name for p in annotated_image_paths(new, root)}, {'old.png', 'new.JPEG'})

    def test_duplicate_copies_and_missing_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['a', 'b']:
                (root / name).mkdir()
                (root / name / 'page.png').write_bytes(b'identical')
            self.assertEqual(len(build_image_index(root)), 1)
            export = root / 'export.ndjson'
            export.write_text(json.dumps({'data_row': {'external_id': 'missing.png'},
                                         'projects': {'p': {'labels': [{}]}}}))
            with self.assertWarns(UserWarning):
                self.assertEqual(annotated_image_paths(export, root), [])
            (root / 'b' / 'page.png').write_bytes(b'different')
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                build_image_index(root)


if __name__ == '__main__':
    unittest.main()
