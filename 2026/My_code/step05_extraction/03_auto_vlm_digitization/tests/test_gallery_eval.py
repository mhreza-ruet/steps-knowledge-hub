"""Regression checks for VLM/extractor schema differences in gallery scoring."""
import unittest

from autodig import gallery_eval as ge


class GallerySchemaTests(unittest.TestCase):
    def test_field_with_heatmap_rows_counts_false_positives(self):
        truth = {"type": "field", "x": [0, 1], "y": [0, 1],
                 "z": [[0, 1], [1, 2]], "z_range": [0, 2]}
        result = {"status": "ok", "table": [{"row": "a", "column": "b", "value": 1}]}
        df = ge.score(truth, result)
        summary = ge.summary("field", df, "ok", truth, result)
        self.assertEqual(summary["extracted items"], 1)
        self.assertEqual(summary["F1 %"], 0)
        result["table"].append({"x": 0.5, "y": 0.5, "z": 1})
        self.assertEqual(ge.ext_field(truth, result, 5), (1, 2))

    def test_histogram_with_line_points_is_missed(self):
        truth = {"type": "histogram", "series": {"A": {"edges": [0, 1, 2], "heights": [1, 2]}}}
        result = {"status": "ok", "series": {"A": {"x": [0.5, 1.5], "y": [1, 2]}}}
        df = ge.score(truth, result)
        self.assertFalse(df["hit"].any())
        self.assertEqual(ge.ext_hist(truth, result, 5), (0, 2))
        result["series"]["A"] = {"left": [0, 1], "right": [1, 2], "height": [1, 2]}
        self.assertTrue(ge.score(truth, result)["hit"].all())
        self.assertEqual(ge.ext_hist(truth, result, 5), (2, 2))

    def test_both_bar_value_schemas(self):
        truth = {"type": "stacked", "stacked": False, "categories": ["a", "b"],
                 "series": {"A": [1, 2]}}
        for key in ("y", "values"):
            with self.subTest(key=key):
                result = {"status": "ok", "series": {"A": {"categories": ["a", "b"], key: [1, 2]}}}
                df = ge.score(truth, result)
                self.assertTrue(df["hit"].all())
                self.assertEqual(ge.ext_stacked(truth, result, 5), (2, 2))


if __name__ == "__main__":
    unittest.main()
