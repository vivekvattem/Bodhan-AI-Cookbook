import unittest

from bodhan_eval.report import calibration


class HumanCalibrationTest(unittest.TestCase):
    def test_sparse_human_labels_and_unavailable_checks_are_visible(self):
        items = [
            {"id": "one", "lang": "hi", "source": "A fact.", "translation": "text",
             "flags": [], "checks": {"number": {"status": "inconclusive"},
                                     "semantic_translation": {"status": "unavailable"},
                                     "markdown_table": {"status": "skipped"}, "latex": {"status": "skipped"}}},
            {"id": "two", "lang": "hi", "source": "More.", "translation": "text",
             "flags": [], "checks": {"number": {"status": "evaluated"},
                                     "semantic_translation": {"status": "unavailable"},
                                     "markdown_table": {"status": "skipped"}, "latex": {"status": "skipped"}}},
            {"id": "unreviewed", "lang": "hi", "source": "Other.", "translation": "text",
             "flags": [], "checks": {}},
        ]
        labels = {("one", "hi"): [{"type": "omission", "field": "source", "span": [0, 1], "severity": "major"}],
                  ("two", "hi"): []}
        records = {key: {"id": key[0], "lang": key[1], "labels": value, "annotation_origin": "human",
                         "human_review": "completed", "reviewer": "reviewer"} for key, value in labels.items()}
        result = calibration(items, labels, 0, "hosted", records)
        omission = result["categories"]["omission"]
        self.assertEqual((omission["support"], omission["false_negative"]), (1, 1))
        self.assertEqual(omission["recall"], {"numerator": 0, "denominator": 1, "value": 0.0})
        self.assertEqual(omission["check_coverage"], {"unavailable": 2})
        self.assertEqual(result["reviewed_metrics"]["unlabelled_items"], 1)
        self.assertEqual(result["reviewed_label_provenance"]["origins"], {"human": 2})
        self.assertEqual(result["threshold_tuning"], "pending")

    def test_bad_span_is_excluded_from_metrics(self):
        item = {"id": "x", "lang": "hi", "source": "Source", "translation": "text",
                "checks": {"number": {"status": "evaluated"}}, "flags": []}
        label = {"type": "number", "field": "source", "span": [0, 99]}
        records = {("x", "hi"): {"id": "x", "lang": "hi", "labels": [label], "annotation_origin": "human"}}
        result = calibration([item], {("x", "hi"): [label]}, 0, "hosted", records)
        self.assertEqual(result["reviewed_metrics"]["invalid_labelled_items"], 1)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["categories"], {})
