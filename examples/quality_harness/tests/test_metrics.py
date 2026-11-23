import copy
import csv
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bodhan_eval.demo import read_jsonl, source_text, write_jsonl
from bodhan_eval.evaluate import HUMAN_FIELDS, review_queue
from bodhan_eval.metrics import evaluate

RECIPE = Path(__file__).resolve().parents[1]


def item(identity, *, flags=(), status="evaluated", lang="hi"):
    return {"id": identity, "lang": lang, "source": "abcdefghij", "translation": "ABCDEFGHIJ", "status": status,
            "flags": list(flags), "checks": {"number": {"status": "evaluated"}, "semantic_translation": {"status": "evaluated"}}}


def flag(category="number", span=None):
    return {"category": category, "field": "translation", "span": span or [2, 3]}


def annotation(prediction, *, labels=None, category="number", clean=False, split="evaluation"):
    return {"id": prediction["id"], "lang": prediction["lang"], "split": split, "annotation_origin": "programmatic",
            "human_review": "pending", "labels": [] if clean else labels or [
                {"type": category, "field": "translation", "span": [2, 3], "text": "C"}]}


class MetricsTest(unittest.TestCase):
    def test_precision_recall_denominators_abstentions_clean_and_unlabelled(self):
        first = item("positive-hit", flags=[flag()])
        missed = item("positive-unavailable", status="inconclusive", lang="bn")
        missed["checks"]["number"]["status"] = "unavailable"
        false_alarm = item("clean-false-alarm", flags=[flag()])
        control = item("clean-inconclusive", status="inconclusive")
        control["checks"]["semantic_translation"]["status"] = "inconclusive"
        unlabelled = item("unlabelled", flags=[flag()])
        records = [first, missed, false_alarm, control, unlabelled]
        labels = [annotation(first), annotation(missed), annotation(false_alarm, clean=True), annotation(control, clean=True)]
        result = evaluate(records, labels)
        score = result["categories"]["number"]
        self.assertEqual([score[k] for k in ("tp", "fp", "fn", "tn")], [1, 1, 1, 1])
        self.assertEqual(score["precision"], {"numerator": 1, "denominator": 2, "value": 0.5})
        self.assertEqual(score["recall"]["value"], 0.5)
        self.assertEqual(score["check_coverage"], {"evaluated": 3, "unavailable": 1})
        self.assertEqual(result["clean_false_alarm_rate"]["value"], 0.5)
        self.assertEqual(result["clean_items_with_incomplete_checks"], 1)
        self.assertEqual((result["unlabelled_items"], result["unlabelled_flagged_items"]), (1, 1))
        self.assertEqual(result["languages"]["bn"]["categories"]["number"]["recall"]["value"], 0)
        self.assertEqual(result["independent_english_sources"], 1)
        self.assertEqual(result["metric_source_support"]["clean_false_alarm_rate"], {"numerator_sources": 1, "denominator_sources": 1})
        self.assertEqual(score["independent_positive_sources"], 1)
        self.assertEqual(result["status"], "unreviewed_label_diagnostic")
        self.assertIsNone(result["categories"]["addition"]["recall"]["value"])

    def test_multiple_errors_cannot_be_explained_by_one_broad_finding(self):
        prediction = item("multiple", flags=[flag(span=[0, 9])])
        labels = [{"type": "number", "field": "translation", "span": [2, 3], "text": "C"},
                  {"type": "number", "field": "translation", "span": [7, 8], "text": "H"}]
        result = evaluate([prediction], [annotation(prediction, labels=labels)])
        self.assertEqual(result["error_detection_recall"]["value"], 0.5)
        self.assertEqual(result["localization_overlap"]["numerator"], 1)
        self.assertEqual(result["localization_overlap"]["denominator"], 1)
        self.assertEqual(result["category_matched_finding_precision"]["value"], 1)

    def test_wrong_category_and_invalid_flag_location(self):
        prediction = item("wrong-category", flags=[flag("mistranslation")])
        result = evaluate([prediction], [annotation(prediction, category="term_entity")])
        self.assertEqual(result["category_accuracy_among_localized_errors"], {"numerator": 0, "denominator": 1, "value": 0})
        self.assertEqual(result["categories"]["term_entity"]["fn"], 1)
        self.assertEqual(result["categories"]["mistranslation"]["fp"], 1)
        self.assertEqual(result["category_accuracy_detected_items"]["value"], 0)
        prediction["flags"] = [flag(span=[0, 999])]
        result = evaluate([prediction], [annotation(prediction)])
        self.assertEqual(result["localization_overlap"]["value"], 0)

    def test_supporting_context_cannot_rescue_a_wrong_primary_span(self):
        prediction = item("broad-context", flags=[flag(span=[7, 8])])
        prediction["flags"][0]["evidence"] = {"locations": [
            {"field": "translation", "span": [7, 8], "text": "H"},
            {"field": "translation", "span": [0, 10], "text": "ABCDEFGHIJ"},
            {"field": "translation", "span": [2, 3], "text": "C"}]}
        result = evaluate([prediction], [annotation(prediction)])
        self.assertEqual(result["localization_overlap"]["value"], 0)
        prediction["flags"][0].update(field="source", span=[2, 3])
        self.assertEqual(evaluate([prediction], [annotation(prediction)])["localization_overlap"]["value"], 0)
        prediction["flags"][0].update(field="translation", span=[2, 3])
        self.assertEqual(evaluate([prediction], [annotation(prediction)])["localization_overlap"]["value"], 1)

    def test_primary_quote_must_match_and_nonempty_spans_are_half_open(self):
        prediction = item("adjacent", flags=[flag(span=[3, 4])])
        self.assertEqual(evaluate([prediction], [annotation(prediction)])["localization_overlap"]["value"], 0)
        prediction["flags"] = [flag()]
        prediction["flags"][0]["evidence"] = {"anchor": {"field": "translation", "span": [2, 3], "text": "wrong"}}
        self.assertEqual(evaluate([prediction], [annotation(prediction)])["localization_overlap"]["value"], 0)

    def test_real_format_deletion_maps_original_span_to_output_boundary(self):
        records = read_jsonl(RECIPE / "data/demo.jsonl")
        truth = next(row for row in read_jsonl(RECIPE / "data/ground_truth.jsonl") if row["operator"] == "format_damage")
        prediction = copy.deepcopy(next(row for row in records if (row["id"], row["lang"]) == (truth["id"], truth["lang"])))
        position = truth["edit"]["output_span"][0]
        prediction.update(status="evaluated", checks={}, flags=[flag("format", [max(0, position - 1), min(len(prediction["translation"]), position + 1)])])
        result = evaluate([prediction], [truth])
        self.assertEqual(result["invalid_annotations"], [])
        self.assertEqual(result["localization_overlap"]["value"], 1)
        bad = copy.deepcopy(truth)
        bad["edit"]["original_text"] = "unverified edit"
        result = evaluate([prediction], [bad])
        self.assertEqual((result["invalid_labelled_items"], result["labelled_items"], result["unlabelled_items"]), (1, 0, 0))

    def test_invalid_and_duplicate_labels_are_never_clean_controls(self):
        prediction = item("bad", flags=[flag()])
        record = annotation(prediction, clean=True)
        for labels in ([record, record], [{**record, "labels": None}], [None],
                       [annotation(prediction, labels=[{"type": "number", "field": "translation", "span": [-1, 3]}])]):
            result = evaluate([prediction], labels)
            self.assertTrue(result["invalid_annotations"])
            self.assertEqual(result["clean_false_alarm_rate"]["denominator"], 0)

    def test_same_source_leakage_is_explicit_even_with_different_ids_and_languages(self):
        calibration = item("calibration")
        held_out = item("held-out", lang="ta")
        result = evaluate([calibration, held_out], [annotation(calibration, clean=True, split="calibration"), annotation(held_out, clean=True)])
        self.assertEqual(result["cross_split_source_overlap"], 1)
        self.assertEqual(result["split_integrity"], "failed")

    def test_calibration_review_is_blinded_blank_and_excludes_evaluation(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "review.csv"
            items = read_jsonl(RECIPE / "data/demo.jsonl")
            labels = read_jsonl(RECIPE / "data/ground_truth.jsonl")
            self.assertEqual(review_queue(items, labels, path), 12)
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({row["lang"] for row in rows}, {"hi", "ta", "bn"})
            self.assertTrue(all(row["split"] == "calibration" for row in rows))
            self.assertTrue(all(row[field] == "" for row in rows for field in HUMAN_FIELDS))
            self.assertFalse({"flags", "programmatic_category", "predictions"} & set(rows[0]))
            seeded = read_jsonl(path.with_suffix(".programmatic.jsonl"))
            self.assertTrue(all(row["annotation_origin"] == "programmatic" and row["human_review"] == "pending" for row in seeded))
            self.assertEqual(sum(row["labels"] == [] for row in seeded), 6)
            rows[0]["human_notes"] = "Test-only saved review entry"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            saved = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "human entries"):
                review_queue(items, labels, path)
            self.assertEqual(path.read_bytes(), saved)

    def test_offline_entrypoint_is_stable_and_preserves_saved_predictions(self):
        code = '''
import runpy,sys
def block(event,args):
    if event in {'socket.connect','socket.getaddrinfo','socket.gethostbyname'}: raise AssertionError('metrics attempted network')
sys.addaudithook(block)
sys.argv=['bodhan_eval.evaluate',*sys.argv[1:]]
runpy.run_module('bodhan_eval.evaluate',run_name='__main__',alter_sys=True)
'''
        with TemporaryDirectory() as temp:
            root = Path(temp)
            prediction = item("offline", flags=[flag()])
            report = root / "report"
            write_jsonl(report / "items.jsonl", [prediction])
            write_jsonl(root / "labels.jsonl", [annotation(prediction)])
            original = (report / "items.jsonl").read_bytes()
            for name in ("one", "two"):
                args = [sys.executable, "-I", "-c", code, "--report", str(report), "--labels", str(root / "labels.jsonl"), "--out", str(root / name)]
                result = subprocess.run(args, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
            for name in ("evaluation.json", "evaluation.md"):
                self.assertEqual((root / "one" / name).read_bytes(), (root / "two" / name).read_bytes())
            self.assertEqual(original, (report / "items.jsonl").read_bytes())

    def test_expansion_is_independent_english_material_and_not_evaluation_support(self):
        original = read_jsonl(RECIPE / "data/sources.jsonl")
        expansion = read_jsonl(RECIPE / "data/source_expansion.jsonl")
        self.assertEqual(len(expansion), 8)
        self.assertEqual(len({source_text(row) for row in [*original, *expansion]}), 16)
        families = {}
        for row in expansion:
            families.setdefault(row["family_id"], set()).add(row["intended_split"])
            self.assertEqual(row["translation_status"], "pending_bodhan")
            self.assertNotIn("translation", row)
        self.assertTrue(all(len(splits) == 1 for splits in families.values()))
        self.assertEqual(sum(row["intended_split"] == "calibration" for row in expansion), 4)


if __name__ == "__main__":
    unittest.main()
