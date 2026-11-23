import copy
import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bodhan_eval.cache import RequestCache
from bodhan_eval.diagnose import item_diagnostic, numeric_trace, write_diagnostics
from bodhan_eval.diagnostic_budget import audit_runtime, requests_for_items


def prediction(identity="case"):
    return {"id": identity, "source": "A shelf has 19 books. Row 7.", "translation": "Shelf 20 books. Row 7.", "lang": "hi",
            "checks": {"number": {"status": "inconclusive", "reason": "numerals could not be paired in the same unit"},
                       "semantic_translation": {"status": "unavailable", "reason": "cost cap"},
                       "back_translation": {"status": "evaluated", "evidence": {"text": "Mock English evidence"}},
                       "judge": {"status": "skipped", "reason": "no number finding to review"}}, "flags": [], "status": "inconclusive"}


def seeded(item, category="number", clean=False):
    start = item["translation"].index("20")
    return {"id": item["id"], "lang": item["lang"], "source_item": "new-unit-fixture", "split": "calibration",
            "annotation_origin": "programmatic", "human_review": "pending", "labels": [] if clean else [
                {"type": category, "field": "translation", "span": [start, start + 2], "text": "20"}]}


class DiagnosticTest(unittest.TestCase):
    def test_traces_keep_original_unicode_spans_and_rejections(self):
        item = prediction()
        item["translation"] = "Shelf ２０ books. Row ７."
        trace = numeric_trace(item)
        self.assertEqual([token["value"] for token in trace["fields"]["translation"]["tokens"]], ["20", "7"])
        self.assertEqual(trace["current_check"]["verdict"], "finding")
        item["translation"] = "Shelf ２０items. Row ７."
        trace = numeric_trace(item)
        issue = trace["fields"]["translation"]["issue"]
        start, end = issue["span"]
        self.assertEqual(item["translation"][start:end], issue["text"])
        self.assertEqual(trace["fields"]["translation"]["lexical_candidates"][0]["folded_decimal_value"], "20")
        self.assertIsNone(trace["fields"]["translation"]["tokens"])

    def test_outcome_categories_coverage_and_read_only_join(self):
        item = prediction()
        annotation = seeded(item)
        original = copy.deepcopy(item)
        result = item_diagnostic(item, annotation)
        case = result["seeded_cases"][0]
        self.assertEqual(case["classification"], "miss")
        self.assertTrue(case["incomplete_coverage_relevant"])
        self.assertEqual(item, original)
        a, b = annotation["labels"][0]["span"]
        item["flags"] = [{"category": "number", "field": "translation", "span": [a, b]}]
        self.assertEqual(item_diagnostic(item, annotation)["classification"], "detection")
        item["flags"][0]["span"] = [0, 5]
        self.assertEqual(item_diagnostic(item, annotation)["classification"], "localization_failure")
        item["flags"][0].update(category="term_entity", span=[a, b])
        self.assertEqual(item_diagnostic(item, annotation)["classification"], "wrong_category")
        item["flags"][0]["span"] = [0, 5]
        self.assertEqual(item_diagnostic(item, annotation)["classification"], "miss")
        self.assertEqual(item_diagnostic(item, None)["classification"], "unlabelled")
        self.assertEqual(item_diagnostic(item, seeded(item, clean=True))["classification"], "clean_control_flagged")

    def test_one_finding_does_not_localize_multiple_seeded_errors(self):
        item = prediction()
        annotation = seeded(item)
        annotation["labels"].append({"type": "number", "field": "translation", "span": [20, 21], "text": "7"})
        item["flags"] = [{"category": "number", "field": "translation", "span": [0, len(item["translation"])]}]
        results = item_diagnostic(item, annotation)["seeded_cases"]
        self.assertEqual(sum(row["classification"] == "detection" for row in results), 1)

    def test_diagnostic_artifacts_are_reproducible_without_provider_access(self):
        item = prediction()
        with TemporaryDirectory() as temp, patch("bodhan_eval.models.HostedTransport.send", side_effect=AssertionError("provider called")):
            for name in ("a", "b"):
                summary = write_diagnostics([item], [seeded(item)], Path(temp) / name)
                self.assertEqual(summary["seeded_errors"], 1)
            for path in (Path(temp) / "a").iterdir():
                self.assertEqual(path.read_bytes(), (Path(temp) / "b" / path.name).read_bytes())
        self.assertEqual(item["flags"], [])

    def test_hash_verified_budget_reconstruction_preserves_unknown_billing(self):
        items = [prediction("short"), prediction("long")]
        items[1]["source"] += " More detail." * 500
        base, model = "https://api.groq.com/openai/v1", "openai/gpt-oss-120b"
        with patch("bodhan_eval.models.HostedTransport.send", side_effect=AssertionError("builder called transport")):
            requests = requests_for_items(items, base, model)
        events = []
        for i, request in enumerate(requests):
            event = {"key": RequestCache.key("hosted-v2", request.exact_input()), "model": request.model, "operation": request.operation,
                     "decision": "provider_dispatch", "status": "provider_response", "original_estimated_inr": "0.0001" if i < 2 else "0.002",
                     "attempts": [{"number": 1, "outcome": "success"}]}
            if i in {0, 2}:
                event["attempts"] = [{"number": 1, "outcome": "unavailable", "reason": "provider HTTP 429",
                                      "provider_error": {"http_status": 429}, "backoff_seconds": 1, "rate_limits": {"retry-after": "1"}},
                                     {"number": 2, "outcome": "success"}]
            if i == 3:
                event.update(attempts=[], decision="budget_block", status="unavailable")
                event.pop("original_estimated_inr")
            events.append(event)
        reserve = requests[0].reserve_inr + requests[2].reserve_inr
        cap = requests[0].reserve_inr + 2 * requests[2].reserve_inr + Decimal("0.01")
        runtime = {"requests": events, "failed_provider_attempts": 2, "incremental_known_inr": "0.0022", "reserved_uncertain_inr": str(reserve)}
        result = audit_runtime(items, runtime, base, cap)
        self.assertEqual(result["status"], "verified_request_reconstruction")
        self.assertTrue(result["matches_reported_known"])
        self.assertTrue(result["matches_reported_reservations"])
        self.assertEqual(result["recovered_failed_attempts"], 2)
        self.assertEqual(result["budget_blocks_due_to_uncertain_reservations"], 1)
        self.assertEqual(result["admission_violations"], [])
        self.assertEqual({row["id"] for row in result["failed_attempts"]}, {"short"})
        self.assertEqual(result["budget_blocks"][0]["id"], "long")
        self.assertEqual(len(result["request_observations"]), 4)
        self.assertTrue(all(row["billing"] == "unknown" for row in result["failed_attempts"]))
        runtime["requests"][0]["key"] = "different-content"
        result = audit_runtime(items, runtime, base, cap)
        self.assertEqual(result["status"], "unverified")
        self.assertEqual(result["reported_failed_attempts"], 2)
        self.assertNotIn("held_uncertain_inr", result)
