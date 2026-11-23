import json
import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from bodhan_eval.cache import RequestCache
from bodhan_eval.cli import main
from bodhan_eval.dataset import load_dataset
from bodhan_eval.demo import read_jsonl, write_jsonl
from bodhan_eval.models import HostedTransport, Models, Unavailable
from bodhan_eval.numbers import compare_numbers, valid_number_finding
from bodhan_eval.replay import plan_key

RECIPE = Path(__file__).resolve().parents[1]
REPORTS = ("summary.md", "summary.json", "items.jsonl", "calibration.json", "diff.md")
ENV = {"BODHAN_TRANSLATE_KEY": "test-only", "LLM_BASE_URL": "https://api.groq.com/openai/v1",
       "LLM_MODEL": "openai/gpt-oss-120b", "LLM_API_KEY": "test-only"}


class QuantityDevelopmentTest(unittest.TestCase):
    def test_independent_development_controls_errors_and_abstentions(self):
        fixtures = read_jsonl(RECIPE / "sample_data/number_development.jsonl")
        self.assertEqual(len(fixtures), 30)
        clean_controls = 0
        expected_findings = {"mass-change", "native-amount-error", "fullwidth-decimal-error", "attached-percent-error", "repeat-elsewhere-error"}
        for row in fixtures:
            with self.subTest(case=row["id"]):
                result = compare_numbers(row["source"], row["translation"])
                finding = result.get("finding")
                if row["expected"] == "match":
                    clean_controls += 1
                    self.assertIsNone(finding)
                    self.assertIn(result.get("verdict"), {"match", "inventory_match"})
                elif row["id"] in expected_findings:
                    self.assertEqual(result.get("verdict"), "finding")
                    self.assertTrue(valid_number_finding(finding, row))
                else:
                    self.assertEqual(result["status"], "inconclusive")
                    self.assertIsNone(finding)
        self.assertEqual(clean_controls, 10)

    def test_partial_and_ambiguous_repeated_values_are_not_fabricated_pairs(self):
        # One value remains after cancellation; both source occurrences are support,
        # and only the unique translation occurrence is the affected location.
        result = compare_numbers("4kg and 4kg", "4kg and 5kg")
        self.assertEqual(result["verdict"], "finding")
        self.assertEqual(result["finding"]["span"], [8, 9])
        self.assertEqual(len(result["finding"]["evidence"]["source_support"]), 2)
        for source, target in (("8m and 12m", "9m and 13m"),
                               ("ID 72; mass 14kg; height 18m", "ID 73; height 19m; mass 15kg")):
            result = compare_numbers(source, target)
            self.assertEqual(result["status"], "inconclusive")
            self.assertNotIn("finding", result)

    def test_original_offsets_survive_compatibility_normalization(self):
        source, target = "⚖ 1.25kg", "⚖ １．３５㎏"
        flag = compare_numbers(source, target)["finding"]
        self.assertEqual(flag["span"], [2, 6])
        self.assertEqual(flag["evidence"]["translation"]["unit_span"], [6, 7])
        self.assertTrue(valid_number_finding(flag, {"source": source, "translation": target}))
        for target in ("2m\u0338", "2kg_extra", "2mg²"):
            self.assertEqual(compare_numbers("2 m", target)["status"], "inconclusive")

    def test_provider_caps_are_independent_and_reservations_are_not_spent_twice(self):
        calls = []

        def wire(request):
            calls.append(request)
            if request.url.host == "api.groq.com":
                return httpx.Response(429, headers={"retry-after": "120"})
            return httpx.Response(200, json={"translation": "Mock", "usage": {"completion_tokens": 5}})

        with TemporaryDirectory() as temp, httpx.Client(transport=httpx.MockTransport(wire)) as http:
            cache = RequestCache(Path(temp) / "cache.sqlite3")
            self.addCleanup(cache.close)
            models = Models(HostedTransport(ENV, http), cache, Decimal("1"),
                            provider_caps={"bodhan": Decimal("0.05"), "judge": Decimal("0.08")})
            with self.assertRaisesRegex(Unavailable, "429"):
                models.judge({"source": "mock"}, smoke=True)
            held = models.provider_uncertain["judge"]
            self.assertGreater(held, 0)
            models.translate("cached", "hi")
            models.translate("cached", "hi")
            self.assertEqual(models.provider_spend["bodhan"], Decimal("0.0001"))
            self.assertEqual(models.provider_uncertain["bodhan"], 0)
            self.assertEqual(models.reserved_uncertain, held)
            self.assertEqual(len(calls), 2)
            self.assertEqual(models.original_cost()["providers"]["judge"]["unpriced_requests"], 1)
            # A new invocation may reuse paid evidence; the historical unknown
            # remains in the old ledger, without becoming new spend/holds.
            replay = Models(HostedTransport(ENV, http), cache, Decimal("0"), provider_caps={"bodhan": Decimal("0")})
            replay.translate("cached", "hi")
            self.assertEqual((replay.incremental, replay.reserved_uncertain), (0, 0))
            self.assertEqual(len(calls), 2)

    def test_failed_bodhan_retry_obeys_provider_cap_and_keeps_judge_funds(self):
        calls = []

        def wire(request):
            calls.append(request)
            if request.url.host == "api.bodhan.ai":
                return httpx.Response(503)
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"verdict":"supports"}'}}],
                                            "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

        with TemporaryDirectory() as temp, httpx.Client(transport=httpx.MockTransport(wire)) as http:
            cache = RequestCache(Path(temp) / "cache.sqlite3")
            self.addCleanup(cache.close)
            models = Models(HostedTransport(ENV, http), cache, Decimal("1"),
                            provider_caps={"bodhan": Decimal("0.07"), "judge": Decimal("0.1")},
                            sleeper=lambda delay: self.fail("unfunded retry slept"))
            with self.assertRaisesRegex(Unavailable, "503"):
                models.translate("mock", "hi")
            self.assertEqual(models.calls, 1)
            self.assertEqual(models.events[0]["budget_scope"], "bodhan")
            self.assertEqual(models.provider_uncertain["bodhan"], Decimal("0.04096"))
            self.assertEqual(models.judge({"source": "mock"}, smoke=True), "supports")
            self.assertEqual(models.calls, 2)

    def test_unpriced_success_remains_unknown_in_provider_accounting(self):
        env = {**ENV, "LLM_BASE_URL": "https://unpriced.invalid/v1", "LLM_MODEL": "mock"}
        answer = {"choices": [{"message": {"content": '{"verdict":"uncertain"}'}}]}
        with TemporaryDirectory() as temp, httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=answer))) as http:
            cache = RequestCache(Path(temp) / "cache.sqlite3")
            self.addCleanup(cache.close)
            models = Models(HostedTransport(env, http), cache, None)
            self.assertEqual(models.judge({"source": "mock"}), "uncertain")
            self.assertEqual(models.telemetry()["providers"]["judge"]["unknown_billing_attempts"], 1)
            self.assertEqual(models.original_cost()["providers"]["judge"]["unpriced_requests"], 1)

    def test_multi_findings_flow_and_all_report_replay(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            row = {"id": "independent", "lang": "hi", "source": "The pole is 11m and the crate is 23kg.",
                   "translation": "Crate 24kg; pole 12m."}
            write_jsonl(root / "data.jsonl", [row])
            judge_inputs = []

            def wire(request):
                data = json.loads(request.content)
                if request.url.host == "api.bodhan.ai":
                    return httpx.Response(200, json={"translation": "Mock English evidence", "usage": {"completion_tokens": 4}})
                judge_inputs.append(json.loads(data["messages"][1]["content"]))
                answer = ({"verdict": "clean", "explanation": "Mock fixture only.", "findings": []}
                          if data["max_tokens"] == 4096 else {"verdict": "uncertain"})
                return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}],
                                                "usage": {"prompt_tokens": 20, "completion_tokens": 10}})

            with httpx.Client(transport=httpx.MockTransport(wire)) as http:
                args = ["--dataset", str(root / "data.jsonl"), "--langs", "hi", "--max-cost-inr", "1",
                        "--bodhan-max-cost-inr", "0.1", "--judge-max-cost-inr", "0.8"]
                self.assertEqual(main([*args, "--out", str(root / "baseline")], transport=HostedTransport(ENV, http)), 0)
                self.assertEqual(main([*args, "--out", str(root / "report"), "--baseline", str(root / "baseline")],
                                      transport=HostedTransport(ENV, http)), 0)
            result = read_jsonl(root / "report/items.jsonl")[0]
            self.assertEqual(result["flags"], [])
            self.assertEqual(result["checks"]["number"]["status"], "inconclusive")
            self.assertTrue(any(len(content.get("quantity_candidates", [])) >= 4 for content in judge_inputs))
            with httpx.Client(transport=httpx.MockTransport(lambda r: self.fail("replay dispatched"))) as http:
                self.assertEqual(main([*args, "--out", str(root / "replay"), "--baseline", str(root / "baseline"),
                                       "--cache-only", "--replay-from", str(root / "report")], transport=HostedTransport(ENV, http)), 0)
            for name in REPORTS:
                self.assertEqual((root / "report" / name).read_bytes(), (root / "replay" / name).read_bytes())
            rows = load_dataset(root / "data.jsonl", {"hi"})
            self.assertTrue(plan_key(rows, HostedTransport(ENV)))
