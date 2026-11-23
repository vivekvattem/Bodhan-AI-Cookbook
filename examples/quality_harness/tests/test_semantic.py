import copy
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from bodhan_eval.cli import main
from bodhan_eval.models import HostedTransport
from bodhan_eval.report import compare_baseline
from bodhan_eval.semantic import RUBRIC, finalize_judgment, parse_judgment

SOURCE = "Ravi bought tea and milk."
TARGET = "रवि ने चाय और कॉफी खरीदी।"
REPORTS = ("summary.md", "summary.json", "items.jsonl", "calibration.json")
ENV = {
    "LLM_BASE_URL": "https://semantic.invalid/v1",
    "LLM_MODEL": "mock-semantic",
    "LLM_API_KEY": "test-only-key",
    "BODHAN_TRANSLATE_KEY": "test-only-translate",
}


def location(field, text, part):
    start = text.index(part)
    return {"field": field, "span": [start, start + len(part)]}


def candidate(category="omission"):
    affected = location("source", SOURCE, "and milk")
    if category == "addition":
        affected = location("translation", TARGET, "कॉफी")
    return {
        "category": category,
        "severity": "major",
        **affected,
        "explanation": "The original meaning differs at the indicated location.",
        "evidence": [affected],
        "confidence": "high",
        "uncertainty": "",
    }


def response(findings=None, verdict=None):
    findings = [] if findings is None else findings
    return {"verdict": verdict or ("findings" if findings else "clean"), "explanation": "Direct source and translation comparison.", "findings": findings}


class SemanticContractTest(unittest.TestCase):
    def test_source_quote_unicode_offsets_preserve_original_codepoints(self):
        entry = candidate()
        entry.pop("span")
        entry["quote"] = "café 25 °C"
        entry["evidence"] = [{"field": "source", "quote": "café 25 °C"}]
        item = {"source": "🌱 A café 25 °C is open.", "translation": TARGET}
        check, flags = finalize_judgment(parse_judgment(response([entry])), item)
        self.assertEqual(check["status"], "evaluated")
        self.assertEqual(item["source"][slice(*flags[0]["span"])], entry["quote"])

    def test_repeated_source_quote_needs_unique_context(self):
        entry = candidate()
        entry.pop("span")
        entry.update(quote="milk", context="tea and milk")
        entry["evidence"] = [{"field": "source", "quote": "milk", "context": "tea and milk"}]
        item = {"source": "Ravi bought tea and milk. Milk and milk are here.", "translation": TARGET}
        check, flags = finalize_judgment(parse_judgment(response([entry])), item)
        self.assertEqual(check["status"], "evaluated")
        self.assertEqual(flags[0]["span"], [20, 24])
        entry["context"] = "milk"
        entry["evidence"][0]["context"] = "milk"
        check, flags = finalize_judgment(parse_judgment(response([entry])), item)
        self.assertEqual((check["status"], flags), ("inconclusive", []))

    def test_source_quote_requires_unique_exact_original_text(self):
        entry = candidate()
        entry.pop("span")
        entry["quote"] = "and milk"
        entry["evidence"] = [{"field": "source", "quote": "and milk"}]
        check, flags = finalize_judgment(parse_judgment(response([entry])), {"source": SOURCE, "translation": TARGET})
        self.assertEqual(check["status"], "evaluated")
        self.assertEqual(SOURCE[slice(*flags[0]["span"])], "and milk")
        for bad_source in ("and milk and milk", "AND MILK", "no matching words"):
            check, flags = finalize_judgment(parse_judgment(response([entry])), {"source": bad_source, "translation": TARGET})
            self.assertEqual((check["status"], flags), ("inconclusive", []))
        entry["field"] = "translation"
        entry["quote"] = "कॉफी"
        self.assertEqual(len(parse_judgment(response([entry]))["rejected"]), 1)

    def test_categories_and_clean(self):
        for category in ("omission", "addition", "mistranslation", "term_entity"):
            with self.subTest(category=category):
                check, flags = finalize_judgment(parse_judgment(response([candidate(category)])), {"source": SOURCE, "translation": TARGET})
                self.assertEqual(check["status"], "evaluated")
                self.assertEqual(flags[0]["category"], category)
                for anchor in flags[0]["evidence"]["locations"]:
                    start, end = anchor["span"]
                    self.assertEqual({"source": SOURCE, "translation": TARGET}[anchor["field"]][start:end], anchor["text"])
        check, flags = finalize_judgment(parse_judgment(response()), {"source": SOURCE, "translation": TARGET})
        self.assertEqual((check["verdict"], flags), ("clean", []))

    def test_multiple_errors_keep_valid_findings_when_an_anchor_fails(self):
        wrong = candidate("mistranslation")
        wrong["span"] = [0, 999]
        check, flags = finalize_judgment(parse_judgment(response([candidate(), wrong, candidate("addition")])), {"source": SOURCE, "translation": TARGET})
        self.assertEqual([flag["category"] for flag in flags], ["omission", "addition"])
        self.assertEqual(check["status"], "inconclusive")
        self.assertEqual(len(check["rejected"]), 1)

    def test_ambiguous_and_invalid_locations_are_inconclusive(self):
        cases = []
        for span in ([True, 5], [-1, 4], [5, 5], [0, 900], [0.0, 4]):
            entry = candidate()
            entry["span"] = span
            cases.append(entry)
        for field in ("translation", "back_translation"):
            entry = candidate()
            entry["field"] = field
            cases.append(entry)
        entry = candidate()
        entry.pop("span")
        entry["anchor"] = "Ravi"
        entry["evidence"] = [{"field": "source", "anchor": "Ravi"}]
        check, flags = finalize_judgment(parse_judgment(response([entry])), {"source": "Ravi saw Ravi", "translation": TARGET})
        self.assertEqual((check["status"], flags), ("inconclusive", []))
        entry["anchor"] = "aa"
        entry["evidence"] = [{"field": "source", "anchor": "aa"}]
        check, flags = finalize_judgment(parse_judgment(response([entry])), {"source": "aaa", "translation": TARGET})
        self.assertEqual((check["status"], flags), ("inconclusive", []))
        for entry in cases:
            with self.subTest(entry=entry):
                check, flags = finalize_judgment(parse_judgment(response([entry])), {"source": SOURCE, "translation": TARGET})
                self.assertEqual((check["status"], flags), ("inconclusive", []))

    def test_unique_source_anchor_and_uncertainty(self):
        entry = candidate()
        entry.pop("span")
        entry["anchor"] = "and milk"
        check, flags = finalize_judgment(parse_judgment(response([entry])), {"source": SOURCE, "translation": TARGET})
        self.assertEqual(check["status"], "evaluated")
        self.assertEqual(flags[0]["span"], location("source", SOURCE, "and milk")["span"])
        entry["confidence"] = "low"
        check, flags = finalize_judgment(parse_judgment(response([entry])), {"source": SOURCE, "translation": TARGET})
        self.assertEqual((check["status"], flags), ("inconclusive", []))
        check, _ = finalize_judgment(parse_judgment(response(verdict="uncertain")), {"source": SOURCE, "translation": TARGET})
        self.assertEqual(check["status"], "inconclusive")

    def test_contract_rejects_invented_text_and_contradictory_clean(self):
        entry = candidate()
        entry["corrected_translation"] = "Invented output"
        parsed = parse_judgment(response([entry]))
        self.assertEqual(len(parsed["rejected"]), 1)
        entry = candidate()
        entry["explanation"] = TARGET
        self.assertEqual(len(parse_judgment(response([entry]))["rejected"]), 1)
        with self.assertRaises(ValueError):
            parse_judgment(response([candidate()], "clean"))


class SemanticFlowTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset.jsonl"
        self.row = {"id": "semantic", "source": SOURCE, "translation": TARGET, "lang": "hi", "labels": []}
        self.dataset.write_text(json.dumps(self.row) + "\n")
        self.requests = []

    def run_flow(self, name, answer, *, environment=None, flags=()):
        def wire(request):
            self.requests.append(json.loads(request.content))
            if request.url.host == "api.bodhan.ai":
                # Deliberately misleading evidence. The judge must use original text.
                return httpx.Response(
                    200, json={"choices": [{"finish_reason": "stop", "message": {"content": "Ravi bought tea and milk."}}], "usage": {"completion_tokens": 8}}
                )
            if isinstance(answer, Exception):
                raise answer
            if isinstance(answer, int):
                return httpx.Response(answer)
            content = answer if isinstance(answer, str) else json.dumps(answer)
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}], "usage": {"completion_tokens": 100}})

        with httpx.Client(transport=httpx.MockTransport(wire)) as client:
            code = main(
                ["--dataset", str(self.dataset), "--langs", self.row["lang"], "--out", str(self.root / name), *flags],
                transport=HostedTransport(ENV if environment is None else environment, client),
            )
        self.assertEqual(code, 0)
        return json.loads((self.root / name / "items.jsonl").read_text())

    def test_graph_multiple_findings_misleading_back_translation_and_replay(self):
        item = self.run_flow("cold", response([candidate(), candidate("addition")]))
        self.assertEqual(len(item["flags"]), 2)
        self.assertEqual(item["checks"]["semantic_translation"]["status"], "evaluated")
        self.assertEqual(item["checks"]["audio_signal"]["status"], "skipped")
        judge = self.requests[-1]
        payload = json.loads(judge["messages"][1]["content"])
        self.assertEqual(set(payload), {"source", "translation", "lang", "back_translation"})
        self.assertEqual(payload["translation"], TARGET)
        self.assertEqual(payload["back_translation"], "Ravi bought tea and milk.")
        self.assertIn("It can be wrong", judge["messages"][0]["content"])
        self.assertEqual(judge["messages"][0]["content"], RUBRIC)
        launch = """
import runpy, sys
def block(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname'}:
        raise AssertionError('network during semantic replay')
sys.addaudithook(block)
sys.argv = ['bodhan_eval', *sys.argv[1:]]
runpy.run_module('bodhan_eval', run_name='__main__', alter_sys=True)
"""
        env = {key: value for key, value in os.environ.items() if key in {"PATH", "TMPDIR", "SYSTEMROOT"}}
        env.update({key: ENV[key] for key in ("LLM_MODEL", "LLM_BASE_URL")})
        result = subprocess.run(
            [sys.executable, "-I", "-c", launch, "--dataset", str(self.dataset), "--langs", "hi", "--out", str(self.root / "replay"), "--cache-only"],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in REPORTS:
            self.assertEqual((self.root / "cold" / name).read_bytes(), (self.root / "replay" / name).read_bytes())
        telemetry = json.loads((self.root / "replay.runtime.json").read_text())
        self.assertEqual((telemetry["provider_attempts"], telemetry["cache_hits"], telemetry["incremental_known_inr"]), (0, 2, "0"))

    def test_each_category_through_mocked_judge_and_graph(self):
        for category in ("omission", "addition", "mistranslation", "term_entity"):
            with self.subTest(category=category):
                item = self.run_flow(f"{category}/report", response([candidate(category)]))
                self.assertEqual([flag["category"] for flag in item["flags"]], [category])

    def test_exact_source_quote_through_graph(self):
        entry = candidate()
        entry.pop("span")
        entry["quote"] = "and milk"
        entry["evidence"] = [{"field": "source", "quote": "and milk"}]
        item = self.run_flow("source-quote", response([entry]))
        self.assertEqual(item["checks"]["semantic_translation"]["status"], "evaluated")
        flag = item["flags"][0]
        self.assertEqual(flag["category"], "omission")
        self.assertEqual(SOURCE[slice(*flag["span"])], "and milk")

    def test_ambiguous_numbers_route_candidates_through_graph_without_labels(self):
        self.row.update(source="Use 25 units and 9 units.", translation="३५ units and ९ units",
                        labels=[{"type": "number"}], injection={"secret": "never-send"})
        self.dataset.write_text(json.dumps(self.row, ensure_ascii=False) + "\n")
        src = {"field": "source", "quantity_id": "S0"}
        dst = {"field": "translation", "quantity_id": "T0"}
        finding = {"category": "number", "severity": "major", **dst,
                   "explanation": "The first quantity changed.", "confidence": "high", "uncertainty": "",
                   "evidence": [src, dst, {"field": "source", "quote": "25 units"}]}
        item = self.run_flow("quantity-route", response([finding]))
        payload = json.loads(self.requests[-1]["messages"][1]["content"])
        self.assertEqual([entry["id"] for entry in payload["quantity_candidates"]], ["S0", "S1", "T0", "T1"])
        self.assertNotIn("never-send", json.dumps(payload))
        self.assertNotIn("labels", payload)
        flag = item["flags"][0]
        self.assertEqual(flag["category"], "number")
        self.assertEqual(self.row["translation"][slice(*flag["span"])], "३५")
        launch = """import runpy, sys
def block(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname'}:
        raise AssertionError('network during replay')
sys.addaudithook(block)
sys.argv = ['bodhan_eval', *sys.argv[1:]]
runpy.run_module('bodhan_eval', run_name='__main__', alter_sys=True)
"""
        env = {key: value for key, value in os.environ.items() if key in {"PATH", "TMPDIR", "SYSTEMROOT"}}
        env.update({key: ENV[key] for key in ("LLM_MODEL", "LLM_BASE_URL")})
        replay = subprocess.run([sys.executable, "-I", "-c", launch, "--dataset", str(self.dataset),
                                 "--langs", "hi", "--out", str(self.root / "quantity-route-replay"),
                                 "--cache-only", "--replay-from", str(self.root / "quantity-route")],
                                cwd=self.root, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
        for name in REPORTS:
            self.assertEqual((self.root / "quantity-route" / name).read_bytes(), (self.root / "quantity-route-replay" / name).read_bytes())
        self.assertEqual(json.loads((self.root / "quantity-route-replay.runtime.json").read_text())["provider_attempts"], 0)

    def test_number_judgment_graph_and_offline_replay(self):
        self.row.update(source="Use 25 objects.", translation="२५ objects")
        self.dataset.write_text(json.dumps(self.row, ensure_ascii=False) + "\n")
        numeric = {"category": "number", "severity": "major", "field": "translation", "quantity_id": "T0",
                   "explanation": "The quantity changed.", "confidence": "high", "uncertainty": "",
                   "evidence": [{"field": "source", "quantity_id": "S0"}, {"field": "translation", "quantity_id": "T0"},
                                {"field": "source", "quote": "25 objects"}]}
        equal = self.run_flow("equal/report", response([numeric]))
        self.assertEqual(equal["checks"]["semantic_translation"]["status"], "inconclusive")
        self.assertFalse(equal["flags"])
        self.row["translation"] = "३५ objects"
        self.dataset.write_text(json.dumps(self.row, ensure_ascii=False) + "\n")
        bad = copy.deepcopy(numeric)
        bad["quantity_id"] = "T99"
        bad["evidence"][1]["quantity_id"] = "T99"
        invalid = self.run_flow("invalid/report", response([bad]))
        self.assertEqual(invalid["checks"]["semantic_translation"]["status"], "inconclusive")
        self.assertEqual([flag["check"] for flag in invalid["flags"]], ["number"])
        uncertain = self.run_flow("uncertain/report", response(verdict="uncertain"))
        self.assertEqual(uncertain["checks"]["semantic_translation"]["status"], "inconclusive")
        cold = self.run_flow("number-cold", response([numeric]))
        self.assertEqual([f["category"] for f in cold["flags"]], ["number"])
        self.assertEqual(cold["flags"][0]["span"], [0, 2])
        original_requests = copy.deepcopy(self.requests[-2:])
        self.row.update(labels=[{"type": "omission"}], injection={"secret_metadata": "DO-NOT-SEND"})
        self.dataset.write_text(json.dumps(self.row, ensure_ascii=False) + "\n")
        changed = self.run_flow("different-cache/report", response([numeric]))
        self.assertEqual(cold["flags"], changed["flags"])
        self.assertEqual(original_requests, self.requests[-2:])
        self.row.pop("injection")
        self.row["labels"] = []
        self.dataset.write_text(json.dumps(self.row, ensure_ascii=False) + "\n")
        launch = """import runpy, sys
def block(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname'}:
        raise AssertionError('network during replay')
sys.addaudithook(block)
sys.argv = ['bodhan_eval', *sys.argv[1:]]
runpy.run_module('bodhan_eval', run_name='__main__', alter_sys=True)
"""
        env = {key: value for key, value in os.environ.items() if key in {"PATH", "TMPDIR", "SYSTEMROOT"}}
        env.update({key: ENV[key] for key in ("LLM_MODEL", "LLM_BASE_URL")})
        replay = subprocess.run([sys.executable, "-I", "-c", launch, "--dataset", str(self.dataset),
                                 "--langs", "hi", "--out", str(self.root / "number-replay"),
                                 "--cache-only", "--replay-from", str(self.root / "number-cold")],
                                cwd=self.root, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
        for name in REPORTS:
            self.assertEqual((self.root / "number-cold" / name).read_bytes(), (self.root / "number-replay" / name).read_bytes())
        telemetry = json.loads((self.root / "number-replay.runtime.json").read_text())
        self.assertEqual(telemetry["provider_attempts"], 0)

    def test_label_isolation_for_actual_judge_requests(self):
        first = self.run_flow("one", response([candidate()]))
        before = copy.deepcopy(self.requests)
        self.requests.clear()
        self.row["labels"] = [{"type": "addition", "severity": "critical"}]
        self.row["injection"] = {"answer": "NEVER-SEND-ME"}
        self.dataset.write_text(json.dumps(self.row) + "\n")
        second = self.run_flow("separate/two", response([candidate()]))
        self.assertEqual(before, self.requests)
        self.assertEqual(first["flags"], second["flags"])
        self.assertEqual(first["checks"], second["checks"])

    def test_failures_timeout_malformed_and_budget_do_not_make_clean(self):
        for index, answer in enumerate([503, httpx.ReadTimeout("mock timeout"), "not JSON", {"verdict": "clean", "findings": []}]):
            with self.subTest(answer=answer):
                item = self.run_flow(f"failure-{index}", answer)
                self.assertEqual(item["checks"]["semantic_translation"]["status"], "unavailable")
                self.assertEqual(item["status"], "inconclusive")
                self.assertEqual(item["flags"], [])
        self.requests.clear()
        capped = self.run_flow("capped", response(), flags=("--max-cost-inr", "0"))
        self.assertEqual(capped["checks"]["semantic_translation"]["status"], "unavailable")
        self.assertEqual(self.requests, [])
        missing = self.run_flow("missing", response(), environment={})
        self.assertEqual(missing["checks"]["semantic_translation"]["status"], "unavailable")

    def test_clean_non_demo_language_and_baseline_unavailable(self):
        self.row.update(lang="ur", translation="روی نے چائے اور دودھ خریدا۔")
        self.dataset.write_text(json.dumps(self.row) + "\n")
        clean = self.run_flow("clean", response())
        self.assertEqual(clean["checks"]["semantic_translation"]["verdict"], "clean")
        self.assertEqual(clean["status"], "evaluated")
        previous = copy.deepcopy(clean)
        previous["flags"] = [
            {"check": "semantic_translation", "category": "omission", "severity": "major", "field": "source", "span": [16, 24], "explanation": "Missing clause"}
        ]
        current = copy.deepcopy(clean)
        current["checks"]["semantic_translation"] = {"status": "unavailable"}
        _, counts = compare_baseline({("semantic", "ur"): previous}, [current])
        self.assertEqual(counts["unavailable"], 1)
