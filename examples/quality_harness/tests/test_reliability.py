import json
import subprocess
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
from bodhan_eval.cache import RequestCache
from bodhan_eval.cli import main
from bodhan_eval.dataset import load_dataset
from bodhan_eval.demo import write_jsonl
from bodhan_eval.models import HostedTransport, Models, Unavailable, retry_after_seconds
from bodhan_eval.replay import plan_key

ENV = {"BODHAN_TRANSLATE_KEY": "private-test-translate", "LLM_BASE_URL": "https://judge.invalid/v1",
       "LLM_MODEL": "mock-judge", "LLM_API_KEY": "private-test-judge"}
REPORTS = ("summary.md", "summary.json", "items.jsonl", "calibration.json")


def translated():
    return httpx.Response(200, json={"translation": "Mock translation", "usage": {"completion_tokens": 5}})


class ReliabilityTest(unittest.TestCase):
    def test_bodhan_translation_honors_retry_after_51_seconds(self):
        attempts, sleeps = [], []

        def wire(request):
            attempts.append(request)
            return httpx.Response(429, headers={"Retry-After": "51"}) if len(attempts) == 1 else translated()

        with httpx.Client(transport=httpx.MockTransport(wire)) as http:
            model = Models(HostedTransport(ENV, http), self.cache, Decimal("1"), sleeper=sleeps.append)
            self.assertEqual(model.translate("short", "hi"), "Mock translation")
        self.assertEqual(sleeps, [30, 21])
        self.assertEqual(len(attempts), 2)
        self.assertEqual(model.events[0]["attempts"][0]["backoff_seconds"], 51)
        self.assertEqual(model.reserved_uncertain, Decimal("0.04096"))

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = RequestCache(self.root / "responses.sqlite3")
        self.addCleanup(self.cache.close)

    def test_retry_after_dates_headers_and_safe_failure_metadata(self):
        self.assertEqual(retry_after_seconds("8"), 8)
        self.assertEqual(retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT", now=1445412470), 10)
        self.assertIsNone(retry_after_seconds("invalid"))
        self.assertIsNone(retry_after_seconds("inf"))
        count, sleeps = [], []

        def wire(request):
            count.append(request)
            if len(count) == 1:
                return httpx.Response(429, headers={"Retry-After": "8", "x-ratelimit-remaining-tokens": "447"},
                                      json={"error": {"code": "rate_limit_exceeded", "message": ENV["BODHAN_TRANSLATE_KEY"]}})
            return translated()

        with httpx.Client(transport=httpx.MockTransport(wire)) as http:
            model = Models(HostedTransport(ENV, http), self.cache, Decimal("1"), sleeper=sleeps.append)
            self.assertEqual(model.translate("hello", "hi"), "Mock translation")
        self.assertEqual(sleeps, [8])
        self.assertEqual((model.calls, model.successes, model.unknown), (2, 1, 1))
        attempt = model.events[0]["attempts"][0]
        self.assertEqual(attempt["rate_limits"]["retry-after"], "8")
        self.assertEqual(attempt["provider_error"]["code"], "rate_limit_exceeded")
        self.assertNotIn(ENV["BODHAN_TRANSLATE_KEY"], json.dumps(model.telemetry()))
        self.assertNotIn(ENV["BODHAN_TRANSLATE_KEY"].encode(), (self.root / "responses.sqlite3").read_bytes())

    def test_permanent_failures_long_retry_after_and_bounded_attempts(self):
        for status, headers, attempts, waits in ((401, {}, 1, []), (403, {}, 1, []), (400, {}, 1, []),
                                                 (429, {"retry-after": "120"}, 1, []), (503, {}, 3, [1, 2])):
            with self.subTest(status=status, headers=headers):
                calls, sleeps = [], []

                def wire(request, calls=calls, status=status, headers=headers):
                    calls.append(request)
                    return httpx.Response(status, headers=headers)

                with httpx.Client(transport=httpx.MockTransport(wire)) as http:
                    model = Models(HostedTransport(ENV, http), self.cache, Decimal("1"), sleeper=sleeps.append)
                    with self.assertRaises(Unavailable):
                        model.translate(str((status, headers)), "hi")
                self.assertEqual(len(calls), attempts)
                self.assertEqual(sleeps, waits)

    def test_retry_budget_keeps_uncertain_reservations_and_cached_spend_is_free(self):
        requests, sleeps = [], []

        def wire(request):
            requests.append(request)
            return httpx.Response(503)

        with httpx.Client(transport=httpx.MockTransport(wire)) as http:
            model = Models(HostedTransport(ENV, http), self.cache, Decimal("0.10"), sleeper=sleeps.append)
            with self.assertRaisesRegex(Unavailable, "503"):
                model.translate("limited", "hi")
            self.assertEqual(model.calls, 2)
            self.assertEqual(model.reserved_uncertain, Decimal("0.08192"))
            self.assertEqual(sleeps, [1])
            self.assertEqual(model.events[0]["retry_blocked"], "budget")
            self.assertEqual([Decimal(attempt["reserved_inr"]) for attempt in model.events[0]["attempts"]], [Decimal("0.04096")] * 2)
            self.assertEqual(Decimal(model.events[0]["attempts"][1]["exposure_before_inr"]), Decimal("0.04096"))
            self.assertTrue(all(attempt["billing_unknown"] for attempt in model.events[0]["attempts"]))
            self.assertEqual(Decimal(model.telemetry()["max_cost_inr"]), Decimal("0.10"))
            with self.assertRaisesRegex(Unavailable, "cost cap"):
                model.translate("other", "hi")
            self.assertEqual(len(requests), 2)
        with httpx.Client(transport=httpx.MockTransport(lambda r: translated())) as http:
            model = Models(HostedTransport(ENV, http), self.cache, Decimal("1"))
            model.translate("cached", "hi")
        with httpx.Client(transport=httpx.MockTransport(lambda r: self.fail("cached replay dispatched"))) as http:
            replay = Models(HostedTransport(ENV, http), self.cache, Decimal("0"))
            replay.translate("cached", "hi")
            self.assertGreater(replay.known, 0)
            self.assertEqual(replay.incremental, 0)

    def test_threaded_calls_share_one_dispatch_slot_and_atomic_cap(self):
        active, peak, calls = 0, 0, 0
        guard = threading.Lock()

        def wire(request):
            nonlocal active, peak, calls
            with guard:
                active += 1
                peak = max(peak, active)
                calls += 1
            time.sleep(0.01)  # Simulated overlap opportunity, no provider network.
            with guard:
                active -= 1
            return httpx.Response(200, json={"translation": "mock", "usage": {"completion_tokens": 2000}})

        with httpx.Client(transport=httpx.MockTransport(wire)) as http:
            model = Models(HostedTransport(ENV, http), self.cache, Decimal("0.07"))

            def evaluate(index):
                try:
                    return model.translate(str(index), "hi")
                except Unavailable:
                    return "blocked"

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(evaluate, range(8)))
            self.assertEqual((peak, calls, results.count("mock")), (1, 1, 1))
            self.assertEqual(model.incremental, Decimal("0.04"))
            self.assertLessEqual(model.incremental + model.reserved_uncertain, model.max_cost)

    def test_interrupted_journal_survives_reopen_and_allows_normal_retry(self):
        run_id = self.cache.start_run("interrupted-plan")
        calls = []

        def wire(request):
            calls.append(request)
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return translated()

        with httpx.Client(transport=httpx.MockTransport(wire)) as http:
            model = Models(HostedTransport(ENV, http), self.cache, Decimal("1"), run_id=run_id)
            model.translate("complete", "hi")
            with self.assertRaises(KeyboardInterrupt):
                model.translate("interrupted", "hi")
        # Open a distinct SQLite connection, as a fresh process would.
        other = RequestCache(self.root / "responses.sqlite3")
        self.addCleanup(other.close)
        with httpx.Client(transport=httpx.MockTransport(lambda r: self.fail("interrupted replay dispatched"))) as http:
            replay = Models(HostedTransport(ENV, http), other, Decimal("0"), cache_only=True, run_id=run_id)
            replay.translate("complete", "hi")
            with self.assertRaisesRegex(Unavailable, "interruption"):
                replay.translate("interrupted", "hi")
            with self.assertRaisesRegex(Unavailable, "response not cached"):
                replay.translate("never reached", "hi")
            self.assertEqual((replay.calls, replay.unknown), (0, 1))
        with httpx.Client(transport=httpx.MockTransport(lambda r: translated())) as http:
            normal = Models(HostedTransport(ENV, http), other, Decimal("1"))
            normal.translate("interrupted", "hi")
            self.assertEqual(normal.calls, 1)

    def test_abrupt_process_exit_commits_uncertain_dispatch_before_send(self):
        code = '''
import os,sys,httpx
from pathlib import Path
from decimal import Decimal
from bodhan_eval.cache import RequestCache
from bodhan_eval.models import Models,HostedTransport
root=Path(sys.argv[1])
cache=RequestCache(root/'responses.sqlite3')
run_id=cache.start_run('killed-process')
(root/'run-id.txt').write_text(run_id)
def terminate(request):
    os._exit(17)
with httpx.Client(transport=httpx.MockTransport(terminate)) as http:
    models=Models(HostedTransport({'BODHAN_TRANSLATE_KEY':'test-only'},http),cache,Decimal('1'),run_id=run_id)
    models.translate('unfinished','hi')
'''
        result = subprocess.run([sys.executable, "-I", "-c", code, str(self.root)], capture_output=True)
        self.assertEqual(result.returncode, 17, result.stderr.decode())
        run_id = (self.root / "run-id.txt").read_text()
        with httpx.Client(transport=httpx.MockTransport(lambda r: self.fail("killed-process replay dispatched"))) as http:
            replay = Models(HostedTransport(ENV, http), self.cache, Decimal("0"), cache_only=True, run_id=run_id)
            with self.assertRaisesRegex(Unavailable, "interruption"):
                replay.translate("unfinished", "hi")
            self.assertEqual((replay.calls, replay.unknown), (0, 1))

    def test_concurrent_retry_failures_cannot_spend_the_same_reservation(self):
        calls = []

        def wire(request):
            calls.append(request)
            return httpx.Response(503)

        with httpx.Client(transport=httpx.MockTransport(wire)) as http:
            model = Models(HostedTransport(ENV, http), self.cache, Decimal("0.07"), sleeper=lambda delay: self.fail("unfunded retry slept"))

            def evaluate(index):
                try:
                    model.translate(str(index), "hi")
                except Unavailable:
                    return
                self.fail("failed provider became clean")

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(evaluate, range(8)))
            self.assertEqual((len(calls), model.calls), (1, 1))
            self.assertEqual(model.reserved_uncertain, Decimal("0.04096"))

    def test_rate_limited_graph_replays_all_reports_and_old_snapshot_survives_recovery(self):
        dataset = self.root / "dataset.jsonl"
        records = [{"id": str(i), "lang": "hi", "source": "Read the lesson.", "translation": text}
                   for i, text in enumerate(("पाठ पढ़ें।", "किताब पढ़ें।", "पाठ सुनें।"))]
        write_jsonl(dataset, records)

        def wire(request):
            if request.url.host == "api.bodhan.ai":
                return httpx.Response(429, headers={"Retry-After": "120"})
            body = {"verdict": "clean", "explanation": "Mock only, no measured quality.", "findings": []}
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(body)}}]})

        def run(name, http, *extra):
            args = ["--dataset", str(dataset), "--langs", "hi", "--out", str(self.root / name), *extra]
            self.assertEqual(main(args, transport=HostedTransport(ENV, http)), 0)

        # Test the optional fifth required file too.
        with httpx.Client(transport=httpx.MockTransport(wire)) as http:
            run("baseline", http)
            run("limited", http, "--baseline", str(self.root / "baseline"))
        with httpx.Client(transport=httpx.MockTransport(lambda r: self.fail("snapshot replay dispatched"))) as http:
            run("replay", http, "--cache-only", "--max-cost-inr", "0", "--baseline", str(self.root / "baseline"))
        for name in (*REPORTS, "diff.md"):
            self.assertEqual((self.root / "limited" / name).read_bytes(), (self.root / "replay" / name).read_bytes())
        telemetry = json.loads((self.root / "replay.runtime.json").read_text())
        self.assertEqual(telemetry["provider_attempts"], 0)
        self.assertEqual(telemetry["incremental_known_inr"], "0")
        self.assertEqual(telemetry["incremental_unpriced_requests"], 0)

        def recovered(request):
            return translated() if request.url.host == "api.bodhan.ai" else wire(request)

        with httpx.Client(transport=httpx.MockTransport(recovered)) as http:
            run("recovered", http)
        result = json.loads((self.root / "recovered/items.jsonl").read_text().splitlines()[0])
        self.assertEqual(result["checks"]["back_translation"]["status"], "evaluated")
        with httpx.Client(transport=httpx.MockTransport(lambda r: self.fail("historical replay dispatched"))) as http:
            run("historical", http, "--cache-only", "--replay-from", str(self.root / "limited"),
                "--baseline", str(self.root / "baseline"))
        for name in (*REPORTS, "diff.md"):
            self.assertEqual((self.root / "limited" / name).read_bytes(), (self.root / "historical" / name).read_bytes())
        # Fingerprints exclude annotation content and secrets.
        rows = load_dataset(dataset, {"hi"})
        before = plan_key(rows, HostedTransport(ENV))
        rows[0].labels = [{"type": "omission", "injection": "do not detect this"}]
        self.assertEqual(before, plan_key(rows, HostedTransport({**ENV, "LLM_API_KEY": "other-key"})))

    def test_budget_and_configuration_skips_replay_without_becoming_failures(self):
        dataset = self.root / "missing.jsonl"
        write_jsonl(dataset, [{"id": "one", "lang": "hi", "source": "Read.", "translation": "पढ़ें।"}])
        for env in (ENV, {"LLM_BASE_URL": ENV["LLM_BASE_URL"], "LLM_MODEL": ENV["LLM_MODEL"]}):
            suffix = "budget" if env == ENV else "configuration"
            with httpx.Client(transport=httpx.MockTransport(lambda r: self.fail("blocked call dispatched"))) as http:
                base = ["--dataset", str(dataset), "--langs", "hi", "--max-cost-inr", "0"]
                with patch("bodhan_eval.models.time.sleep", side_effect=AssertionError("blocked call slept")):
                    self.assertEqual(main([*base, "--out", str(self.root / suffix)], transport=HostedTransport(env, http)), 0)
                    self.assertEqual(main([*base, "--out", str(self.root / (suffix + "-replay")), "--cache-only"],
                                          transport=HostedTransport(env, http)), 0)
            for name in REPORTS:
                self.assertEqual((self.root / suffix / name).read_bytes(), (self.root / (suffix + "-replay") / name).read_bytes())
            telemetry = json.loads((self.root / (suffix + "-replay.runtime.json")).read_text())
            self.assertEqual(telemetry["incremental_unpriced_requests"], 0)


if __name__ == "__main__":
    unittest.main()
