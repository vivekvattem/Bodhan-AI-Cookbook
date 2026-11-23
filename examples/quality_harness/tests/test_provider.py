import json
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from bodhan_eval.cache import RequestCache
from bodhan_eval.models import HostedTransport, Models, Request, Unavailable, judge_rates, judge_reserve, parse_response


class ProviderTest(unittest.TestCase):
    def test_configured_estimated_token_pacing_and_oversized_request(self):
        current, sent = [0.0], []

        def sleep(seconds):
            current[0] += seconds

        def respond(request):
            sent.append(current[0])
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"verdict":"supports"}'}}]})

        env = {"LLM_BASE_URL": "https://judge.invalid/v1", "LLM_MODEL": "mock", "LLM_API_KEY": "mock"}
        with TemporaryDirectory() as folder, httpx.Client(transport=httpx.MockTransport(respond)) as http:
            cache = RequestCache(Path(folder) / "cache.sqlite3")
            self.addCleanup(cache.close)
            model = Models(HostedTransport(env, http), cache, None, sleeper=sleep, clock=lambda: current[0],
                           judge_rpm=20, judge_estimated_tpm=1700)
            model.judge({"source": "first"})
            model.judge({"source": "second"})
            self.assertEqual(sent, [0, 60])
            with self.assertRaisesRegex(Unavailable, "estimated request exceeds"):
                model.judge({"source": "x" * 3000})
            self.assertEqual(len(sent), 2)

    def test_local_ollama_profile_is_zero_cost_only_for_exact_endpoint_and_model(self):
        endpoint = "http://127.0.0.1:11434/v1/chat/completions"
        self.assertEqual(judge_rates(endpoint, "qwen3:1.7b")[:2], (Decimal(0), Decimal(0)))
        self.assertIsNone(judge_rates("http://example.com:11434/v1/chat/completions", "qwen3:1.7b"))
        self.assertIsNone(judge_rates(endpoint, "other-model"))

    def test_optional_groq_rate_limit_wait_retries_past_default_and_honors_header(self):
        current, attempts, sleeps = [0.0], [], []

        def sleep(seconds):
            sleeps.append(seconds)
            current[0] += seconds

        def respond(request):
            attempts.append(current[0])
            if len(attempts) <= 4:
                return httpx.Response(429, headers={"retry-after": "31"})
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"verdict":"supports"}'}}],
                                             "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

        environment = {"LLM_BASE_URL": "https://api.groq.com/openai/v1", "LLM_MODEL": "openai/gpt-oss-20b", "LLM_API_KEY": "mock"}
        with TemporaryDirectory() as folder, httpx.Client(transport=httpx.MockTransport(respond)) as http:
            cache = RequestCache(Path(folder) / "cache.sqlite3")
            self.addCleanup(cache.close)
            model = Models(HostedTransport(environment, http), cache, None, sleeper=sleep, clock=lambda: current[0],
                           wait_for_groq_rate_limit=True)
            self.assertEqual(model.judge({"source": "test"}), "supports")
            self.assertEqual(len(attempts), 5)
            self.assertTrue(all(b - a >= 31 for a, b in zip(attempts, attempts[1:], strict=False)))
            self.assertTrue(all(s <= 30 for s in sleeps))
            self.assertEqual(model.incremental_unknown, 4)

    def test_groq_local_limit_counts_dispatches_and_retries_but_not_cache_hits(self):
        current = [0.0]
        dispatches = []

        def sleep(seconds):
            current[0] += seconds

        def respond(request):
            dispatches.append(current[0])
            if len(dispatches) == 2:
                return httpx.Response(429, headers={"retry-after": "1"})
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"verdict":"supports"}'}}],
                                             "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

        environment = {"LLM_BASE_URL": "https://api.groq.com/openai/v1", "LLM_MODEL": "openai/gpt-oss-20b", "LLM_API_KEY": "mock"}
        with TemporaryDirectory() as folder, httpx.Client(transport=httpx.MockTransport(respond)) as http:
            cache = RequestCache(Path(folder) / "cache.sqlite3")
            self.addCleanup(cache.close)
            model = Models(HostedTransport(environment, http), cache, Decimal("10"), sleeper=sleep, clock=lambda: current[0])
            for index in range(27):
                self.assertEqual(model.judge({"source": str(index)}), "supports")
            before = current[0]
            model.judge({"source": "0"})
            self.assertEqual(current[0], before)
            self.assertEqual(len(dispatches), 28)
            self.assertTrue(all(b - a >= 3.0 - 1e-8 for a, b in zip(dispatches, dispatches[1:], strict=False)))
            self.assertGreaterEqual(dispatches[20] - dispatches[0], 60 - 1e-8)
            self.assertEqual(model.events[1]["attempts"][0]["backoff_seconds"], 1)

    def test_groq_small_model_profile_uses_json_and_published_tariff(self):
        from bodhan_eval.judge_providers import request_options

        endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.assertEqual(judge_rates(endpoint, "openai/gpt-oss-20b")[:2],
                         (Decimal("0.075") / 1000000, Decimal("0.30") / 1000000))
        options = request_options(endpoint, "openai/gpt-oss-20b", semantic=True)
        self.assertEqual(options["response_format"], {"type": "json_object"})
        self.assertEqual(options["reasoning_effort"], "low")

    def test_generic_openai_compatible_provider_fails_closed_on_unknown_tariff(self):
        environment = {"LLM_BASE_URL": "https://api.openai.com/v1", "LLM_MODEL": "example-model", "LLM_API_KEY": "test-only"}
        calls = []

        def respond(request):
            calls.append(request)
            payload = json.loads(request.content)
            self.assertEqual(str(request.url), "https://api.openai.com/v1/chat/completions")
            self.assertEqual(payload["model"], "example-model")
            self.assertNotIn("reasoning_effort", payload)
            self.assertNotIn("response_format", payload)
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"verdict":"supports"}'}}]})

        with TemporaryDirectory() as folder, httpx.Client(transport=httpx.MockTransport(respond)) as http:
            cache = RequestCache(Path(folder) / "cache.sqlite3")
            self.addCleanup(cache.close)
            transport = HostedTransport(environment, http)
            with self.assertRaisesRegex(Unavailable, "cost cap"):
                Models(transport, cache, Decimal("1")).judge({"source": "test"})
            self.assertEqual(len(calls), 0)
            uncapped = Models(transport, cache, None)
            self.assertEqual(uncapped.judge({"source": "test"}), "supports")
            self.assertEqual(uncapped.telemetry()["providers"]["judge"]["unknown_billing_attempts"], 1)
            replay = Models(transport, cache, Decimal("0"), cache_only=True)
            self.assertEqual(replay.judge({"source": "test"}), "supports")
            self.assertEqual(len(calls), 1)

    def test_provider_error_body_uses_only_fixed_safe_classifications(self):
        secret = "secret-test-value"
        environment = {"LLM_BASE_URL": "https://example.invalid/v1", "LLM_MODEL": "mock", "LLM_API_KEY": secret}
        for message, expected in (("No endpoints found for " + secret, "No matching provider endpoint found."),
                                  ("Unrecognized input " + secret, "[provider message redacted]")):
            with self.subTest(message=expected), httpx.Client(transport=httpx.MockTransport(
                lambda request, body=message: httpx.Response(404, json={"error": {"message": body}})
            )) as client:
                transport = HostedTransport(environment, client)
                request = Request("judge", "mock", transport.judge_endpoint, {}, Decimal("0"))
                with self.assertRaises(Unavailable):
                    transport.send(request)
                body = transport.last_error["redacted_body"]
                self.assertEqual(body["error"]["message"], expected)
                self.assertNotIn(secret, json.dumps(transport.last_error))

    def test_openrouter_free_route_has_zero_reservation_without_pricing_paid_route(self):
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        transport = HostedTransport({"LLM_BASE_URL": "https://openrouter.ai/api/v1", "LLM_MODEL": "openai/gpt-oss-120b:free"})
        self.assertEqual(judge_reserve(transport, {"messages": [], "max_tokens": 64}), Decimal("0"))
        self.assertIsNone(judge_rates(endpoint, "openai/gpt-oss-120b"))
        self.assertIsNone(judge_rates("https://other.example/v1/chat/completions", "openai/gpt-oss-120b:free"))

    def test_groq_judge_price_and_account_rate_headers(self):
        environment = {
            "LLM_BASE_URL": "https://api.groq.com/openai/v1",
            "LLM_MODEL": "openai/gpt-oss-120b",
            "LLM_API_KEY": "test-only",
        }
        calls = []

        def respond(request):
            calls.append(request)
            self.assertEqual(json.loads(request.content)["response_format"], {"type": "json_object"})
            self.assertEqual(json.loads(request.content)["reasoning_effort"], "low")
            return httpx.Response(
                200,
                headers={"x-ratelimit-limit-requests": "1000", "x-ratelimit-limit-tokens": "8000"},
                json={
                    "choices": [{"message": {"content": '{"verdict":"supports"}'}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                },
            )

        with TemporaryDirectory() as folder, httpx.Client(transport=httpx.MockTransport(respond)) as http:
            cache = RequestCache(Path(folder) / "cache.sqlite3")
            self.addCleanup(cache.close)
            transport = HostedTransport(environment, http)
            blocked = Models(transport, cache, Decimal("0"))
            with self.assertRaisesRegex(Unavailable, "cost cap"):
                blocked.judge({"source": "short"}, smoke=True)
            self.assertEqual(len(calls), 0)
            allowed = Models(transport, cache, Decimal("1"))
            self.assertEqual(allowed.judge({"source": "short"}, smoke=True), "supports")
            self.assertEqual(allowed.telemetry()["incremental_known_inr"], "0.0021")
            self.assertEqual(allowed.events[-1]["rate_limits"]["x-ratelimit-limit-tokens"], "8000")
            self.assertEqual(len(calls), 1)

    def test_openrouter_smoke_price_admission_and_usage_cost(self):
        environment = {
            "LLM_BASE_URL": "https://openrouter.ai/api/v1",
            "LLM_MODEL": "google/gemini-2.5-pro",
            "LLM_API_KEY": "test-only",
        }
        calls = []

        def respond(request):
            calls.append(request)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": '{"verdict":"supports"}'}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                },
            )

        with TemporaryDirectory() as folder, httpx.Client(transport=httpx.MockTransport(respond)) as http:
            cache = RequestCache(Path(folder) / "cache.sqlite3")
            self.addCleanup(cache.close)
            blocked = Models(HostedTransport(environment, http), cache, Decimal("0"))
            with self.assertRaisesRegex(Unavailable, "cost cap"):
                blocked.judge({"source": "short"}, smoke=True)
            self.assertEqual(len(calls), 0)
            allowed = Models(HostedTransport(environment, http), cache, Decimal("1"))
            self.assertEqual(allowed.judge({"source": "short"}, smoke=True), "supports")
            self.assertEqual(allowed.telemetry()["incremental_known_inr"], "0.0225")
            replay = Models(HostedTransport(environment, http), cache, Decimal("1"), cache_only=True)
            self.assertEqual(replay.judge({"source": "short"}, smoke=True), "supports")
            self.assertEqual(len(calls), 1)
            self.assertEqual(replay.telemetry()["incremental_known_inr"], "0")

    def test_cache_only_missing_response_is_not_a_dispatch(self):
        with TemporaryDirectory() as folder:
            cache = RequestCache(Path(folder) / "cache.sqlite3")
            self.addCleanup(cache.close)
            models = Models(HostedTransport({}), cache, None, cache_only=True)
            with self.assertRaisesRegex(Unavailable, "response not cached"):
                models.judge({"source": "source"})
            self.assertEqual(models.telemetry()["provider_attempts"], 0)
            self.assertEqual(models.events[0]["decision"], "cache_miss")
            self.assertFalse(models.events[0]["dispatch_attempted"])

    def test_chat_translation_rejects_incomplete_output(self):
        request = Request(
            "translate",
            "indic-translate",
            "https://api.bodhan.ai/v1/chat/completions",
            {},
            Decimal("0.01"),
        )
        for reason in ("length", "content_filter", None):
            with self.subTest(reason=reason):
                with self.assertRaises(Unavailable):
                    parse_response(
                        request,
                        {
                            "choices": [
                                {
                                    "finish_reason": reason,
                                    "message": {"content": "partial text"},
                                }
                            ]
                        },
                    )

    def test_exact_key_includes_content_provider_settings_and_audio(self):
        request = Request(
            "judge",
            "model",
            "https://provider.invalid/api",
            {"temperature": 0, "text": "25"},
            Decimal(0),
            "rubric-v1",
            b"audio A",
        )
        exact = RequestCache.key("production", request.exact_input())
        for change in (
            {"model": "other"},
            {"endpoint": "https://else.invalid"},
            {"prompt_version": "rubric-v2"},
            {"audio": b"audio B"},
            {"payload": {"temperature": 0.1, "text": "25"}},
            {"payload": {"temperature": 0, "text": "२५"}},
        ):
            self.assertNotEqual(
                exact,
                RequestCache.key("production", replace(request, **change).exact_input()),
            )
        self.assertNotEqual(exact, RequestCache.key("stub", request.exact_input()))

    def test_hosted_http_and_failure_are_injectable(self):
        calls = []

        def respond(request):
            calls.append(request)
            self.assertEqual(request.url, "https://api.bodhan.ai/v1/chat/completions")
            if len(calls) == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"translation": "25 kg", "usage": {"completion_tokens": 5}})

        with (
            TemporaryDirectory() as temp,
            httpx.Client(transport=httpx.MockTransport(respond)) as http,
        ):
            cache = RequestCache(Path(temp) / "requests.sqlite3")
            self.addCleanup(cache.close)
            transport = HostedTransport({"BODHAN_TRANSLATE_KEY": "test-only"}, http)
            models = Models(transport, cache, None)
            # A transient 503 is retried within the bounded request policy.
            self.assertEqual(models.translate("25 kg", "hi"), "25 kg")
            self.assertEqual(models.translate("25 kg", "hi"), "25 kg")
            self.assertEqual(len(calls), 2)
            self.assertEqual(json.loads(calls[0].content)["target_language_code"], "hi")
            self.assertNotIn(b"test-only", (Path(temp) / "requests.sqlite3").read_bytes())
