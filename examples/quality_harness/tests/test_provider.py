import json
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from bodhan_eval.cache import RequestCache
from bodhan_eval.models import HostedTransport, Models, Request, Unavailable


class ProviderTest(unittest.TestCase):
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
                RequestCache.key(
                    "production", replace(request, **change).exact_input()
                ),
            )
        self.assertNotEqual(exact, RequestCache.key("stub", request.exact_input()))

    def test_hosted_http_and_failure_are_injectable(self):
        calls = []

        def respond(request):
            calls.append(request)
            self.assertEqual(request.url, "https://api.bodhan.ai/translate")
            if len(calls) == 1:
                return httpx.Response(503)
            return httpx.Response(
                200, json={"translation": "25 kg", "usage": {"completion_tokens": 5}}
            )

        with (
            TemporaryDirectory() as temp,
            httpx.Client(transport=httpx.MockTransport(respond)) as http,
        ):
            cache = RequestCache(Path(temp) / "requests.sqlite3")
            self.addCleanup(cache.close)
            transport = HostedTransport({"BODHAN_TRANSLATE_KEY": "test-only"}, http)
            models = Models(transport, cache, None)
            with self.assertRaisesRegex(Unavailable, "HTTP 503"):
                models.translate("25 kg", "hi")
            self.assertEqual(models.translate("25 kg", "hi"), "25 kg")
            self.assertEqual(models.translate("25 kg", "hi"), "25 kg")
            self.assertEqual(len(calls), 2)
            self.assertEqual(json.loads(calls[0].content)["target_language"], "hi")
            self.assertNotIn(
                b"test-only", (Path(temp) / "requests.sqlite3").read_bytes()
            )
