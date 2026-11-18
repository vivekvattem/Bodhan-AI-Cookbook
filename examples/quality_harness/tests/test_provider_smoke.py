import io
import json
import os
import subprocess
import sys
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from bodhan_eval.cli import main
from bodhan_eval.models import HostedTransport

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data/provider_smoke.jsonl"
REPORTS = ("summary.md", "summary.json", "items.jsonl", "calibration.json")
MOCK_ENV = {
    "BODHAN_TRANSLATE_KEY": "test-only-translate-secret",
    "BODHAN_SPEAK_KEY": "test-only-speak-secret",
    "BODHAN_TRANSCRIBE_KEY": "test-only-transcribe-secret",
    "LLM_BASE_URL": "https://judge.invalid/v1",
    "LLM_MODEL": "wire-test-judge",
    "LLM_API_KEY": "test-only-judge-secret",
}
BLOCK_NETWORK = """
import runpy
import sys

def block(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname'}:
        raise AssertionError('unexpected network access during cache replay')

sys.addaudithook(block)
sys.argv = ['bodhan_eval', *sys.argv[1:]]
runpy.run_module('bodhan_eval', run_name='__main__', alter_sys=True)
"""


def fixture_audio():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * 240)
    return buffer.getvalue()


class ProviderSmokeTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.calls = []
        self.wav = fixture_audio()

    def cli_args(self, name, *flags, dataset=SAMPLE):
        return [
            "--dataset",
            str(dataset),
            "--langs",
            "hi",
            "--out",
            str(self.root / name),
            "--provider-smoke",
            *flags,
        ]

    def read(self, path):
        return json.loads((self.root / path).read_text(encoding="utf-8"))

    def mock_http(self, request):
        path = request.url.path
        self.calls.append(path)
        if path == "/translate":
            payload = json.loads(request.content)
            self.assertEqual(request.url.host, "api.bodhan.ai")
            self.assertEqual(
                request.headers["Authorization"],
                "Bearer " + MOCK_ENV["BODHAN_TRANSLATE_KEY"],
            )
            target = payload["target_language"]
            self.assertEqual(
                payload["text"], "The box weighs 5 kg." if target == "hi" else "५ kg"
            )
            return httpx.Response(
                200,
                json={
                    "translation": "५ kg" if target == "hi" else "The box weighs 5 kg.",
                    "truncated": False,
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            )
        if path == "/v1/audio/speech":
            self.assertEqual(request.url.host, "api.bodhan.ai")
            self.assertEqual(
                request.headers["Authorization"],
                "Bearer " + MOCK_ENV["BODHAN_SPEAK_KEY"],
            )
            payload = json.loads(request.content)
            self.assertEqual(payload["input"], "५ kg")
            self.assertEqual(payload["model"], "indic-speak")
            self.assertEqual(payload["voice"], "Kavya")
            self.assertEqual(json.loads(payload["instructions"]), {"lang": "hi"})
            return httpx.Response(
                200, content=self.wav, headers={"Content-Type": "audio/wav"}
            )
        if path == "/v1/audio/transcriptions":
            self.assertEqual(request.url.host, "api.bodhan.ai")
            self.assertEqual(
                request.headers["Authorization"],
                "Bearer " + MOCK_ENV["BODHAN_TRANSCRIBE_KEY"],
            )
            self.assertIn("multipart/form-data", request.headers["Content-Type"])
            self.assertIn(self.wav, request.content)
            self.assertIn(b"indic-transcribe", request.content)
            self.assertIn(b"\r\n\r\nhi\r\n", request.content)
            return httpx.Response(200, json={"text": "५ kg"})
        self.assertEqual(str(request.url), "https://judge.invalid/v1/chat/completions")
        self.assertEqual(
            request.headers["Authorization"], "Bearer " + MOCK_ENV["LLM_API_KEY"]
        )
        payload = json.loads(request.content)
        content = json.loads(payload["messages"][1]["content"])
        self.assertEqual(payload["model"], MOCK_ENV["LLM_MODEL"])
        self.assertEqual(content["translation"], "५ kg")
        self.assertEqual(content["back_translation"], "The box weighs 5 kg.")
        self.assertEqual(content["transcription"], "५ kg")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"verdict":"supports"}'}}],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 7,
                    "total_tokens": 27,
                },
            },
        )

    def run_isolated(self, launch, *args, environment=None):
        env = {
            name: os.environ[name]
            for name in ("PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP")
            if name in os.environ
        }
        result = subprocess.run(
            [sys.executable, "-I", *launch, *args],
            cwd=self.root,
            env={**env, **(environment or {})},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_isolated_cli_reports_unavailable_without_credentials(self):
        self.run_isolated(
            ["-m", "bodhan_eval"],
            *self.cli_args("missing", "--generate", "--max-cost-inr", "1"),
        )
        item = self.read("missing/items.jsonl")
        self.assertIsNone(item["translation"])
        self.assertIsNone(item["audio"])
        for name in (
            "translation_generation",
            "speech_generation",
            "back_translation",
            "transcription",
            "provider_smoke_judge",
        ):
            self.assertEqual(item["checks"][name]["status"], "unavailable")
        telemetry = self.read("missing.runtime.json")
        self.assertEqual(telemetry["mode"], "hosted")
        self.assertEqual(telemetry["provider_calls"], 0)

    def test_hosted_wire_mock_and_isolated_network_blocked_replay(self):
        with httpx.Client(transport=httpx.MockTransport(self.mock_http)) as client:
            self.assertEqual(
                main(
                    self.cli_args("cold", "--generate"),
                    transport=HostedTransport(MOCK_ENV, client),
                ),
                0,
            )
        self.assertEqual(
            self.calls,
            [
                "/translate",
                "/v1/audio/speech",
                "/translate",
                "/v1/audio/transcriptions",
                "/v1/chat/completions",
            ],
        )
        item = self.read("cold/items.jsonl")
        self.assertEqual(
            item["checks"]["provider_smoke_judge"],
            {"status": "evaluated", "verdict": "supports"},
        )
        self.assertEqual(item["checks"]["speech_quality"]["status"], "pending")
        self.assertEqual(
            item["checks"]["other_translation_checks"]["status"], "pending"
        )
        self.assertEqual(item["flags"], [])
        telemetry = self.read("cold.runtime.json")
        requests = {request["purpose"]: request for request in telemetry["requests"]}
        self.assertEqual(
            requests["translation_generation"]["usage"]["completion_tokens"], 5
        )
        self.assertEqual(
            requests["speak"]["audio"],
            {
                "duration_seconds": 0.01,
                "sample_rate_hz": 24000,
                "channels": 1,
                "bits_per_sample": 16,
            },
        )
        self.assertEqual(
            requests["provider_smoke_judge"]["usage"]["completion_tokens"], 7
        )
        self.assertIsNone(requests["provider_smoke_judge"]["original_estimated_inr"])
        self.assertEqual(telemetry["incremental_unpriced_requests"], 1)
        self.assertEqual(telemetry["provider_calls"], 5)
        for path in (
            *sorted((self.root / "cold").iterdir()),
            self.root / "cold.runtime.json",
            self.root / ".bodhan-eval/responses.sqlite3",
        ):
            self.assertNotIn(b"test-only-", path.read_bytes())

        judge_env = {key: MOCK_ENV[key] for key in ("LLM_BASE_URL", "LLM_MODEL")}
        self.run_isolated(
            ["-c", BLOCK_NETWORK],
            *self.cli_args("warm", "--generate", "--cache-only"),
            environment=judge_env,
        )
        replay = self.read("warm.runtime.json")
        self.assertEqual(
            (
                replay["provider_calls"],
                replay["cache_hits"],
                replay["incremental_known_inr"],
                replay["incremental_unpriced_requests"],
            ),
            (0, 5, "0", 0),
        )
        for report in REPORTS:
            self.assertEqual(
                (self.root / "cold" / report).read_bytes(),
                (self.root / "warm" / report).read_bytes(),
            )

        audio = self.root / "supplied.wav"
        audio.write_bytes(self.wav)
        supplied = self.root / "supplied.jsonl"
        record = json.loads(SAMPLE.read_text(encoding="utf-8"))
        supplied.write_text(
            json.dumps(
                {**record, "translation": item["translation"], "audio": audio.name}
            )
            + "\n",
            encoding="utf-8",
        )
        self.run_isolated(
            ["-c", BLOCK_NETWORK],
            *self.cli_args("supplied", "--cache-only", dataset=supplied),
            environment=judge_env,
        )
        result = self.read("supplied/items.jsonl")
        self.assertEqual(
            supplied.read_text(encoding="utf-8"),
            json.dumps(
                {**record, "translation": item["translation"], "audio": audio.name}
            )
            + "\n",
        )
        self.assertEqual(
            (result["translation"], result["audio"], audio.read_bytes()),
            (item["translation"], audio.name, self.wav),
        )
        self.assertEqual(
            result["checks"]["translation_generation"]["status"], "skipped"
        )
        self.assertEqual(result["checks"]["speech_generation"]["status"], "skipped")

    def test_total_cap_blocks_unpriced_judge(self):
        with httpx.Client(transport=httpx.MockTransport(self.mock_http)) as client:
            self.assertEqual(
                main(
                    self.cli_args("capped", "--generate", "--max-cost-inr", "1"),
                    transport=HostedTransport(MOCK_ENV, client),
                ),
                0,
            )
        telemetry = self.read("capped.runtime.json")
        self.assertEqual(telemetry["provider_calls"], 4)
        self.assertLess(float(telemetry["incremental_known_inr"]), 1)
        check = self.read("capped/items.jsonl")["checks"]["provider_smoke_judge"]
        self.assertEqual(check["status"], "unavailable")
        self.assertIn("unpriced", check["reason"])
