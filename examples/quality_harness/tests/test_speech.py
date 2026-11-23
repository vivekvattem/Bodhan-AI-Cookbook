import hashlib
import json
import math
import struct
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

from bodhan_eval.cli import main
from bodhan_eval.models import Unavailable
from bodhan_eval.speech import signal_check, signal_findings, text_units, transcript_check


def make_wav(path, segments, rate=8000):
    with wave.open(str(path), "wb") as target:
        target.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        samples = []
        for seconds, kind in segments:
            for index in range(round(seconds * rate)):
                value = {"silence": 0, "clipped": 32767, "tone": round(7000 * math.sin(index * 0.2))}[kind]
                samples.append(struct.pack("<h", value))
        target.writeframes(b"".join(samples))


class SpeechTest(unittest.TestCase):
    def test_live_repetition_inside_replacement_and_intentional_controls(self):
        expected = "डिब्बे का वज़न 5 किग्रा है।"
        observed = "डिब्बे का वज़न डिब्बे का वज़न फाइव के ग्रा है"
        result = transcript_check(expected, observed, {})
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual([candidate["type"] for candidate in result["candidates"]], ["repetition", "numeric_rendering"])
        evidence = result["candidates"][0]["evidence"]
        start, end = evidence["span"]
        self.assertEqual(observed[start:end], "डिब्बे का वज़न")
        intentional = "डिब्बे का वज़न डिब्बे का वज़न 5 किग्रा है।"
        control = transcript_check(intentional, observed, {})
        self.assertNotIn("repetition", [candidate["type"] for candidate in control["candidates"]])
        for text in ("go go home", "one two one two stop", intentional):
            self.assertEqual(transcript_check(text, text, {})["verdict"], "match")

    def test_numeric_rendering_stays_unresolved_and_retains_both_texts(self):
        expected = "डिब्बे का वज़न 5 किग्रा है।"
        transcript = "डिब्बे का वज़न फाइव के ग्रा है"
        result = transcript_check(expected, transcript, {})
        candidate = result["candidates"][0]
        self.assertEqual(candidate["type"], "numeric_rendering")
        self.assertEqual(candidate["evidence"]["text"], "5 किग्रा")
        self.assertEqual(candidate["transcript_evidence"]["text"], "फाइव के ग्रा")
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(transcript_check("25 kg", "२५ kg", {})["verdict"], "match")
        self.assertEqual(transcript_check("We need tea", "We need coffee", {})["candidates"][0]["type"], "substitution")

    def test_general_unicode_controls_and_unresolved_spoken_numbers(self):
        for expected, observed in (("２５ KG", "25 kg."), ("٢٥ kg", "२५ KG"), ("café", "cafe\u0301")):
            self.assertEqual(transcript_check(expected, observed, {})["verdict"], "match")
        for expected, observed in (("5 kg", "five kilograms"), ("5 किग्रा", "फाइव के ग्रा")):
            result = transcript_check(expected, observed, {})
            self.assertEqual(result["status"], "inconclusive")
            evidence = result["candidates"][0]["evidence"]
            start, end = evidence["span"]
            self.assertEqual(expected[start:end], evidence["text"])

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def audio(self, name, segments):
        path = self.root / name
        make_wav(path, segments)
        return signal_check(path)

    def test_signal_clean_silent_clipping_short_and_corrupt(self):
        clean = self.audio("clean.wav", [(1, "tone")])
        self.assertEqual(clean["status"], "evaluated")
        self.assertEqual(clean["measurements"]["duration_seconds"], 1)
        self.assertEqual(signal_findings(clean), [])
        silent = self.audio("silent.wav", [(2, "silence")])
        self.assertEqual(silent["measurements"]["silence_fraction"], 1)
        self.assertEqual(signal_findings(silent)[0]["interval_seconds"], [0, 2])
        clipped = self.audio("clipped.wav", [(0.5, "tone"), (0.2, "clipped"), (0.3, "tone")])
        self.assertEqual(signal_findings(clipped)[0]["interval_seconds"], [0.5, 0.7])
        short = self.audio("short.wav", [(0.05, "tone")])
        self.assertEqual(short["status"], "inconclusive")
        broken = self.root / "broken.wav"
        broken.write_bytes(b"invalid")
        self.assertEqual(signal_check(broken)["status"], "unavailable")
        empty = self.root / "empty.wav"
        make_wav(empty, [])
        self.assertEqual(signal_check(empty)["status"], "unavailable")
        unsupported = self.root / "unsupported.wav"
        make_wav(unsupported, [(0.2, "tone")])
        wav_bytes = bytearray(unsupported.read_bytes())
        wav_bytes[20:22] = (3).to_bytes(2, "little")
        unsupported.write_bytes(wav_bytes)
        self.assertEqual(signal_check(unsupported)["status"], "unavailable")
        self.assertEqual(signal_check(self.root / "missing.wav")["status"], "unavailable")
        self.assertEqual(signal_check(None)["status"], "skipped")

    def test_transcript_alignment_clean_omission_truncation_and_repetition(self):
        signal = self.audio("tail.wav", [(1, "tone"), (1.5, "silence")])
        expected = "पहला दूसरा तीसरा चौथा"
        self.assertEqual(transcript_check(expected, expected, signal)["verdict"], "match")
        truncation = transcript_check(expected, "पहला दूसरा", signal)
        candidate = truncation["candidates"][0]
        self.assertEqual((truncation["status"], candidate["type"]), ("inconclusive", "truncation"))
        self.assertEqual(candidate["supporting_audio_interval_seconds"], [1, 2.5])
        start, end = candidate["evidence"]["span"]
        self.assertEqual(expected[start:end], candidate["evidence"]["text"])
        omission = transcript_check(expected, "पहला तीसरा चौथा", signal)
        self.assertEqual(omission["candidates"][0]["type"], "omission")
        repetition = transcript_check(expected, "पहला दूसरा दूसरा तीसरा चौथा", signal)
        self.assertEqual(repetition["candidates"][0]["type"], "repetition")
        start, end = repetition["candidates"][0]["evidence"]["span"]
        self.assertEqual("पहला दूसरा दूसरा तीसरा चौथा"[start:end], repetition["candidates"][0]["evidence"]["text"])
        self.assertEqual(transcript_check(expected, None, signal)["status"], "unavailable")

    def test_normalization_and_fallible_transcription(self):
        self.assertEqual([x[0] for x in text_units("२५ किग्रा")], ["25", "किग्रा"])
        self.assertEqual([x[0] for x in text_units("é")], [x[0] for x in text_units("e\u0301")])
        clean_signal = self.audio("healthy.wav", [(2, "tone")])
        mismatch = transcript_check("एक दो तीन", "एक तीन", clean_signal)
        self.assertEqual(mismatch["status"], "inconclusive")
        self.assertNotIn("supporting_audio_interval_seconds", mismatch["candidates"][0])
        self.assertEqual(signal_findings(clean_signal), [])

    def test_full_graph_isolates_bad_rows_and_replays_cached_transcripts(self):
        make_wav(self.root / "clean.wav", [(1.5, "tone")])
        make_wav(self.root / "silent.wav", [(1.2, "silence")])
        make_wav(self.root / "truncated.wav", [(1, "tone"), (1.5, "silence")])
        make_wav(self.root / "repeated.wav", [(1.7, "tone")])
        (self.root / "bad.wav").write_bytes(b"broken")
        source = "One two three."
        rows = [
            {"id": "clean", "source": source, "lang": "hi", "translation": "एक दो तीन", "audio": "clean.wav"},
            {"id": "silent", "source": source, "lang": "hi", "translation": "एक दो तीन", "audio": "silent.wav"},
            {"id": "truncated", "source": source, "lang": "hi", "translation": "एक दो तीन", "audio": "truncated.wav"},
            {"id": "repeated", "source": source, "lang": "hi", "translation": "एक दो तीन", "audio": "repeated.wav"},
            {"id": "bad", "source": source, "lang": "hi", "translation": "एक दो तीन", "audio": "bad.wav"},
            {"id": "missing", "source": source, "lang": "hi", "translation": "एक दो तीन", "audio": "absent.wav"},
        ]
        dataset = self.root / "items.jsonl"
        dataset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        original = hashlib.sha256((self.root / "clean.wav").read_bytes()).hexdigest()
        transcripts = {
            (self.root / "truncated.wav").read_bytes(): "एक दो",
            (self.root / "repeated.wav").read_bytes(): "एक दो दो तीन",
        }

        class Transport:
            namespace, mode = "speech-test", "fixture-stub"
            judge_model, judge_endpoint, judge_reserve_inr = "judge", "https://test.invalid", None
            calls = []
            blocked = False

            def missing_configuration(self, request):
                return "test judge disabled" if request.operation == "judge" else None

            def send(self, request):
                if self.blocked:
                    raise AssertionError("replay attempted a provider call")
                self.calls.append(request.operation)
                if request.operation == "translate":
                    return {"translation": source, "usage": {"completion_tokens": 3}}
                if request.operation == "transcribe":
                    return {"text": transcripts.get(request.audio, "एक दो तीन")}
                raise Unavailable("unsupported test operation")

        transport = Transport()

        def run(name):
            return main(["--dataset", str(dataset), "--langs", "hi", "--out", str(self.root / name), "--max-cost-inr", "1"], transport=transport)

        self.assertEqual(run("cold"), 0)
        items = {item["id"]: item for item in map(json.loads, (self.root / "cold/items.jsonl").read_text().splitlines())}
        self.assertEqual(items["clean"]["checks"]["speech_alignment"]["verdict"], "match")
        self.assertTrue(any(flag["category"] == "speech_silence" for flag in items["silent"]["flags"]))
        truncation = items["truncated"]["checks"]["speech_alignment"]
        self.assertEqual(truncation["status"], "inconclusive")
        self.assertEqual(truncation["candidates"][0]["type"], "truncation")
        self.assertEqual(truncation["candidates"][0]["supporting_audio_interval_seconds"], [1, 2.5])
        self.assertEqual(items["repeated"]["checks"]["speech_alignment"]["candidates"][0]["type"], "repetition")
        self.assertEqual(items["bad"]["checks"]["audio_signal"]["status"], "unavailable")
        self.assertEqual(items["missing"]["checks"]["audio_signal"]["status"], "unavailable")
        self.assertEqual(items["clean"]["checks"]["speech_generation"]["status"], "skipped")
        self.assertEqual(hashlib.sha256((self.root / "clean.wav").read_bytes()).hexdigest(), original)
        transport.blocked = True
        self.assertEqual(run("warm"), 0)
        for name in ("summary.md", "summary.json", "items.jsonl", "calibration.json"):
            self.assertEqual((self.root / "cold" / name).read_bytes(), (self.root / "warm" / name).read_bytes())
        telemetry = json.loads((self.root / "warm.runtime.json").read_text())
        self.assertEqual(telemetry["provider_attempts"], 0)
