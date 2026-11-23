import copy
import hashlib
import json
import struct
import unittest
import wave
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from bodhan_eval.cache import RequestCache
from bodhan_eval.cli import main
from bodhan_eval.dataset import load_dataset
from bodhan_eval.demo import LANGUAGES, build, generate, read_jsonl, write_bundle, write_jsonl
from bodhan_eval.formatting import check_latex, check_tables
from bodhan_eval.inject import NotApplicable, audio_variant, split_sources, text_variant
from bodhan_eval.models import HostedTransport, Models, RetryPolicy, Unavailable

RECIPE = Path(__file__).resolve().parents[1]
REPORTS = ("summary.md", "summary.json", "items.jsonl", "calibration.json")


def test_bases(sources):
    # Test-only target text; never used to prepare the real Bodhan dataset.
    return [{"source_item": source["id"], "lang": lang, "parts": ["मात्रा २५ है।", "यह दूसरा वाक्य है।"]}
            for source in sources for lang in LANGUAGES]


class InjectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sources = read_jsonl(RECIPE / "data/sources.jsonl")

    def test_reproducible_bundle_and_source_group_split(self):
        bases = test_bases(self.sources)
        first = build(self.sources, bases, 42)
        self.assertEqual(first, build(list(reversed(self.sources)), list(reversed(bases)), 42))
        self.assertNotEqual(first, build(self.sources, bases, 43))
        items, truth = first
        self.assertEqual(len(items), 120)
        self.assertEqual({item["lang"] for item in items}, {"hi", "ta", "bn"})
        groups = {}
        for row in truth:
            groups.setdefault(row["source_group"], set()).add(row["split"])
        self.assertEqual(len(groups), 8)
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertEqual(sum(row["split"] == "calibration" for row in truth), 30)
        self.assertTrue(all(set(item) == {"id", "lang", "source", "translation"} for item in items))
        # Duplicate exact source text is grouped even if caller IDs differ.
        split = split_sources(["same", "same", "other"], 7)
        self.assertEqual(len(split), 2)
        for name in ("a", "b"):
            out = self.root / name
            out.mkdir()
            write_jsonl(out / "sources.jsonl", self.sources)
            write_jsonl(out / "base_translations.jsonl", bases)
            write_bundle(out, self.sources, bases, 42)
        for path in (self.root / "a").iterdir():
            self.assertEqual(path.read_bytes(), (self.root / "b" / path.name).read_bytes())

    def test_shipped_dataset_manifest_spans_and_split_isolation(self):
        data = RECIPE / "data"
        manifest = json.loads((data / "manifest.json").read_text())
        for name, digest in manifest["files_sha256"].items():
            self.assertEqual(hashlib.sha256((data / name).read_bytes()).hexdigest(), digest)
        rebuilt = build(self.sources, read_jsonl(data / "base_translations.jsonl"), manifest["seed"])
        self.assertEqual(rebuilt, (read_jsonl(data / "demo.jsonl"), read_jsonl(data / "ground_truth.jsonl")))
        self.assertGreaterEqual(len(rebuilt[0]), 100)
        self.assertEqual({lang: sum(item["lang"] == lang for item in rebuilt[0]) for lang in LANGUAGES}, {"hi": 40, "ta": 40, "bn": 40})
        for split in ("calibration", "evaluation"):
            labels = read_jsonl(data / (split + "_labels.jsonl"))
            records = read_jsonl(data / (split + ".jsonl"))
            self.assertEqual({row["id"] for row in labels}, {row["id"] for row in records})
            self.assertTrue(all(row["split"] == split for row in labels))
        before = {row["source"] for row in read_jsonl(data / "calibration.jsonl")}
        after = {row["source"] for row in read_jsonl(data / "evaluation.jsonl")}
        self.assertFalse(before & after)

    def test_edit_locations_controls_and_original_source_omission(self):
        items, truth = build(self.sources, test_bases(self.sources), 42)
        controls = {(row["source_item"], row["lang"]): item for item, row in zip(items, truth, strict=True) if row["operator"] == "clean"}
        for item, row in zip(items, truth, strict=True):
            clean = controls[row["source_item"], row["lang"]]
            if row["operator"] == "clean":
                self.assertEqual(row["labels"], [])
                self.assertIsNone(row["edit"])
                self.assertFalse(row["base_translation_verified"])
                for checker in (check_tables, check_latex):
                    self.assertNotEqual(checker(item["source"], item["translation"]).get("verdict"), "finding")
                continue
            edit = row["edit"]
            a, b = edit["original_span"]
            self.assertEqual(clean["translation"][a:b], edit["original_text"])
            self.assertEqual(item["translation"], clean["translation"][:a] + edit["replacement"] + clean["translation"][b:])
            a, b = edit["output_span"]
            self.assertEqual(item["translation"][a:b], edit["replacement"])
            label = row["labels"][0]
            field = clean["translation"] if label["field"] == "clean_translation" else item[label["field"]]
            a, b = label["span"]
            self.assertEqual(field[a:b], label["text"])
            if row["operator"] == "omission":
                self.assertEqual(label["field"], "source")
                source = next(source for source in self.sources if source["id"] == row["source_item"])
                self.assertEqual(label["text"], source["parts"][0])
            if row["operator"] == "format_damage":
                self.assertTrue(any(checker(item["source"], item["translation"]).get("verdict") == "finding" for checker in (check_tables, check_latex)))

    def test_digit_changes_preserve_script_and_unsupported_inputs_abstain(self):
        for text in ("२५", "٢٥", "２５", "25"):
            item = {"id": "digits", "source": "25", "translation": text, "lang": "hi"}
            changed, truth = text_variant(item, "digit_change", 4)
            self.assertNotEqual(changed["translation"], text)
            self.assertEqual(len(changed["translation"]), len(text))
            self.assertEqual(len(truth["edit"]["original_text"]), 1)
        for operator in ("digit_change", "omission", "term_swap", "format_damage"):
            with self.assertRaises(NotApplicable):
                text_variant({"id": "none", "source": "text", "translation": "text", "lang": "ta"}, operator, 4)

    def test_generation_uses_only_bodhan_adapter_and_preserves_failures(self):
        class FakeModels:
            def __init__(self):
                self.calls = []

            def translate(self, text, lang, **kwargs):
                self.calls.append((text, lang, kwargs))
                if lang == "ta":
                    raise Unavailable("mock provider failure")
                return "TEST ONLY: " + text

        model = FakeModels()
        bases, failures = generate(self.sources[:1], model)
        self.assertEqual(len(bases), 2)
        self.assertEqual(len(failures), 2)
        self.assertEqual(len(model.calls), 6)
        self.assertEqual({call[2]["purpose"] for call in model.calls}, {"demo_base_translation"})

    def test_429_pauses_endpoint_without_counting_blocked_calls_as_attempts(self):
        requests = []

        def wire(request):
            requests.append(request)
            return httpx.Response(429)

        cache = RequestCache(self.root / "cache.sqlite3")
        self.addCleanup(cache.close)
        with httpx.Client(transport=httpx.MockTransport(wire)) as client:
            transport = HostedTransport({"BODHAN_TRANSLATE_KEY": "test-only"}, client)
            model = Models(transport, cache, Decimal("1"), retry_policy=RetryPolicy(max_attempts=1))
            with self.assertRaisesRegex(Unavailable, "HTTP 429"):
                model.translate("first", "hi")
            with self.assertRaisesRegex(Unavailable, "endpoint paused"):
                model.translate("second", "ta")
            self.assertEqual(len(requests), 1)
            self.assertEqual(model.telemetry()["provider_attempts"], 1)
            self.assertEqual(model.events[-1]["decision"], "rate_limit_block")
            # Fresh invocations may retry; a rate limit is not a permanent cache entry.
            recovered = Models(HostedTransport({"BODHAN_TRANSLATE_KEY": "test-only"}, client), cache, Decimal("1"),
                               retry_policy=RetryPolicy(max_attempts=1))
            with self.assertRaisesRegex(Unavailable, "HTTP 429"):
                recovered.translate("second", "ta")
            self.assertEqual(len(requests), 2)

        class RateLimited:
            calls = 0

            def translate(self, *args, **kwargs):
                self.calls += 1
                raise Unavailable("provider HTTP 429")

        paused = RateLimited()
        bases, failures = generate(self.sources, paused)
        self.assertEqual((bases, len(failures), paused.calls), ([], 1, 1))

    def test_audio_edits_are_exact_reproducible_and_separate(self):
        source = self.root / "source.wav"
        samples = b"".join(struct.pack("<h", i % 2000 - 1000) for i in range(16000))
        with wave.open(str(source), "wb") as wav:
            wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            wav.writeframes(samples)
        expected = {
            "truncate": samples[:22400],
            "repeat_segment": samples[:19200] + samples[6400:19200] + samples[19200:],
            "insert_silence": samples[:6400] + bytes(24000) + samples[6400:],
        }
        for operator in expected:
            end = 1.4 if operator == "truncate" else 1.2
            first, second = self.root / (operator + ".wav"), self.root / (operator + "-again.wav")
            result = audio_variant(source, first, operator, 42, source_item="control", start=0.4, end=end)
            self.assertEqual(result, audio_variant(source, second, operator, 42, source_item="control", start=0.4, end=end))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with wave.open(str(first), "rb") as wav:
                self.assertEqual(wav.readframes(wav.getnframes()), expected[operator])
            self.assertEqual(result["linguistic_boundaries"], "unverified")
            self.assertEqual(result["source_item"], "control")
        with self.assertRaises(ValueError):
            audio_variant(source, source, "truncate", 1, source_item="control")

    def test_detector_requests_ignore_injection_metadata_and_cache_replays(self):
        items, truth = build(self.sources[:2], test_bases(self.sources[:2]), 42)
        # Include both clean controls and every operator through the real graph.
        chosen = [item for item in items if item["lang"] == "hi"]
        env = {"BODHAN_TRANSLATE_KEY": "test-only", "LLM_API_KEY": "test-only",
               "LLM_BASE_URL": "https://test.invalid/v1", "LLM_MODEL": "mock"}
        requests = []

        def wire(request):
            data = json.loads(request.content)
            requests.append((str(request.url), data))
            if request.url.host == "api.bodhan.ai":
                answer = "A test-only English back translation."
            else:
                verdict = {"verdict": "clean", "explanation": "Test stub, not a quality measurement.", "findings": []}
                answer = json.dumps(verdict) if data["max_tokens"] == 4096 else '{"verdict":"uncertain"}'
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": answer}}], "usage": {"completion_tokens": 10}})

        for run in ("plain", "annotated"):
            folder = self.root / run
            folder.mkdir()
            records = copy.deepcopy(chosen)
            if run == "annotated":
                by_key = {(row["id"], row["lang"]): row for row in truth}
                for item in records:
                    item.update(by_key[item["id"], item["lang"]])
                    item["injection_secret"] = "DETECTION_MUST_NOT_SEE_THIS"
            write_jsonl(folder / "dataset.jsonl", records)
            with httpx.Client(transport=httpx.MockTransport(wire)) as client:
                args = ["--dataset", str(folder / "dataset.jsonl"), "--langs", "hi", "--out", str(folder / "report")]
                self.assertEqual(main(args, transport=HostedTransport(env, client)), 0)
            if run == "plain":
                first_requests = requests.copy()
            else:
                self.assertEqual(requests, first_requests)
                self.assertNotIn("DETECTION_MUST_NOT_SEE_THIS", json.dumps(requests))
            requests.clear()
        def findings(name):
            return [row["flags"] for row in read_jsonl(self.root / name / "report/items.jsonl")]
        self.assertEqual(findings("plain"), findings("annotated"))
        rows = load_dataset(self.root / "annotated/dataset.jsonl", {"hi"})
        self.assertTrue(all(set(row.fields) <= {"lang", "source", "translation", "reference", "audio", "issues"} for row in rows))
        with httpx.Client(transport=httpx.MockTransport(lambda request: self.fail("cache replay attempted a provider call"))) as client:
            args = ["--dataset", str(self.root / "plain/dataset.jsonl"), "--langs", "hi", "--out", str(self.root / "plain/replay"), "--cache-only"]
            self.assertEqual(main(args, transport=HostedTransport(env, client)), 0)
        for name in REPORTS:
            self.assertEqual((self.root / "plain/report" / name).read_bytes(), (self.root / "plain/replay" / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
