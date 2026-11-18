import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "sample_data/milestone.jsonl"
OFFLINE = ROOT / "tests/offline"
REQUIRED = ("summary.md", "summary.json", "items.jsonl", "calibration.json")


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class HarnessIntegrationTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.dataset = self.folder / "data" / "milestone.jsonl"
        self.dataset.parent.mkdir()
        shutil.copyfile(FIXTURE, self.dataset)
        subprocess.run(
            [
                sys.executable,
                str(OFFLINE / "fixture_transport.py"),
                str(self.dataset.parent / "audio/fixture.wav"),
            ],
            check=True,
            cwd=self.folder,
        )

    def run_cli(self, name, *flags, dataset=None, offline=True, extra_env=None):
        out = self.folder / name
        env = {
            name: os.environ[name]
            for name in ("PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP")
            if name in os.environ
        }
        env.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        entrypoint = (
            [str(OFFLINE / "run_fixture.py")]
            if offline
            else ["-I", "-m", "bodhan_eval"]
        )
        env.update(extra_env or {})
        completed = subprocess.run(
            [
                sys.executable,
                *entrypoint,
                "--dataset",
                str(dataset or self.dataset),
                "--langs",
                "hi,ta,bn,ur",
                "--out",
                str(out),
                *map(str, flags),
            ],
            cwd=self.folder,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return out

    def edit_dataset(self, name, edit):
        path = self.dataset.with_name(name)
        records = []
        for line in self.dataset.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                records.append(line)
                continue
            edit(value)
            records.append(json.dumps(value, ensure_ascii=False))
        path.write_text("\n".join(records) + "\n", encoding="utf-8")
        return path

    def test_end_to_end_and_baseline_replay(self):
        original_audio = (self.dataset.parent / "audio/fixture.wav").read_bytes()
        plain = self.run_cli("plain")
        original = self.run_cli("generated", "--generate", "--max-cost-inr", "50")
        plain_items = {item["id"]: item for item in jsonl(plain / "items.jsonl")}
        items = {item["id"]: item for item in jsonl(original / "items.jsonl")}
        summary = json.loads((original / "summary.json").read_text())
        self.assertEqual(summary["mode"], "fixture-stub")
        self.assertEqual(summary["rows"], 10)
        self.assertEqual(
            summary["items"],
            {
                "evaluated": 7,
                "failed": 1,
                "skipped": 1,
                "inconclusive": 1,
                "unavailable": 0,
            },
        )
        self.assertEqual(summary["categories"]["number"]["flags"], 3)
        self.assertEqual(
            (self.dataset.parent / "audio/fixture.wav").read_bytes(), original_audio
        )
        self.assertEqual(plain_items["hi-clean"]["audio"], items["hi-clean"]["audio"])
        self.assertEqual(
            plain_items["hi-changed"]["translation"], items["hi-changed"]["translation"]
        )
        self.assertEqual(
            items["hi-clean"]["checks"]["transcription"]["status"], "evaluated"
        )
        self.assertEqual(
            items["ta-missing-audio"]["checks"]["transcription"]["status"],
            "unavailable",
        )
        self.assertEqual(items["hi-clean"]["label_status"], "clean")
        self.assertEqual(items["ta-decimal"]["label_status"], "absent")
        self.assertEqual(
            items["ur-changed"]["flags"][0]["evidence"]["translation"]["text"], "۴۵"
        )
        self.assertIsNone(plain_items["bn-missing"]["translation"])
        self.assertEqual(
            items["bn-missing"]["checks"]["translation_generation"]["status"],
            "evaluated",
        )
        self.assertEqual(items["bn-missing"]["audio"]["origin"], "generated")
        self.assertTrue(
            (
                self.folder
                / ".bodhan-eval/audio"
                / (items["bn-missing"]["audio"]["sha256"] + ".wav")
            ).is_file()
        )
        self.assertFalse((original / "diff.md").exists())

        def revise(row):
            if row["id"] == "hi-changed":
                row["translation"] = "२५ kg"
            elif row["id"] == "bn-changed":
                row["translation"] = {"invalid": True}
            elif row["id"] == "ur-changed":
                row["translation"] = "۵۰ km"

        edited = self.edit_dataset("revised.jsonl", revise)
        flags = ("--baseline", str(original))
        revised = self.run_cli("revised", *flags, dataset=edited)
        diff = (revised / "diff.md").read_text()
        self.assertIn("hi-changed (hi) · resolved", diff)
        self.assertIn("bn-changed (bn) · prior", diff)
        self.assertNotIn("bn-changed (bn) · resolved", diff)
        self.assertIn("ur-changed (ur) · 40 km → ۴۵ km [major] to 40 km → ۵۰ km", diff)
        replay = self.run_cli(
            "replay",
            *flags,
            dataset=edited,
            extra_env={"BODHAN_TEST_FAIL_ON_CALL": "1"},
        )
        telemetry = json.loads(replay.with_name("replay.runtime.json").read_text())
        self.assertEqual(telemetry["provider_calls"], 0)
        self.assertEqual(telemetry["incremental_known_inr"], "0")
        self.assertGreater(telemetry["cache_hits"], 0)
        for name in (*REQUIRED, "diff.md"):
            self.assertEqual(
                (revised / name).read_bytes(), (replay / name).read_bytes(), name
            )

    def test_annotations_are_absent_from_model_requests_and_evidence(self):
        trace_one = self.folder / "requests1.jsonl"
        original = self.run_cli(
            "original", extra_env={"BODHAN_TEST_TRACE": str(trace_one)}
        )

        def change_labels(row):
            row.pop("labels", None)
            row["injection"] = {"expected": "DO-NOT-SEND-THIS"}

        edited = self.edit_dataset("unlabelled.jsonl", change_labels)
        external = self.folder / "annotations.jsonl"
        external.write_text(
            '{"id":"hi-changed","lang":"hi","labels":[]}\n', encoding="utf-8"
        )
        (self.folder / "another").mkdir()
        trace_two = self.folder / "requests2.jsonl"
        changed = self.run_cli(
            "another/changed",
            "--calibration",
            external,
            dataset=edited,
            extra_env={"BODHAN_TEST_TRACE": str(trace_two)},
        )
        self.assertEqual(trace_one.read_bytes(), trace_two.read_bytes())
        for before, after in zip(
            jsonl(original / "items.jsonl"), jsonl(changed / "items.jsonl"), strict=True
        ):
            self.assertEqual(before["flags"], after["flags"])
            self.assertEqual(before["checks"], after["checks"])
        evaluation = json.loads((changed / "calibration.json").read_text())
        self.assertEqual(evaluation["mode"], "fixture-stub")
        self.assertEqual(evaluation["categories"]["number"]["false_positive"], 1)
        unavailable = self.run_cli("unlabelled", dataset=edited, offline=False)
        report = json.loads((unavailable / "calibration.json").read_text())
        self.assertEqual(report["status"], "unavailable")
        item = jsonl(unavailable / "items.jsonl")[1]
        self.assertEqual(item["checks"]["back_translation"]["status"], "unavailable")
        self.assertEqual(item["checks"]["judge"]["status"], "unavailable")
        self.assertEqual(len(item["flags"]), 1)

    def test_cost_cap_blocks_paid_fixture_operations(self):
        out = self.run_cli("budget", "--generate", "--max-cost-inr", "0")
        telemetry = json.loads(out.with_name("budget.runtime.json").read_text())
        self.assertEqual(
            {
                entry["operation"]
                for entry in telemetry["requests"]
                if entry["status"] == "provider_response"
            },
            {"judge"},
        )
        item = next(
            item for item in jsonl(out / "items.jsonl") if item["id"] == "bn-missing"
        )
        self.assertIn("cost cap", item["checks"]["translation_generation"]["reason"])
        self.assertEqual(item["status"], "unavailable")
