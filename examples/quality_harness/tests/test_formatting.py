import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bodhan_eval.formatting import check_latex, check_tables, valid_format_finding
from bodhan_eval.report import calibration, compare_baseline
from bodhan_eval.stages import deterministic, validate_results

TABLE = "| Name | Value |\n| --- | ---: |\n| Box | 5 |"


class FormattingTest(unittest.TestCase):
    def test_actual_cli_reports_format_flags_without_network(self):
        launch = """
import runpy, sys
def block(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname'}:
        raise AssertionError('unexpected network access')
sys.addaudithook(block)
sys.argv = ['bodhan_eval', *sys.argv[1:]]
runpy.run_module('bodhan_eval', run_name='__main__', alter_sys=True)
"""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "dataset.jsonl"
            row = {
                "id": "format", "lang": "hi",
                "source": TABLE + "\n\n" + r"Literal \n and \newcommand. \[x\]",
                "translation": TABLE.replace("| Box | 5 |", "| Box |") + "\n\n" + r"Literal \n and \newcommand. \[x",
            }
            data.write_text(json.dumps(row) + "\n")
            process = subprocess.run(
                [sys.executable, "-I", "-c", launch, "--dataset", str(data), "--langs", "hi", "--out", str(root / "report"), "--cache-only"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            item = json.loads((root / "report/items.jsonl").read_text())
            self.assertEqual(item["source"], row["source"])
            self.assertEqual(item["translation"], row["translation"])
            self.assertEqual({flag["check"] for flag in item["flags"]}, {"markdown_table", "latex"})
            self.assertEqual(item["status"], "evaluated")
            self.assertEqual(item["checks"]["audio_signal"]["status"], "skipped")
            self.assertEqual(item["checks"]["other_translation_checks"]["status"], "pending")
            telemetry = json.loads((root / "report.runtime.json").read_text())
            self.assertEqual(telemetry["provider_attempts"], 0)
            summary = json.loads((root / "report/summary.json").read_text())
            self.assertGreaterEqual(summary["categories"]["format"]["flags"], 2)

    def test_tables_controls_and_escaped_pipes(self):
        for text in (TABLE, r"| A \| B | `x|y` |" + "\n| --- | --- |\n| a | b |"):
            self.assertEqual(check_tables(text, text)["verdict"], "match")
        self.assertEqual(check_tables("prose", "प्राकृतिक पाठ")["status"], "skipped")
        for fence in ("```", "~~~~"):
            text = f"{fence}\n{TABLE}\n{fence}\n"
            self.assertEqual(check_tables(text, text)["status"], "skipped")

    def test_table_damage_and_original_spans(self):
        for target in (TABLE.replace("| Box | 5 |", "| Box |"), TABLE.replace("| --- | ---: |", "| --- | bad |"), "Plain text"):
            result = check_tables(TABLE, target)
            self.assertEqual(result["verdict"], "finding")
            for candidate in result["candidates"]:
                self.assertTrue(valid_format_finding(candidate, {"source": TABLE, "translation": target}))
        self.assertEqual(check_tables(TABLE.replace("| Box | 5 |", "| Box |"), TABLE)["status"], "inconclusive")

    def test_latex_controls_escapes_and_code(self):
        controls = [
            r"Value $x_{i}+\frac{a}{b}$.",
            r"\[x=\{a,b\}\] and \(y\)",
            r"\begin{align}x&=\begin{matrix}a&b\end{matrix}\end{align}",
            r"Escaped \$ and braces {ordinary prose} and price $5.",
            "```latex\n$broken{\n```\n`$also broken`",
            "~~~\n\\[broken\n~~~",
        ]
        for text in controls:
            with self.subTest(text=text):
                self.assertFalse(check_latex(text, text).get("candidates"))

    def test_latex_damage_and_unicode_spans(self):
        source = r"Equation \[x_{i}\]"
        for target in (r"📦 समीकरण \[x_{i}", r"📦 \[x_{i\]", r"📦 \begin{align}x\end{equation}", r"📦 \)"):
            with self.subTest(target=target):
                result = check_latex(source, target)
                self.assertEqual(result["verdict"], "finding")
                for candidate in result["candidates"]:
                    item = {"source": source, "translation": target}
                    self.assertTrue(valid_format_finding(candidate, item))
                    candidate["span"][0] += 1
                    self.assertFalse(valid_format_finding(candidate, item))
        self.assertEqual(check_latex(source, "text only")["status"], "inconclusive")

    def test_stages_baseline_and_calibration(self):
        item = {
            "source": "25 kg\n" + TABLE,
            "translation": "25 kg\n" + TABLE.replace("| Box | 5 |", "| Box |"),
            "issues": {},
            "checks": {"translation_generation": {"status": "skipped"}},
        }
        deterministic([item])
        validate_results([item])
        self.assertEqual(item["flags"][0]["category"], "format")
        self.assertEqual(item["checks"]["other_translation_checks"]["status"], "pending")
        item.update(id="format", lang="hi")
        baseline = {("format", "hi"): copy.deepcopy(item)}
        unavailable = copy.deepcopy(item)
        unavailable["flags"] = []
        unavailable["checks"]["markdown_table"] = {"status": "unavailable"}
        _, counts = compare_baseline(baseline, [unavailable])
        self.assertEqual(counts["unavailable"], 1)
        clean = copy.deepcopy(item)
        clean["flags"] = []
        clean["checks"]["markdown_table"] = {"status": "evaluated", "verdict": "match"}
        _, counts = compare_baseline(baseline, [clean])
        self.assertEqual(counts["improved"], 1)
        # Format findings must never count as number false positives.
        item["checks"]["number"] = {"status": "evaluated", "verdict": "match"}
        result = calibration([item], {("format", "hi"): []}, 0, "fixture-stub")
        self.assertEqual(result["categories"]["number"]["false_positive"], 0)

    def test_missing_translation(self):
        for checker in (check_tables, check_latex):
            self.assertEqual(checker("source", None)["status"], "unavailable")
