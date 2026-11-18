import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


RUNNER = Path(__file__).with_name("run.py").resolve()


class HeadlessFlowTest(unittest.TestCase):
    def test_digit_comparisons_in_fresh_processes(self) -> None:
        environment = {
            name: os.environ[name]
            for name in ("PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP")
            if name in os.environ
        }
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"

        with TemporaryDirectory() as workdir:
            for digits, normalized, equal in (("२५", "25", True), ("३५", "35", False)):
                with self.subTest(digits=digits):
                    result = subprocess.run(
                        [sys.executable, str(RUNNER), digits, "25"],
                        cwd=workdir,
                        env=environment,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=60,
                    )
                    self.assertEqual(
                        json.loads(result.stdout),
                        {"left": normalized, "right": "25", "equal": equal},
                    )
