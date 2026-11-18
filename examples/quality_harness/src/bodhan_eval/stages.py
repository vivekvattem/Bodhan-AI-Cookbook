import hashlib
import os
from contextvars import ContextVar
from pathlib import Path
from tempfile import NamedTemporaryFile

from .models import Unavailable
from .numbers import compare_numbers, valid_number_finding


class Evaluation:
    def __init__(self, models, generate, artifacts, *, provider_smoke=False):
        self.models = models
        self.generate = generate
        self.artifacts = artifacts
        self.provider_smoke = provider_smoke

    def save_audio(self, data):
        digest = hashlib.sha256(data).hexdigest()
        path = self.artifacts / f"{digest}.wav"
        if not path.is_file() or path.read_bytes() != data:
            with NamedTemporaryFile(dir=self.artifacts, delete=False) as handle:
                temp = Path(handle.name)
                try:
                    handle.write(data)
                    handle.close()
                    os.replace(temp, path)
                finally:
                    temp.unlink(missing_ok=True)
        return path, digest


evaluation: ContextVar[Evaluation] = ContextVar("evaluation")


def check(status, reason=None, **details):
    return {"status": status, **({"reason": reason} if reason else {}), **details}


def prepare(items):
    session = evaluation.get()
    for item in items:
        item.setdefault("translation", None)
        item.setdefault("audio", None)
        item["checks"] = {}
        checks = item["checks"]
        issue = item["issues"].get("translation")
        if issue or item["translation"] is not None or not session.generate:
            checks["translation_generation"] = (
                check("unavailable", issue)
                if issue
                else check(
                    "skipped",
                    "supplied translation"
                    if item["translation"] is not None
                    else "generation disabled",
                )
            )
        else:
            try:
                item["translation"] = session.models.translate(
                    item["source"], item["lang"]
                )
                checks["translation_generation"] = check(
                    "evaluated", origin="generated"
                )
            except Unavailable as exc:
                checks["translation_generation"] = check("unavailable", str(exc))
        issue = item["issues"].get("audio")
        if (
            issue
            or item["audio"] is not None
            or not session.generate
            or item["translation"] is None
        ):
            if issue or (
                session.generate
                and item["audio"] is None
                and item["translation"] is None
            ):
                checks["speech_generation"] = check(
                    "unavailable", issue or "no synthesis input"
                )
            else:
                checks["speech_generation"] = check(
                    "skipped",
                    "supplied audio"
                    if item["audio"] is not None
                    else "generation disabled",
                )
        else:
            try:
                data = session.models.speak(item["translation"], item["lang"])
                path, digest = session.save_audio(data)
                item["audio_path"] = str(path)
                item["audio"] = {"origin": "generated", "sha256": digest}
                checks["speech_generation"] = check("evaluated", origin="generated")
            except Unavailable as exc:
                checks["speech_generation"] = check("unavailable", str(exc))
            except OSError:
                checks["speech_generation"] = check(
                    "unavailable", "could not save generated WAV"
                )
    return items


def deterministic(items):
    for item in items:
        checks = item["checks"]
        checks["number"] = compare_numbers(item["source"], item["translation"])
        if checks["translation_generation"]["status"] == "unavailable":
            checks["number"] = check("unavailable", "no usable translation")
        checks["reference_score"] = (
            check("unavailable", item["issues"]["reference"])
            if "reference" in item["issues"]
            else check("pending", "reference scoring is outside this milestone")
            if item.get("reference")
            else check("skipped", "no reference")
        )
        checks["other_translation_checks"] = check(
            "pending", "detectors are outside this milestone"
        )
        checks["speech_quality"] = check(
            "pending", "speech detectors are outside this milestone"
        )
    return items


def model_evidence(items):
    models = evaluation.get().models
    for item in items:
        checks = item["checks"]
        if item["translation"] is None or item["lang"] == "en":
            checks["back_translation"] = check(
                "unavailable" if item["translation"] is None else "skipped",
                "no usable translation"
                if item["translation"] is None
                else "translation is already English",
            )
        else:
            try:
                text = models.translate(
                    item["translation"], "en", purpose="back_translation"
                )
                checks["back_translation"] = check("evaluated", evidence={"text": text})
            except Unavailable as exc:
                checks["back_translation"] = check("unavailable", str(exc))
        if not item.get("audio_path"):
            checks["transcription"] = check(
                "unavailable"
                if checks["speech_generation"]["status"] == "unavailable"
                else "skipped",
                "no usable audio",
            )
        else:
            try:
                text = models.transcribe(Path(item["audio_path"]), item["lang"])
                checks["transcription"] = check("evaluated", evidence={"text": text})
            except Unavailable as exc:
                checks["transcription"] = check("unavailable", str(exc))
    return items


def judging(items):
    session = evaluation.get()
    if session.provider_smoke:
        for item in items:
            smoke_judgment(session.models, item)
        return items
    for item in items:
        checks = item["checks"]
        candidate = checks["number"].get("candidate")
        if candidate is None:
            checks["judge"] = check("skipped", "no number finding to review")
            continue
        content = {name: item[name] for name in ("lang", "source", "translation")}
        content["discrepancy"] = candidate["evidence"]
        back = checks["back_translation"]
        if back["status"] == "evaluated":
            content["back_translation"] = back["evidence"]["text"]
        try:
            verdict = session.models.judge(content)
            checks["judge"] = check(
                "evaluated" if verdict != "uncertain" else "inconclusive",
                verdict=verdict,
            )
        except Unavailable as exc:
            checks["judge"] = check("unavailable", str(exc))
    return items


def smoke_judgment(models, item):
    checks = item["checks"]
    checks["judge"] = check(
        "skipped", "provider smoke uses a separate structured judgment"
    )
    if item["translation"] is None:
        checks["provider_smoke_judge"] = check("unavailable", "no usable translation")
        return
    content = {name: item[name] for name in ("lang", "source", "translation")}
    for name in ("back_translation", "transcription"):
        if checks[name]["status"] == "evaluated":
            content[name] = checks[name]["evidence"]["text"]
    try:
        verdict = models.judge(content, smoke=True)
        checks["provider_smoke_judge"] = check("evaluated", verdict=verdict)
    except Unavailable as exc:
        checks["provider_smoke_judge"] = check("unavailable", str(exc))


def validate_results(items):
    for item in items:
        checks = item["checks"]
        number = checks["number"]
        candidate = number.pop("candidate", None)
        item["flags"] = []
        if candidate is not None:
            if valid_number_finding(candidate, item):
                item["flags"].append(candidate)
            else:
                checks["number"] = check(
                    "inconclusive", "number evidence did not map to original text"
                )
        item.pop("audio_path", None)
        item.pop("issues", None)
    return items
