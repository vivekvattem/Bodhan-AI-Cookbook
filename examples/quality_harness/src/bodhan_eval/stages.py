import hashlib
import os
from contextvars import ContextVar
from pathlib import Path
from tempfile import NamedTemporaryFile

from .formatting import check_latex, check_tables, valid_format_finding
from .models import Unavailable
from .numbers import compare_numbers, quantity_candidates, valid_number_finding
from .semantic import finalize_judgment
from .speech import signal_check, signal_findings, transcript_check


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
            checks["number"] = check("unavailable", "no usable translation", policy_version=checks["number"]["policy_version"])
        checks["markdown_table"] = check_tables(item["source"], item["translation"])
        checks["latex"] = check_latex(item["source"], item["translation"])
        checks["reference_score"] = (
            check("unavailable", item["issues"]["reference"])
            if "reference" in item["issues"]
            else check("pending", "reference scoring is outside this milestone")
            if item.get("reference")
            else check("skipped", "no reference")
        )
        checks["other_translation_checks"] = check(
            "pending", "wrong-language and script detectors are outside this milestone"
        )
        if item.get("audio_path"):
            checks["audio_signal"] = signal_check(item["audio_path"])
        elif checks.get("speech_generation", {}).get("status") == "unavailable":
            checks["audio_signal"] = check("unavailable", "no usable audio")
        else:
            checks["audio_signal"] = check("skipped", "no audio supplied")
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
        if checks["transcription"]["status"] == "evaluated":
            checks["speech_alignment"] = transcript_check(
                item["translation"], checks["transcription"]["evidence"]["text"], checks["audio_signal"]
            )
        elif checks["transcription"]["status"] == "skipped":
            checks["speech_alignment"] = check("skipped", "no transcript available")
        else:
            checks["speech_alignment"] = check("unavailable", "transcription unavailable")
    return items


def judging(items):
    session = evaluation.get()
    if session.provider_smoke:
        for item in items:
            smoke_judgment(session.models, item)
            item["checks"]["semantic_translation"] = check("skipped", "provider connectivity smoke")
        return items
    for item in items:
        checks = item["checks"]
        if item["translation"] is None:
            checks["semantic_translation"] = check("unavailable", "no usable translation")
            continue
        content = {name: item[name] for name in ("lang", "source", "translation")}
        source_numbers = quantity_candidates(item["source"], "source")
        translation_numbers = quantity_candidates(item["translation"], "translation")
        if checks["number"]["status"] == "inconclusive" or max(len(source_numbers), len(translation_numbers)) > 1:
            content["quantity_candidates"] = source_numbers + translation_numbers
        if checks["back_translation"]["status"] == "evaluated":
            content["back_translation"] = checks["back_translation"]["evidence"]["text"]
        try:
            checks["semantic_translation"] = {"status": "pending", "judgment": session.models.semantic_judge(content)}
        except Unavailable as exc:
            checks["semantic_translation"] = check("unavailable", str(exc))
    for item in items:
        checks = item["checks"]
        candidates = checks["number"].get("candidates") or ([checks["number"]["candidate"]] if checks["number"].get("candidate") else [])
        if not candidates:
            checks["judge"] = check("skipped", "no number finding to review")
            continue
        checks["judge"] = check("skipped", "quantity correspondence is reviewed by semantic judge")
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
        candidate = number.get("candidate")
        candidates = number.get("candidates") or ([candidate] if candidate else [])
        item["flags"] = []
        finding = number.get("finding")
        if finding:
            if valid_number_finding(finding, item):
                item["flags"].append(finding)
            else:
                checks["number"] = {**number, "status": "inconclusive", "reason": "number evidence did not map to original text"}
        for candidate in candidates:
            if not valid_number_finding(candidate, item):
                checks["number"] = {**number, "status": "inconclusive", "reason": "number evidence did not map to original text"}
        for name in ("markdown_table", "latex"):
            candidates = checks[name].pop("candidates", [])
            if all(valid_format_finding(candidate, item) for candidate in candidates):
                item["flags"].extend(candidates)
            else:
                checks[name] = check("inconclusive", "format evidence did not map to original text")
        semantic = checks.get("semantic_translation", {})
        if "judgment" in semantic:
            checks["semantic_translation"], flags = finalize_judgment(semantic["judgment"], item)
            existing = {(flag["category"], flag["field"], tuple(flag["span"])) for flag in item["flags"]}
            item["flags"].extend(flag for flag in flags if (flag["category"], flag["field"], tuple(flag["span"])) not in existing)
        item["flags"].extend(signal_findings(checks["audio_signal"]))
        item.pop("audio_path", None)
        item.pop("issues", None)
    return items
