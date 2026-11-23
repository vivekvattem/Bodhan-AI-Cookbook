"""Seeded perturbations. Return detection content and separate edit ground truth.

An injected error is known relative to the base output. Neither the base output
nor the linguistic validity of a perturbation is automatically human verified.
"""

import hashlib
import math
import random
import re
import unicodedata
import wave

from .models import canonical_json

OPERATORS = ("clean", "omission", "digit_change", "term_swap", "format_damage")


class NotApplicable(ValueError):
    """The requested operator cannot make a supported edit to this input."""


def seeded(seed, *identity):
    digest = hashlib.sha256(canonical_json([seed, *identity]).encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def split_sources(sources, seed, fraction=0.25):
    """Group by exact source content, across languages and all derived variants."""
    if not 0 < fraction < 1:
        raise ValueError("calibration fraction must be between zero and one")
    groups = sorted(set(sources))
    if len(groups) < 2:
        raise ValueError("at least two independent sources are required")
    seeded(seed, "source-split").shuffle(groups)
    count = max(1, min(len(groups) - 1, round(len(groups) * fraction)))
    return {text: "calibration" if i < count else "evaluation" for i, text in enumerate(groups)}


def text_variant(item, operator, seed, *, omission=None, term_pairs=()):
    """Apply one edit; spans are half-open Unicode code points, never bytes.

    Explicit omission mapping names paired source/translation spans. Without
    that mapping, a deleted target sentence has only a clean_translation anchor,
    which is unsuitable for measuring source-side omission localization.
    Term pairs must be supplied by the caller, never invented by a judge.
    """
    if operator not in OPERATORS:
        raise ValueError("unknown text injection operator")
    content = {key: item[key] for key in ("id", "source", "lang", "translation")}
    text = content["translation"]
    rng = seeded(seed, item["source"], item["lang"], item["id"], operator)
    identity = hashlib.sha256(canonical_json([seed, item["id"], item["lang"], operator]).encode()).hexdigest()[:20]
    content["id"] = identity
    truth = {
        "id": identity, "lang": item["lang"], "source_item": item["id"],
        "operator": operator, "annotation_origin": "programmatic",
        "human_review": "pending", "base_translation_verified": False,
        "labels": [], "edit": None,
        "base_translation_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    if operator == "clean":
        truth["category"] = None
        return content, truth
    category = {"omission": "omission", "digit_change": "number", "term_swap": "term_entity", "format_damage": "format"}[operator]
    replacement = ""
    if operator == "omission":
        if omission:
            start, end = omission["translation"]
        else:
            sentences = list(re.finditer(r"[^.!?।\n]+[.!?।](?:\s|$)", text.split("\n\n", 1)[0]))
            if len(sentences) < 2:
                raise NotApplicable("need two sentences or an explicit omission span")
            start, end = rng.choice(sentences).span()
    elif operator == "digit_change":
        choices = [i for i, char in enumerate(text) if char.isdecimal()]
        if not choices:
            raise NotApplicable("no Unicode decimal digit")
        start = rng.choice(choices)
        end = start + 1
        old = unicodedata.decimal(text[start])
        value = rng.choice([value for value in range(10) if value != old])
        # Unicode decimal blocks are contiguous; preserve the original script.
        replacement = chr(ord(text[start]) - old + value)
    elif operator == "term_swap":
        choices = [(m.start(), m.end(), new) for old, new in term_pairs if old and new and old != new
                   for m in re.finditer(r"(?<!\w)" + re.escape(old) + r"(?!\w)", text)]
        if not choices:
            raise NotApplicable("no supplied literal term pair occurs")
        start, end, replacement = rng.choice(choices)
    else:
        # Delete a supported structural marker outside fenced and inline code.
        from .formatting import outside_code

        masked = outside_code(text)
        choices = [m.span() for m in re.finditer(r"(?<!\\)\\\)|(?<!\\)\\\]|(?<!\\)\$", masked)]
        choices += [m.span() for m in re.finditer(r"(?m)(?<=\|)\s*-{3,}\s*(?=\|)", masked)]
        if not choices:
            raise NotApplicable("no supported math or table marker")
        start, end = rng.choice(choices)
    if not (type(start) is int and type(end) is int and 0 <= start < end <= len(text)):
        raise ValueError("edit span does not select original text")
    content["translation"] = text[:start] + replacement + text[end:]
    truth["category"] = category
    truth["edit"] = {
        "field": "translation", "original_span": [start, end],
        "original_text": text[start:end], "replacement": replacement,
        "output_span": [start, start + len(replacement)],
    }
    label = {"type": category, "severity": "minor" if category == "format" else "major"}
    if operator == "omission" and omission and "source" in omission:
        a, b = omission["source"]
        if not (type(a) is int and type(b) is int and 0 <= a < b <= len(item["source"])):
            raise ValueError("invalid paired source omission span")
        label.update(field="source", span=[a, b], text=item["source"][a:b])
    elif not replacement:
        label.update(field="clean_translation", span=[start, end], text=text[start:end])
    else:
        label.update(field="translation", span=[start, start + len(replacement)], text=replacement)
    truth["labels"] = [label]
    return content, truth


def audio_variant(source, target, operator, seed, *, source_item, start=None, end=None, silence_seconds=1.5):
    """Edit PCM frames only. Metadata proves signal edits, not spoken words."""
    if operator not in {"truncate", "repeat_segment", "insert_silence"}:
        raise ValueError("unknown audio injection operator")
    if source.resolve() == target.resolve():
        raise ValueError("audio output must preserve the original file")
    with wave.open(str(source), "rb") as wav:
        params = wav.getparams()
        data = wav.readframes(params.nframes)
    width = params.nchannels * params.sampwidth
    if params.comptype != "NONE" or len(data) != params.nframes * width or params.nframes < 4:
        raise NotApplicable("need a nonempty complete PCM WAV")
    duration = params.nframes / params.framerate
    rng = seeded(seed, source_item, hashlib.sha256(data).hexdigest(), operator)
    start = duration * rng.uniform(0.2, 0.4) if start is None else start
    end = duration * rng.uniform(0.5, 0.7) if end is None else end
    if not all(math.isfinite(x) for x in (start, end, silence_seconds)) or not 0 <= start < end <= duration or silence_seconds <= 0:
        raise ValueError("invalid audio edit interval")
    a, b = round(start * params.framerate), round(end * params.framerate)
    if a >= b or b >= params.nframes:
        raise NotApplicable("edit must retain audio on both sides of its endpoint")
    effective = [a / params.framerate, b / params.framerate]
    if operator == "truncate":
        output = data[:b * width]
        affected, inserted, category = [effective[1], duration], None, "speech_omission"
    elif operator == "repeat_segment":
        segment = data[a * width:b * width]
        output = data[:b * width] + segment + data[b * width:]
        affected, inserted, category = effective, [effective[1], (2 * b - a) / params.framerate], "speech_repetition"
    else:
        frames = round(silence_seconds * params.framerate)
        if frames < 1:
            raise NotApplicable("silence is shorter than one sample")
        # Unsigned 8-bit PCM silence is 128, signed multibyte PCM silence is 0.
        silence = (b"\x80" if params.sampwidth == 1 else b"\0" * params.sampwidth) * params.nchannels * frames
        output = data[:a * width] + silence + data[a * width:]
        affected, inserted, category = [effective[0], effective[0]], [effective[0], (a + frames) / params.framerate], "speech_silence"
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as wav:
        wav.setparams(params)
        wav.writeframes(output)
    return {
        "source_item": source_item, "operator": operator, "category": category,
        "annotation_origin": "programmatic_signal_edit", "human_review": "pending",
        "original_interval_seconds": affected, "output_interval_seconds": inserted,
        "original_audio_sha256": hashlib.sha256(data).hexdigest(),
        "linguistic_boundaries": "unverified",
    }
