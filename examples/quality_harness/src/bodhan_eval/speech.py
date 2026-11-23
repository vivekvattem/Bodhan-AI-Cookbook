"""Provisional speech checks for PCM WAV and fallible transcript evidence.

Thresholds are engineering defaults and have not been calibrated on speech data.
Transcript offsets refer to the transcript evidence, not timestamps in audio.
"""

import difflib
import math
import unicodedata
import wave
from pathlib import Path

SILENCE_LEVEL = 0.01
CLIP_LEVEL = 0.99
WINDOW_SECONDS = 0.05
LONG_SILENCE_SECONDS = 1.0
MIN_AUDIO_SECONDS = 0.1


def signal_check(path):
    if not path:
        return {"status": "skipped", "reason": "no audio supplied"}
    try:
        with wave.open(str(Path(path)), "rb") as audio:
            channels, width, rate, frames = audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()
            if audio.getcomptype() != "NONE" or width not in (1, 2, 3, 4) or channels < 1 or rate < 1:
                return {"status": "unavailable", "reason": "unsupported WAV encoding"}
            raw = audio.readframes(frames)
            if len(raw) != frames * channels * width or frames == 0:
                return {"status": "unavailable", "reason": "empty or truncated WAV"}
    except (OSError, EOFError, wave.Error, ValueError):
        return {"status": "unavailable", "reason": "unreadable or unsupported WAV"}

    duration = frames / rate
    peak_value = float(1 << (width * 8 - 1))
    window_frames = max(1, round(rate * WINDOW_SECONDS))
    silence, clipping, clipped_samples, silent_frames = [], [], 0, 0
    for frame_start in range(0, frames, window_frames):
        frame_end = min(frames, frame_start + window_frames)
        values = []
        for index in range(frame_start * channels, frame_end * channels):
            start = index * width
            sample = int.from_bytes(raw[start : start + width], "little", signed=width != 1)
            values.append((sample - 128 if width == 1 else sample) / peak_value)
        rms = math.sqrt(sum(value * value for value in values) / len(values))
        clips = sum(abs(value) >= CLIP_LEVEL for value in values)
        clipped_samples += clips
        if rms < SILENCE_LEVEL:
            silent_frames += frame_end - frame_start
            silence.append((frame_start, frame_end))
        if clips:
            clipping.append((frame_start, frame_end))

    def merge(windows):
        ranges = []
        for start, end in windows:
            if ranges and ranges[-1][1] == start:
                ranges[-1][1] = end
            else:
                ranges.append([start, end])
        return [[round(start / rate, 6), round(end / rate, 6)] for start, end in ranges]

    silence_intervals, clipping_intervals = merge(silence), merge(clipping)
    long_silence = [interval for interval in silence_intervals if interval[1] - interval[0] >= LONG_SILENCE_SECONDS]
    measurements = {
        "duration_seconds": round(duration, 6),
        "sample_rate_hz": rate,
        "channels": channels,
        "bits_per_sample": width * 8,
        "silence_fraction": round(silent_frames / frames, 6),
        "clipped_sample_fraction": round(clipped_samples / (frames * channels), 6),
        "silence_intervals_seconds": silence_intervals,
        "clipping_intervals_seconds": clipping_intervals,
        "long_silence_intervals_seconds": long_silence,
        "thresholds": {
            "window_seconds": WINDOW_SECONDS,
            "silence_rms": SILENCE_LEVEL,
            "clipping_amplitude": CLIP_LEVEL,
            "long_silence_seconds": LONG_SILENCE_SECONDS,
        },
    }
    if duration < MIN_AUDIO_SECONDS:
        return {"status": "inconclusive", "reason": "audio too short for a reliable signal assessment", "measurements": measurements}
    return {
        "status": "evaluated",
        "verdict": "finding" if long_silence or clipping_intervals else "match",
        "measurements": measurements,
    }


def text_units(text):
    """Normalize character clusters, retaining each token's original offsets."""
    clusters = []
    for index, character in enumerate(text):
        if clusters and (unicodedata.combining(character) or unicodedata.category(character).startswith("M")):
            clusters[-1][1] = index + 1
        else:
            clusters.append([index, index + 1])
    chars = []
    for start, end in clusters:
        for character in unicodedata.normalize("NFKC", text[start:end]).casefold():
            if character.isdecimal():
                character = str(unicodedata.decimal(character))
            chars.append((character, start, end))
    tokens, current = [], []
    for character, start, end in chars:
        if character.isalnum() or unicodedata.category(character).startswith("M"):
            current.append((character, start, end))
        elif current:
            tokens.append(("".join(c for c, _, _ in current), current[0][1], current[-1][2]))
            current = []
    if current:
        tokens.append(("".join(c for c, _, _ in current), current[0][1], current[-1][2]))
    return tokens


def repeated_prefix(expected_tokens, observed_tokens, start, end):
    """Find a surplus adjacent copy at the start of a changed alignment block."""
    for size in range(end - start, 0, -1):
        phrase = observed_tokens[start : start + size]
        previous = observed_tokens[max(0, start - size) : start]
        following = observed_tokens[start + size : start + size * 2]
        if phrase != previous and phrase != following:
            continue
        before = sum(expected_tokens[index : index + size] == phrase for index in range(len(expected_tokens) - size + 1))
        after = sum(observed_tokens[index : index + size] == phrase for index in range(len(observed_tokens) - size + 1))
        if after > before:
            return size
    return 0


def transcript_check(expected, transcript, signal):
    if not expected:
        return {"status": "unavailable", "reason": "no text intended for speech"}
    if transcript is None:
        return {"status": "unavailable", "reason": "transcript unavailable"}
    left, right = text_units(expected), text_units(transcript)
    if not left or not right:
        return {"status": "inconclusive", "reason": "empty normalized speech text or transcript"}
    a, b = [entry[0] for entry in left], [entry[0] for entry in right]
    candidates = []
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for operation, i, j, k, end_index in matcher.get_opcodes():
        if operation == "equal":
            continue
        if operation in {"replace", "insert"}:
            while k < end_index:
                repeated_size = repeated_prefix(a, b, k, end_index)
                if not repeated_size:
                    break
                start, end = right[k][1], right[k + repeated_size - 1][2]
                candidates.append(
                    {
                        "type": "repetition",
                        "evidence": {"field": "transcript", "span": [start, end], "text": transcript[start:end]},
                        "reason": "surplus adjacent transcript copy; recognizer repetition remains possible",
                    }
                )
                k += repeated_size
            if k == end_index and operation == "insert":
                continue
        if operation in {"delete", "replace"}:
            start, end = left[i][1], left[j - 1][2]
            if k < end_index:
                kind = "numeric_rendering" if any(character.isdecimal() for character in expected[start:end]) else "substitution"
            else:
                kind = "truncation" if j == len(left) and end_index == len(right) else "omission"
            evidence = {"field": "translation", "span": [start, end], "text": expected[start:end]}
        else:
            start, end = right[k][1], right[end_index - 1][2]
            kind = "insertion"
            evidence = {"field": "transcript", "span": [start, end], "text": transcript[start:end]}
        candidate = {"type": kind, "evidence": evidence, "reason": "transcription mismatch may originate in the speech recognizer"}
        if operation in {"delete", "replace"}:
            transcript_start = right[k][1] if k < len(right) else len(transcript)
            transcript_end = right[end_index - 1][2] if k < end_index else transcript_start
            candidate["transcript_evidence"] = {
                "field": "transcript",
                "span": [transcript_start, transcript_end],
                "text": transcript[transcript_start:transcript_end],
            }
        measurements = signal.get("measurements", {})
        if kind == "numeric_rendering":
            candidate["reason"] = "numeric or unit rendering differs; spoken-word equivalence is unverified"
        if kind == "truncation":
            candidate["audio_context"] = {
                "duration_seconds": measurements.get("duration_seconds"),
                "silence_intervals_seconds": measurements.get("silence_intervals_seconds", []),
                "word_timing": "unavailable",
            }
        if kind == "truncation" and measurements.get("long_silence_intervals_seconds"):
            trailing = measurements["long_silence_intervals_seconds"][-1]
            if trailing[1] >= measurements["duration_seconds"] - WINDOW_SECONDS:
                candidate["supporting_audio_interval_seconds"] = trailing
        candidates.append(candidate)
    if not candidates:
        return {"status": "evaluated", "verdict": "match", "expected_tokens": len(left), "transcript_tokens": len(right)}
    return {
        "status": "inconclusive",
        "reason": "transcript mismatch needs independent corroboration",
        "candidates": candidates,
        "expected_tokens": len(left),
        "transcript_tokens": len(right),
    }


def signal_findings(check):
    if check.get("status") != "evaluated":
        return []
    measurements = check["measurements"]
    result = []
    for interval in measurements["long_silence_intervals_seconds"]:
        result.append(
            {
                "category": "speech_silence",
                "severity": "major",
                "check": "audio_signal",
                "field": "audio",
                "interval_seconds": interval,
                "evidence": {"reason": "sustained low-energy audio", "measurements": measurements},
            }
        )
    for interval in measurements["clipping_intervals_seconds"]:
        result.append(
            {
                "category": "speech_clipping",
                "severity": "minor",
                "check": "audio_signal",
                "field": "audio",
                "interval_seconds": interval,
                "evidence": {"reason": "samples near full scale", "measurements": measurements},
            }
        )
    return result
