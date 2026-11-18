import re
import unicodedata
from decimal import Decimal

TOKEN = re.compile(r"[+−-]?\d+(?:[.٫]\d+)?")
UNIT = re.compile(r" *(kg|mg|km|cm|mm|mL|°C|°F|g|L|m|s|h|%)(?!\w)")
UNSUPPORTED = re.compile(r"\d(?:[,٬:/⁄‐‑‒–—−-]|[^\S\n])+[+−-]?\d|\d\s*\.\s*\.")


def number_tokens(text, field):
    if UNSUPPORTED.search(text) or any(
        c.isnumeric() and not c.isdecimal() for c in text
    ):
        return None
    tokens = []
    for match in TOKEN.finditer(text):
        start, end = match.span()
        remainder = text[end:]
        unit = UNIT.match(remainder)
        before = text[start - 1] if start else ""
        after = remainder[:1]
        if before and (before.isalnum() or before in ".,٫_+−-$₹€£¥"):
            return None
        if after and (after in "_+−-" or after.isalpha()) and not unit:
            return None
        if unit and remainder[unit.end() :].startswith(("/", "^")):
            return None
        raw = match[0]
        folded = "".join(str(unicodedata.decimal(c, c)) for c in raw)
        value = Decimal(folded.replace("−", "-").replace("٫", "."))
        tokens.append(
            {
                "field": field,
                "span": [start, end],
                "text": raw,
                "value": str(value),
                "unit": unit[1] if unit else None,
            }
        )
    return tokens


def compare_numbers(source, translation):
    if translation is None:
        return {"status": "skipped", "reason": "no usable translation"}
    expected = number_tokens(source, "source")
    observed = number_tokens(translation, "translation")
    if expected is None or observed is None:
        return {
            "status": "inconclusive",
            "reason": "unsupported numeric representation",
        }
    if not expected and not observed:
        return {"status": "skipped", "reason": "no decimal numerals"}
    left = sorted(Decimal(token["value"]) for token in expected)
    right = sorted(Decimal(token["value"]) for token in observed)
    if left == right:
        if len(expected) == len(observed) == 1:
            before, after = expected[0], observed[0]
            if before["unit"] != after["unit"]:
                return {
                    "status": "inconclusive",
                    "reason": "numerals have different unit context",
                }
            return {
                "status": "evaluated",
                "verdict": "match",
                "paired_unit": before["unit"],
            }
        return {"status": "evaluated", "verdict": "match"}
    if len(expected) == len(observed) == 1:
        before, after = expected[0], observed[0]
        if before["unit"] and before["unit"] == after["unit"]:
            return {
                "status": "evaluated",
                "verdict": "finding",
                "candidate": {
                    "category": "number",
                    "severity": "major",
                    "check": "number",
                    "field": after["field"],
                    "span": after["span"],
                    "evidence": {"source": before, "translation": after},
                },
            }
    return {
        "status": "inconclusive",
        "reason": "numerals could not be paired in the same unit",
    }


def valid_number_finding(finding, item):
    for field in ("source", "translation"):
        evidence = finding["evidence"][field]
        start, end = evidence["span"]
        text = item[field]
        if evidence["field"] != field or not 0 <= start < end <= len(text):
            return False
        if text[start:end] != evidence["text"]:
            return False
    return (
        finding["field"] == "translation"
        and finding["span"] == finding["evidence"]["translation"]["span"]
    )
