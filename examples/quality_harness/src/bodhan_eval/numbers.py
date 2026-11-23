"""Conservative quantity inventories with original code-point evidence."""
import re
import unicodedata
from collections import Counter
from decimal import Decimal

VERSION = "quantity-inventory-v5-residual-promotion"
TOKEN = re.compile(r"[+−-]?\d+(?:[.٫]\d+)?")
UNSUPPORTED = re.compile(r"\d(?:[,٬:/⁄‐‑‒–—−-]|[^\S\n])+[+−-]?\d|\d\s*\.\s*\.")
# Multiplicative conversions only. Temperature scales remain distinct.
UNIT_SPECS = {
    "kg": ("mass", "1000"), "g": ("mass", "1"), "mg": ("mass", "0.001"), "μg": ("mass", "0.000001"),
    "km": ("length", "1000"), "m": ("length", "1"), "cm": ("length", "0.01"), "mm": ("length", "0.001"),
    "L": ("volume", "1"), "mL": ("volume", "0.001"),
    "s": ("time", "1"), "min": ("time", "60"), "h": ("time", "3600"),
    "%": ("percentage", "1"), "°C": ("temperature_c", "1"), "°F": ("temperature_f", "1"),
}
UNIT_ALIASES = {name: name for name in UNIT_SPECS}
UNIT_ALIASES.update({"ml": "mL", "l": "L"})
for word, unit in (("kilogram", "kg"), ("gram", "g"), ("milligram", "mg"), ("microgram", "μg"),
                   ("kilometre", "km"), ("kilometer", "km"), ("metre", "m"), ("meter", "m"),
                   ("centimetre", "cm"), ("centimeter", "cm"), ("millimetre", "mm"), ("millimeter", "mm"),
                   ("litre", "L"), ("liter", "L"), ("millilitre", "mL"), ("milliliter", "mL"),
                   ("second", "s"), ("minute", "min"), ("hour", "h")):
    UNIT_ALIASES[word] = unit
    UNIT_ALIASES[word + "s"] = unit
UNIT = re.compile(r" *([A-Za-zμ°%]+)(?!\w)")


def normalized_view(text):
    clusters = []
    for i, character in enumerate(text):
        if clusters and unicodedata.category(character).startswith("M") and not text[clusters[-1][0]].isdecimal():
            clusters[-1][1] = i + 1
        else:
            clusters.append([i, i + 1])
    characters, offsets = [], []
    for a, b in clusters:
        for character in unicodedata.normalize("NFKC", text[a:b]):
            characters.append(str(unicodedata.decimal(character)) if character.isdecimal() else character)
            offsets.append((a, b))
    return "".join(characters), offsets


def decimal_value(raw):
    folded = "".join(str(unicodedata.decimal(c, c)) for c in raw)
    return Decimal(folded.replace("−", "-").replace("٫", "."))


def quantity_candidates(text, field):
    """Enumerate decimal spans for judge selection, without pairing quantities."""
    if not isinstance(text, str) or field not in {"source", "translation"}:
        return []
    result = []
    normalized, mapping = normalized_view(text)
    recognized = {tuple(token["span"]): token for token in (number_tokens(text, field) or [])}
    for match in TOKEN.finditer(normalized):
        a, b = match.span()
        start, end = mapping[a][0], mapping[b - 1][1]
        original = text[start:end]
        if any(c.isnumeric() and not c.isdecimal() for c in original):
            continue
        before = normalized[a - 1] if a else ""
        after = normalized[b] if b < len(normalized) else ""
        if before and (before.isalnum() or before in "_.,٫/⁄^"):
            continue
        if after and (after.isnumeric() and not after.isdecimal() or after in "_/⁄^"):
            continue
        if after and after.isascii() and after.isalpha():
            unit = UNIT.match(normalized[b:])
            if unit is None or unit[1] not in UNIT_ALIASES:
                continue
        # Do not represent a component of fractions, ranges, dates or grouped numbers as a quantity.
        if after in ",٬:/⁄‐‑‒–—−-" and after:
            continue
        if before and before in ",٬:/⁄‐‑‒–—−-":
            continue
        context_start = max(0, start - 48)
        context_end = min(len(text), end + 48)
        candidate = {"id": ("S" if field == "source" else "T") + str(len(result)),
                     "field": field, "span": [start, end], "value": str(decimal_value(match[0])),
                     "context_span": [context_start, context_end]}
        token = recognized.get((start, end), {})
        for key in ("unit", "unit_span", "dimension", "base_value", "suffix_span", "suffix_interpretation"):
            if token.get(key) is not None:
                candidate[key] = token[key]
        result.append(candidate)
        if len(result) > 100:
            return []
    return result


def analyze_numbers(text, field):
    def rejected(reason, a, b):
        return {"tokens": None, "issue": {"reason": reason, "field": field, "span": [a, b], "text": text[a:b]}}

    # NFKC must not turn fractions or superscripts into ordinary quantities.
    for i, character in enumerate(text):
        if character.isnumeric() and not character.isdecimal():
            return rejected("non-decimal numeric character", i, i + 1)
    normalized, mapping = normalized_view(text)

    def original(a, b):
        return mapping[a][0], mapping[b - 1][1]

    unsupported = UNSUPPORTED.search(normalized)
    if unsupported:
        return rejected("unsupported composite numeric notation", *original(*unsupported.span()))
    tokens = []
    for match in TOKEN.finditer(normalized):
        a, b = match.span()
        start, end = original(a, b)
        remainder = normalized[b:]
        unit_match = UNIT.match(remainder)
        raw_unit = unit_match[1] if unit_match else None
        unit = UNIT_ALIASES.get(raw_unit)
        if unit is None and raw_unit and len(raw_unit) > 2:
            # Full English unit names may start a sentence; SI symbols remain case sensitive.
            unit = UNIT_ALIASES.get(raw_unit.lower()) if raw_unit.lower() not in UNIT_SPECS else None
        before = normalized[a - 1] if a else ""
        after = remainder[:1]
        if before and (before.isalnum() or before in "^*.,٫_+−-$₹€£¥"):
            return rejected("ambiguous numeric prefix", max(0, start - 1), end)
        opaque_suffix = bool(after and not after.isascii() and unicodedata.category(after)[:1] in {"L", "M"} and unit is None)
        if after and (after in "_+−-" or after.isalpha()) and unit is None and not opaque_suffix:
            return rejected("unsupported adjacent suffix or sign", start, min(len(text), end + 1))
        if unit is not None and remainder[unit_match.end():].startswith(("/", "^")):
            return rejected("compound or powered unit", start, original(b + unit_match.end() - 1, b + unit_match.end())[1] + 1)
        if unit is not None and remainder[unit_match.end():] and unicodedata.category(remainder[unit_match.end()]).startswith("M"):
            return rejected("unsupported decorated unit", start, original(b + unit_match.end() - 1, b + unit_match.end())[1])
        value = decimal_value(match[0])
        token = {"field": field, "span": [start, end], "text": text[start:end], "value": str(value), "unit": unit}
        if opaque_suffix:
            suffix_end = b
            while suffix_end < len(normalized) and unicodedata.category(normalized[suffix_end])[:1] in {"L", "M"}:
                suffix_end += 1
            suffix_start, suffix_end_original = original(b, suffix_end)
            token.update(suffix_span=[suffix_start, suffix_end_original],
                         suffix_text=text[suffix_start:suffix_end_original], suffix_interpretation="unknown",
                         unknown_suffix=text[suffix_start:suffix_end_original])
        if unit is not None:
            unit_start, unit_end = original(b + unit_match.start(1), b + unit_match.end(1))
            dimension, factor = UNIT_SPECS[unit]
            token.update(unit_span=[unit_start, unit_end], unit_text=text[unit_start:unit_end],
                         dimension=dimension, base_value=str(value * Decimal(factor)))
        tokens.append(token)
    return {"tokens": tokens, "issue": None}


def number_tokens(text, field):
    return analyze_numbers(text, field)["tokens"]


def _finding(before, after):
    return {"category": "number", "severity": "major", "check": "number", "field": after["field"], "span": after["span"],
            "evidence": {"source": before, "translation": after, "pairing": "unverified residual values in a shared dimension"}}


def compare_numbers(source, translation):
    def result(status, **details):
        return {"status": status, "policy_version": VERSION, **details}

    if translation is None:
        return result("skipped", reason="no usable translation")
    exponent = re.compile(r"\b([A-Za-zμ]+)(?:\s*\^\s*|\s*\*\*\s*|(?=\d))(\d+)\b")
    powers = []
    for text in (source, translation):
        powers.append([(m.group(1), m.group(2)) for m in exponent.finditer(text) if m.group(1) in UNIT_ALIASES])
    if len(powers[0]) == len(powers[1]) == 1 and powers[0][0][0] == powers[1][0][0] and powers[0] != powers[1]:
        return result("inconclusive", subtype="unit_exponent", reason="unit exponent differs")
    if any(c in text for text in (source, translation) for c in ("^", "**")):
        return result("inconclusive", reason="powers require review")
    expected, observed = number_tokens(source, "source"), number_tokens(translation, "translation")
    if expected is None or observed is None:
        return result("inconclusive", reason="unsupported numeric representation")
    if not expected and not observed:
        return result("skipped", reason="no decimal numerals")
    def inventory(tokens):
        return Counter((token.get("dimension"), Decimal(token.get("base_value", token["value"]))) for token in tokens)
    if inventory(expected) == inventory(observed):
        if any(token.get("unknown_suffix") for token in expected + observed):
            return result("inconclusive", verdict="inventory_match", reason="unknown suffix remains unverified")
        if max(len(expected), len(observed)) > 1:
            return result("inconclusive", verdict="inventory_match", scope="quantity inventory, not semantic role alignment")
        return result("evaluated", verdict="match", scope="supported quantity inventory")
    source_counts = Counter(Decimal(token["value"]) for token in expected)
    target_counts = Counter(Decimal(token["value"]) for token in observed)
    missing, extra = source_counts - target_counts, target_counts - source_counts
    if not missing and not extra:
        return result("inconclusive", verdict="inventory_match", reason="unit associations remain unverified")
    if sum(missing.values()) == sum(extra.values()) == 1:
        old_value, new_value = next(iter(missing)), next(iter(extra))
        old_tokens = [token for token in expected if Decimal(token["value"]) == old_value]
        new_tokens = [token for token in observed if Decimal(token["value"]) == new_value]
        if len(new_tokens) == 1:
            old, new = old_tokens[0], new_tokens[0]
            def suffix_safe(token):
                suffix = token.get("unknown_suffix")
                if not suffix:
                    return True
                digit = next((c for c in token["text"] if c.isdecimal()), "")
                script = unicodedata.name(digit, "").split(" DIGIT")[0]
                if script in {"ARABIC-INDIC", "EXTENDED ARABIC-INDIC"}:
                    script = "ARABIC"
                characters = [c for c in suffix if unicodedata.category(c)[:1] in {"L", "M"}]
                return bool(script and characters and all(unicodedata.name(c, "").startswith(script + " ") for c in characters))
            words = re.compile(
                r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
                r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
                r"thousand|million|billion|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
                r"page|room|route|section|id)\b", re.I)
            def nearby_words(text, token):
                a, b = token["span"]
                return bool(words.search(text[max(0, a - 24):min(len(text), b + 24)]))
            if not all(suffix_safe(token) for token in old_tokens + new_tokens):
                return result("inconclusive", reason="unsupported suffix")
            if any(nearby_words(source, token) for token in old_tokens) or nearby_words(translation, new):
                return result("inconclusive", reason="nearby number word, ordinal, or identifier")
            if any(token.get("dimension") != new.get("dimension") for token in old_tokens):
                return result("inconclusive", reason="units cannot be compared")
            if old.get("dimension") and any(Decimal(token["base_value"]) == Decimal(new["base_value"]) for token in old_tokens):
                return result("inconclusive", reason="conversion may make values equivalent")
            finding = _finding(old, new)
            finding["evidence"]["source_support"] = old_tokens
            finding["evidence"]["pairing"] = "single residual value pair"
            return result("evaluated", verdict="finding", finding=finding)
    return result("inconclusive", reason="quantity residuals cannot be paired uniquely")


def valid_number_finding(finding, item):
    for evidence in finding["evidence"].get("source_support", []):
        start, end = evidence["span"]
        if evidence["field"] != "source" or not 0 <= start < end <= len(item["source"]) or item["source"][start:end] != evidence["text"]:
            return False
    for field in ("source", "translation"):
        evidence = finding["evidence"][field]
        start, end = evidence["span"]
        text = item[field]
        if evidence["field"] != field or not 0 <= start < end <= len(text):
            return False
        if text[start:end] != evidence["text"]:
            return False
        if "unit_span" in evidence:
            a, b = evidence["unit_span"]
            if not 0 <= a < b <= len(text) or text[a:b] != evidence["unit_text"]:
                return False
        if "suffix_span" in evidence:
            a, b = evidence["suffix_span"]
            if not 0 <= a < b <= len(text) or text[a:b] != evidence["suffix_text"]:
                return False
    return (
        finding["field"] == "translation"
        and finding["span"] == finding["evidence"]["translation"]["span"]
    )
