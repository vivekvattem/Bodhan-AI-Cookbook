"""Semantic rubric, response contract, and original-field evidence resolution."""

from pathlib import Path

VERSION = "semantic-translation-v5-validated-quantity-ids"
CATEGORIES = ("omission", "addition", "mistranslation", "term_entity", "number")
RUBRIC = Path(__file__).with_name("semantic_rubric.md").read_text(encoding="utf-8")


def english(value, *, empty=False):
    return isinstance(value, str) and value.isascii() and len(value) <= 1200 and (empty or bool(value.strip()))


def source_quote(value):
    """Accept Unicode English-source evidence without asking for Indic scripts."""
    if not isinstance(value, str) or not value.strip() or len(value) > 1200:
        return False
    return all(not (0x0900 <= ord(char) <= 0x0DFF or 0x11000 <= ord(char) <= 0x11FFF) for char in value)


def location_shape(location):
    shapes = ({"field", "span"}, {"field", "anchor"}, {"field", "quote"},
              {"field", "quote", "context"}, {"field", "quantity_id"})
    if not isinstance(location, dict) or set(location) not in shapes:
        return False
    if not isinstance(location["field"], str) or location["field"] not in {"source", "translation"}:
        return False
    if "anchor" in location or "quote" in location:
        return (location["field"] == "source" and source_quote(location.get("anchor", location.get("quote")))
                and ("context" not in location or source_quote(location["context"])))
    if "quantity_id" in location:
        value = location["quantity_id"]
        return isinstance(value, str) and value.startswith("S" if location["field"] == "source" else "T") and value[1:].isascii() and value[1:].isdigit()
    span = location["span"]
    return isinstance(span, list) and len(span) == 2 and all(type(i) is int for i in span)


def parse_judgment(value):
    if not isinstance(value, dict) or set(value) != {"verdict", "explanation", "findings"}:
        raise ValueError("invalid semantic response shape")
    if not isinstance(value["verdict"], str) or value["verdict"] not in {"clean", "findings", "uncertain"} or not english(value["explanation"]):
        raise ValueError("invalid semantic verdict")
    findings = value["findings"]
    if not isinstance(findings, list) or len(findings) > 50:
        raise ValueError("invalid semantic findings list")
    if (value["verdict"] == "clean" and findings) or (value["verdict"] == "findings" and not findings):
        raise ValueError("semantic verdict contradicts findings")
    accepted, rejected = [], []
    required = {"category", "severity", "field", "explanation", "evidence", "confidence", "uncertainty"}
    for index, candidate in enumerate(findings):
        shapes = (required | {"span"}, required | {"anchor"}, required | {"quote"},
                  required | {"quote", "context"}, required | {"quantity_id"})
        if not isinstance(candidate, dict) or set(candidate) not in shapes:
            rejected.append({"index": index, "reason": "invalid finding shape"})
            continue
        if (
            candidate["category"] not in CATEGORIES
            or not isinstance(candidate["severity"], str)
            or candidate["severity"] not in {"minor", "major", "critical"}
            or not isinstance(candidate["confidence"], str)
            or candidate["confidence"] not in {"high", "medium", "low"}
            or not english(candidate["explanation"])
            or not english(candidate["uncertainty"], empty=True)
            or not isinstance(candidate["evidence"], list)
            or not candidate["evidence"]
        ):
            rejected.append({"index": index, "reason": "invalid finding fields"})
            continue
        key = next(key for key in ("span", "anchor", "quote", "quantity_id") if key in candidate)
        location = {"field": candidate["field"], key: candidate[key]}
        if "context" in candidate:
            location["context"] = candidate["context"]
        if not location_shape(location) or not all(location_shape(location) for location in candidate["evidence"]):
            rejected.append({"index": index, "reason": "invalid original location shape"})
            continue
        # Locations are resolved later against immutable original text.
        accepted.append({**candidate, "index": index})
    return {"verdict": value["verdict"], "explanation": value["explanation"], "candidates": accepted, "rejected": rejected}


def resolve_location(location, item):
    if not location_shape(location):
        raise ValueError("invalid location shape")
    field = location["field"]
    if field not in {"source", "translation"} or not isinstance(item.get(field), str):
        raise ValueError("unknown original field")
    text = item[field]
    if "quantity_id" in location:
        from .numbers import quantity_candidates

        candidates = quantity_candidates(text, field)
        selected = next((entry for entry in candidates if entry["id"] == location["quantity_id"]), None)
        if selected is None:
            raise ValueError("unknown or unsupported quantity identifier")
        start, end = selected["span"]
    elif "span" in location:
        span = location["span"]
        if not isinstance(span, list) or len(span) != 2 or not all(type(i) is int for i in span):
            raise ValueError("invalid code-point span")
        start, end = span
        if not 0 <= start < end <= len(text):
            raise ValueError("span outside original text")
    else:
        anchor = location.get("anchor", location.get("quote"))
        if field != "source" or not source_quote(anchor):
            raise ValueError("source anchor is absent or ambiguous")
        context = location.get("context")
        if context is not None:
            offset = text.find(context)
            if offset < 0 or text.find(context, offset + 1) >= 0:
                raise ValueError("source context is absent or ambiguous")
            relative = context.find(anchor)
            if relative < 0 or context.find(anchor, relative + 1) >= 0:
                raise ValueError("source quote is absent or ambiguous in context")
            start = offset + relative
        else:
            start = text.find(anchor)
        if start < 0 or (context is None and text.find(anchor, start + 1) >= 0):
            raise ValueError("source anchor is absent or ambiguous")
        end = start + len(anchor)
    return {"field": field, "span": [start, end], "text": text[start:end]}


def finalize_judgment(judgment, item):
    rejected = list(judgment["rejected"])
    flags = []
    for candidate in judgment["candidates"]:
        try:
            key = next(key for key in ("span", "anchor", "quote", "quantity_id") if key in candidate)
            location = {"field": candidate["field"], key: candidate[key]}
            if "context" in candidate:
                location["context"] = candidate["context"]
            affected = resolve_location(location, item)
            expected_field = {"omission": "source", "addition": "translation"}.get(candidate["category"])
            if expected_field and affected["field"] != expected_field:
                raise ValueError("category anchored to the wrong field")
            evidence = [resolve_location(location, item) for location in candidate["evidence"]]
            if affected not in evidence:
                raise ValueError("evidence does not include affected location")
            if candidate["category"] == "number":
                from .numbers import TOKEN, decimal_value, normalized_view, number_tokens, quantity_candidates

                if "quantity_id" not in candidate:
                    raise ValueError("number finding requires original quantity identifiers")
                quantity_evidence = [loc for loc in candidate["evidence"] if "quantity_id" in loc]
                if any(sum(loc["field"] == field for loc in quantity_evidence) != 1 for field in ("source", "translation")):
                    raise ValueError("number finding requires one quantity identifier in each field")

                numeric = {}
                for location in evidence:
                    normalized, _ = normalized_view(location["text"])
                    if TOKEN.fullmatch(normalized):
                        numeric.setdefault(location["field"], []).append(decimal_value(normalized))
                if affected["field"] != "translation" or any(len(numeric.get(field, [])) != 1 for field in ("source", "translation")):
                    raise ValueError("number finding requires one exact numeric span in each original field")
                if not TOKEN.fullmatch(normalized_view(affected["text"])[0]):
                    raise ValueError("affected span is not the numeric translation evidence")
                if numeric["source"][0] == numeric["translation"][0]:
                    raise ValueError("numeric values are equivalent")
                if any("quantity_id" in loc for loc in candidate["evidence"]):
                    source_number = next(loc for loc in evidence if loc["field"] == "source"
                                         and TOKEN.fullmatch(normalized_view(loc["text"])[0]))
                    if not any(loc["field"] == "source" and loc["span"] != source_number["span"]
                               and loc["span"][0] <= source_number["span"][0]
                               and loc["span"][1] >= source_number["span"][1] for loc in evidence):
                        raise ValueError("number pairing lacks original source context")
                identified = {}
                for field in ("source", "translation"):
                    tokens = number_tokens(item[field], field) or []
                    evidence_span = next(loc["span"] for loc in evidence if loc["field"] == field
                                         and TOKEN.fullmatch(normalized_view(loc["text"])[0]))
                    identified[field] = next((token for token in tokens if token["span"] == evidence_span), {})
                source_unit, target_unit = identified["source"], identified["translation"]
                if not source_unit or not target_unit:
                    raise ValueError("quantity representation cannot be verified")
                if any(token.get("suffix_interpretation") for token in (source_unit, target_unit)):
                    raise ValueError("quantity suffix is unsupported")
                if source_unit.get("dimension") != target_unit.get("dimension"):
                    raise ValueError("quantity units cannot be compared")
                if (source_unit.get("dimension") and source_unit.get("dimension") == target_unit.get("dimension")
                        and decimal_value(source_unit["base_value"]) == decimal_value(target_unit["base_value"])):
                    raise ValueError("quantities are equivalent after supported unit conversion")
            if candidate["confidence"] == "low" or candidate["uncertainty"].strip():
                raise ValueError("tentative finding needs review")
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append(
                {
                    "index": candidate["index"],
                    "reason": str(exc),
                    "category": candidate["category"],
                    "confidence": candidate["confidence"],
                    "explanation": candidate["explanation"],
                    "uncertainty": candidate["uncertainty"],
                }
            )
            continue
        details = {"locations": evidence}
        if candidate["category"] == "number":
            contexts = []
            for loc in quantity_evidence:
                entry = next(entry for entry in quantity_candidates(item[loc["field"]], loc["field"])
                             if entry["id"] == loc["quantity_id"])
                a, b = entry["context_span"]
                contexts.append({"field": loc["field"], "span": [a, b], "text": item[loc["field"]][a:b],
                                 "quantity_id": loc["quantity_id"]})
            details["quantity_contexts"] = contexts
        flags.append(
            {
                "category": candidate["category"],
                "severity": candidate["severity"],
                "check": "semantic_translation",
                "field": affected["field"],
                "span": affected["span"],
                "explanation": candidate["explanation"],
                "confidence": candidate["confidence"],
                "uncertainty": candidate["uncertainty"],
                "evidence": details,
            }
        )
    incomplete = bool(rejected) or judgment["verdict"] == "uncertain"
    result = {
        "status": "inconclusive" if incomplete else "evaluated",
        "verdict": "uncertain" if incomplete else judgment["verdict"],
        "explanation": judgment["explanation"],
        "rejected": rejected,
        "accepted_findings": len(flags),
        "rubric_version": VERSION,
    }
    return result, flags
