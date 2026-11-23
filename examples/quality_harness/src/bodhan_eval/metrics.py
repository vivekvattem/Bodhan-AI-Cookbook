"""Post-detection diagnostics. These functions never call a model or tune rules.

Precision/recall use item-category pairs. Recall includes abstentions and failed
checks in its denominator. Error localization uses one-to-one finding matches.
Programmatic, unreviewed labels do not establish measured translation quality.
"""

import hashlib
from collections import Counter, defaultdict

CHECKS = {
    "number": ("number",), "format": ("markdown_table", "latex"),
    "omission": ("semantic_translation",), "addition": ("semantic_translation",),
    "mistranslation": ("semantic_translation",), "term_entity": ("semantic_translation",),
    "speech_silence": ("audio_signal",), "speech_clipping": ("audio_signal",),
    "speech_omission": ("speech_alignment",), "speech_repetition": ("speech_alignment",),
}


def ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def source_group(item):
    return hashlib.sha256(item["source"].encode()).hexdigest() if isinstance(item.get("source"), str) else None


def groups(items):
    return {group for item in items if (group := source_group(item)) is not None}


def checked_span(span, text, quote=None):
    if not isinstance(span, list) or len(span) != 2 or not all(type(i) is int for i in span):
        raise ValueError("span must use integer code-point offsets")
    a, b = span
    if not isinstance(text, str) or not 0 <= a < b <= len(text) or (quote is not None and text[a:b] != quote):
        raise ValueError("span or quote does not resolve against original text")
    return a, b


def label_location(label, annotation, item):
    field = label.get("field")
    if field in {"source", "translation"}:
        a, b = checked_span(label.get("span"), item.get(field), label.get("text"))
        return field, a, b
    if field != "clean_translation":
        raise ValueError("unsupported label location")
    edit = annotation.get("edit") or {}
    before = edit.get("original_text")
    replacement = edit.get("replacement")
    start, stop = edit.get("output_span", [None, None])
    old_start, old_stop = edit.get("original_span", [None, None])
    text = item.get("translation")
    if (not isinstance(text, str) or not isinstance(before, str) or not isinstance(replacement, str)
            or any(type(n) is not int for n in (start, stop, old_start, old_stop))
            or not 0 <= start <= stop <= len(text) or start != old_start
            or stop - start != len(replacement) or old_stop - old_start != len(before)
            or text[start:stop] != replacement):
        raise ValueError("invalid original-to-output edit mapping")
    clean = text[:start] + before + text[stop:]
    digest = annotation.get("base_translation_sha256")
    if digest and hashlib.sha256(clean.encode()).hexdigest() != digest:
        raise ValueError("base translation fingerprint does not match")
    a, b = checked_span(label.get("span"), clean, label.get("text"))

    def mapped(position, right):
        if position <= old_start:
            return position
        if position >= old_stop:
            return position + len(replacement) - len(before)
        return stop if right else start

    return "translation", mapped(a, False), mapped(b, True)


def flag_locations(flag, item):
    """Score the declared affected span, never a broad supporting context quote."""
    evidence = flag.get("evidence") or {}
    field = flag.get("field")
    if field not in {"source", "translation"}:
        return set()
    try:
        a, b = checked_span(flag.get("span"), item.get(field), flag.get("text"))
    except ValueError:
        return set()
    candidates = list(evidence.get("locations", []))
    candidates += [evidence[name] for name in ("anchor", "source", "translation") if isinstance(evidence.get(name), dict)]
    for location in candidates:
        if location.get("field") == field and location.get("span") == [a, b]:
            try:
                checked_span([a, b], item[field], location.get("text"))
            except ValueError:
                return set()
    return {(field, a, b)}


def overlaps(first, second):
    field, a, b = first
    other, c, d = second
    if field != other:
        return False
    # Deleted characters map to an output boundary, not an invented character.
    if a == b:
        return c <= a <= d
    return max(a, c) < min(b, d)


def matching_pairs(labels, flags, predicate):
    """Maximum bipartite matching; a finding cannot explain two seeded errors."""
    assigned = {}

    def visit(index, seen):
        for j, flag in enumerate(flags):
            if j in seen or not predicate(labels[index], flag):
                continue
            seen.add(j)
            if j not in assigned or visit(assigned[j], seen):
                assigned[j] = index
                return True
        return False

    for i in range(len(labels)):
        visit(i, set())
    return {i: j for j, i in assigned.items()}


def matching_count(labels, flags, predicate):
    return len(matching_pairs(labels, flags, predicate))


def category_status(item, category):
    statuses = {item.get("checks", {}).get(check, {}).get("status", "absent") for check in CHECKS.get(category, ())}
    if not statuses:
        return "unsupported"
    statuses.discard("skipped")
    if not statuses:
        return "skipped"
    return next(iter(statuses)) if len(statuses) == 1 else "partial"


def annotation_index(annotations):
    result, invalid, duplicates = {}, [], set()
    for record in annotations:
        if not isinstance(record, dict):
            invalid.append({"id": None, "lang": None, "reason": "annotation must be an object"})
            continue
        key = record.get("id"), record.get("lang")
        if not all(isinstance(s, str) and s for s in key) or not isinstance(record.get("labels"), list):
            invalid.append({"id": record.get("id"), "lang": record.get("lang"), "reason": "missing identity or explicit labels array"})
        elif key in result or key in duplicates:
            duplicates.add(key)
            result.pop(key, None)
        else:
            result[key] = record
    invalid.extend({"id": key[0], "lang": key[1], "reason": "duplicate annotation"} for key in sorted(duplicates))
    return result, invalid


def summarize(items, annotations):
    index, invalid = annotation_index(annotations)
    invalid_keys = {(row["id"], row["lang"]) for row in invalid}
    labelled, unlabelled = [], []
    for item in items:
        record = index.get((item.get("id"), item.get("lang")))
        if record is None:
            if (item.get("id"), item.get("lang")) not in invalid_keys:
                unlabelled.append(item)
            continue
        try:
            labels = record["labels"]
            if any(not isinstance(label, dict) or not isinstance(label.get("type"), str) or not label["type"] for label in labels):
                raise ValueError("invalid label category")
            locations = [label_location(label, record, item) for label in labels]
        except (ValueError, TypeError, KeyError):
            invalid.append({"id": item.get("id"), "lang": item.get("lang"), "reason": "label evidence or edit mapping invalid"})
            continue
        labelled.append((item, record, list(zip(labels, locations, strict=True))))
    categories = sorted(set(CHECKS) | {flag["category"] for item in items for flag in item.get("flags", [])}
                        | {label["type"] for _, _, pairs in labelled for label, _ in pairs})
    scores = {}
    for category in categories:
        tp = fp = fn = tn = 0
        positive_sources, flagged_sources = set(), set()
        coverage = Counter()
        for item, _, pairs in labelled:
            positive = any(label["type"] == category for label, _ in pairs)
            predicted = any(flag["category"] == category for flag in item.get("flags", []))
            tp += positive and predicted
            fp += predicted and not positive
            fn += positive and not predicted
            tn += not positive and not predicted
            if positive and source_group(item) is not None:
                positive_sources.add(source_group(item))
            if predicted and source_group(item) is not None:
                flagged_sources.add(source_group(item))
            coverage[category_status(item, category)] += 1
        scores[category] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "labelled_items": len(labelled), "positive_items": tp + fn,
            "independent_positive_sources": len(positive_sources), "independent_flagged_sources": len(flagged_sources),
            "check_coverage": dict(sorted(coverage.items())),
        }
    detected = localized = any_localized = 0
    category_correct_items = detected_items = 0
    detected_groups, localized_groups, aligned_groups = set(), set(), set()
    detected_item_groups, category_correct_groups, false_alarm_groups = set(), set(), set()
    clean, false_alarms, clean_inconclusive = [], 0, 0
    for item, _, pairs in labelled:
        flags = [(flag, flag_locations(flag, item)) for flag in item.get("flags", [])]
        def same_category(label, flag):
            return label[0]["type"] == flag[0]["category"]

        def aligned(label, flag):
            return any(overlaps(label[1], location) for location in flag[1])
        item_detected = matching_count(pairs, flags, same_category)
        item_localized = matching_count(pairs, flags, lambda label, flag: same_category(label, flag) and aligned(label, flag))
        item_aligned = matching_count(pairs, flags, aligned)
        detected += item_detected
        localized += item_localized
        any_localized += item_aligned
        group = source_group(item)
        for count, support in ((item_detected, detected_groups), (item_localized, localized_groups), (item_aligned, aligned_groups)):
            if count and group is not None:
                support.add(group)
        if pairs and flags:
            detected_items += 1
            correct = {label["type"] for label, _ in pairs} == {flag["category"] for flag, _ in flags}
            category_correct_items += correct
            if group is not None:
                detected_item_groups.add(group)
                if correct:
                    category_correct_groups.add(group)
        if not pairs:
            clean.append(item)
            false_alarms += bool(flags)
            if flags and group is not None:
                false_alarm_groups.add(group)
            clean_inconclusive += any(check.get("status") in {"inconclusive", "unavailable", "pending"} for check in item.get("checks", {}).values())
    checks = defaultdict(Counter)
    for item in items:
        for name, check in item.get("checks", {}).items():
            checks[name][check.get("status", "absent")] += 1
    origins = Counter(record.get("annotation_origin", "unspecified") for _, record, _ in labelled)
    all_human = bool(labelled) and all(record.get("annotation_origin") == "human" and record.get("human_review") == "completed"
                                     for _, record, _ in labelled)
    return {
        "status": "human_label_diagnostic" if all_human else "unreviewed_label_diagnostic",
        "human_calibration": "not_performed", "threshold_tuning": "not_performed",
        "provenance": dict(origins), "items": len(items), "labelled_items": len(labelled),
        "unlabelled_items": len(unlabelled), "unlabelled_flagged_items": sum(bool(item.get("flags")) for item in unlabelled),
        "invalid_annotations": invalid, "invalid_labelled_items": len(items) - len(labelled) - len(unlabelled),
        "independent_english_sources": len(groups(items)),
        "independent_labelled_sources": len(groups([item for item, _, _ in labelled])),
        "categories": scores, "item_statuses": dict(Counter(item.get("status", "absent") for item in items)),
        "checks": {name: dict(sorted(counts.items())) for name, counts in sorted(checks.items())},
        "clean_false_alarm_rate": ratio(false_alarms, len(clean)),
        "independent_clean_sources": len(groups(clean)),
        "clean_items_with_incomplete_checks": clean_inconclusive,
        "localization_overlap": ratio(localized, detected),
        "category_matched_finding_precision": ratio(detected, sum(len(item.get("flags", [])) for item, _, _ in labelled)),
        "error_detection_recall": ratio(detected, sum(len(pairs) for _, _, pairs in labelled)),
        "category_accuracy_among_localized_errors": ratio(localized, any_localized),
        "category_accuracy_detected_items": ratio(category_correct_items, detected_items),
        "metric_source_support": {
            "clean_false_alarm_rate": {"denominator_sources": len(groups(clean)), "numerator_sources": len(false_alarm_groups)},
            "localization_overlap": {"denominator_sources": len(detected_groups), "numerator_sources": len(localized_groups)},
            "category_accuracy_among_localized_errors": {"denominator_sources": len(aligned_groups), "numerator_sources": len(localized_groups)},
            "category_accuracy_detected_items": {"denominator_sources": len(detected_item_groups), "numerator_sources": len(category_correct_groups)},
        },
        "definitions": {
            "precision_recall": "item-category pairs; missing or inconclusive checks remain in the end-to-end recall denominator",
            "localization": "one-to-one correct-category findings with same-field span overlap / correct-category detected errors",
            "affected_location": "only the primary declared affected field/span is scored; supporting context cannot rescue a misplaced finding",
            "category_accuracy": "correct-category localized errors / localized errors regardless of category; one-to-one matching",
            "detected_item_category_accuracy": "exact predicted category set / labelled error category set, among error items with any flag",
            "deletions": "clean-translation spans project through the edit to output boundaries; touching that boundary counts as overlap",
            "controls": "clean means explicitly empty labels, not absent labels; incomplete checks do not certify correctness",
            "independence": "distinct exact English source content, not translated or injected variants",
        },
    }


def evaluate(items, annotations):
    result = summarize(items, annotations)
    annotations = [record for record in annotations if isinstance(record, dict)]
    result["languages"] = {lang: summarize([item for item in items if item.get("lang") == lang],
                                           [row for row in annotations if row.get("lang") == lang])
                           for lang in sorted({item["lang"] for item in items if item.get("lang")})}
    index, _ = annotation_index(annotations)
    result["splits"] = {}
    source_splits = defaultdict(set)
    for item in items:
        record = index.get((item.get("id"), item.get("lang")))
        if record and record.get("split") and source_group(item) is not None:
            source_splits[source_group(item)].add(record["split"])
    result["cross_split_source_overlap"] = sum(len(splits) > 1 for splits in source_splits.values())
    result["split_integrity"] = "failed" if result["cross_split_source_overlap"] else "passed"
    for split in sorted({record.get("split") for record in annotations if record.get("split")}):
        subset = [item for item in items if index.get((item.get("id"), item.get("lang")), {}).get("split") == split]
        subset_labels = [row for row in annotations if row.get("split") == split]
        score = summarize(subset, subset_labels)
        score["languages"] = {lang: summarize([item for item in subset if item.get("lang") == lang],
                                              [row for row in subset_labels if row.get("lang") == lang])
                              for lang in sorted({item["lang"] for item in subset if item.get("lang")})}
        result["splits"][split] = score
    return result
