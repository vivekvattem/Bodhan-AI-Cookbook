import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from .semantic import CATEGORIES

ITEM_STATUSES = ("evaluated", "failed", "skipped", "inconclusive", "unavailable")


def pretty_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def load_baseline(directory):
    items = {}
    with (directory / "items.jsonl").open(encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            if "source" in item:
                items[item["id"], item["lang"]] = item
    return items


def make_items(rows, results, external):
    ordered_results = iter(results)
    items = []
    for row in rows:
        item = {
            "id": row.item_id,
            "lang": row.lang,
            "line": row.line,
            "status": row.status,
            "checks": {},
            "flags": [],
        }
        if row.status == "ready":
            result = next(ordered_results)
            item.update(result)
            item["status"] = result["checks"]["number"]["status"]
            format_statuses = [result["checks"][name]["status"] for name in ("markdown_table", "latex", "audio_signal", "speech_alignment")]
            if item["flags"] or "evaluated" in format_statuses:
                item["status"] = "evaluated"
            elif item["status"] == "skipped" and "inconclusive" in format_statuses:
                item["status"] = "inconclusive"
            semantic = result["checks"].get("semantic_translation", {})
            if not item["flags"]:
                if semantic.get("status") in {"unavailable", "inconclusive"} and item["status"] != "unavailable":
                    item["status"] = "inconclusive"
                elif semantic.get("status") == "evaluated" and item["status"] == "skipped":
                    item["status"] = "evaluated"
            labels = external.get(row.key, row.labels)
            item["label_status"] = (
                "absent" if labels is None else "clean" if not labels else "labelled"
            )
        else:
            item["reason"] = row.reason
        items.append(item)
    return items


def calibration(items, labels_by_key, invalid_external, mode, records=None):
    counts = Counter(
        true_positive=0, false_positive=0, false_negative=0, true_negative=0
    )
    annotated, support = 0, 0
    for item in items:
        if "source" not in item:
            continue
        labels = labels_by_key.get((item["id"], item["lang"]))
        if labels is None:
            continue
        annotated += 1
        positive = any(label["type"] == "number" for label in labels)
        if item.get("checks", {}).get("number", {}).get("status") != "evaluated" or (
            labels and not positive
        ):
            continue
        support += 1
        predicted = any(flag["category"] == "number" for flag in item["flags"])
        if positive:
            counts["true_positive" if predicted else "false_negative"] += 1
        else:
            counts["false_positive" if predicted else "true_negative"] += 1
    result = {
        "mode": mode,
        "status": "descriptive_only" if support else "unavailable",
        "annotated_items": annotated,
        "eligible_number_items": support,
        "invalid_external_rows": invalid_external,
        "threshold_tuning": "pending",
        "judge_calibration": "pending",
        "categories": {},
    }
    if records:
        from .metrics import summarize

        evaluated = summarize([item for item in items if "source" in item], list(records.values()))
        result["reviewed_metrics"] = evaluated
        result["reviewed_label_provenance"] = {
            "origins": evaluated["provenance"],
            "reviewers": sorted({str(record.get("reviewer")) for record in records.values() if record.get("reviewer")}),
            "scope": "descriptive; thresholds unchanged; sparse samples are not reliable calibration",
        }
        if evaluated["labelled_items"]:
            result["status"] = "descriptive_only"
            for category, metric in evaluated["categories"].items():
                result["categories"][category] = {
                    "true_positive": metric["tp"], "false_positive": metric["fp"],
                    "false_negative": metric["fn"], "true_negative": metric["tn"],
                    "precision": metric["precision"], "recall": metric["recall"],
                    "support": metric["positive_items"], "labelled_items": metric["labelled_items"],
                    "check_coverage": metric["check_coverage"],
                    "scope": "item-category; unavailable checks remain in recall denominator",
                }
            result["judge_calibration"] = "descriptive_only; no threshold tuning"
            return result
        result["status"] = "unavailable"
        result["reason"] = "no valid reviewed labels resolved against evaluated items"
        return result
    if not support:
        result["reason"] = "no eligible labelled number evaluations"
        return result
    tp = counts["true_positive"]
    predicted = tp + counts["false_positive"]
    actual = tp + counts["false_negative"]
    result["categories"]["number"] = {
        **counts,
        "precision": tp / predicted if predicted else None,
        "recall": tp / actual if actual else None,
        "scope": "item-level deterministic number check",
    }
    return result


def counts_for(items):
    counts = Counter(item["status"] for item in items)
    return {status: counts[status] for status in ITEM_STATUSES}


def describe_flag(flag):
    if flag["check"] == "audio_signal":
        return f"{flag['category']} · audio {flag['interval_seconds']} seconds · {flag['evidence']['reason']}"
    if flag["check"] == "semantic_translation":
        return f"{flag['category']} · {flag['field']} {flag['span']} · {flag['explanation']}"
    if flag["category"] == "format":
        return f"{flag['check']} · {flag['evidence']['reason']}"
    expected = flag["evidence"]["source"]
    observed = flag["evidence"]["translation"]
    return (
        f"{expected['text']} {expected['unit']} → {observed['text']} {observed['unit']}"
    )


def describe_item(key):
    return " ".join(f"{key[0]} ({key[1]})".split())


def compare_baseline(baseline, items):
    current = {(item["id"], item["lang"]): item for item in items if "source" in item}
    changes = defaultdict(list)
    for key in sorted(baseline.keys() | current.keys()):
        before, after = baseline.get(key), current.get(key)
        name = describe_item(key)
        if before is None or after is None:
            changes["coverage"].append(
                f"{name} · {'added item' if before is None else 'missing item'}"
            )
            continue
        if before["source"] != after["source"]:
            changes["coverage"].append(f"{name} · source changed")
            continue
        previous = [flag for flag in before["flags"] if flag["check"] == "number"]
        present = [flag for flag in after["flags"] if flag["check"] == "number"]
        old_check = before["checks"].get("number", {})
        new_check = after["checks"].get("number", {})
        if old_check.get("policy_version") != new_check.get("policy_version"):
            changes["coverage"].append(f"{name} · number policy changed; findings are not a like-for-like quality comparison")
        elif previous and not present:
            source_unit = previous[0]["evidence"]["source"]["unit"]
            if (
                new_check.get("verdict") == "match"
                and new_check.get("status") == "evaluated"
                and new_check.get("paired_unit") == source_unit
            ):
                changes["improved"].append(
                    f"{name} · resolved {describe_flag(previous[0])}"
                )
            else:
                reason = new_check.get("reason", "check unavailable")
                changes["unavailable"].append(
                    f"{name} · prior {describe_flag(previous[0])} · {reason}"
                )
        elif present and not previous:
            if old_check.get("status") == "evaluated":
                changes["regressed"].append(f"{name} · new {describe_flag(present[0])}")
            else:
                changes["coverage"].append(
                    f"{name} · new coverage found {describe_flag(present[0])}"
                )
        elif previous and present and previous != present:
            changes["changed"].append(
                f"{name} · {describe_flag(previous[0])} [{previous[0]['severity']}] to "
                f"{describe_flag(present[0])} [{present[0]['severity']}]"
            )
        elif old_check.get("status") != new_check.get("status"):
            changes["coverage"].append(
                f"{name} · number coverage {old_check.get('status', 'absent')} to {new_check['status']}"
            )
        for check_name in ("markdown_table", "latex", "semantic_translation", "audio_signal"):
            previous = [flag for flag in before["flags"] if flag["check"] == check_name]
            present = [flag for flag in after["flags"] if flag["check"] == check_name]
            old = before["checks"].get(check_name, {})
            new = after["checks"].get(check_name, {})
            if previous and not present:
                section = "improved" if new.get("status") == "evaluated" and new.get("verdict") in {"match", "clean"} else "unavailable"
                changes[section].append(f"{name} · {check_name} · prior {describe_flag(previous[0])} · {new.get('reason', new.get('verdict', 'check absent'))}")
            elif present and not previous:
                section = "regressed" if old.get("status") == "evaluated" else "coverage"
                changes[section].append(f"{name} · new {describe_flag(present[0])}")
            elif previous != present:
                changes["changed"].append(f"{name} · {check_name} findings changed")
    sections = ["# Baseline comparison", ""]
    for name in ("improved", "regressed", "changed", "unavailable", "coverage"):
        sections.extend([f"## {name.capitalize()} ({len(changes[name])})", ""])
        sections.extend([*(f"- {entry}" for entry in changes[name]), ""])
    return "\n".join(sections), {
        name: len(entries) for name, entries in changes.items()
    }


def render_summary(items, mode, original_cost, baseline_counts, calibration_result):
    by_language = defaultdict(list)
    checks = defaultdict(Counter)
    findings = Counter()
    for item in items:
        if item["lang"] is not None:
            by_language[item["lang"]].append(item)
        for name, check in item["checks"].items():
            checks[name][check["status"]] += 1
        findings.update(flag["category"] for flag in item["flags"])
    summary = {
        "mode": mode,
        "rows": len(items),
        "items": counts_for(items),
        "languages": {
            lang: {**counts_for(rows), "flags": sum(len(row["flags"]) for row in rows)}
            for lang, rows in by_language.items()
        },
        "categories": {
            **{category: {"flags": findings[category], "checks": dict(checks["semantic_translation"])} for category in CATEGORIES},
            "number": {"flags": findings["number"], "checks": dict(checks["number"])},
            "format": {
                "flags": findings["format"],
                "checks": {name: dict(checks[name]) for name in ("markdown_table", "latex")},
            },
            "speech_silence": {"flags": findings["speech_silence"], "checks": dict(checks["audio_signal"])},
            "speech_clipping": {"flags": findings["speech_clipping"], "checks": dict(checks["audio_signal"])},
        },
        "checks": {name: dict(counts) for name, counts in checks.items()},
        "original_evaluation_cost": original_cost,
        "calibration": calibration_result["status"],
    }
    if baseline_counts is not None:
        summary["baseline"] = baseline_counts
    source = (
        "Offline fixture stubs. Simulated results and charges."
        if mode == "fixture-stub"
        else "Hosted adapters."
    )
    lines = [
        "# Bodhan quality harness",
        "",
        source,
        "",
        f"Read {len(items)} rows. "
        + ", ".join(f"{count} {status}" for status, count in summary["items"].items())
        + ".",
        f"Number findings: {findings['number']}.",
        f"Format findings: {findings['format']}.",
        f"Semantic findings: {sum(findings[category] for category in CATEGORIES)}.",
        f"Speech signal findings: {findings['speech_silence'] + findings['speech_clipping']}.",
        (
            f"Original evaluation cost estimate: ₹{original_cost['estimated_known_inr']} known, "
            f"{original_cost['unpriced_requests']} unpriced or uncertain requests."
        ),
        f"Calibration: {calibration_result['status']}.",
    ]
    if baseline_counts is not None:
        lines.append(
            f"Baseline regressions: {baseline_counts['regressed']}. See diff.md for examples."
        )
    lines.extend(["", "## Findings", ""])
    lines.extend(
        f"- {describe_item((item['id'], item['lang']))} · {describe_flag(flag)}"
        for item in items
        for flag in item["flags"]
    )
    lines.extend(
        [
            "",
            "Speech transcript mismatches are candidates requiring corroboration. Signal thresholds and judge calibration are pending.",
            "",
        ]
    )
    return "\n".join(lines), summary


def write_report(directory, items, mode, original_cost, baseline, calibration_result):
    diff, baseline_counts = (
        (None, None) if baseline is None else compare_baseline(baseline, items)
    )
    markdown, summary = render_summary(
        items, mode, original_cost, baseline_counts, calibration_result
    )
    files = {
        "items.jsonl": "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in items
        ),
        "calibration.json": pretty_json(calibration_result),
        "summary.json": pretty_json(summary),
        "summary.md": markdown,
    }
    if diff is not None:
        files["diff.md"] = diff
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".writing-", dir=directory) as staging:
        temporary = Path(staging)
        for name, contents in files.items():
            (temporary / name).write_text(contents, encoding="utf-8", newline="\n")
        for name in files:
            (temporary / name).replace(directory / name)
    if diff is None:
        (directory / "diff.md").unlink(missing_ok=True)
    return summary
