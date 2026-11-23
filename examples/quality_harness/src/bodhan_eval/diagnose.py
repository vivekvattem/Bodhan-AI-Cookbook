"""Join saved predictions to annotations after detection. Never dispatch providers."""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

from .cli import nonnegative_cost
from .demo import read_jsonl, write_jsonl
from .diagnostic_budget import audit_runtime
from .metrics import CHECKS, annotation_index, category_status, flag_locations, label_location, matching_pairs, overlaps, source_group
from .numbers import TOKEN, analyze_numbers, compare_numbers, decimal_value
from .report import pretty_json


def numeric_trace(item):
    fields = {}
    for field in ("source", "translation"):
        text = item[field]
        fields[field] = {**analyze_numbers(text, field), "lexical_candidates": [
            {"span": list(match.span()), "text": match[0], "folded_decimal_value": str(decimal_value(match[0])),
             "note": "lexical only; not accepted if the field parser rejects context"}
            for match in TOKEN.finditer(text)]}
    return {"historical_check": item["checks"].get("number"), "current_check": compare_numbers(item["source"], item["translation"]),
            "fields": fields, "category_mapping": {"seed": "number", "detector": "number"},
            "policy": "changed multi-number inventories or unpaired units abstain; a digit mismatch alone does not confirm a quantity error"}


def legacy_context_locations(flag, item):
    """Read-only comparator for the old permissive localization calculation."""
    from .metrics import checked_span

    locations = set(flag_locations(flag, item))
    evidence = flag.get("evidence") or {}
    anchors = list(evidence.get("locations", [])) + [evidence[name] for name in ("anchor", "source", "translation")
                                                   if isinstance(evidence.get(name), dict)]
    for anchor in anchors:
        field = anchor.get("field")
        if field in {"source", "translation"}:
            try:
                a, b = checked_span(anchor.get("span"), item.get(field), anchor.get("text"))
                locations.add((field, a, b))
            except ValueError:
                pass
    return locations


def item_diagnostic(item, annotation):
    result = {key: item.get(key) for key in ("id", "lang", "source", "translation", "status")}
    result.update(checks=item.get("checks", {}), flags=item.get("flags", []), source_group=source_group(item))
    if annotation is None:
        return {**result, "classification": "unlabelled", "seeded_cases": []}
    result.update(source_item=annotation.get("source_item"), split=annotation.get("split"),
                  annotation_origin=annotation.get("annotation_origin", "unspecified"), human_review=annotation.get("human_review"))
    labels = annotation["labels"]
    result["original_edit"] = annotation.get("edit")
    if not labels:
        return {**result, "classification": "clean_control_flagged" if result["flags"] else "clean_control_no_flag", "seeded_cases": []}
    pairs = [(label, label_location(label, annotation, item)) for label in labels]
    flags = [(flag, flag_locations(flag, item)) for flag in result["flags"]]
    assigned, used, outcomes = {}, set(), {}
    policies = (
        ("detection", lambda label, flag: label[0]["type"] == flag[0]["category"] and any(overlaps(label[1], loc) for loc in flag[1])),
        ("localization_failure", lambda label, flag: label[0]["type"] == flag[0]["category"]),
        ("wrong_category", lambda label, flag: any(overlaps(label[1], loc) for loc in flag[1])),
    )
    for outcome, predicate in policies:
        left, right = [i for i in range(len(pairs)) if i not in assigned], [j for j in range(len(flags)) if j not in used]
        matches = matching_pairs([pairs[i] for i in left], [flags[j] for j in right], predicate)
        for i, j in matches.items():
            assigned[left[i]], outcomes[left[i]] = right[j], outcome
            used.add(right[j])
    cases = []
    for i, (label, location) in enumerate(pairs):
        same = [flag for flag in result["flags"] if flag["category"] == label["type"]]
        old_overlap = any(overlaps(location, anchor) for flag in same for anchor in legacy_context_locations(flag, item))
        primary_overlap = any(overlaps(location, anchor) for flag in same for anchor in flag_locations(flag, item))
        relevant = {name: item.get("checks", {}).get(name, {"status": "absent"}) for name in CHECKS.get(label["type"], ())}
        incomplete = category_status(item, label["type"]) != "evaluated"
        cases.append({"seed": label, "evaluation_coordinate": {"field": location[0], "span": list(location[1:]), "zero_width": location[1] == location[2]},
                      "classification": outcomes.get(i, "miss"), "matched_flag": result["flags"][assigned[i]] if i in assigned else None,
                      "relevant_checks": relevant, "incomplete_coverage_relevant": incomplete,
                      "coverage_interpretation": ("relevant checker abstained, failed, or did not run; counterfactual result unknown"
                                                  if incomplete else "relevant check completed"),
                      "legacy_context_overlap": old_overlap, "primary_span_overlap": primary_overlap,
                      "coordinate_rule": ("original code points; clean-translation deletions project to an output boundary; "
                                          "boundary touch is explicitly credited")})
    result.update(classification=cases[0]["classification"] if len(cases) == 1 else "multiple_seeded_errors", seeded_cases=cases)
    if any(label["type"] == "number" for label in labels):
        result["numeric_trace"] = numeric_trace(item)
    return result


def format_patterns(diagnostics):
    patterns = defaultdict(lambda: {"cases": 0, "sources": set(), "languages": Counter(), "localized": 0})
    for row in diagnostics:
        if not any(case["seed"]["type"] == "format" for case in row["seeded_cases"]):
            continue
        edit = row["original_edit"]
        original = row["translation"][:edit["output_span"][0]] + edit["original_text"] + row["translation"][edit["output_span"][1]:]
        if edit["original_text"] == "$":
            name = "opening_dollar_deleted" if edit["original_span"][0] == original.find("$") else "closing_dollar_deleted"
        elif "---" in edit["original_text"]:
            name = "pipe_separator_cell_deleted"
        else:
            name = "other_structural_deletion"
        value = patterns[name]
        value["cases"] += 1
        value["sources"].add(row["source_group"])
        value["languages"][row["lang"]] += 1
        value["localized"] += any(case["primary_span_overlap"] for case in row["seeded_cases"])
    return {name: {**value, "sources": sorted(value["sources"]), "independent_sources": len(value["sources"]), "languages": dict(value["languages"])}
            for name, value in sorted(patterns.items())}


def write_diagnostics(items, annotations, out, *, provider_observations=()):
    index, invalid = annotation_index(annotations)
    if invalid:
        raise ValueError("diagnostic requires unique explicit annotations")
    rows = [item_diagnostic(item, index.get((item["id"], item["lang"]))) for item in items]
    events_by_item = defaultdict(list)
    for observation in provider_observations:
        events_by_item[observation["id"], observation["lang"]].append(observation)
    for row in rows:
        row["provider_observations"] = events_by_item[row["id"], row["lang"]]
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "items-diagnostic.jsonl", rows)
    cases = [{"id": row["id"], "lang": row["lang"], "source_item": row.get("source_item"), **case}
             for row in rows for case in row["seeded_cases"]]
    write_jsonl(out / "localization-audit.jsonl", [case for case in cases if case["classification"] in {"detection", "localization_failure"}])
    write_jsonl(out / "localization-failures.jsonl", [case for case in cases if case["classification"] == "localization_failure"])
    write_jsonl(out / "numeric-trace.jsonl", [row for row in rows if "numeric_trace" in row])
    summary = {"items": len(rows), "seeded_errors": len(cases), "classification": dict(Counter(case["classification"] for case in cases)),
               "categories": {category: dict(Counter(case["classification"] for case in cases if case["seed"]["type"] == category))
                              for category in sorted({case["seed"]["type"] for case in cases})},
               "incomplete_relevant_checks": dict(Counter(case["seed"]["type"] for case in cases if case["incomplete_coverage_relevant"])),
               "format_patterns": format_patterns(rows),
               "metric_fix_changed_localization_cases": sum(case["legacy_context_overlap"] != case["primary_span_overlap"] for case in cases),
               "labels": "programmatic and unreviewed; no tuning or new provider calls"}
    (out / "summary.json").write_text(pretty_json(summary), encoding="utf-8")
    with (out / "cases.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=("id", "lang", "source_item", "category", "seed_field", "seed_span", "scoring_field", "scoring_span",
                                                   "classification", "incomplete_coverage_relevant", "seed_text", "check_statuses", "number_reason",
                                                   "judge_verdict", "judge_explanation", "judge_rejected_evidence", "number_review", "back_translation",
                                                   "matched_flag_field", "matched_flag_span", "matched_flag_evidence", "provider_observations",
                                                   "source", "translation"))
        writer.writeheader()
        by_key = {(row["id"], row["lang"]): row for row in rows}
        for case in cases:
            row = by_key[case["id"], case["lang"]]
            checks = row["checks"]
            judge = checks.get("semantic_translation", {})
            back = checks.get("back_translation", {})
            flag = case.get("matched_flag") or {}
            writer.writerow({key: case[key] for key in ("id", "lang", "source_item", "classification", "incomplete_coverage_relevant")} | {
                "category": case["seed"]["type"], "seed_field": case["seed"]["field"], "seed_span": case["seed"]["span"],
                "scoring_field": case["evaluation_coordinate"]["field"], "scoring_span": case["evaluation_coordinate"]["span"],
                "seed_text": case["seed"].get("text", ""),
                "check_statuses": json.dumps({name: check["status"] for name, check in checks.items()}, sort_keys=True),
                "number_reason": checks.get("number", {}).get("reason", ""), "judge_verdict": judge.get("verdict", judge.get("status")),
                "judge_explanation": judge.get("explanation", judge.get("reason", "")), "judge_rejected_evidence": json.dumps(judge.get("rejected", [])),
                "number_review": json.dumps(checks.get("judge", {}), sort_keys=True),
                "back_translation": back.get("evidence", {}).get("text", back.get("reason", "")),
                "matched_flag_field": flag.get("field", ""), "matched_flag_span": flag.get("span", ""),
                "matched_flag_evidence": json.dumps(flag.get("evidence", {}), ensure_ascii=False, sort_keys=True),
                "provider_observations": json.dumps(row["provider_observations"], sort_keys=True),
                "source": row["source"], "translation": row["translation"]})
    lines = ["# Saved-run diagnostic", "", "Programmatic labels are joined after detection. Original reports are unchanged. No held-out tuning.", "",
             "| Category | Localized detection | Localization failure | Miss | Wrong category |", "| --- | ---: | ---: | ---: | ---: |"]
    for category, counts in summary["categories"].items():
        lines.append(f"| {category} | {counts.get('detection', 0)} | {counts.get('localization_failure', 0)} | "
                     f"{counts.get('miss', 0)} | {counts.get('wrong_category', 0)} |")
    lines.extend(["", "Each row in items-diagnostic.jsonl retains source, translation, original edit, all check statuses/evidence, judge verdict/rejections, "
                  "and per-seed scoring coordinates. cases.csv is a compact index. Numeric traces include parser rejection anchors. "
                  "Localization failure means a correctly categorized finding missed the seeded span under the stated coordinate rule; "
                  "it does not imply the finding's evidence is out of bounds.", ""])
    (out / "diagnostic.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-cost-inr", type=nonnegative_cost, required=True, help="Recorded historical cap; this command dispatches no requests")
    args = parser.parse_args(argv)
    items = read_jsonl(args.report / "items.jsonl")
    runtime = json.loads(args.report.with_name(args.report.name + ".runtime.json").read_text())
    load_dotenv(Path.cwd() / ".env", override=False)
    budget = audit_runtime(items, runtime, os.environ.get("LLM_BASE_URL", ""), args.max_cost_inr)
    summary = write_diagnostics(items, read_jsonl(args.labels), args.out, provider_observations=budget.get("request_observations", []))
    (args.out / "provider-budget-audit.json").write_text(pretty_json(budget), encoding="utf-8")
    print(f"Diagnosed {summary['items']} items / {summary['seeded_errors']} injected errors; no provider dispatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
