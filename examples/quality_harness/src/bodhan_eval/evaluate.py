"""Evaluate saved predictions against separate annotations, without model calls."""

import argparse
import csv
from pathlib import Path

from .demo import read_jsonl, write_jsonl
from .metrics import evaluate
from .report import pretty_json

HUMAN_FIELDS = ("human_verdict", "human_categories", "human_severity", "human_field",
                "human_span_start", "human_span_end", "human_notes", "reviewer", "reviewed_at")


def review_queue(items, annotations, path, limit=12):
    """Calibration-only, paired controls/corruptions; seeded answers kept separate."""
    if limit < 1:
        raise ValueError("review limit must be positive")
    if path.exists():
        with path.open(encoding="utf-8", newline="") as existing:
            if any(row.get(field, "").strip() for row in csv.DictReader(existing) for field in HUMAN_FIELDS):
                raise ValueError("review file has human entries; choose a new output path")
    by_key = {(item["id"], item["lang"]): item for item in items}
    eligible = [row for row in annotations if row.get("split") == "calibration" and row.get("annotation_origin") == "programmatic"
                and (row.get("id"), row.get("lang")) in by_key]
    controls = sorted([row for row in eligible if row.get("labels") == []], key=lambda row: (row.get("source_item", ""), row["lang"], row["id"]))
    selected, seen = [], set()
    operators = ("omission", "digit_change", "term_swap", "format_damage")
    for i, control in enumerate(controls):
        options = [row for row in eligible if row.get("source_group") == control.get("source_group") and row["lang"] == control["lang"]
                   and row.get("operator") == operators[i % len(operators)]]
        for row in [control, *options[:1]]:
            key = row["id"], row["lang"]
            if key not in seen:
                selected.append(row)
                seen.add(key)
    selected = selected[:limit]
    rows = []
    for row in selected:
        item = by_key[row["id"], row["lang"]]
        rows.append({"id": row["id"], "lang": row["lang"], "source_item": row.get("source_item", ""),
                     "split": "calibration", "annotation_provenance": "programmatic_unreviewed",
                     "source": item["source"], "translation": item["translation"], **dict.fromkeys(HUMAN_FIELDS, "")})
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("id", "lang", "source_item", "split", "annotation_provenance", "source", "translation", *HUMAN_FIELDS)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_jsonl(path.with_suffix(".programmatic.jsonl"), selected)
    path.with_suffix(".instructions.md").write_text(
        "# Independent calibration review\n\n"
        "The CSV contains only calibration inputs and blank human fields. It has no detector predictions. "
        "All existing labels are programmatic and unreviewed; their answers are in the separate .programmatic.jsonl file. "
        "Review the English source and translation before looking at those answers.\n\n"
        "Fill human_verdict (clean/error/uncertain), categories, severity, affected original field, half-open Unicode "
        "code-point span, notes, reviewer and date. For omissions select English source text. "
        "Do not infer a clean verdict from unavailable or inconclusive model checks. "
        "Judge base translation correctness as well as the injected change. Leave uncertain judgments unresolved. "
        "This export does not import human answers or tune thresholds. Held-out items are excluded.\n",
        encoding="utf-8",
    )
    return len(rows)


def render(result):
    lines = ["# Injected-label diagnostics", "", "Labels are programmatic and unreviewed unless explicitly stated otherwise. "
             "These results do not establish human-calibrated accuracy. No thresholds were tuned.", "",
             f"Items: {result['items']}; labelled: {result['labelled_items']}; unlabelled: {result['unlabelled_items']}; "
             f"invalid labelled items: {result['invalid_labelled_items']}. "
             f"Independent English sources: {result['independent_english_sources']}.", "",
             "| Category | TP | FP | FN | Precision | Recall | Positive source groups |",
             "| --- | ---: | ---: | ---: | --- | --- | ---: |"]
    for category, score in result["categories"].items():
        if not score["positive_items"] and not score["fp"]:
            continue
        precision, recall = score["precision"], score["recall"]
        lines.append(f"| {category} | {score['tp']} | {score['fp']} | {score['fn']} | "
                     f"{precision['numerator']}/{precision['denominator']} | {recall['numerator']}/{recall['denominator']} | "
                     f"{score['independent_positive_sources']} |")
    for name in ("clean_false_alarm_rate", "localization_overlap", "category_accuracy_detected_items", "category_accuracy_among_localized_errors"):
        score = result[name]
        lines.extend(["", f"{name}: {score['numerator']}/{score['denominator']} (undefined if denominator is zero)."])
    lines.extend(["", "Recall includes incomplete checks. Check coverage, per-language and per-split denominators, "
                  "unlabelled items, invalid anchors, and matching definitions are in evaluation.json. "
                  "Related translations and injected variants are not independent sources.", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--review-out", type=Path)
    parser.add_argument("--review-limit", type=int, default=12)
    parser.add_argument("--dataset-status", choices=("unspecified", "development", "previously_inspected"), default="unspecified")
    args = parser.parse_args(argv)
    items = read_jsonl(args.report / "items.jsonl")
    annotations = read_jsonl(args.labels)
    result = evaluate(items, annotations)
    result["dataset_status"] = args.dataset_status
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "evaluation.json").write_text(pretty_json(result), encoding="utf-8")
    markdown = render(result)
    if args.dataset_status == "previously_inspected":
        markdown = "Previously inspected diagnostic set: these are not fresh held-out accuracy results.\n\n" + markdown
    (args.out / "evaluation.md").write_text(markdown, encoding="utf-8")
    if args.review_out:
        count = review_queue(items, annotations, args.review_out, args.review_limit)
        print(f"Prepared {count} calibration review rows; human fields are blank")
    print(f"Evaluated {result['items']} saved predictions; no model calls")
    return 1 if result["cross_split_source_overlap"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
