import json
from dataclasses import dataclass
from pathlib import Path

VOICES = {
    "en": "Amit",
    "hi": "Kavya",
    "bn": "Ishita",
    "mr": "Anagha",
    "te": "Sravani",
    "ta": "Anitha",
    "gu": "Dhara",
    "kn": "Deepika",
    "ml": "Lakshmi",
    "or": "Itishree",
    "pa": "Kaur",
    "as": "Prastuti",
    "ur": "Saba",
}


@dataclass
class Row:
    line: int
    item_id: str | None = None
    lang: str | None = None
    fields: dict | None = None
    labels: list | None = None
    status: str = "failed"
    reason: str | None = None

    @property
    def key(self):
        return self.item_id, self.lang


def valid_labels(value):
    return isinstance(value, list) and all(
        isinstance(label, dict) and isinstance(label.get("type"), str)
        for label in value
    )


def parse_line(raw):
    def reject_constant(value):
        raise ValueError(f"invalid JSON constant {value}")

    value = json.loads(raw, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError("row must be a JSON object")
    return value


def has_text(value):
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def load_dataset(path: Path, languages: set[str]):
    rows = []
    seen = set()
    with path.open("rb") as dataset:
        for line, raw in enumerate(dataset, 1):
            row = Row(line)
            rows.append(row)
            try:
                data = parse_line(raw.decode("utf-8"))
            except (ValueError, TypeError, RecursionError):
                row.reason = "invalid JSONL object"
                continue
            row.item_id = data.get("id") if has_text(data.get("id")) else None
            row.lang = data.get("lang") if has_text(data.get("lang")) else None
            if (
                row.item_id is None
                or row.lang not in VOICES
                or not has_text(data.get("source"))
            ):
                row.reason = "expected a nonempty id, source, and supported lang"
                continue
            if row.lang not in languages:
                row.status, row.reason = "skipped", "language filter"
                continue
            if row.key in seen:
                row.reason = "duplicate id and lang"
                continue
            seen.add(row.key)
            row.status = "ready"
            row.labels = data["labels"] if valid_labels(data.get("labels")) else None
            issues = {}
            row.fields = {"lang": row.lang, "source": data["source"], "issues": issues}
            for name in ("reference", "translation", "audio"):
                value = data.get(name)
                if value is not None and not has_text(value):
                    issues[name] = "expected a nonempty UTF-8 string"
                else:
                    row.fields[name] = value
            if row.fields.get("audio"):
                row.fields["audio_path"] = str(path.parent / row.fields["audio"])
    return rows


def load_calibration(path: Path, *, include_records=False):
    labels, records, duplicates = {}, {}, set()
    invalid = 0
    with path.open("rb") as source:
        for raw in source:
            try:
                data = parse_line(raw.decode("utf-8"))
                item_id, lang = data.get("id"), data.get("lang")
                if not has_text(item_id) or not has_text(lang) or lang not in VOICES:
                    raise ValueError("missing id or lang")
                if not valid_labels(data.get("labels")):
                    raise ValueError("invalid labels array")
            except (ValueError, TypeError, RecursionError):
                invalid += 1
                continue
            key = item_id, lang
            if key in labels or key in duplicates:
                duplicates.add(key)
                labels.pop(key, None)
                records.pop(key, None)
                invalid += 1
                continue
            labels[key] = data["labels"]
            records[key] = data
    return (labels, invalid, records) if include_records else (labels, invalid)
