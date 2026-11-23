"""Prepare Bodhan base translations, then build a seeded, label-isolated demo.

Run `python -m bodhan_eval.demo --help`. This preparation tool does not evaluate
or calibrate a detector. The normal bodhan_eval CLI evaluates its output.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from .cache import RequestCache
from .cli import nonnegative_cost
from .inject import OPERATORS, split_sources, text_variant
from .models import Models, Unavailable, default_transport
from .report import pretty_json

LANGUAGES = ("hi", "ta", "bn")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def source_text(source):
    return "\n\n".join([*source["parts"], source["markup"]])


def generate(sources, models, *, min_interval=0):
    """Translate individual source paragraphs so omission mappings are known.

    Scaffold markup is supplied identically on both sides; no judge writes text.
    Retain failures separately and omit incomplete bases instead of faking text.
    """
    bases, failures = [], []
    for source in sources:
        for lang in LANGUAGES:
            parts = []
            for index, text in enumerate(source["parts"]):
                try:
                    before = getattr(models, "calls", 0)
                    parts.append(models.translate(text, lang, purpose="demo_base_translation"))
                except Unavailable as exc:
                    failures.append({"source_item": source["id"], "lang": lang, "part": index, "reason": str(exc)})
                    if str(exc) == "provider HTTP 429":
                        return bases, failures
                if min_interval and getattr(models, "calls", 0) > before:
                    time.sleep(min_interval)
            if len(parts) == len(source["parts"]):
                bases.append({"source_item": source["id"], "lang": lang, "parts": parts})
    return bases, failures


def build(sources, bases, seed):
    source_by_id = {source["id"]: source for source in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("duplicate source IDs")
    assignment = split_sources([source_text(source) for source in sources], seed)
    items, truth = [], []
    seen = set()
    for base in sorted(bases, key=lambda row: (row["source_item"], row["lang"])):
        source = source_by_id[base["source_item"]]
        key = base["source_item"], base["lang"]
        if key in seen or base["lang"] not in LANGUAGES:
            raise ValueError("duplicate or unsupported demo base language")
        seen.add(key)
        parts = base["parts"]
        if len(parts) != len(source["parts"]) or any(not isinstance(part, str) or not part.strip() for part in parts):
            raise ValueError("base translation parts do not match the source")
        clean = {
            "id": source["id"], "lang": base["lang"], "source": source_text(source),
            "translation": "\n\n".join([*parts, source["markup"]]),
        }
        group = hashlib.sha256(clean["source"].encode()).hexdigest()
        for operator in OPERATORS:
            item, record = text_variant(
                clean, operator, seed,
                omission={"source": [0, len(source["parts"][0])], "translation": [0, len(parts[0]) + 2]},
                term_pairs=source["term_pairs"],
            )
            record.update(source_group=group, split=assignment[clean["source"]], seed=seed)
            items.append(item)
            truth.append(record)
    return items, truth


def write_bundle(out, sources, bases, seed):
    items, truth = build(sources, bases, seed)
    write_jsonl(out / "demo.jsonl", items)
    write_jsonl(out / "ground_truth.jsonl", truth)
    for split in ("calibration", "evaluation"):
        keys = {(row["id"], row["lang"]) for row in truth if row["split"] == split}
        write_jsonl(out / (split + ".jsonl"), [row for row in items if (row["id"], row["lang"]) in keys])
        write_jsonl(out / (split + "_labels.jsonl"), [row for row in truth if row["split"] == split])
    manifest = {
        "version": 1, "seed": seed, "items": len(items), "languages": list(LANGUAGES),
        "source_groups": len({row["source_group"] for row in truth}),
        "splits": {split: sum(row["split"] == split for row in truth) for split in ("calibration", "evaluation")},
        "operators": {operator: sum(row["operator"] == operator for row in truth) for operator in OPERATORS},
        "base_translation_review": "pending; clean means no injection, not certified correct",
        "human_annotations": 0, "threshold_tuning": "not performed",
        "source_material": "Original synthetic English educational passages; see README.md",
        "files_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in (
            "sources.jsonl", "base_translations.jsonl", "demo.jsonl", "ground_truth.jsonl",
            "calibration.jsonl", "evaluation.jsonl", "calibration_labels.jsonl", "evaluation_labels.jsonl",
        )},
    }
    (out / "manifest.json").write_text(pretty_json(manifest), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("generate", help="Bodhan translation only, never synthesis or judging")
    prepare.add_argument("--sources", type=Path, default=Path("data/sources.jsonl"))
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--max-cost-inr", type=nonnegative_cost, required=True)
    prepare.add_argument("--cache-only", action="store_true")
    prepare.add_argument("--min-interval-seconds", type=float, default=4,
                         help="Pause after dispatched requests; stop preparation on the first HTTP 429")
    build_parser = sub.add_parser("build", help="Offline deterministic text edits and grouped splits")
    build_parser.add_argument("--sources", type=Path, default=Path("data/sources.jsonl"))
    build_parser.add_argument("--bases", type=Path, default=Path("data/base_translations.jsonl"))
    build_parser.add_argument("--out", type=Path, default=Path("data"))
    build_parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args(argv)
    if args.command == "generate":
        if not 0 <= args.min_interval_seconds <= 60:
            parser.error("request interval must be between 0 and 60 seconds")
        load_dotenv(Path.cwd() / ".env", override=False)
        args.out.mkdir(parents=True, exist_ok=True)
        cache_dir = args.out.parent / ".bodhan-eval"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = RequestCache(cache_dir / "responses.sqlite3")
        try:
            models = Models(default_transport(), cache, args.max_cost_inr, cache_only=args.cache_only)
            bases, failures = generate(read_jsonl(args.sources), models, min_interval=args.min_interval_seconds)
            write_jsonl(args.out / "base_translations.jsonl", bases)
            (args.out / "generation.runtime.json").write_text(pretty_json({"failures": failures, **models.telemetry()}), encoding="utf-8")
        finally:
            cache.close()
        print(f"Prepared {len(bases)} complete base translations; {len(failures)} unavailable paragraphs")
        return 0 if not failures else 1
    args.out.mkdir(parents=True, exist_ok=True)
    # Preserve reusable inputs alongside the derived, text-only public fixture.
    for name, path in (("sources.jsonl", args.sources), ("base_translations.jsonl", args.bases)):
        if path.resolve() != (args.out / name).resolve():
            (args.out / name).write_bytes(path.read_bytes())
    manifest = write_bundle(args.out, read_jsonl(args.sources), read_jsonl(args.bases), args.seed)
    print(f"Built {manifest['items']} items; splits={manifest['splits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
