import argparse
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from dotenv import load_dotenv

from .cache import RequestCache
from .dataset import VOICES, load_calibration, load_dataset
from .flow import run_flow
from .models import Models
from .models import default_transport as hosted_transport
from .report import calibration, load_baseline, make_items, pretty_json, write_report
from .stages import Evaluation


def nonnegative_cost(value):
    try:
        amount = Decimal(value)
        if amount.is_finite() and amount >= 0:
            return amount
    except InvalidOperation:
        pass
    raise argparse.ArgumentTypeError("expected a finite nonnegative INR amount")


def language_filter(value):
    codes = {code.strip() for code in value.split(",")}
    if codes - VOICES.keys():
        raise argparse.ArgumentTypeError(
            "expected comma-separated Indic-Speak production language codes"
        )
    return codes


def arguments(argv):
    parser = argparse.ArgumentParser(
        description="Evaluate Bodhan translation and speech outputs."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--langs", required=True, type=language_filter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--max-cost-inr", type=nonnegative_cost)
    parser.add_argument(
        "--provider-smoke",
        action="store_true",
        help="Check provider connectivity for one non-English item",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Use cached responses and disable provider dispatch",
    )
    return parser.parse_args(argv)


def main(argv=None, *, transport=None):
    args = arguments(argv)
    started = time.monotonic()
    if transport is None:
        load_dotenv(Path.cwd() / ".env")
        transport = hosted_transport()
    try:
        rows = load_dataset(args.dataset.absolute(), args.langs)
        if args.provider_smoke and (
            len(rows) != 1 or rows[0].status != "ready" or rows[0].lang == "en"
        ):
            raise ValueError(
                "--provider-smoke requires one valid non-English item selected by --langs"
            )
        baseline = load_baseline(args.baseline) if args.baseline else None
        external, invalid = (
            load_calibration(args.calibration) if args.calibration else ({}, 0)
        )
        cache_dir = args.out.parent / ".bodhan-eval"
        artifacts = cache_dir / "audio"
        artifacts.mkdir(parents=True, exist_ok=True)
        cache = RequestCache(cache_dir / "responses.sqlite3")
        try:
            models = Models(
                transport, cache, args.max_cost_inr, cache_only=args.cache_only
            )
            records = [row.fields for row in rows if row.status == "ready"]
            results = run_flow(
                records,
                Evaluation(
                    models, args.generate, artifacts, provider_smoke=args.provider_smoke
                ),
            )
            if len(results) != len(records):
                raise RuntimeError(
                    "flow returned a different number of items than it received"
                )
            items = make_items(rows, results, external)
            labels = {
                row.key: external.get(row.key, row.labels)
                for row in rows
                if row.status == "ready"
            }
            calibration_result = calibration(items, labels, invalid, transport.mode)
            summary = write_report(
                args.out,
                items,
                transport.mode,
                models.original_cost(),
                baseline,
                calibration_result,
            )
            telemetry = {
                "mode": transport.mode,
                "provider_smoke": args.provider_smoke,
                "cache_only": args.cache_only,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "rows": [
                    {
                        "line": item["line"],
                        "status": item["status"],
                        "checks": {
                            name: value["status"]
                            for name, value in item["checks"].items()
                        },
                    }
                    for item in items
                ],
                **models.telemetry(),
            }
            telemetry_path = args.out.with_name(args.out.name + ".runtime.json")
            telemetry_path.write_text(
                pretty_json(telemetry), encoding="utf-8", newline="\n"
            )
        finally:
            cache.close()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"bodhan_eval: {exc}", file=sys.stderr)
        return 2
    print(
        f"Wrote {args.out} · {summary['rows']} rows · {transport.mode} · {telemetry['provider_calls']} provider calls"
    )
    return 0
