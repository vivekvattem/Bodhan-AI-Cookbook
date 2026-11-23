import argparse
import json
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
from .replay import plan_key
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


def positive_integer(value):
    try:
        result = int(value)
        if result > 0:
            return result
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("expected a positive integer")


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
    parser.add_argument("--bodhan-max-cost-inr", type=nonnegative_cost, help="Additional cap for new Bodhan spend and uncertain attempt reservations")
    parser.add_argument("--judge-max-cost-inr", type=nonnegative_cost, help="Additional independent cap for the configured judge")
    parser.add_argument("--wait-for-groq-rate-limit", action="store_true",
                        help="Keep retrying Groq HTTP 429 responses with backoff; other failures retain bounded retries")
    parser.add_argument("--judge-requests-per-minute", type=positive_integer,
                        help="Optional per-process judge request pacing limit")
    parser.add_argument("--judge-estimated-tokens-per-minute", type=positive_integer,
                        help="Optional per-process conservative estimated-token pacing limit")
    parser.add_argument(
        "--provider-smoke",
        action="store_true",
        help="Check provider connectivity for one non-English item",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Replay saved responses and unavailable observations without provider dispatch",
    )
    parser.add_argument("--replay-from", type=Path,
                        help="With --cache-only, select a report whose adjacent .run.json identifies its original run")
    return parser.parse_args(argv)


def main(argv=None, *, transport=None):
    args = arguments(argv)
    if args.replay_from and not args.cache_only:
        print("bodhan_eval: --replay-from requires --cache-only", file=sys.stderr)
        return 2
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
        external, invalid, calibration_records = (
            load_calibration(args.calibration, include_records=True) if args.calibration else ({}, 0, {})
        )
        cache_dir = args.out.parent / ".bodhan-eval"
        artifacts = cache_dir / "audio"
        artifacts.mkdir(parents=True, exist_ok=True)
        cache = RequestCache(cache_dir / "responses.sqlite3")
        try:
            plan = plan_key(rows, transport, args.generate, args.provider_smoke)
            selected = None
            if args.replay_from:
                reference = args.replay_from.with_name(args.replay_from.name + ".run.json")
                selected = json.loads(reference.read_text(encoding="utf-8"))["run_id"]
            run_id = cache.find_run(plan, selected) if args.cache_only else cache.start_run(plan)
            if args.replay_from and run_id is None:
                raise ValueError("saved run does not match the detection inputs or provider settings")
            models = Models(
                transport, cache, args.max_cost_inr, cache_only=args.cache_only, run_id=run_id,
                provider_caps={"bodhan": args.bodhan_max_cost_inr, "judge": args.judge_max_cost_inr},
                wait_for_groq_rate_limit=args.wait_for_groq_rate_limit,
                judge_rpm=args.judge_requests_per_minute,
                judge_estimated_tpm=args.judge_estimated_tokens_per_minute,
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
            calibration_result = calibration(items, labels, invalid, transport.mode, calibration_records)
            summary = write_report(
                args.out,
                items,
                transport.mode,
                models.original_cost(),
                baseline,
                calibration_result,
            )
            if run_id:
                if not args.cache_only:
                    cache.finish_run(run_id)
                reference = args.out.with_name(args.out.name + ".run.json")
                reference.write_text(pretty_json({"run_id": run_id, "plan": plan}), encoding="utf-8")
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
