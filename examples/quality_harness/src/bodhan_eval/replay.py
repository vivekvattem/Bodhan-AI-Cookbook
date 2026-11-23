"""Run identities exclude labels and secrets; journals preserve invocation decisions."""

import hashlib
from pathlib import Path

from .models import canonical_json, format_inr
from .numbers import VERSION as NUMBER_VERSION
from .semantic import VERSION as SEMANTIC_VERSION


def plan_key(rows, transport, generate=False, provider_smoke=False):
    records = []
    for row in rows:
        fields = dict(row.fields or {})
        if fields.get("audio_path"):
            try:
                fields["audio_sha256"] = hashlib.sha256(Path(fields["audio_path"]).read_bytes()).hexdigest()
            except OSError:
                fields["audio_sha256"] = "unreadable"
        records.append([row.line, row.item_id, row.lang, row.status, row.reason, fields])
    value = ["evaluation-plan-v3", NUMBER_VERSION, SEMANTIC_VERSION, records, transport.namespace, transport.judge_model,
             transport.judge_endpoint, generate, provider_smoke]
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def import_legacy_run(cache, plan, telemetry):
    """Recover pre-journal decisions from retained telemetry and immutable responses.

    Explicit migration only. Does not infer successful responses or edit a report.
    Old failure billing was unknown; blocked requests were never dispatched.
    """
    reasons = {
        "rate_limit_block": "endpoint paused after HTTP 429; resume in a later invocation",
        "budget_block": "cost cap: request is unpriced or exceeds remaining budget",
    }
    decisions = []
    seen = set()
    for position, event in enumerate(telemetry["requests"]):
        key = event["key"]
        cached = cache.get(key)
        if event["status"] in {"cache_hit", "provider_response"}:
            if cached is None:
                raise ValueError("legacy successful response is missing; cannot import")
            result, cost = cached
            if event["original_estimated_inr"] != (format_inr(cost) if cost is not None else None):
                raise ValueError("legacy response cost differs")
            value = {"result": result, "cost": str(cost) if cost is not None else None,
                     "known_delta": str(cost) if cost is not None and key not in seen else "0",
                     "unknown_delta": int(cost is None and key not in seen)}
        else:
            decision = event["decision"]
            reason = cache.get_failure(key) if decision == "provider_failure" else reasons.get(decision)
            if reason is None:
                raise ValueError("legacy decision cannot be reconstructed")
            value = {"reason": reason, "decision": decision, "known_delta": "0",
                     "unknown_delta": int(decision == "provider_failure" and key not in seen)}
        decisions.append((position, key, value))
        seen.add(key)
    run_id = cache.start_run(plan)
    for position, key, value in decisions:
        cache.put_decision(run_id, position, key, value)
    cache.finish_run(run_id)
    return run_id
