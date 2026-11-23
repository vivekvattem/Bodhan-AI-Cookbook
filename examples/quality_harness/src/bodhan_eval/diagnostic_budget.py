"""Read-only reconstruction of a supplied-text run's budget from saved artifacts."""

from collections import Counter
from decimal import Decimal

from .cache import RequestCache
from .models import HostedTransport, Models, RetryPolicy


class CapturedRequest(Exception):
    def __init__(self, request):
        self.request = request


class RequestRecorder(Models):
    """Reuse request builders, stopping at call() before cache or transport I/O."""

    def __init__(self, base_url, model):
        self.transport = HostedTransport({"LLM_BASE_URL": base_url, "LLM_MODEL": model})

    def call(self, request):
        raise CapturedRequest(request)


def recorded_request(builder, *args, **kwargs):
    try:
        builder(*args, **kwargs)
    except CapturedRequest as captured:
        return captured.request
    raise ValueError("request builder did not produce a request")


def requests_for_items(items, base_url, model, *, with_owners=False):
    recorder = RequestRecorder(base_url, model)
    requests, owners = [], []
    for item in items:
        if item.get("translation") and item.get("lang") != "en":
            requests.append(recorded_request(recorder.translate, item["translation"], "en", purpose="back_translation"))
            owners.append({"id": item["id"], "lang": item["lang"], "check": "back_translation"})
    for item in items:
        if not item.get("translation"):
            continue
        content = {key: item[key] for key in ("source", "translation", "lang")}
        back = item["checks"].get("back_translation", {})
        if back.get("status") == "evaluated":
            content["back_translation"] = back["evidence"]["text"]
        requests.append(recorded_request(recorder.semantic_judge, content))
        owners.append({"id": item["id"], "lang": item["lang"], "check": "semantic_translation"})
    return (requests, owners) if with_owners else requests


def audit_runtime(items, telemetry, base_url, cap):
    """Never infer a reservation if reconstructed exact request keys differ.

    This supports the saved supplied-text run, whose two model stages each make
    one request per usable item. Other run shapes are reported as unverified.
    """
    events = telemetry["requests"]
    reported = {"reported_failed_attempts": telemetry.get("failed_provider_attempts"),
                "reported_incremental_known_inr": telemetry.get("incremental_known_inr"),
                "reported_held_uncertain_inr": telemetry.get("reserved_uncertain_inr")}
    models = {event["model"] for event in events if event["operation"] == "judge"}
    if len(models) != 1 or telemetry.get("provider_smoke"):
        return {**reported, "status": "unverified", "reason": "unsupported judge identities or connectivity-smoke run"}
    requests, owners = requests_for_items(items, base_url, next(iter(models)), with_owners=True)
    if len(requests) != len(events):
        return {**reported, "status": "unverified", "reason": "saved request sequence differs from supplied-text stage shape"}
    mismatches = [i for i, (request, event) in enumerate(zip(requests, events, strict=True))
                  if RequestCache.key("hosted-v2", request.exact_input()) != event["key"]]
    if mismatches:
        return {**reported, "status": "unverified", "reason": "request identity mismatch; reservations not reconstructed", "positions": mismatches}
    known = uncertain = Decimal(0)
    failures, blocks, violations, observations = [], [], [], []
    for i, (request, event) in enumerate(zip(requests, events, strict=True)):
        owner = owners[i]
        observations.append({**owner, "position": i, "operation": request.operation, "model": request.model,
                             "decision": event.get("decision", event["status"]), "status": event["status"], "attempts": event.get("attempts", [])})
        reserve = request.reserve_inr
        if reserve is None:
            return {**reported, "status": "unverified", "reason": "unpriced request"}
        if event.get("decision") == "budget_block":
            blocks.append({**owner, "position": i, "reserve_inr": str(reserve), "current_exposure_inr": str(known + uncertain),
                           "affordable_ignoring_uncertain_billing": known + reserve <= cap})
        attempts = event.get("attempts", [])
        for j, attempt in enumerate(attempts):
            if known + uncertain + reserve > cap:
                violations.append({"position": i, "attempt": j + 1})
            if attempt["outcome"] == "success":
                known += Decimal(event["original_estimated_inr"])
                continue
            uncertain += reserve
            status = attempt.get("provider_error", {}).get("http_status")
            condition = ("retryable_rate_limit" if status == 429 else "retryable_provider_failure"
                         if status in {408, 500, 502, 503, 504} else "permanent_http_failure"
                         if status and 400 <= status < 500 else "unknown")
            recovered = any(later["outcome"] == "success" for later in attempts[j + 1:])
            stop = None
            if not recovered and j == len(attempts) - 1:
                if event.get("retry_blocked"):
                    stop = event["retry_blocked"]
                else:
                    stop = "no recovery observed within request"
                    try:
                        if float(attempt.get("rate_limits", {}).get("retry-after", "0")) > RetryPolicy().max_delay:
                            stop = "provider delay exceeds bounded wait"
                    except ValueError:
                        # An old HTTP-date needs its original receipt time, which
                        # early telemetry did not retain. Don't infer a delay now.
                        pass
            failures.append({**owner, "position": i, "attempt": attempt["number"], "operation": request.operation,
                             "model": request.model, "request_key": event["key"], "http_status": status,
                             "condition": condition, "recovered_within_request": recovered, "billing": "unknown",
                             "observed_classification": ("transient_recovered" if recovered else "permanent_error_response"
                                                         if condition == "permanent_http_failure" else "unresolved_after_bounded_policy"),
                             "held_reservation_inr": str(reserve), "rate_limits": attempt.get("rate_limits", {}),
                             "backoff_seconds": attempt.get("backoff_seconds"), "retry_stop_reason": stop,
                             "request_content_reconstruction": "exact key verified"})
    return {
        **reported,
        "status": "verified_request_reconstruction", "exact_request_keys_verified": len(requests),
        "cap_inr": str(cap), "cap_provenance": "recorded command supplied to audit",
        "new_known_success_inr": str(known), "held_uncertain_inr": str(uncertain),
        "remaining_after_reservations_inr": str(cap - known - uncertain),
        "matches_reported_known": known == Decimal(telemetry["incremental_known_inr"]),
        "matches_reported_reservations": uncertain == Decimal(telemetry["reserved_uncertain_inr"]),
        "failed_attempts": failures, "failure_conditions": dict(Counter(row["condition"] for row in failures)),
        "observed_failure_classifications": dict(Counter(row["observed_classification"] for row in failures)),
        "recovered_failed_attempts": sum(row["recovered_within_request"] for row in failures),
        "budget_blocks": blocks, "budget_blocks_due_to_uncertain_reservations": sum(row["affordable_ignoring_uncertain_billing"] for row in blocks),
        "admission_violations": violations,
        "request_observations": observations,
        "counterfactual_limit": ("Blocked requests were individually affordable without held reservations; "
                                 "this does not forecast funding or success for all of them."),
        "limitations": ("failed billing remains unknown; historical telemetry lacks per-attempt reserves, "
                        "so current request builders and exact hashes verify reconstruction"),
    }
