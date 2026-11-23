import base64
import io
import json
import os
import sys
import unicodedata
import wave
from decimal import Decimal
from pathlib import Path

from bodhan_eval.models import Unavailable


def synthetic_wav():
    target = io.BytesIO()
    with wave.open(target, "wb") as audio:
        audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * 80)
    return target.getvalue()


class FixtureTransport:
    namespace = "milestone-fixture-v1"
    mode = "fixture-stub"
    judge_model = "fixture-judge"
    judge_endpoint = "https://fixture.invalid/chat/completions"
    judge_reserve_inr = Decimal(0)

    def missing_configuration(self, request):
        return None

    def send(self, request):
        if os.environ.get("BODHAN_TEST_FAIL_ON_CALL"):
            raise AssertionError("cache replay attempted a provider call")
        trace = os.environ.get("BODHAN_TEST_TRACE")
        if trace:
            with Path(trace).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(request.exact_input(), sort_keys=True) + "\n")
        if request.operation == "translate":
            original = request.payload["messages"][0]["content"]
            if request.payload["target_language_code"] == "en":
                folded = "".join(str(unicodedata.decimal(c, c)) for c in original)
                return {
                    "translation": "Stub evidence: " + folded,
                    "usage": {"completion_tokens": 6},
                }
            if (
                request.payload["target_language_code"] == "bn"
                and original == "The box weighs 7 kg."
            ):
                return {
                    "translation": chr(0x09E6 + 7) + " kg",
                    "usage": {"completion_tokens": 4},
                }
            raise Unavailable("no translation stub for this payload")
        if request.operation == "speak":
            return {"audio_b64": base64.b64encode(synthetic_wav()).decode("ascii")}
        if request.operation == "transcribe":
            return {"text": "Stub transcript"}
        if request.operation == "judge":
            if request.purpose == "semantic_translation":
                judgment = {
                    "verdict": "clean",
                    "explanation": "Synthetic clean verdict for fixture transport only.",
                    "findings": [],
                }
                return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(judgment)}}]}
            return {"choices": [{"message": {"content": '{"verdict":"supports"}'}}]}
        raise AssertionError(f"unexpected stub operation: {request.operation}")


if __name__ == "__main__":
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(synthetic_wav())
