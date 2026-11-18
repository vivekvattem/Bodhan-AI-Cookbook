import os
from pathlib import Path
from tempfile import TemporaryDirectory

from .models import canonical_json
from .stages import evaluation

FLOW = Path(__file__).with_name("quality_harness.json")
FIRST_NODE = "Preparation-harness"
LAST_NODE = "ResultValidation-harness"


def run_flow(records, session):
    if not records:
        return []
    token = evaluation.set(session)
    old_config = os.environ.get("LANGFLOW_CONFIG_DIR")
    try:
        with TemporaryDirectory(prefix="bodhan-lfx-") as config_dir:
            os.environ["LANGFLOW_CONFIG_DIR"] = config_dir
            from lfx.load import run_flow_from_json

            runs = run_flow_from_json(
                flow=FLOW,
                input_value="",
                output_component=LAST_NODE,
                tweaks={FIRST_NODE: {"rows_json": canonical_json(records)}},
            )
            return runs[0].outputs[0].outputs["results"]["message"]["items"]
    finally:
        evaluation.reset(token)
        if old_config is None:
            os.environ.pop("LANGFLOW_CONFIG_DIR", None)
        else:
            os.environ["LANGFLOW_CONFIG_DIR"] = old_config
