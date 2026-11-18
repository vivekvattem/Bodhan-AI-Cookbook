import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from lfx.load import run_flow_from_json


FLOW = Path(__file__).with_name("digit_normalization.json")
COMPONENT_ID = "DigitNormalization-proof"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the exported digit comparison flow.")
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()

    with TemporaryDirectory(prefix="bodhan-lfx-") as config_dir:
        # Keep LFX's scratch files separate from any editor installation.
        os.environ["LANGFLOW_CONFIG_DIR"] = config_dir
        runs = run_flow_from_json(
            flow=FLOW,
            input_value="",
            output_component=COMPONENT_ID,
            tweaks={COMPONENT_ID: {"left_text": args.left, "right_text": args.right}},
        )

    comparison = runs[0].outputs[0].outputs["comparison"]["message"]
    print(json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    main()
