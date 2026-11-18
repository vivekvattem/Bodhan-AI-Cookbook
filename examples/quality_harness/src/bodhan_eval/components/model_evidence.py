from typing import ClassVar

from lfx.custom import Component
from lfx.io import DataInput, Output
from lfx.schema import Data

from bodhan_eval.stages import model_evidence


class ModelEvidence(Component):
    name = "ModelEvidence"
    display_name = "Gather Model Evidence"
    inputs: ClassVar = [
        DataInput(name="checked", display_name="Deterministic evidence")
    ]
    outputs: ClassVar = [
        Output(name="batch", display_name="Model evidence", method="process")
    ]

    async def process(self) -> Data:
        return Data(data={"items": model_evidence(self.checked.data["items"])})
