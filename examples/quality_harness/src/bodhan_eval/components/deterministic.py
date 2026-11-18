from typing import ClassVar

from lfx.custom import Component
from lfx.io import DataInput, Output
from lfx.schema import Data

from bodhan_eval.stages import deterministic


class DeterministicChecks(Component):
    name = "DeterministicChecks"
    display_name = "Run Deterministic Checks"
    inputs: ClassVar = [DataInput(name="prepared", display_name="Prepared items")]
    outputs: ClassVar = [
        Output(name="batch", display_name="Deterministic evidence", method="process")
    ]

    async def process(self) -> Data:
        return Data(data={"items": deterministic(self.prepared.data["items"])})
