from typing import ClassVar

from lfx.custom import Component
from lfx.io import DataInput, Output
from lfx.schema import Data

from bodhan_eval.stages import judging


class Judging(Component):
    name = "Judging"
    display_name = "Judge Number Findings"
    inputs: ClassVar = [DataInput(name="evidence", display_name="Model evidence")]
    outputs: ClassVar = [
        Output(name="batch", display_name="Reviewed evidence", method="process")
    ]

    async def process(self) -> Data:
        return Data(data={"items": judging(self.evidence.data["items"])})
