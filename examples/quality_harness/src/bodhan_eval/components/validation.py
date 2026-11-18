from typing import ClassVar

from lfx.custom import Component
from lfx.io import DataInput, Output
from lfx.schema import Data

from bodhan_eval.stages import validate_results


class ResultValidation(Component):
    name = "ResultValidation"
    display_name = "Validate Result Spans"
    inputs: ClassVar = [DataInput(name="reviewed", display_name="Reviewed evidence")]
    outputs: ClassVar = [
        Output(name="results", display_name="Validated results", method="process")
    ]

    async def process(self) -> Data:
        return Data(data={"items": validate_results(self.reviewed.data["items"])})
