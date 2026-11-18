import json
from typing import ClassVar

from lfx.custom import Component
from lfx.io import MultilineInput, Output
from lfx.schema import Data

from bodhan_eval.stages import prepare


class Preparation(Component):
    name = "Preparation"
    display_name = "Prepare Outputs"
    inputs: ClassVar = [
        MultilineInput(name="rows_json", display_name="Detection batch")
    ]
    outputs: ClassVar = [
        Output(name="batch", display_name="Prepared items", method="process")
    ]

    async def process(self) -> Data:
        return Data(data={"items": prepare(json.loads(self.rows_json))})
