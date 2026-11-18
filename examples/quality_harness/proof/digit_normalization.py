from lfx.custom import Component
from lfx.io import Output, StrInput
from lfx.schema import Data

from digit_text import normalize_digits


class DigitNormalization(Component):
    display_name = "Digit Normalization"
    description = "Compare two strings after folding Unicode decimal digits."
    name = "DigitNormalization"

    inputs = [
        StrInput(name="left_text", display_name="Left text"),
        StrInput(name="right_text", display_name="Right text"),
    ]
    outputs = [Output(name="comparison", display_name="Comparison", method="compare")]

    def compare(self) -> Data:
        left = normalize_digits(self.left_text)
        right = normalize_digits(self.right_text)
        return Data(data={"left": left, "right": right, "equal": left == right})
