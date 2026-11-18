from unicodedata import decimal


def normalize_digits(text: str) -> str:
    """Fold Unicode decimal digits to ASCII while preserving other characters."""
    return "".join(str(decimal(character, character)) for character in text)
