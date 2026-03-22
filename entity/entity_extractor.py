"""
Regex-based entity extraction for NAVIGATE commands.
Extracts distance (meters) and angle (degrees).
"""

import re
from typing import Dict, Optional

UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

SCALES = {
    "hundred": 100,
    "thousand": 1000,
}

DIGIT_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}

WORD_NUMBER_PATTERN = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|point|"
    r"and|a|an|half|quarter)"
)

# Matches either numeric values (e.g. 19, 19.5) or spoken numbers,
# including values like "one thousand twenty" and "one and a half".
NUMBER_TOKEN_PATTERN = rf"(?:\\d+(?:\\.\\d+)?|{WORD_NUMBER_PATTERN}(?:[\\s-]+{WORD_NUMBER_PATTERN})*)"


def extract_entities(text: str) -> Dict[str, Optional[float]]:
    """
    Extract distance and angle from a NAVIGATE command.

    Parameters
    ----------
    text : str
        The recognized speech text.

    Returns
    -------
    dict with keys:
        distance : float or None
        angle : float or None
    """
    distance = _extract_distance(text)
    angle = _extract_angle(text)
    return {"distance": distance, "angle": angle}


def has_required_entities(entities: Dict[str, Optional[float]]) -> bool:
    """Check that at least distance or angle was extracted."""
    return entities.get("distance") is not None or entities.get("angle") is not None


def _extract_distance(text: str) -> Optional[float]:
    """Extract distance in meters from text."""
    patterns = [
        rf"({NUMBER_TOKEN_PATTERN})\s*(?:meters?|m)\b",
        rf"(?:move|go|drive|travel|advance|proceed|navigate|head)\s+(?:forward\s+)?({NUMBER_TOKEN_PATTERN})",
        rf"({NUMBER_TOKEN_PATTERN})\s*(?:meters?\s+)?(?:forward|ahead|straight)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _parse_numeric_value(match.group(1))
    return None


def _extract_angle(text: str) -> Optional[float]:
    """Extract angle in degrees from text."""
    patterns = [
        rf"({NUMBER_TOKEN_PATTERN})\s*(?:degrees?|°)\b",
        rf"(?:turn|rotate)\s+(?:left|right)?\s*({NUMBER_TOKEN_PATTERN})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _parse_numeric_value(match.group(1))
    return None


def _parse_numeric_value(raw: str) -> Optional[float]:
    """Parse a numeric string or spoken number into a float."""
    raw = raw.strip().lower()

    # Digit-based values: 19, 19.5
    if re.fullmatch(r"\d+(?:\.\d+)?", raw):
        return float(raw)

    normalized = raw.replace("-", " ")

    # Fractional words: "half", "a half", "one and a half", "three and a quarter"
    for fraction_word, fraction_value in (("half", 0.5), ("quarter", 0.25)):
        if normalized in (fraction_word, f"a {fraction_word}", f"an {fraction_word}"):
            return fraction_value

        suffix = f"and a {fraction_word}"
        if normalized.endswith(suffix):
            prefix = normalized[: -len(suffix)].strip()
            whole = _parse_integer_words(prefix.split())
            if whole is not None:
                return float(whole) + fraction_value

    # Decimal words: "nineteen point five", "one point zero"
    if " point " in normalized:
        left, right = normalized.split(" point ", 1)
        whole = _parse_integer_words(left.split())
        if whole is None:
            return None

        right_tokens = right.split()
        if not right_tokens:
            return None

        decimal_digits = []
        for token in right_tokens:
            if token in DIGIT_WORDS:
                decimal_digits.append(str(DIGIT_WORDS[token]))
            elif token.isdigit() and len(token) == 1:
                decimal_digits.append(token)
            else:
                return None

        return float(f"{whole}.{''.join(decimal_digits)}")

    # Integer words: nineteen, forty five, one thousand twenty
    whole = _parse_integer_words(normalized.split())
    return float(whole) if whole is not None else None


def _parse_integer_words(tokens) -> Optional[int]:
    """Parse integer number words up to thousands."""
    if not tokens:
        return None

    total = 0
    current = 0
    seen_number = False

    for token in tokens:
        if token in {"and", "a", "an"}:
            continue
        if token in UNITS:
            current += UNITS[token]
            seen_number = True
            continue
        if token in TENS:
            current += TENS[token]
            seen_number = True
            continue
        if token == "hundred":
            current = (current or 1) * SCALES[token]
            seen_number = True
            continue
        if token == "thousand":
            total += (current or 1) * SCALES[token]
            current = 0
            seen_number = True
            continue
        return None

    value = total + current
    return value if seen_number else None


if __name__ == "__main__":
    test_cases = [
        "move forward 2 meters and turn 90 degrees",
        "move forward two meters and turn nineteen degrees",
        "move forward one and a half meters",
        "turn one thousand twenty degrees",
        "go ahead 5 meters",
        "turn left 45 degrees",
        "move forward",
        "hello world",
    ]
    for text in test_cases:
        entities = extract_entities(text)
        valid = has_required_entities(entities)
        print(f"'{text}' → {entities}  valid={valid}")
