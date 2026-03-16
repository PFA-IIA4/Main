"""
Regex-based entity extraction for NAVIGATE commands.
Extracts distance (meters) and angle (degrees).
"""

import re
from typing import Dict, Optional


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
        r"(\d+(?:\.\d+)?)\s*(?:meters?|m)\b",
        r"(?:move|go|drive|travel|advance|proceed|navigate|head)\s+(?:forward\s+)?(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*(?:meters?\s+)?(?:forward|ahead|straight)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _extract_angle(text: str) -> Optional[float]:
    """Extract angle in degrees from text."""
    patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:degrees?|°)\b",
        r"(?:turn|rotate)\s+(?:left|right)?\s*(\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


if __name__ == "__main__":
    test_cases = [
        "move forward 2 meters and turn 90 degrees",
        "go ahead 5 meters",
        "turn left 45 degrees",
        "move forward",
        "hello world",
    ]
    for text in test_cases:
        entities = extract_entities(text)
        valid = has_required_entities(entities)
        print(f"'{text}' → {entities}  valid={valid}")
