"""
LLM-only Intent Classifier for Voice-Controlled Robot System.

This module is a thin production wrapper around the local llama.cpp-based
classifier and returns a stable dictionary interface for the rest of the app.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

from intent.llm_classifier import get_classifier

logger = logging.getLogger(__name__)


@dataclass
class IntentClassificationResult:
    """Structured intent classification result."""

    intent: str
    confidence: float
    reason: str = ""
    model_used: str = "llm"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "reason": self.reason,
            "model_used": self.model_used,
        }


class IntentClassifier:
    """LLM-only intent classifier."""

    def __init__(self):
        self.llm_classifier = get_classifier()
        logger.debug("LLM intent classifier initialized successfully")

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict intent using the local LLM and return a dictionary."""
        result = self.llm_classifier.classify(text)
        return {
            "intent": result.intent,
            "confidence": result.confidence,
            "reason": result.reason,
            "tokens_generated": result.tokens_generated,
            "inference_time_ms": result.inference_time_ms,
            "model_used": result.model_used,
        }


def predict(text: str) -> Dict[str, Any]:
    """Predict intent from text."""
    return IntentClassifier().predict(text)


__all__ = ["IntentClassifier", "IntentClassificationResult", "predict"]
