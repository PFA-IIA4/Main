"""
Intent classifier using a pre-trained TF-IDF + Logistic Regression model.
Returns intent label, confidence, and margin information.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import joblib
import numpy as np

MODEL_DIR = os.path.dirname(__file__)
CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.58"))
MARGIN_THRESHOLD = float(os.getenv("INTENT_MARGIN_THRESHOLD", "0.12"))
NAVIGATE_BOOST = float(os.getenv("INTENT_NAVIGATE_BOOST", "0.06"))
BREAK_BOOST = float(os.getenv("INTENT_BREAK_BOOST", "0.35"))
FATIGUE_BREAK_BOOST = float(os.getenv("INTENT_FATIGUE_BREAK_BOOST", "0.8"))
RAG_QUERY_BOOST = float(os.getenv("INTENT_RAG_BOOST", "0.06"))
RAG_QUERY_STRONG_BOOST = float(os.getenv("INTENT_RAG_STRONG_BOOST", "0.45"))

NAVIGATE_VERBS = (
    "move",
    "go",
    "turn",
    "rotate",
    "drive",
    "advance",
    "proceed",
    "navigate",
    "head",
)

NAVIGATE_HINTS = (
    "meter",
    "meters",
    "degree",
    "degrees",
    "left",
    "right",
    "forward",
    "ahead",
    "straight",
    "north",
    "south",
    "east",
    "west",
)

SESSION_TERMS = (
    "session",
    "study",
    "studying",
    "focus mode",
    "study mode",
    "break",
)

RAG_TRIGGERS = (
    "what is",
    "explain",
    "define",
    "summarize",
    "tell me about",
    "what does",
)

RAG_SUMMARY_PATTERNS = (
    "turn this into a summary",
    "summarize this",
    "summary of this",
    "summary of the document",
)

RAG_CONTEXT_HINTS = (
    "document",
    "pdf",
    "notes",
    "chapter",
    "lesson",
    "page",
    "course",
    "file",
)

BREAK_COMMAND_HINTS = (
    "take a break",
    "need a break",
    "break time",
    "pause",
    "rest",
)

FATIGUE_HINTS = (
    "i am tired",
    "i'm tired",
    "im tired",
    "tired",
    "exhausted",
    "sleepy",
    "worn out",
    "burned out",
)


class IntentClassifier:
    """Predict intent and confidence from text."""

    def __init__(
        self,
        model_dir: str = MODEL_DIR,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        margin_threshold: float = MARGIN_THRESHOLD,
    ):
        vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
        classifier_path = os.path.join(model_dir, "classifier.joblib")

        if not os.path.exists(vectorizer_path) or not os.path.exists(classifier_path):
            raise FileNotFoundError(
                "Intent models not found. Run 'python intent/train_intent.py' first."
            )

        self.vectorizer = joblib.load(vectorizer_path)
        self.classifier = joblib.load(classifier_path)
        self.classes_ = [str(label) for label in self.classifier.classes_]
        self._class_to_index = {
            label: index for index, label in enumerate(self.classes_)
        }
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold

    def _boost_probability(self, probabilities: np.ndarray, label: str, boost: float) -> None:
        index = self._class_to_index.get(label)
        if index is None:
            return
        probabilities[index] += boost * (1.0 - probabilities[index])

    def _matches_navigation_rule(self, lowered: str) -> bool:
        has_navigation_verb = any(term in lowered for term in NAVIGATE_VERBS)
        has_navigation_hint = any(term in lowered for term in NAVIGATE_HINTS)
        has_numbered_units = bool(
            re.search(
                r"\b\d+(?:\.\d+)?\s*(?:m|meter|meters|degree|degrees|°)\b",
                lowered,
            )
        )

        # Avoid over-biasing session statements like "turn off session".
        mentions_session = any(term in lowered for term in SESSION_TERMS)
        if mentions_session and not (has_navigation_hint or has_numbered_units):
            return False

        return (has_navigation_verb and has_navigation_hint) or has_numbered_units

    def _matches_rag_rule(self, lowered: str) -> bool:
        has_trigger = any(term in lowered for term in RAG_TRIGGERS)
        has_summary_pattern = any(pattern in lowered for pattern in RAG_SUMMARY_PATTERNS)
        has_context_hint = any(term in lowered for term in RAG_CONTEXT_HINTS)
        return has_trigger or has_summary_pattern or (
            has_context_hint and ("what" in lowered or "explain" in lowered)
        )

    def _matches_strong_rag_rule(self, lowered: str) -> bool:
        return any(pattern in lowered for pattern in RAG_SUMMARY_PATTERNS)

    def _matches_break_rule(self, lowered: str) -> bool:
        has_break_command = any(term in lowered for term in BREAK_COMMAND_HINTS)
        has_break_token = bool(re.search(r"\b(?:break|breaks|brake|brakes|rest|pause)\b", lowered))

        return has_break_command or has_break_token

    def _matches_fatigue_rule(self, lowered: str) -> bool:
        return any(term in lowered for term in FATIGUE_HINTS)

    def _apply_rule_boosts(self, text: str, probabilities: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        boosted = probabilities.astype(float).copy()
        lowered = text.lower()
        applied_rules: List[str] = []

        fatigue_rule_matched = self._matches_fatigue_rule(lowered)
        break_rule_matched = self._matches_break_rule(lowered)
        if fatigue_rule_matched:
            self._boost_probability(boosted, "BREAK", FATIGUE_BREAK_BOOST)
            applied_rules.append("BREAK_FATIGUE")
        elif break_rule_matched:
            self._boost_probability(boosted, "BREAK", BREAK_BOOST)
            applied_rules.append("BREAK")

        if self._matches_navigation_rule(lowered):
            self._boost_probability(boosted, "NAVIGATE", NAVIGATE_BOOST)
            applied_rules.append("NAVIGATE")

        if fatigue_rule_matched or break_rule_matched:
            # Break/fatigue commands should not be pulled into RAG by broad triggers.
            pass
        elif self._matches_strong_rag_rule(lowered):
            self._boost_probability(boosted, "RAG_QUERY", RAG_QUERY_STRONG_BOOST)
            applied_rules.append("RAG_QUERY_STRONG")
        elif self._matches_rag_rule(lowered):
            self._boost_probability(boosted, "RAG_QUERY", RAG_QUERY_BOOST)
            applied_rules.append("RAG_QUERY")

        boosted = np.clip(boosted, 1e-9, None)
        boosted /= boosted.sum()
        return boosted, applied_rules

    def _top_k_predictions(
        self, probabilities: np.ndarray, k: int = 3
    ) -> List[Tuple[str, float]]:
        top_indices = np.argsort(probabilities)[::-1][:k]
        return [
            (self.classes_[index], float(probabilities[index]))
            for index in top_indices
        ]

    def predict(self, text: str) -> Dict[str, object]:
        """
        Classify text into an intent.

        Returns
        -------
        dict with keys:
            intent : str
            confidence : float
            margin : float
            top3 : list
            unknown_reason : str | None
            rule_boosts : list
        """
        X = self.vectorizer.transform([text])
        raw_probabilities = self.classifier.predict_proba(X)[0]
        probabilities, rule_boosts = self._apply_rule_boosts(text, raw_probabilities)

        top3 = self._top_k_predictions(probabilities, k=3)
        top_intent, top_confidence = top3[0]
        second_confidence = top3[1][1] if len(top3) > 1 else 0.0
        margin = float(top_confidence - second_confidence)

        intent = top_intent
        unknown_reason = None
        if margin < self.margin_threshold:
            fatigue_break_applied = "BREAK_FATIGUE" in rule_boosts
            if not (
                top_intent == "BREAK"
                and fatigue_break_applied
                and top_confidence >= 0.4
            ):
                unknown_reason = "margin"
                intent = "UNKNOWN"
        elif top_confidence < self.confidence_threshold:
            rag_rule_applied = (
                "RAG_QUERY" in rule_boosts or "RAG_QUERY_STRONG" in rule_boosts
            )
            rag_relaxed_floor = max(0.35, self.confidence_threshold - 0.2)
            if not (
                top_intent == "RAG_QUERY"
                and rag_rule_applied
                and top_confidence >= rag_relaxed_floor
            ):
                unknown_reason = "confidence"
                intent = "UNKNOWN"

        return {
            "intent": intent,
            "confidence": round(float(top_confidence), 2),
            "margin": round(margin, 2),
            "unknown_reason": unknown_reason,
            "rule_boosts": rule_boosts,
            "top3": [
                {"intent": label, "confidence": round(confidence, 4)}
                for label, confidence in top3
            ],
        }


if __name__ == "__main__":
    clf = IntentClassifier()
    test_sentences = [
        "start session",
        "move forward 2 meters and turn 90 degrees",
        "show me my statistics",
        "take a break",
        "stop the session",
        "hello how are you",
        "what is machine learning from my notes",
    ]
    for sentence in test_sentences:
        result = clf.predict(sentence)
        print(f"'{sentence}' → {result}")
