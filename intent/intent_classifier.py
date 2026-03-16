"""
Intent classifier using a pre-trained TF-IDF + Logistic Regression model.
Returns intent label and confidence score.
"""

import os
from typing import Dict

import joblib
import numpy as np

MODEL_DIR = os.path.dirname(__file__)
CONFIDENCE_THRESHOLD = 0.6


class IntentClassifier:
    """Predict intent and confidence from text."""

    def __init__(self, model_dir: str = MODEL_DIR):
        vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
        classifier_path = os.path.join(model_dir, "classifier.joblib")

        if not os.path.exists(vectorizer_path) or not os.path.exists(classifier_path):
            raise FileNotFoundError(
                "Intent models not found. Run 'python intent/train_intent.py' first."
            )

        self.vectorizer = joblib.load(vectorizer_path)
        self.classifier = joblib.load(classifier_path)

    def predict(self, text: str) -> Dict[str, object]:
        """
        Classify text into an intent.

        Returns
        -------
        dict with keys:
            intent : str
            confidence : float
        """
        X = self.vectorizer.transform([text])
        proba = self.classifier.predict_proba(X)[0]
        max_idx = int(np.argmax(proba))
        confidence = float(proba[max_idx])
        intent = self.classifier.classes_[max_idx]

        if confidence < CONFIDENCE_THRESHOLD:
            intent = "UNKNOWN"

        return {"intent": intent, "confidence": round(confidence, 2)}


if __name__ == "__main__":
    clf = IntentClassifier()
    test_sentences = [
        "start session",
        "move forward 2 meters and turn 90 degrees",
        "show me my statistics",
        "take a break",
        "stop the session",
        "hello how are you",
    ]
    for sentence in test_sentences:
        result = clf.predict(sentence)
        print(f"'{sentence}' → {result}")
