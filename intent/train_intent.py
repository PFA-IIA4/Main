"""
Train a TF-IDF + Logistic Regression intent classifier.
The dataset is generated at training time using structured augmentation.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from intent.data_augmentation import (
    DEFAULT_RANDOM_SEED,
    INTENT_ORDER,
    TARGET_SAMPLES_PER_INTENT,
    generate_dataset,
)

MODEL_DIR = Path(__file__).resolve().parent
GENERATED_DATASET_PATH = MODEL_DIR / "generated_dataset.json"
SAVE_GENERATED_DATASET = "--no-save-dataset" not in sys.argv

AMBIGUOUS_PROBES: List[Tuple[str, str]] = [
    ("go ahead a bit then turn right", "NAVIGATE"),
    ("start studying now please", "START_SESSION"),
    ("turn this into a summary", "RAG_QUERY (main pre-route), classifier may be UNKNOWN"),
    ("move forward", "NAVIGATE (classifier), UNKNOWN after entity validation"),
    ("explain pid controller", "RAG_QUERY (main pre-route), classifier may be low-confidence"),
]


def _print_confusion_matrix(labels: Sequence[str], matrix) -> None:
    print("[EVAL] Confusion matrix (rows=true, cols=pred):")
    header = "true\\pred".ljust(16) + " ".join(label.ljust(14) for label in labels)
    print(header)
    for label, row in zip(labels, matrix):
        row_text = " ".join(str(value).ljust(14) for value in row)
        print(label.ljust(16) + row_text)


def _evaluate_classifier(eval_data: Sequence[Tuple[str, str]]) -> None:
    from intent.intent_classifier import IntentClassifier

    clf = IntentClassifier(model_dir=str(MODEL_DIR))

    true_labels: List[str] = []
    predicted_labels: List[str] = []
    correct_per_intent: Dict[str, int] = Counter()
    total_per_intent: Dict[str, int] = Counter()

    unknown_count = 0
    unknown_margin_count = 0
    unknown_confidence_count = 0

    navigate_predicted = 0
    navigate_true_positive = 0

    for text, true_intent in eval_data:
        result = clf.predict(text)
        predicted_intent = result["intent"]
        unknown_reason = result.get("unknown_reason")

        true_labels.append(true_intent)
        predicted_labels.append(predicted_intent)

        total_per_intent[true_intent] += 1
        if predicted_intent == true_intent:
            correct_per_intent[true_intent] += 1

        if predicted_intent == "UNKNOWN":
            unknown_count += 1
            if unknown_reason == "margin":
                unknown_margin_count += 1
            elif unknown_reason == "confidence":
                unknown_confidence_count += 1

        if predicted_intent == "NAVIGATE":
            navigate_predicted += 1
            if true_intent == "NAVIGATE":
                navigate_true_positive += 1

    total_samples = max(len(eval_data), 1)
    overall_accuracy = sum(
        1 for true_intent, predicted_intent in zip(true_labels, predicted_labels) if true_intent == predicted_intent
    ) / total_samples
    unknown_rate = unknown_count / total_samples
    false_unknown_rate = unknown_count / total_samples
    margin_trigger_rate = unknown_margin_count / total_samples
    navigate_precision = (
        navigate_true_positive / navigate_predicted if navigate_predicted else 0.0
    )

    print("[EVAL] Overall accuracy:", round(overall_accuracy, 4))
    print("[EVAL] Unknown rate:", round(unknown_rate, 4))
    print("[EVAL] False unknown rate:", round(false_unknown_rate, 4))
    print("[EVAL] Margin-trigger UNKNOWN rate:", round(margin_trigger_rate, 4))
    print("[EVAL] Confidence-trigger UNKNOWN count:", unknown_confidence_count)
    print("[EVAL] NAVIGATE precision:", round(navigate_precision, 4))

    print("[EVAL] Per-intent accuracy:")
    for intent in INTENT_ORDER:
        total = total_per_intent.get(intent, 0)
        correct = correct_per_intent.get(intent, 0)
        accuracy = (correct / total) if total else 0.0
        print(f"  - {intent}: {accuracy:.4f} ({correct}/{total})")

    matrix_labels = list(INTENT_ORDER) + ["UNKNOWN"]
    matrix = confusion_matrix(true_labels, predicted_labels, labels=matrix_labels)
    _print_confusion_matrix(matrix_labels, matrix)

    print("[EVAL] Ambiguous phrase probes:")
    for phrase, expected in AMBIGUOUS_PROBES:
        prediction = clf.predict(phrase)
        print(
            f"  - '{phrase}' -> {prediction['intent']} "
            f"(expected~{expected}, conf={prediction['confidence']}, margin={prediction['margin']}, reason={prediction['unknown_reason']})"
        )


def train_and_save(
    target_per_intent: int = TARGET_SAMPLES_PER_INTENT,
    random_seed: int = DEFAULT_RANDOM_SEED,
    save_dataset: bool = SAVE_GENERATED_DATASET,
) -> None:
    """Train the classifier and save models to disk."""
    training_data = generate_dataset(
        target_per_intent=target_per_intent,
        random_seed=random_seed,
        save_path=GENERATED_DATASET_PATH if save_dataset else None,
    )

    texts = [text for text, _ in training_data]
    labels = [label for _, label in training_data]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    X = vectorizer.fit_transform(texts)

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_seed,
    )
    classifier.fit(X, labels)

    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.joblib")
    joblib.dump(classifier, MODEL_DIR / "classifier.joblib")

    label_counts = Counter(labels)
    print(f"[TRAIN] Generated dataset size: {len(texts)}")
    print(f"[TRAIN] Intents: {sorted(set(labels))}")
    print(f"[TRAIN] Samples per intent: {dict(sorted(label_counts.items()))}")
    if save_dataset:
        print(f"[TRAIN] Generated dataset saved to {GENERATED_DATASET_PATH}")
    print(f"[TRAIN] Models saved to {MODEL_DIR}/")

    eval_target_per_intent = max(120, target_per_intent // 20)
    evaluation_data = generate_dataset(
        target_per_intent=eval_target_per_intent,
        random_seed=random_seed + 999,
        save_path=None,
    )
    _evaluate_classifier(evaluation_data)


if __name__ == "__main__":
    train_and_save()
