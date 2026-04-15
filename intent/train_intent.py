"""
Train a TF-IDF + Logistic Regression intent classifier.
Saves vectorizer.joblib and classifier.joblib to the intent/ directory.
"""

import os
from collections import Counter

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TARGET_SAMPLES_PER_INTENT = 120

# ---------------------------------------------------------------------------
# Base intent examples.
# RAG_QUERY includes 90 explicit samples and is augmented to 120.
# ---------------------------------------------------------------------------
INTENT_EXAMPLES = {
    "START_SESSION": [
        "start session",
        "begin session",
        "start studying",
        "begin study mode",
        "start the session",
        "let's start",
        "start working",
        "begin working",
        "start a new session",
        "initiate session",
        "open session",
        "launch session",
        "kick off session",
        "start study session",
        "begin a study session",
        "start my session",
        "time to study",
        "let's begin",
        "start the study timer",
        "activate session",
        "commence session",
        "start focus mode",
        "begin focus session",
        "turn on study mode",
        "i want to start studying",
    ],
    "STOP_SESSION": [
        "stop session",
        "end session",
        "stop studying",
        "end study mode",
        "finish session",
        "terminate session",
        "close session",
        "stop the session",
        "end the session",
        "quit session",
        "shut down session",
        "i'm done studying",
        "done for now",
        "finish studying",
        "stop working",
        "end working",
        "wrap up session",
        "that's enough",
        "stop the study timer",
        "deactivate session",
        "i want to stop",
        "let's stop",
        "time to stop",
        "halt session",
        "cancel session",
    ],
    "GET_STATS": [
        "get stats",
        "show statistics",
        "display stats",
        "how am i doing",
        "show my progress",
        "what are my stats",
        "give me statistics",
        "show session stats",
        "how long have i studied",
        "session statistics",
        "progress report",
        "show report",
        "get my statistics",
        "display statistics",
        "what's my progress",
        "tell me my stats",
        "how much time",
        "show time spent",
        "study report",
        "performance stats",
        "check stats",
        "view statistics",
        "how many sessions",
        "summary of sessions",
        "give me a summary",
    ],
    "BREAK": [
        "take a break",
        "i need a break",
        "break time",
        "let's take a break",
        "pause session",
        "pause",
        "rest",
        "i need rest",
        "time for a break",
        "short break",
        "break please",
        "give me a break",
        "pause the session",
        "let me rest",
        "take five",
        "pause studying",
        "hold on",
        "rest time",
        "stop for a break",
        "quick break",
        "i want a break",
        "break now",
        "chill for a bit",
        "relax time",
        "timeout",
    ],
    "NAVIGATE": [
        "move forward 2 meters",
        "go forward 5 meters",
        "turn left 90 degrees",
        "turn right 45 degrees",
        "move forward 3 meters and turn 60 degrees",
        "go ahead 1 meter",
        "drive forward 10 meters",
        "navigate forward 4 meters",
        "move 2 meters ahead",
        "go 7 meters forward",
        "move forward 1.5 meters and turn left 30 degrees",
        "turn 180 degrees",
        "rotate 90 degrees right",
        "go straight 6 meters",
        "move ahead 8 meters",
        "travel forward 3.5 meters",
        "advance 2 meters",
        "proceed 5 meters forward",
        "move forward and turn 45 degrees",
        "go 3 meters and rotate 120 degrees",
        "drive 4 meters ahead",
        "move to the left 2 meters",
        "go right 3 meters",
        "navigate 10 meters north",
        "head forward 1 meter and turn right 90 degrees",
    ],
    "RAG_QUERY": [
        "what is PID",
        "define pid",
        "explain pid controller",
        "can you explain the concept of pid controller from my uploaded course notes",
        "what is machine learning from my notes",
        "summarize chapter 2",
        "summarize chapter two from my course pdf",
        "what does this document say",
        "what is written in the pdf",
        "give me a summary of this lesson",
        "what did I upload",
        "tell me about the content of my files",
        "explain the topic from the document",
        "summarize the uploaded pdf",
        "what is explained in page 5",
        "give details about this chapter",
        "define convolution",
        "explain convolution from my notes",
        "what is backpropagation in my course material",
        "summarize lesson 4",
        "explain this chapter in simple words",
        "what does page 10 talk about",
        "can you summarize my document",
        "summarize my document",
        "explain from my course",
        "tell me what this file is about",
        "what are the key points in this pdf",
        "extract the main idea from chapter 1",
        "what is the definition of gradient descent in the notes",
        "give me the explanation from the uploaded file",
        "explain the content of page 3",
        "what does the lesson say about neural networks",
        "tell me about chapter 6 in the document",
        "summarize the section about classification",
        "what did the pdf mention about overfitting",
        "explain regularization according to my notes",
        "what is in the uploaded document",
        "what does this course file explain",
        "give me a quick summary of the document",
        "provide a detailed summary of chapter 7",
        "answer from my uploaded notes what is reinforcement learning",
        "explain this topic based on my pdf",
        "in my notes what is linear regression",
        "what is described on page 2",
        "tell me the important points from this lesson",
        "summarize the notes i uploaded",
        "can you read my file and explain the topic",
        "explain the meaning of convolutional layer from my notes",
        "what does chapter 3 cover",
        "clarify what is written in my course document",
        "define entropy from the document",
        "explain accuracy versus precision from my notes",
        "give me the summary for page 8",
        "tell me what the uploaded pdf says about data preprocessing",
        "from my files explain supervised learning",
        "what is the topic of this chapter",
        "explain this paragraph from the lesson",
        "summarize the content i uploaded",
        "what is discussed in lesson 2",
        "give details from page 12",
        "explain the formula in chapter 5 from my notes",
        "what does my document say about logistic regression",
        "summarize this chapter for me",
        "define matrix multiplication from my pdf notes",
        "tell me about the concept explained in the uploaded file",
        "what does the course document mention about activation functions",
        "explain the uploaded lesson in brief",
        "give me a concise summary of my notes",
        "i uploaded a pdf explain what it contains",
        "what did my notes say about model evaluation",
        "summarize the pdf chapter about optimization",
        "explain the chapter on feature engineering",
        "what is this lesson saying about bias and variance",
        "can you answer using my uploaded document what is svm",
        "tell me the content of page 15 in my file",
        "explain from the document what clustering means",
        "what is explained in my lecture notes about normalization",
        "give me the main takeaway from this course file",
        "what does the uploaded material say about cross validation",
        "define overfitting from my uploaded notes",
        "explain underfitting using my course document",
        "summarize all key points from this lesson file",
        "what are the definitions provided in chapter 9",
        "tell me about the text in my pdf",
        "explain from my notes what a confusion matrix is",
        "what is written in my uploaded course notes about recall",
        "summarize the topic discussed on page 20",
        "give details about the uploaded chapter on probability",
        "explain the content of my study document",
        "what does this uploaded file say about machine learning basics",
    ],
}

INTENT_AUGMENTATION = {
    "START_SESSION": {
        "prefixes": ["please", "can you", "robot", "kindly", "assistant"],
        "suffixes": ["now", "please", "for me"],
    },
    "STOP_SESSION": {
        "prefixes": ["please", "can you", "robot", "kindly", "assistant"],
        "suffixes": ["now", "please", "for me"],
    },
    "GET_STATS": {
        "prefixes": ["please", "can you", "robot", "kindly", "assistant"],
        "suffixes": ["for me", "please", "right now"],
    },
    "BREAK": {
        "prefixes": ["please", "can you", "robot", "kindly", "assistant"],
        "suffixes": ["now", "please", "for me"],
    },
    "NAVIGATE": {
        "prefixes": ["please", "can you", "robot", "quickly", "assistant"],
        "suffixes": ["now", "please", "for me"],
    },
    "RAG_QUERY": {
        "prefixes": ["please", "can you", "based on my notes"],
        "suffixes": ["from my uploaded pdf", "using the document"],
    },
}

INTENT_ORDER = [
    "START_SESSION",
    "STOP_SESSION",
    "GET_STATS",
    "BREAK",
    "NAVIGATE",
    "RAG_QUERY",
]

MODEL_DIR = os.path.dirname(__file__)


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _deduplicate_keep_order(texts):
    unique = []
    seen = set()
    for text in texts:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            continue
        key = _normalize_text(cleaned)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def _augment_examples(intent: str, examples, target_count: int):
    if len(examples) >= target_count:
        return examples[:target_count]

    config = INTENT_AUGMENTATION.get(intent, {})
    prefixes = config.get("prefixes", [])
    suffixes = config.get("suffixes", [])

    augmented = list(examples)
    seen = {_normalize_text(text) for text in augmented}

    def _add(candidate: str):
        normalized = _normalize_text(candidate)
        if normalized in seen:
            return
        seen.add(normalized)
        augmented.append(candidate)

    for prefix in prefixes:
        for text in examples:
            _add(f"{prefix} {text}")
            if len(augmented) >= target_count:
                return augmented[:target_count]

    for suffix in suffixes:
        for text in examples:
            _add(f"{text} {suffix}")
            if len(augmented) >= target_count:
                return augmented[:target_count]

    for prefix in prefixes:
        for suffix in suffixes:
            for text in examples:
                _add(f"{prefix} {text} {suffix}")
                if len(augmented) >= target_count:
                    return augmented[:target_count]

    raise ValueError(
        f"Could not build {target_count} unique samples for intent '{intent}'. "
        f"Generated only {len(augmented)} samples."
    )


def _build_training_data():
    rows = []
    for intent in INTENT_ORDER:
        examples = _deduplicate_keep_order(INTENT_EXAMPLES[intent])
        if intent == "RAG_QUERY" and len(examples) < 80:
            raise ValueError("RAG_QUERY must include at least 80 unique examples.")

        examples = _augment_examples(intent, examples, TARGET_SAMPLES_PER_INTENT)
        rows.extend((text, intent) for text in examples)

    return rows


def _validate_dataset(training_data):
    seen = {}
    for text, intent in training_data:
        key = _normalize_text(text)
        if key in seen:
            previous_intent = seen[key]
            raise ValueError(
                f"Duplicate text sample found ('{text}') with intents "
                f"'{previous_intent}' and '{intent}'."
            )
        seen[key] = intent

    counts = Counter(label for _, label in training_data)
    expected = set(INTENT_ORDER)
    if set(counts.keys()) != expected:
        raise ValueError("Dataset intents do not match expected intent set.")

    min_count = min(counts.values())
    max_count = max(counts.values())
    if min_count != max_count:
        raise ValueError(
            f"Dataset is not balanced. Min samples={min_count}, max samples={max_count}."
        )


TRAINING_DATA = _build_training_data()
_validate_dataset(TRAINING_DATA)


def train_and_save():
    """Train the classifier and save models to disk."""
    texts = [text for text, _ in TRAINING_DATA]
    labels = [label for _, label in TRAINING_DATA]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X, labels)

    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
    joblib.dump(classifier, os.path.join(MODEL_DIR, "classifier.joblib"))

    label_counts = Counter(labels)
    print(f"[TRAIN] Trained on {len(texts)} samples.")
    print(f"[TRAIN] Intents: {sorted(set(labels))}")
    print(f"[TRAIN] Samples per intent: {dict(sorted(label_counts.items()))}")
    print(f"[TRAIN] Models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    train_and_save()
