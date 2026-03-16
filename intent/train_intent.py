"""
Train a TF-IDF + Logistic Regression intent classifier.
Saves vectorizer.joblib and classifier.joblib to the intent/ directory.
"""

import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# ---------------------------------------------------------------------------
# Training data – minimum 20-30 examples per intent
# ---------------------------------------------------------------------------
TRAINING_DATA = [
    # START_SESSION
    ("start session", "START_SESSION"),
    ("begin session", "START_SESSION"),
    ("start studying", "START_SESSION"),
    ("begin study mode", "START_SESSION"),
    ("start the session", "START_SESSION"),
    ("let's start", "START_SESSION"),
    ("start working", "START_SESSION"),
    ("begin working", "START_SESSION"),
    ("start a new session", "START_SESSION"),
    ("initiate session", "START_SESSION"),
    ("open session", "START_SESSION"),
    ("launch session", "START_SESSION"),
    ("kick off session", "START_SESSION"),
    ("start study session", "START_SESSION"),
    ("begin a study session", "START_SESSION"),
    ("start my session", "START_SESSION"),
    ("time to study", "START_SESSION"),
    ("let's begin", "START_SESSION"),
    ("start the study timer", "START_SESSION"),
    ("activate session", "START_SESSION"),
    ("commence session", "START_SESSION"),
    ("start focus mode", "START_SESSION"),
    ("begin focus session", "START_SESSION"),
    ("turn on study mode", "START_SESSION"),
    ("i want to start studying", "START_SESSION"),

    # STOP_SESSION
    ("stop session", "STOP_SESSION"),
    ("end session", "STOP_SESSION"),
    ("stop studying", "STOP_SESSION"),
    ("end study mode", "STOP_SESSION"),
    ("finish session", "STOP_SESSION"),
    ("terminate session", "STOP_SESSION"),
    ("close session", "STOP_SESSION"),
    ("stop the session", "STOP_SESSION"),
    ("end the session", "STOP_SESSION"),
    ("quit session", "STOP_SESSION"),
    ("shut down session", "STOP_SESSION"),
    ("i'm done studying", "STOP_SESSION"),
    ("done for now", "STOP_SESSION"),
    ("finish studying", "STOP_SESSION"),
    ("stop working", "STOP_SESSION"),
    ("end working", "STOP_SESSION"),
    ("wrap up session", "STOP_SESSION"),
    ("that's enough", "STOP_SESSION"),
    ("stop the study timer", "STOP_SESSION"),
    ("deactivate session", "STOP_SESSION"),
    ("i want to stop", "STOP_SESSION"),
    ("let's stop", "STOP_SESSION"),
    ("time to stop", "STOP_SESSION"),
    ("halt session", "STOP_SESSION"),
    ("cancel session", "STOP_SESSION"),

    # GET_STATS
    ("get stats", "GET_STATS"),
    ("show statistics", "GET_STATS"),
    ("display stats", "GET_STATS"),
    ("how am i doing", "GET_STATS"),
    ("show my progress", "GET_STATS"),
    ("what are my stats", "GET_STATS"),
    ("give me statistics", "GET_STATS"),
    ("show session stats", "GET_STATS"),
    ("how long have i studied", "GET_STATS"),
    ("session statistics", "GET_STATS"),
    ("progress report", "GET_STATS"),
    ("show report", "GET_STATS"),
    ("get my statistics", "GET_STATS"),
    ("display statistics", "GET_STATS"),
    ("what's my progress", "GET_STATS"),
    ("tell me my stats", "GET_STATS"),
    ("how much time", "GET_STATS"),
    ("show time spent", "GET_STATS"),
    ("study report", "GET_STATS"),
    ("performance stats", "GET_STATS"),
    ("check stats", "GET_STATS"),
    ("view statistics", "GET_STATS"),
    ("how many sessions", "GET_STATS"),
    ("summary of sessions", "GET_STATS"),
    ("give me a summary", "GET_STATS"),

    # BREAK
    ("take a break", "BREAK"),
    ("i need a break", "BREAK"),
    ("break time", "BREAK"),
    ("let's take a break", "BREAK"),
    ("pause session", "BREAK"),
    ("pause", "BREAK"),
    ("rest", "BREAK"),
    ("i need rest", "BREAK"),
    ("time for a break", "BREAK"),
    ("short break", "BREAK"),
    ("break please", "BREAK"),
    ("give me a break", "BREAK"),
    ("pause the session", "BREAK"),
    ("let me rest", "BREAK"),
    ("take five", "BREAK"),
    ("pause studying", "BREAK"),
    ("hold on", "BREAK"),
    ("rest time", "BREAK"),
    ("stop for a break", "BREAK"),
    ("quick break", "BREAK"),
    ("i want a break", "BREAK"),
    ("break now", "BREAK"),
    ("chill for a bit", "BREAK"),
    ("relax time", "BREAK"),
    ("timeout", "BREAK"),

    # NAVIGATE
    ("move forward 2 meters", "NAVIGATE"),
    ("go forward 5 meters", "NAVIGATE"),
    ("turn left 90 degrees", "NAVIGATE"),
    ("turn right 45 degrees", "NAVIGATE"),
    ("move forward 3 meters and turn 60 degrees", "NAVIGATE"),
    ("go ahead 1 meter", "NAVIGATE"),
    ("drive forward 10 meters", "NAVIGATE"),
    ("navigate forward 4 meters", "NAVIGATE"),
    ("move 2 meters ahead", "NAVIGATE"),
    ("go 7 meters forward", "NAVIGATE"),
    ("move forward 1.5 meters and turn left 30 degrees", "NAVIGATE"),
    ("turn 180 degrees", "NAVIGATE"),
    ("rotate 90 degrees right", "NAVIGATE"),
    ("go straight 6 meters", "NAVIGATE"),
    ("move ahead 8 meters", "NAVIGATE"),
    ("travel forward 3.5 meters", "NAVIGATE"),
    ("advance 2 meters", "NAVIGATE"),
    ("proceed 5 meters forward", "NAVIGATE"),
    ("move forward and turn 45 degrees", "NAVIGATE"),
    ("go 3 meters and rotate 120 degrees", "NAVIGATE"),
    ("drive 4 meters ahead", "NAVIGATE"),
    ("move to the left 2 meters", "NAVIGATE"),
    ("go right 3 meters", "NAVIGATE"),
    ("navigate 10 meters north", "NAVIGATE"),
    ("head forward 1 meter and turn right 90 degrees", "NAVIGATE"),
]

MODEL_DIR = os.path.dirname(__file__)


def train_and_save():
    """Train the classifier and save models to disk."""
    texts = [t for t, _ in TRAINING_DATA]
    labels = [l for _, l in TRAINING_DATA]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X, labels)

    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
    joblib.dump(classifier, os.path.join(MODEL_DIR, "classifier.joblib"))

    print(f"[TRAIN] Trained on {len(texts)} samples.")
    print(f"[TRAIN] Intents: {sorted(set(labels))}")
    print(f"[TRAIN] Models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    train_and_save()
