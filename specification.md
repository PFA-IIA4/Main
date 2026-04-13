This project is an offline-first voice-controlled robotic system that integrates speech recognition, intent classification, entity extraction, and task execution. It is designed to run primarily on Raspberry Pi 4/5 with optional ESP32 for low-level actuation. A chatbot fallback handles unrecognized commands using the Hugging Face cloud inference API.

Features

Offline Speech Recognition

Uses Vosk 0.22 for converting live microphone audio to text.

Continuous real-time STT with partial result feedback.

Fully offline operation.

Intent Classification

Lightweight ML-based intent classifier (TF-IDF + LogisticRegression).

Recognizes intents:

START_SESSION

STOP_SESSION

GET_STATS

BREAK

NAVIGATE

UNKNOWN

Returns confidence scores.

Entity Extraction

Regex-based extraction for NAVIGATE:

Distance (meters)

Angle / Orientation (degrees)

Returns UNKNOWN if required entities are missing.

Action Dispatcher

Executes commands based on intent and extracted entities.

Example: NAVIGATE → move robot; START_SESSION → begin study mode.

Chatbot Fallback

For UNKNOWN intent, chatbot flow is:

1) Hugging Face cloud API call if HUGGINGFACE_API_KEY is configured.

2) Rule-based response if API is unavailable.

Enables conversational responses and guidance.

Console Behavior

Logs recognized speech, intent, confidence, and extracted entities.

Partial results displayed during live speech recognition.

Architecture
+----------------+
|  Microphone    |
+-------+--------+
        |
        v
+----------------+      +-------------------+
|   Speech-to-   | ---> | Intent Classifier  |
|    Text (Vosk) |      |  + Entity Extract |
+----------------+      +--------+----------+
                                |
                                v
                      +-------------------+
                      | Action Dispatcher |
                      +---------+---------+
                                |
       +------------------------+-------------------------+
       |                        |                         |
   Robot Motion              Study Mode                 Chatbot
   (via ESP32)               (session control)         (UNKNOWN intent)
Modules & File Structure
/project_root
│
├─ /stt/
│   ├─ vosk_stt.py           # Live microphone -> text
│   └─ vosk_model/           # Offline model files
│
├─ /intent/
│   ├─ train_intent.py       # Train classifier
│   ├─ intent_classifier.py  # Predict intent + confidence
│   └─ intent_model.joblib    # Saved ML model
│
├─ /entity/
│   └─ entity_extractor.py   # Regex-based entity extraction
│
├─ /action/
│   └─ dispatcher.py         # Maps intent+entity -> robot actions
│
├─ /chatbot/
│   └─ chatbot_handler.py    # Calls Hugging Face cloud API for UNKNOWN
│
├─ main.py                   # Integrates STT -> Intent -> Entities -> Dispatcher
└─ specification.md
Training Dataset Requirements

Minimum 20–30 examples per intent (expandable).

Text samples should reflect natural voice commands.

Separate training set for each intent.

Saved using joblib:

vectorizer.joblib → TF-IDF vectorizer

classifier.joblib → Logistic Regression classifier

Speech-to-Text Integration

Library: vosk

Functionality:

Continuous streaming from microphone.

Partial and final transcription.

Output: text → passed to intent classifier.

Intent Classification Details

Input: Text from STT

Model: TF-IDF + Logistic Regression

Output: { intent, confidence }

Confidence threshold = 0.6 → otherwise fallback to UNKNOWN.

Entity Extraction Details

Regex-based parser for NAVIGATE commands.

Extracted Fields:

Distance (float, meters)

Angle (float, degrees)

Failure: missing or invalid → intent set to UNKNOWN.

Action Dispatcher

Maps { intent, entities } → robot actions.

Examples:

NAVIGATE → call movement routines (via ESP32)

START_SESSION → log session start

STOP_SESSION → end session

GET_STATS → display session statistics

BREAK → trigger break sequence

UNKNOWN → call chatbot fallback

Chatbot Fallback

Trigger: Intent = UNKNOWN

Implementation:

Hugging Face cloud chat-completions endpoint (if API key configured)

Rule-based fallback responder

Accepts user text → returns chatbot response

Purpose: Handles unrecognized commands and casual conversation

Console / Log Behavior
User: "move forward 2 meters and turn 90 degrees"
You said: move forward 2 meters and turn 90 degrees
Intent: NAVIGATE
Confidence: 0.91
Distance: 2.0
Angle: 90.0

User: "move forward"
Intent: UNKNOWN
Confidence: 0.88
Action: Chatbot invoked
Response: "Please provide distance and angle."
Future Enhancements

Multi-turn dialogue memory

Text-to-Speech responses

ONNX model deployment for heavier ML models

Integration of additional sensors for more precise navigation

Adaptive learning: model improves from user corrections

Technical Stack

Python 3.11+

Raspberry Pi OS

Raspberry Pi 4/5 target devices

Vosk 0.22 – offline speech recognition

scikit-learn – intent classification

joblib – model persistence

Regex – entity extraction

Hugging Face Inference API (API key) – cloud chatbot fallback

Optional: ESP32 for low-level actuation

Raspberry Pi Deployment Notes

Pi 5 is recommended for best responsiveness.

Pi 4 is supported and works well with cloud chatbot mode.

Use Raspberry Pi OS 64-bit for better package compatibility and memory behavior.

Performance / Optimization Notes

Use lightweight ML to maintain real-time inference on Raspberry Pi.

Keep offline-first approach for STT and primary intent recognition.

Modular architecture allows easy replacement of:

Intent classifier

Entity parser

Chatbot backend