# Offline/Cloud Voice-Controlled Robotic System

An offline-STT voice-controlled robot that listens to spoken commands and handles intent classification, parameter extraction, and conversational responses using a unified Hugging Face API call on a Raspberry Pi.

---

## Architecture

```
🎤 Microphone
     │
     ▼
┌─────────────┐     ┌──────────────────┐
│  1. STT     │────▶│  2. Brain        │
│  (Vosk)     │     │  (HF Cloud API)  │
│  audio→text │     │  text→JSON       │
└─────────────┘     └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  3. Dispatcher   │
                    │  JSON→action     │
                    └────────┬─────────┘
                             │
            ┌────────────┬───┴────────┬────────────┐
            ▼            ▼            ▼            ▼
       🚗 Move     📚 Session    📊 Stats        💬 Chatbot
       (ESP32)     (start/stop)  (display)       (JSON response)
```

## How It Works

### Stage 1: Speech-to-Text
`stt/vosk_stt.py` uses Vosk and `sounddevice` to stream microphone audio into text. It runs fully offline.

### Stage 2: The Brain (Intent & Entity Extraction)
`intent/llm_classifier.py` and `intent/intent_classifier.py` use the Hugging Face Inference API (via `requests`) to classify the text into one of these intents:

- `START_SESSION`
- `STOP_SESSION`
- `RESUME_SESSION`
- `GET_STATS`
- `BREAK`
- `NAVIGATE`
- `RAG_QUERY`
- `CHATBOT`
- `UNKNOWN`

The API is instructed to return strict JSON containing the `intent`, `parameters` (like distance and angle for NAVIGATE), and a potential `response` (for conversational talk).

### Stage 3: Action Dispatcher
`action/dispatcher.py` handles the JSON output. If the intent is `CHATBOT`, it directly replies to the user using the provided `response`. Otherwise, it maps the intent and parameters to a robot or session action.

---

## Project Structure

```
├── stt/
│   ├── vosk_stt.py
│   └── vosk_model/
│
├── intent/
│   ├── llm_classifier.py
│   └── intent_classifier.py
│
├── action/
│   └── dispatcher.py
│
├── RAG-/
│   └── ...
│
├── main.py
├── requirements.txt
├── spec_v2.md
├── spec_v3.md
└── README.md
```

---

## Installation

### 1. Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up the API Key

You will need a Hugging Face API key to enable intent classification and conversational processing.

---

## Environment Variables

```bash
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
HUGGINGFACE_API_URL=https://router.huggingface.co/v1/chat/completions
HUGGINGFACE_MODEL=Qwen/Qwen2.5-7B-Instruct
HUGGINGFACE_TIMEOUT_SECONDS=90

RAG_BASE_URL=http://127.0.0.1:8000
RAG_ASK_PATH=/ask
RAG_TIMEOUT_SECONDS=10
```

Replace `HUGGINGFACE_API_KEY` with your own secret value.

---

## Running The App

### Voice Mode

```bash
python main.py
```

### Text Mode

```bash
python main.py --text
```

Text mode is the easiest way to verify the full API pipeline without a microphone.

---

## Quick Tests

```bash
python -c "from intent.intent_classifier import IntentClassifier; print(IntentClassifier().predict('start session'))"
python -c "from intent.intent_classifier import IntentClassifier; print(IntentClassifier().predict('move forward 3 meters'))"
pytest test_llm_classifier.py -v
```

---

## Raspberry Pi Notes

- Raspberry Pi 4/5 are supported targets.
- Speech-to-Text handles audio locally using Vosk.
- Intent classification offloads to Hugging Face, keeping Pi CPU usage very low.
- If the network fails, the classifier safely returns `UNKNOWN` to avoid crashes.

---

## Status

- STT: local and offline
- Intent classification: cloud API (Hugging Face)
- Entity extraction: cloud API parameter parsing
- Dispatch: local Python
- Chatbot: cloud API native response
