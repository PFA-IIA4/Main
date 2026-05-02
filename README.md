# Offline Voice-Controlled Robotic System

An offline-first voice-controlled robot that listens to spoken commands, understands them locally with a lightweight LLM, and acts on them on a Raspberry Pi.

The system keeps speech recognition, intent classification, entity extraction, and action dispatch fully local. Only the chatbot path can optionally call a cloud API when the classifier returns `UNKNOWN`.

---

## Architecture

```
🎤 Microphone
     │
     ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  1. STT     │────▶│  2. Intent       │────▶│  3. Entity       │
│  (Vosk)     │     │  Classifier      │     │  Extractor       │
│  audio→text │     │  (TinyLlama)     │     │  text→params     │
└─────────────┘     └──────────────────┘     └────────┬─────────┘
                                                       │
                                                       ▼
                                             ┌──────────────────┐
                                             │  4. Dispatcher   │
                                             │  intent→action   │
                                             └────────┬─────────┘
                                                      │
                    ┌────────────┬────────────┬────────┴────────┐
                    ▼            ▼            ▼                 ▼
               🚗 Move     📚 Session    📊 Stats        💬 Chatbot
               (ESP32)     (start/stop)  (display)       (UNKNOWN)
```

## How It Works

### Stage 1: Speech-to-Text
`stt/vosk_stt.py` uses Vosk and `sounddevice` to stream microphone audio into text. It runs fully offline.

### Stage 2: Intent Classification
`intent/llm_classifier.py` and `intent/intent_classifier.py` use TinyLlama 1.1B Chat through `llama.cpp` to classify the text into one of these intents:

- `START_SESSION`
- `STOP_SESSION`
- `GET_STATS`
- `BREAK`
- `NAVIGATE`
- `RAG_QUERY`
- `UNKNOWN`

The classifier returns a structured result with the predicted intent, confidence, reason, token count, inference time, and `model_used`.

The model is prompted to return strict JSON so the app can parse it reliably. The wrapper also caches repeated inputs in memory so identical commands do not trigger a second model run.

### Stage 3: Entity Extraction
`entity/entity_extractor.py` only runs for `NAVIGATE`. It extracts distance and angle values from the recognized text.

### Stage 4: Action Dispatcher
`action/dispatcher.py` maps the intent to a robot action or session action.

### Stage 5: Chatbot Handling
`chatbot/chatbot_handler.py` is only used when the dispatcher returns `CHATBOT_FALLBACK`. It can call Hugging Face if configured, otherwise it uses a local rule-based response.

---

## Project Structure

```
├── stt/
│   ├── vosk_stt.py
│   └── vosk_model/
│
├── intent/
│   ├── llm_classifier.py
│   ├── intent_classifier.py
│   └── README.md
│
├── entity/
│   └── entity_extractor.py
│
├── action/
│   └── dispatcher.py
│
├── chatbot/
│   └── chatbot_handler.py
│
├── llama.cpp/
│   ├── build/
│   ├── models/
│   └── ...
│
├── RAG-/
│   └── ...
│
├── main.py
├── requirements.txt
├── requirements_llm.txt
├── spec_v2.md
└── README.md
```

---

## Installation

### 1. Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build llama.cpp

From the `llama.cpp` directory:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### 3. Download the Model

The model is already present in this workspace at:

`llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`

If you need to download it again, use the Hugging Face GGUF variant of TinyLlama 1.1B Chat.

---

## Environment Variables

```bash
USE_LLM_CLASSIFIER=true
LLM_MODEL_PATH=./llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
LLAMA_BIN_PATH=./llama.cpp/build/bin/llama-cli
LLM_MAX_TOKENS=150
LLM_TEMPERATURE=0.3
LLM_INFERENCE_TIMEOUT=5

RAG_BASE_URL=http://127.0.0.1:8000
RAG_ASK_PATH=/ask
RAG_TIMEOUT_SECONDS=10

HUGGINGFACE_API_KEY=your_huggingface_api_key_here
HUGGINGFACE_API_URL=https://router.huggingface.co/v1/chat/completions
HUGGINGFACE_MODEL=Qwen/Qwen2.5-7B-Instruct
HUGGINGFACE_TIMEOUT_SECONDS=20
```

Replace `HUGGINGFACE_API_KEY` with your own secret value before using the chatbot API.

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

Text mode is the easiest way to verify the full local pipeline.

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
- The intent classifier runs locally through `llama.cpp`.
- The cache keeps repeated commands cheaper.
- If the model times out or returns invalid JSON, the classifier returns `UNKNOWN` and the chatbot path can take over.

---

## Status

- STT: local and offline
- Intent classification: local LLM only
- Entity extraction: local regex
- Dispatch: local Python
- Chatbot: optional cloud call, otherwise local rule-based response
