# 🤖 Offline Voice-Controlled Robotic System

An offline-first voice-controlled robot that listens to spoken commands, understands what you want, and acts on it — all running on a Raspberry Pi. The core speech pipeline stays local, while document-style questions can be routed to an external RAG backend when configured.

Say *"move forward 3 meters and turn 90 degrees"* and the robot moves. Say *"start session"* and it begins tracking your study time. Say *"what is PID"* and it forwards the question to the RAG service. Say something it doesn't understand, and a chatbot responds naturally.

---

## How It Works

The system is a **core 5-stage pipeline** with an additional rule-based RAG pre-route for document-style questions before intent classification:

```
🎤 Microphone
     │
     ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  1. STT     │────▶│  2. Intent       │────▶│  3. Entity       │
│  (Vosk)     │     │  Classifier      │     │  Extractor       │
│  audio→text │     │  text→intent     │     │  text→params     │
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
               (ESP32)     (start/stop)  (display)       (fallback)
```

### Stage 1 — Speech-to-Text (`stt/vosk_stt.py`)

**Technology:** [Vosk 0.22](https://alphacephei.com/vosk/) + [sounddevice](https://python-sounddevice.readthedocs.io/)

Your microphone streams audio at 16 kHz into the Vosk speech recognizer. Vosk runs **completely offline** — no cloud APIs, no internet required. As you speak, it produces:

- **Partial results** — shown live as you talk (e.g., `(listening) move forward two me...`)
- **Final results** — the complete sentence, passed to Stage 2

**How it connects to the next stage:** The final transcription text is sent to `process_text()` in `main.py`, which feeds it to the intent classifier.

### Stage 2 — Intent Classification (`intent/`)

**Technology:** [scikit-learn](https://scikit-learn.org/) (TF-IDF + Logistic Regression) + [joblib](https://joblib.readthedocs.io/)

The transcribed text is vectorized using **TF-IDF** (turns words into numbers based on importance), then classified by a **Logistic Regression** model into one of these intents:

| Intent | Example commands |
|--------|-----------------|
| `START_SESSION` | *"start session"*, *"begin studying"*, *"let's start"* |
| `STOP_SESSION` | *"stop session"*, *"I'm done studying"*, *"end session"* |
| `GET_STATS` | *"show statistics"*, *"how am I doing"*, *"progress report"* |
| `BREAK` | *"take a break"*, *"pause"*, *"I need rest"* |
| `NAVIGATE` | *"move forward 3 meters"*, *"turn left 90 degrees"* |
| `RAG_QUERY` | *"what is PID"*, *"summarize chapter 2"*, *"explain this document"* |
| `UNKNOWN` | Ambiguous or low-confidence predictions |

The model is trained on a generated dataset with **2000 samples per intent** (**12000 total** across six intents). Obvious document-style questions may be labeled as `RAG_QUERY` by `main.py` before the classifier runs. Each prediction returns a **confidence score** and a **margin**; by default, if the margin is below **0.12** or confidence drops below **0.58**, the intent is forced to `UNKNOWN`.

The rule-boosting layer is additive (not hard override), with small context-aware boosts for likely NAVIGATE and RAG queries.

**How it connects to the next stage:** The `{ intent, confidence, margin, unknown_reason }` result is passed to the entity extractor (for `NAVIGATE`) or directly to the dispatcher (for all other intents).

### Stage 3 — Entity Extraction (`entity/entity_extractor.py`)

**Technology:** Python `re` (regex)

Only activated when the intent is `NAVIGATE`. Uses regex patterns + number-word parsing to pull out:

| Entity | Unit | Pattern examples |
|--------|------|-----------------|
| **Distance** | meters | `"3 meters"`, `"move forward five"`, `"2.5m"`, `"one and a half meters"` |
| **Angle** | degrees | `"90 degrees"`, `"turn left forty five"`, `"rotate 180°"`, `"nineteen degrees"` |

Supported spoken-number forms include values like `"nineteen"`, `"forty five"`, `"one thousand twenty"`, `"one and a half"`, and `"three and a quarter"`.

**Critical rule:** If neither distance nor angle can be extracted, the intent is **downgraded to `UNKNOWN`**. This triggers the chatbot to ask *"Please provide distance and angle."*

**How it connects to the next stage:** The `{ distance, angle }` dict is passed alongside the intent to the dispatcher.

### Stage 4 — Action Dispatcher (`action/dispatcher.py`)

**Technology:** Pure Python (dict-based routing + in-memory state)

Maps each intent to a concrete action:

| Intent | What happens |
|--------|-------------|
| `START_SESSION` | Records start time, resets break counter |
| `STOP_SESSION` | Calculates elapsed time, accumulates total study seconds |
| `GET_STATS` | Returns total study time, break count, session status |
| `BREAK` | Increments break counter (only during active session) |
| `NAVIGATE` | Formats a movement command string (ESP32 placeholder) |
| `RAG_QUERY` | Forwards the text to an external HTTP RAG backend during an active session; otherwise asks the user to start a session first |
| `UNKNOWN` | Returns `"CHATBOT_FALLBACK"` sentinel → triggers Stage 5 |

Session state is kept in memory:

```python
{ "active": bool, "start_time": datetime, "breaks": int, "total_study_seconds": float }
```

The RAG backend is not part of this repository; configure it with `RAG_BASE_URL`, `RAG_ASK_PATH`, and `RAG_TIMEOUT_SECONDS`.

RAG queries are only answered while a session is active.

**How it connects to the next stage:** For `UNKNOWN` intents, the dispatcher returns the sentinel `"CHATBOT_FALLBACK"`, which tells `main.py` to invoke the chatbot.

### Stage 5 — Chatbot Fallback (`chatbot/chatbot_handler.py`)

**Technology:** Hugging Face cloud inference API + rule-based fallback

When the system can't understand a command, this module takes over:

1. **Primary:** Calls a Hugging Face cloud model with your API key.
2. **Fallback:** If the API call fails or is not configured, a rule-based responder handles greetings, help requests, and incomplete navigation commands.

> No local `transformers` model is loaded in the current implementation.

> The chatbot accepts both `HUGGINGFACE_*` and `CHATBOT_*` environment variable names.

---

## Console Output

Every command produces structured output:

```
You said: move forward 2 meters and turn 90 degrees
Intent: NAVIGATE
Confidence: 0.91
Distance: 2.0
Angle: 90.0
Action: Moving 2.0m forward | Turning 90.0° [command sent to ESP32]
```

```
You said: explain chapter 2 from my notes
Intent: RAG_QUERY
Confidence: 1.00
Action: [answer from the external RAG service]
```

```
You said: hello there
Intent: UNKNOWN
Confidence: 0.35
Action: Chatbot invoked
Response: Hello! I'm your study robot assistant. Try commands like 'start session' or 'move forward 2 meters'.
```

---

## Project Structure

```
├── stt/
│   ├── vosk_stt.py              # Microphone → text (Vosk streaming)
│   └── vosk_model/              # Place downloaded Vosk model here
│
├── intent/
│   ├── data_augmentation.py     # Structured dataset generator and noise/synonym utilities
│   ├── train_intent.py          # Train the classifier (run once)
│   ├── intent_classifier.py     # IntentClassifier — predict intent + confidence + margin
│   ├── vectorizer.joblib        # [generated] TF-IDF vectorizer
│   └── classifier.joblib        # [generated] Logistic Regression model
│
├── entity/
│   └── entity_extractor.py      # Regex-based distance/angle extraction
│
├── action/
│   └── dispatcher.py            # Intent → action mapping + session state + RAG forwarding
│
├── chatbot/
│   └── chatbot_handler.py       # Hugging Face API chatbot + rule-based fallback
│
├── main.py                      # Entry point — wires all stages together and pre-routes document questions
├── test_augmentation.py         # Prints augmentation samples, sizes, and noise examples
├── specification.md             # Detailed technical specification
└── README.md                    # This file
```

The external RAG backend is separate from this repository and is reached over HTTP.

---

## Technology Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Speech-to-Text | **Vosk 0.22** | Offline speech recognition from microphone |
| Audio capture | **sounddevice** | Streams mic audio to Vosk |
| Intent classification | **scikit-learn** | TF-IDF vectorization + Logistic Regression |
| External RAG backend | **requests** | HTTP forwarding for document-style questions |
| Model persistence | **joblib** | Save/load trained ML models |
| Entity extraction | **Python re** | Regex-based parameter parsing |
| Chatbot | **Hugging Face Inference API** | Cloud chatbot fallback for UNKNOWN intents |
| Numerical ops | **NumPy** | Probability array handling |
| Target platform | **Raspberry Pi 4/5** | Primary deployment target |
| Actuation | **ESP32** *(optional)* | Low-level motor control |

---

## Getting Started

### 0. Raspberry Pi 4/5 System Packages

Install audio and build prerequisites first:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv portaudio19-dev libatlas-base-dev
```

### 1. Install Python Dependencies

```bash
pip install vosk sounddevice scikit-learn joblib numpy requests
```

Or install from the project file:

```bash
pip install -r requirements.txt
```

### 2. Download the Vosk Model

Download a model from [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) — for example:

```bash
cd stt/
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip
mv vosk-model-en-us-0.22/* vosk_model/
```

### 3. Train the Intent Classifier

```bash
python intent/train_intent.py
```

Output:
```
[TRAIN] Generated dataset size: 12000
[TRAIN] Intents: ['BREAK', 'GET_STATS', 'NAVIGATE', 'RAG_QUERY', 'START_SESSION', 'STOP_SESSION']
[TRAIN] Samples per intent: {'BREAK': 2000, 'GET_STATS': 2000, 'NAVIGATE': 2000, 'RAG_QUERY': 2000, 'START_SESSION': 2000, 'STOP_SESSION': 2000}
[TRAIN] Generated dataset saved to intent/generated_dataset.json
[TRAIN] Models saved to intent/
[EVAL] Overall accuracy: ...
[EVAL] Unknown rate: ...
[EVAL] NAVIGATE precision: ...
[EVAL] Confusion matrix (rows=true, cols=pred):
```

### 4. Run the System

**Voice mode** (requires microphone + Vosk model):
```bash
python main.py
```

**Text mode** (no microphone needed — great for testing):
```bash
python main.py --text
```

> If voice mode fails (missing model or no mic), the system automatically falls back to text mode.

### 5. Configure Optional Cloud Integrations

Set the variables in your shell or in a local `.env` file before launching the app.

```bash
# Chatbot fallback
HUGGINGFACE_API_KEY=your_hf_api_key_here
HUGGINGFACE_API_URL=https://router.huggingface.co/v1/chat/completions
HUGGINGFACE_MODEL=Qwen/Qwen2.5-7B-Instruct
HUGGINGFACE_TIMEOUT_SECONDS=20

# External RAG backend
RAG_BASE_URL=http://127.0.0.1:8000
RAG_ASK_PATH=/ask
RAG_TIMEOUT_SECONDS=10
```

The chatbot also accepts `CHATBOT_API_KEY`, `CHATBOT_MODEL`, and `CHATBOT_TIMEOUT_SECONDS` as fallbacks.

You can switch models by changing `HUGGINGFACE_MODEL`.

### Raspberry Pi Readiness Notes

- **OS recommendation:** Use Raspberry Pi OS 64-bit for smoother package installs and better memory handling.
- **Raspberry Pi 5 (recommended):** Should run voice + intent + entity + dispatcher reliably, and cloud chatbot mode works well.
- **Raspberry Pi 4:** Works for this architecture and benefits from cloud chatbot mode.
- **Memory guidance:** 4 GB is workable; 8 GB is more comfortable for multitasking.
- **Offline behavior:** STT/intent/entity/dispatcher stay fully offline. Cloud chatbot mode requires internet only for UNKNOWN-intent responses.

---

## Testing Individual Modules

You can test each stage independently:

```bash
# Test speech-to-text (requires mic + Vosk model)
python stt/vosk_stt.py

# Test intent classification (requires trained model)
python intent/intent_classifier.py

# Test entity extraction
python entity/entity_extractor.py

# Test chatbot fallback
python chatbot/chatbot_handler.py
```

### Quick Test with Text Mode

The fastest way to test the full pipeline end-to-end:

```bash
# 1. Train the model
python intent/train_intent.py

# 2. Launch in text mode
python main.py --text

# 3. Inspect augmentation output
python test_augmentation.py
```

Then try these commands:

```
You: start session
You: move forward 3 meters and turn 90 degrees
You: move forward one and a half meters and turn nineteen degrees
You: turn one thousand twenty degrees
You: explain chapter 2 from my notes
You: how am I doing
You: take a break
You: hello there
You: stop session
You: quit
```

---

## How the Stages Interact (Data Flow)

Document-style questions are detected in `main.py` before intent classification and sent to the external RAG backend. Everything else follows the local intent → entity → dispatcher path shown below.

```
User speaks: "move forward 3 meters and turn 90 degrees"
                                    │
                                    ▼
           ┌─ vosk_stt.py ─────────────────────────────┐
           │  audio stream → "move forward 3 meters     │
           │                  and turn 90 degrees"       │
           └─────────────────────────┬──────────────────┘
                                     │  text (str)
                                     ▼
           ┌─ intent_classifier.py ─────────────────────┐
           │  TF-IDF transform → LogReg predict          │
           │  → { intent: "NAVIGATE", confidence: 0.91 } │
           └─────────────────────────┬──────────────────┘
                                     │  intent == NAVIGATE
                                     ▼
           ┌─ entity_extractor.py ──────────────────────┐
           │  regex match → { distance: 3.0,             │
           │                  angle: 90.0 }              │
           └─────────────────────────┬──────────────────┘
                                     │  { intent, entities }
                                     ▼
           ┌─ dispatcher.py ────────────────────────────┐
           │  NAVIGATE handler → "Moving 3.0m forward    │
           │  | Turning 90.0° [command sent to ESP32]"   │
           └─────────────────────────┬──────────────────┘
                                     │
                                     ▼
                              Console output
```

---

## License

See [LICENSE](LICENSE) for details.