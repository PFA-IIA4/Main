# 🤖 Offline Voice-Controlled Robotic System

An offline-first voice-controlled robot that listens to spoken commands, understands what you want, and acts on it — all running on a Raspberry Pi.

Say *"move forward 3 meters and turn 90 degrees"* and the robot moves. Say *"start session"* and it begins tracking your study time. Say something it doesn't understand, and a chatbot responds naturally.

---

## How It Works

The system is a **5-stage pipeline** where each stage feeds the next:

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
| `UNKNOWN` | Anything not recognized with ≥ 60% confidence |

The model is trained on **125 labeled examples** (25 per intent). Each prediction returns a **confidence score** — if it's below **0.6**, the intent is forced to `UNKNOWN` regardless of the predicted class.

**How it connects to the next stage:** The `{ intent, confidence }` result is passed to the entity extractor (for `NAVIGATE`) or directly to the dispatcher (for all other intents).

### Stage 3 — Entity Extraction (`entity/entity_extractor.py`)

**Technology:** Python `re` (regex)

Only activated when the intent is `NAVIGATE`. Uses regex patterns to pull out:

| Entity | Unit | Pattern examples |
|--------|------|-----------------|
| **Distance** | meters | `"3 meters"`, `"move forward 5"`, `"2.5m"` |
| **Angle** | degrees | `"90 degrees"`, `"turn left 45"`, `"rotate 180°"` |

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
| `UNKNOWN` | Returns `"CHATBOT_FALLBACK"` sentinel → triggers Stage 5 |

Session state is kept in memory:

```python
{ "active": bool, "start_time": datetime, "breaks": int, "total_study_seconds": float }
```

**How it connects to the next stage:** For `UNKNOWN` intents, the dispatcher returns the sentinel `"CHATBOT_FALLBACK"`, which tells `main.py` to invoke the chatbot.

### Stage 5 — Chatbot Fallback (`chatbot/chatbot_handler.py`)

**Technology:** [Hugging Face Transformers](https://huggingface.co/docs/transformers/) (DialoGPT-small)

When the system can't understand a command, this module takes over:

1. **Primary:** Loads `microsoft/DialoGPT-small` (lazy-loaded on first use) and generates a conversational response.
2. **Fallback:** If HuggingFace is unavailable (no internet/library), a rule-based responder handles greetings, help requests, and incomplete navigation commands.

A placeholder exists for future **Google AI API** integration when a free tier becomes available.

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
│   ├── train_intent.py          # Train the classifier (run once)
│   ├── intent_classifier.py     # IntentClassifier — predict intent + confidence
│   ├── vectorizer.joblib        # [generated] TF-IDF vectorizer
│   └── classifier.joblib        # [generated] Logistic Regression model
│
├── entity/
│   └── entity_extractor.py      # Regex-based distance/angle extraction
│
├── action/
│   └── dispatcher.py            # Intent → action mapping + session state
│
├── chatbot/
│   └── chatbot_handler.py       # HuggingFace chatbot + rule-based fallback
│
├── main.py                      # Entry point — wires all stages together
├── specification.md             # Detailed technical specification
└── README.md                    # This file
```

---

## Technology Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Speech-to-Text | **Vosk 0.22** | Offline speech recognition from microphone |
| Audio capture | **sounddevice** | Streams mic audio to Vosk |
| Intent classification | **scikit-learn** | TF-IDF vectorization + Logistic Regression |
| Model persistence | **joblib** | Save/load trained ML models |
| Entity extraction | **Python re** | Regex-based parameter parsing |
| Chatbot | **Hugging Face Transformers** | Conversational fallback (DialoGPT-small) |
| Numerical ops | **NumPy** | Probability array handling |
| Target platform | **Raspberry Pi** | Primary deployment target |
| Actuation | **ESP32** *(optional)* | Low-level motor control |

---

## Getting Started

### 1. Install Dependencies

```bash
pip install vosk sounddevice scikit-learn joblib numpy transformers
```

### 2. Download the Vosk Model

Download a model from [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) — for example:

```bash
cd stt/
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
mv vosk-model-small-en-us-0.15/* vosk_model/
```

### 3. Train the Intent Classifier

```bash
python intent/train_intent.py
```

Output:
```
[TRAIN] Trained on 125 samples.
[TRAIN] Intents: ['BREAK', 'GET_STATS', 'NAVIGATE', 'START_SESSION', 'STOP_SESSION']
[TRAIN] Models saved to intent/
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
```

Then try these commands:

```
You: start session
You: move forward 3 meters and turn 90 degrees
You: how am I doing
You: take a break
You: hello there
You: stop session
You: quit
```

---

## How the Stages Interact (Data Flow)

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