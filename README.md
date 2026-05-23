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
├── tts/
│   └── engine.py
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

## Raspberry Pi Notes & Integration Steps

- **Supported Hardware:** Raspberry Pi 4/5 are supported targets.
- **Speech-to-Text:** Handles audio locally using Vosk.
- **Intent Classification & Chatbot:** Offloads to Hugging Face via API to keep Pi CPU usage very low and ensure real-time responsiveness.
- **Connection Loss:** If the network fails, the classifier safely returns `UNKNOWN` to avoid crashes.

### Hardware Wiring (I2S Interface)

To get audio working, wire the INMP441 microphone and MAX98357A amplifier directly to the Raspberry Pi's GPIO pins:

**1. INMP441 Microphone (Input)**
| INMP441 Pin | Raspberry Pi GPIO | Physical Pin |
| ----------- | ----------------- | ------------ |
| VDD | 3.3V | Pin 1 |
| GND | GND | Pin 6 |
| SCK | GPIO18 (PCM_CLK) | Pin 12 |
| WS | GPIO19 (PCM_FS) | Pin 35 |
| SD | GPIO20 (PCM_DIN) | Pin 38 |
| L/R | GND | — |

**2. MAX98357A Amplifier (Output)**
| MAX98357A Pin | Raspberry Pi GPIO | Physical Pin |
| ------------- | ----------------- | ------------ |
| VIN | 5V | Pin 2 or 4 |
| GND | GND | Pin 9 |
| BCLK | GPIO18 (PCM_CLK) | Pin 12 |
| LRC | GPIO19 (PCM_FS) | Pin 35 |
| DIN | GPIO21 (PCM_DOUT) | Pin 40 |
| SD | Leave unconnected | — |
| GAIN | Leave unconnected | — |

**Speaker** → Connect directly to the `SPK+` and `SPK-` terminals on the MAX98357A.

### Deployment Steps on the Pi

1. Clone your repository containing this `main` branch to your Raspberry Pi.
2. **Enable I2S in Boot Config:** Open `/boot/config.txt` (or `/boot/firmware/config.txt` for newer Raspberry Pi OS) and ensure the following lines are uncommented or added to enable the I2S audio drivers:
   ```ini
   dtparam=i2s=on
   ```
   *Note: Reboot your Pi after making this change!*
3. Ensure you have missing system packages for audio processing (`sudo apt-get install python3-pyaudio portaudio19-dev ffmpeg`).
4. Set your hardware audio card as the system default by copying the provided ALSA config into your home directory:
   ```bash
   cp .asoundrc_template ~/.asoundrc
   ```
   *Test that the hardware runs properly using `arecord -l` and `aplay -l`.*
5. Install Python dependencies using `pip install -r requirements.txt`.
4. Set up your environment variables permanently:

   **For Linux / Raspberry Pi (Bash):**
   ```bash
   cat << 'EOF' >> ~/.bashrc
   export HUGGINGFACE_API_KEY="your_huggingface_api_key_here"
   export HUGGINGFACE_API_URL="https://router.huggingface.co/v1/chat/completions"
   export HUGGINGFACE_MODEL="Qwen/Qwen2.5-7B-Instruct"
   export HUGGINGFACE_TIMEOUT_SECONDS="90"
   export RAG_BASE_URL="http://127.0.0.1:8000"
   export RAG_ASK_PATH="/ask"
   export RAG_TIMEOUT_SECONDS="10"
   EOF
   
   source ~/.bashrc
   ```

   **For Windows (PowerShell):**
   ```powershell
   [Environment]::SetEnvironmentVariable("HUGGINGFACE_API_KEY", "your_huggingface_api_key_here", "User")
   [Environment]::SetEnvironmentVariable("HUGGINGFACE_API_URL", "https://router.huggingface.co/v1/chat/completions", "User")
   [Environment]::SetEnvironmentVariable("HUGGINGFACE_MODEL", "Qwen/Qwen2.5-7B-Instruct", "User")
   [Environment]::SetEnvironmentVariable("HUGGINGFACE_TIMEOUT_SECONDS", "90", "User")
   [Environment]::SetEnvironmentVariable("RAG_BASE_URL", "http://127.0.0.1:8000", "User")
   [Environment]::SetEnvironmentVariable("RAG_ASK_PATH", "/ask", "User")
   [Environment]::SetEnvironmentVariable("RAG_TIMEOUT_SECONDS", "10", "User")
   ```
   *(Note: Restart your PowerShell terminal to apply the Windows changes.)*

5. Run the script: `python main.py` or create a systemd service to run it automatically on boot.

---

## Status

- STT: local and offline
- Intent classification: cloud API (Hugging Face)
- Entity extraction: cloud API parameter parsing
- Dispatch: local Python
- Chatbot: cloud API native response
