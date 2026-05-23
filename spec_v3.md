**Status:** Production implementation complete  
**Date:** May 2, 2026  
**Architecture:** Unified Cloud Inference (Hugging Face API)

---

## Goal

Replace the local `llama.cpp` inference architecture (TinyLlama 1.1B on Raspberry Pi) from v2 with a remote Hugging Face API calls. 

This unifies Intent Classification, Entity Extraction, and the Chatbot into a single, high-quality network request, heavily freeing up local resources on the Raspberry Pi.

---

## Final Architecture

### Pipeline

1. **Speech-to-text** with local Vosk.
2. **Unified Brain** via `requests` to Hugging Face (`Qwen2.5-7B-Instruct` or similar) doing three jobs at once:
   - Intent Classification
   - Parameter extraction (formerly Regex entity extraction)
   - Chatbot generation (formerly a separate fallback module)
3. **Dispatch** to robot/session actions based on JSON output.

### Behavior

- No local `llama-cli` subprocess runs.
- The model outputs highly structured JSON due to sophisticated instruction following of 7B+ cloud models.
- If the network fails, the API gracefully sets the intent to `UNKNOWN` with a network error reason, preserving system stability.
- Entities (`distance`, `angle`) for `NAVIGATE` are natively returned inside the `"parameters"` dictionary embedded in the JSON.
- Conversational answers are natively returned inside the `"response"` string in the JSON.

---

## Implementation Status

### Completed

- `intent/llm_classifier.py` is rewritten using `requests`.
- `intent/intent_classifier.py` parses parameters and responses from the output dictionary.
- `main.py` is refactored to remove separate subroutine steps for entities and chatbot handling.
- `test_llm_classifier.py` rewritten to use Mock requests to completely simulate network logic.

### Removed
To significantly reduce disk clutter and technical debt, the following were deleted:
- `llama.cpp/` (including all GGUF models and the local build binary)
- `entity/entity_extractor.py` and directory
- `chatbot/chatbot_handler.py` and directory
- `requirements_llm.txt`

---

## API Classifier Design

### Output Contract

The API system prompt enforces exactly this JSON schema:

```json
{
  "intent": "NAVIGATE",
  "confidence": 0.95,
  "reason": "Move the robot",
  "parameters": {
    "distance": 3,
    "angle": -90
  },
  "response": ""
}
```

For CHATBOT intents, `"parameters"` is empty and `"response"` contains the conversational string.

### Failure Handling

- JSON Decode error -> `UNKNOWN`
- Network Timeout or Error -> `UNKNOWN`
- The system will NOT crash if internet disconnects, the user will trigger the `UNKNOWN` pipeline.

---

## Deployment Configuration

```bash
HUGGINGFACE_API_KEY=your_key
HUGGINGFACE_API_URL=https://router.huggingface.co/v1/chat/completions
HUGGINGFACE_MODEL=Qwen/Qwen2.5-7B-Instruct
HUGGINGFACE_TIMEOUT_SECONDS=60
```

Recommended Raspberry Pi settings:
- Reliable WiFi connection is necessary.
- A fast TTS implementation is encouraged to quickly read the generated `response` when `CHATBOT` is triggered.

---

## Operational Summary

This version solves the `JSON parse error` bugs frequently seen on smaller models by delegating logic to significantly larger, more intelligent cloud models. It reduces pipeline stages from 5 down to 3, strips heavy local build requirements, and minimizes memory load on the Raspberry Pi host.
