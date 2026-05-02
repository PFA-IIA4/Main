# LLM Migration Specification v2

**Status:** Production implementation complete  
**Date:** April 30, 2026  
**Architecture:** LLM-only intent classification with local llama.cpp inference

---

## Goal

Replace the old TF-IDF + Logistic Regression intent pipeline with a local lightweight LLM running through llama.cpp on the Raspberry Pi.

The new architecture keeps the system offline-first for the core voice pipeline and removes the old ML fallback architecture entirely.

---

## Final Architecture

### Pipeline

1. Speech-to-text with Vosk
2. Intent classification with TinyLlama 1.1B Chat via llama.cpp
3. Entity extraction with regex for `NAVIGATE`
4. Dispatch to robot/session actions
5. Chatbot handling only when the dispatcher returns `CHATBOT_FALLBACK`

### Behavior

- The classifier is LLM-only.
- There is no TF-IDF vectorizer.
- There is no Logistic Regression model.
- There is no ML fallback path.
- If the LLM fails, times out, or returns invalid JSON, the classifier returns `UNKNOWN`.
- The chatbot path remains as the user-facing fallback for `UNKNOWN` intents.

---

## Implementation Status

### Completed

- `llama.cpp` is cloned and built successfully.
- TinyLlama 1.1B Chat Q4_K_M is downloaded.
- `intent/llm_classifier.py` is implemented.
- `intent/intent_classifier.py` is reduced to a thin LLM-only wrapper.
- `main.py` is wired to the new classifier output.
- `test_llm_classifier.py` covers the LLM-only flow.
- Old augmentation and training files are removed.

### Removed

- `intent/data_augmentation.py`
- `intent/train_intent.py`
- `test_augmentation.py`
- legacy joblib model artifacts

---

## LLM Classifier Design

### Model

- TinyLlama 1.1B Chat
- Quantization: Q4_K_M
- Runtime: llama.cpp `llama-cli`

### Classifier Output

The wrapper returns a dictionary with:

- `intent`
- `confidence`
- `reason`
- `tokens_generated`
- `inference_time_ms`
- `model_used`

### Cache

The classifier keeps an in-memory cache of previous inputs. This means repeated identical text can return immediately without another llama.cpp call.

### Failure Handling

- Invalid JSON response -> `UNKNOWN`
- Timeout -> `UNKNOWN`
- Runtime error -> `UNKNOWN`

No ML fallback is attempted.

---

## Files of Interest

- `intent/llm_classifier.py` - llama.cpp subprocess wrapper
- `intent/intent_classifier.py` - app-facing LLM-only wrapper
- `main.py` - pipeline integration
- `README.md` - user-facing setup and usage
- `test_llm_classifier.py` - validation

---

## Deployment Configuration

```bash
USE_LLM_CLASSIFIER=true
LLM_MODEL_PATH=./llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
LLAMA_BIN_PATH=./llama.cpp/build/bin/llama-cli
LLM_MAX_TOKENS=150
LLM_TEMPERATURE=0.3
LLM_INFERENCE_TIMEOUT=5
```

Recommended Raspberry Pi settings:

- Keep `LLM_TEMPERATURE` low for stable output.
- Keep `LLM_INFERENCE_TIMEOUT` strict so failures are fast.
- Keep the model file on fast local storage.

---

## Performance Notes

- First model load is slower than cached inference.
- Cached repeated queries are inexpensive.
- The main cost is the active model inference when a new command arrives.
- No background ML model sits resident anymore.

---

## Validation Checklist

- `llama-cli` builds successfully
- TinyLlama model loads successfully
- Classifier returns valid JSON-derived results
- `main.py` prints the selected model as `llm`
- `pytest test_llm_classifier.py -v` passes
- README reflects the LLM-only workflow
- No ML fallback references remain in the active code path

---

## Operational Summary

This version is intentionally simpler than the previous architecture:

- one classifier path
- one model family
- one runtime for intent decisions
- one user-visible fallback path via chatbot for `UNKNOWN`

That keeps the Raspberry Pi footprint focused on the actual LLM inference workload instead of maintaining a second legacy classifier.
