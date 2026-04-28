# Intent Dataset Generation

## 1. Overview

This module builds a balanced intent dataset for training the intent classifier.

Pipeline stages:
- Template slot filling
- Synonym substitution
- NAVIGATE-specific clause permutation
- Noise injection (speech-like + typo-level)
- Optional LLM paraphrasing (dataset generation only)
- Deduplication and shuffling

Main file: `intent/data_augmentation.py`

## 2. Augmentation Methods

### Templates
Seed templates are expanded with prefixes, suffixes, numeric phrases, and intent-specific slots.

### Synonyms
Controlled synonym replacement broadens lexical coverage while staying near the base meaning.

### Noise
The generator injects ASR-like noise and typo-level character noise:
- filler insertion
- optional word drop
- unit drop
- repeat word
- spelling variation
- number confusion
- typo noise (deletion, swap, duplication, missing letters)

### LLM Augmentation
If enabled and configured, semantic paraphrases are generated via LLM and filtered to remain consistent with the target intent.

## 3. Enable LLM Augmentation

1. Install optional dependency:

```bash
pip install openai
```

2. Set API key:

```bash
export LLM_API_KEY="your_api_key"
```

3. Optional knobs:

```bash
export LLM_MODEL="gpt-4o-mini"
export USE_LLM_AUGMENTATION="true"
export LLM_TIMEOUT_SECONDS="20"
```

## 4. Disable LLM (Safe Mode)

Any of these will disable LLM usage:
- Do not set `LLM_API_KEY` (default safe behavior)
- Set `USE_LLM_AUGMENTATION=false`

When disabled, generation runs with local deterministic augmentation only.

## 5. Performance Note

LLM augmentation is used only during dataset generation and is never used by runtime intent prediction on the robot.

## 6. Example Usage

Generate and save dataset:

```bash
python intent/train_intent.py
```

Generate dataset only (from Python):

```python
from intent.data_augmentation import build_generated_dataset

dataset, stats = build_generated_dataset(save_path="intent/generated_dataset.json")
print(len(dataset), stats.per_intent_counts)
```

## 7. Tips for New Intents

- Add the new label to `INTENT_ORDER`
- Add templates in `TEMPLATES`
- Add synonym constraints in `INTENT_SYNONYM_KEYS` when needed
- Add guard patterns for LLM filtering in `INTENT_GUARD_PATTERNS` and optional negatives in `INTENT_FORBIDDEN_PATTERNS`
- Add intent-specific entity slots in `_iter_structured_samples` if the new intent requires structured parameters
- Re-train models after template updates
