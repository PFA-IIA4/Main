OFFLINE VOICE ROBOT INTENT SYSTEM UPGRADE SPECIFICATION
Goal

Improve language understanding, dataset scalability, and intent classification quality while keeping full compatibility with Raspberry Pi 4 and Raspberry Pi 5.

System must remain offline-first for STT and intent inference.

No heavy deep learning models allowed on-device.

Keep runtime lightweight and deterministic.

Current System Context

Existing pipeline

STT using Vosk (offline)
Intent classification using TF-IDF + Logistic Regression
Entity extraction using regex
Dispatcher executes robot actions
RAG routing handled separately
Chatbot fallback via Hugging Face API or rule-based system

Current limitation

dataset too small and static
weak generalization to speech variation
no structured augmentation pipeline
confidence handling is rigid
synonym and phrase variation not modeled
Required Upgrade Overview

Implement a structured dataset generation and augmentation system for intent training.

Improve classification robustness using hybrid rule + ML signals.

Keep inference unchanged on Raspberry Pi (TF-IDF + Logistic Regression stays).

New Module to Add

Create new file

intent/data_augmentation.py

This module must generate expanded training datasets from a small seed dataset.

Dataset Design Rules

The system must support:

intent templates
slot-based parameterization
synonym substitution
sentence permutation
noise injection
deduplication
Step 1 Seed Dataset Format

Define training samples in structured template format

Example format:

TEMPLATES = {
  "NAVIGATE": [
    "move forward {distance} meters and turn {angle} degrees",
    "go forward {distance} meters then rotate {angle} degrees"
  ]
}
Step 2 Slot System

Define parameter pools

DISTANCES = [0.5, 1, 2, 3, 5, 10]

ANGLES = [15, 30, 45, 90, 120, 180]

Add spoken variants

DISTANCE_WORDS = {
  1: ["one", "1"],
  2: ["two", "2"],
  3: ["three", "3"],
  5: ["five", "5"]
}
Step 3 Synonym Engine

Create synonym dictionary

SYNONYMS = {
  "move": ["go", "walk", "advance", "proceed"],
  "forward": ["ahead"],
  "turn": ["rotate", "spin", "pivot"],
  "stop": ["end", "halt"],
  "start": ["begin", "initiate"]
}

Implement function

apply_synonyms(sentence) -> list of variants

Step 4 Sentence Permutation Engine

Implement reordering logic for NAVIGATE type sentences

Rules:

preserve meaning
allow swapping independent clauses
allow insertion of connectors: "then", "after that", "and"

Example outputs:

Input:
move forward 3 meters and turn 90 degrees

Outputs:

turn 90 degrees then move forward 3 meters
move forward 3 meters then rotate 90 degrees
go forward 3 meters and then turn 90 degrees
Step 5 Noise Injection Engine

Simulate speech recognition errors

Implement:

word dropout
filler words insertion
spelling variation
number confusion

Examples:

"move uh forward 3 meters"
"move forward three meter"
"move forward 3 meters and and turn 90 degrees"

Noise probability: 0.1 to 0.3 per sample

Step 6 Full Dataset Generator Pipeline

Implement function:

generate_dataset(seed_templates) -> list of (text, intent)

Pipeline order:

expand templates with slot values
apply synonyms
apply permutation
apply noise injection
deduplicate outputs
shuffle dataset

Output must be balanced across intents

Target scaling:

minimum 2000 samples per intent equivalent
expandable to 10k+
Step 7 Integration with Training Script

Modify:

intent/train_intent.py

Replace static dataset loading with:

from intent.data_augmentation import generate_dataset

Dataset must be generated at training time

Save output dataset optionally to:

intent/generated_dataset.json

Step 8 Improve Intent Classification Logic

Modify:

intent/intent_classifier.py

Add hybrid confidence logic:

get top 3 probabilities
compute margin = top1 - top2

Decision rules:

IF margin < 0.15:
return UNKNOWN

ELSE:
return top intent

Keep existing threshold fallback but reduce reliance on fixed 0.6 rule

Step 9 Rule Boosting Layer (Important)

Before classification, apply lightweight rule signals:

Examples:

IF text contains "meters" OR "turn" OR "rotate":
boost NAVIGATE probability

IF text contains "what is" OR "explain":
boost RAG_QUERY probability

This is NOT replacement of ML

This is probability adjustment layer

Step 10 Deduplication Requirement

Ensure dataset generator removes duplicates using:

exact string match
normalized lowercase match
whitespace normalization
Performance Constraints

System must:

run training on Raspberry Pi OR laptop
inference unchanged speed (<50ms target per prediction)
avoid transformers or embeddings at runtime
Expected Outcome

After implementation:

dataset size increases from ~720 to 10k+ equivalent samples
intent classification becomes robust to speech variation
NAVIGATE becomes resilient to ordering and phrasing
UNKNOWN rate decreases significantly
system remains fully offline and lightweight
Testing Requirements

Add test script:

test_augmentation.py

Must output:

sample generated sentences per intent
dataset size per intent
duplicate rate
noise injection examples
Key Design Principle

Do not increase model complexity for now.

Increase data diversity instead.