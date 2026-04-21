"""
Structured intent dataset generation and augmentation utilities.

The module expands small seed templates into a larger balanced dataset using:
- template slot filling
- synonym substitution
- NAVIGATE clause permutation
- speech-like noise injection
- deduplication and shuffling
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

INTENT_ORDER = [
    "START_SESSION",
    "STOP_SESSION",
    "GET_STATS",
    "SMALL_TALK",
    "BREAK",
    "NAVIGATE",
    "RAG_QUERY",
]

TARGET_SAMPLES_PER_INTENT = 2000
DEFAULT_RANDOM_SEED = 42
NOISE_PROBABILITY_RANGE = (0.1, 0.3)
MAX_AUGMENTED_VARIANTS_PER_BASE = 96
NEAR_DUPLICATE_TOKEN_THRESHOLD = 0.94
NEAR_DUPLICATE_BIGRAM_THRESHOLD = 0.84

PREFIXES = [
    "",
    "please",
    "robot",
    "assistant",
    "hey robot",
    "hey assistant",
    "can you",
    "could you",
    "kindly",
    "quickly",
]

SUFFIXES = [
    "",
    "please",
    "for me",
    "right now",
    "if you can",
    "now",
    "today",
    "for this session",
]

DISTANCES = [0.5, 1, 2, 3, 5, 10]
ANGLES = [15, 30, 45, 90, 120, 180]

DISTANCE_WORDS = {
    0.5: ["0.5", "half"],
    1: ["1", "one"],
    2: ["2", "two"],
    3: ["3", "three"],
    5: ["5", "five"],
    10: ["10", "ten"],
}

ANGLE_WORDS = {
    15: ["15", "fifteen"],
    30: ["30", "thirty"],
    45: ["45", "forty five"],
    90: ["90", "ninety"],
    120: ["120", "one hundred twenty"],
    180: ["180", "one hundred eighty"],
}

PAGE_VALUES = [2, 3, 5, 8, 10, 12, 15, 20]
CHAPTER_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]

NUMBER_WORDS = {
    0: ["0", "zero"],
    1: ["1", "one"],
    2: ["2", "two"],
    3: ["3", "three"],
    4: ["4", "four"],
    5: ["5", "five"],
    6: ["6", "six"],
    7: ["7", "seven"],
    8: ["8", "eight"],
    9: ["9", "nine"],
    10: ["10", "ten"],
    12: ["12", "twelve"],
    15: ["15", "fifteen"],
    20: ["20", "twenty"],
}

SYNONYMS = {
    "move": ["go", "walk", "advance", "proceed"],
    "go": ["move", "advance"],
    "forward": ["ahead"],
    "ahead": ["forward"],
    "turn": ["rotate", "spin", "pivot"],
    "rotate": ["turn", "spin"],
    "stop": ["end", "halt"],
    "end": ["stop"],
    "start": ["begin", "initiate"],
    "begin": ["start", "initiate"],
}

INTENT_SYNONYM_KEYS = {
    "START_SESSION": {"start", "begin"},
    "STOP_SESSION": {"stop", "end"},
    "GET_STATS": set(),
    "SMALL_TALK": set(),
    "BREAK": set(),
    "NAVIGATE": {"move", "go", "forward", "ahead", "turn", "rotate"},
    "RAG_QUERY": set(),
}

FILLER_WORDS = ["uh", "um", "like", "just"]

SPELLING_VARIATIONS = {
    "meters": ["meter", "metres"],
    "meter": ["meters", "metres"],
    "degrees": ["degree"],
    "degree": ["degrees"],
}

UNIT_VARIATIONS = {
    "meters": ["meter"],
    "meter": ["meters"],
    "degrees": ["degree"],
    "degree": ["degrees"],
}

NAVIGATION_CONNECTORS = ["and then", "after that", "then", "and"]
OPTIONAL_DROP_WORDS = {
    "please",
    "now",
    "right",
    "kindly",
    "assistant",
    "robot",
    "the",
    "a",
    "an",
    "to",
    "for",
}

REPEATABLE_WORDS = {"and", "turn", "move", "forward", "rotate", "then"}

TEMPLATES = {
    "START_SESSION": [
        "{prefix} start session {suffix}",
        "{prefix} begin session {suffix}",
        "{prefix} start studying {suffix}",
        "{prefix} begin studying {suffix}",
        "{prefix} start study mode {suffix}",
        "{prefix} begin study mode {suffix}",
        "{prefix} start focus mode {suffix}",
        "{prefix} activate session {suffix}",
        "{prefix} initiate session {suffix}",
        "{prefix} launch session {suffix}",
        "{prefix} open session {suffix}",
        "{prefix} turn on study mode {suffix}",
        "{prefix} start the study timer {suffix}",
        "{prefix} kick off the session {suffix}",
    ],
    "STOP_SESSION": [
        "{prefix} stop session {suffix}",
        "{prefix} end session {suffix}",
        "{prefix} stop studying {suffix}",
        "{prefix} end study mode {suffix}",
        "{prefix} finish session {suffix}",
        "{prefix} terminate session {suffix}",
        "{prefix} close session {suffix}",
        "{prefix} stop the study timer {suffix}",
        "{prefix} finish studying {suffix}",
        "{prefix} i am done studying {suffix}",
        "{prefix} wrap up the session {suffix}",
        "{prefix} shut down session {suffix}",
    ],
    "GET_STATS": [
        "{prefix} show statistics {suffix}",
        "{prefix} display stats {suffix}",
        "{prefix} check stats {suffix}",
        "{prefix} show my progress {suffix}",
        "{prefix} how am i doing {suffix}",
        "{prefix} what are my stats {suffix}",
        "{prefix} session statistics {suffix}",
        "{prefix} progress report {suffix}",
        "{prefix} study report {suffix}",
        "{prefix} performance stats {suffix}",
        "{prefix} summary of sessions {suffix}",
        "{prefix} give me a summary of my sessions {suffix}",
        "{prefix} summarize my study progress {suffix}",
    ],
    "SMALL_TALK": [
        "{prefix} hello {suffix}",
        "{prefix} hi {suffix}",
        "{prefix} hey {suffix}",
        "{prefix} how are you {suffix}",
        "{prefix} how are you doing {suffix}",
        "{prefix} how is it going {suffix}",
        "{prefix} whats up {suffix}",
        "{prefix} good morning {suffix}",
        "{prefix} good afternoon {suffix}",
        "{prefix} good evening {suffix}",
        "{prefix} who are you {suffix}",
        "{prefix} nice to meet you {suffix}",
    ],
    "BREAK": [
        "{prefix} take a break {suffix}",
        "{prefix} i need a break {suffix}",
        "{prefix} break time {suffix}",
        "{prefix} pause session {suffix}",
        "{prefix} pause the session {suffix}",
        "{prefix} pause studying {suffix}",
        "{prefix} break now {suffix}",
        "{prefix} rest time {suffix}",
        "{prefix} quick break {suffix}",
        "{prefix} i want a break {suffix}",
        "{prefix} take five {suffix}",
        "{prefix} stop for a break {suffix}",
    ],
    "NAVIGATE": [
        "{prefix} move forward {distance_phrase} and turn {angle_phrase} {suffix}",
        "{prefix} go forward {distance_phrase} then rotate {angle_phrase} {suffix}",
        "{prefix} move {distance_phrase} forward and turn {angle_phrase} {suffix}",
        "{prefix} advance {distance_phrase} then pivot {angle_phrase} {suffix}",
        "{prefix} drive forward {distance_phrase} and turn {angle_phrase} {suffix}",
        "{prefix} go ahead {distance_phrase} and rotate {angle_phrase} {suffix}",
        "{prefix} turn {angle_phrase} then move forward {distance_phrase} {suffix}",
        "{prefix} rotate {angle_phrase} then move {distance_phrase} forward {suffix}",
        "{prefix} move forward {distance_phrase} after that turn {angle_phrase} {suffix}",
        "{prefix} go {distance_phrase} ahead and then turn {angle_phrase} {suffix}",
    ],
    "RAG_QUERY": [
        "{prefix} what is {topic} {suffix}",
        "{prefix} explain {topic} {suffix}",
        "{prefix} define {topic} {suffix}",
        "{prefix} summarize {topic} {suffix}",
        "{prefix} turn this into a summary {suffix}",
        "{prefix} summarize this section {suffix}",
        "{prefix} summarize this for me {suffix}",
        "{prefix} give me a summary of {topic} {suffix}",
        "{prefix} tell me about {topic} {suffix}",
        "{prefix} give details about {topic} {suffix}",
        "{prefix} what does {topic} say {suffix}",
        "{prefix} what is written about {topic} {suffix}",
        "{prefix} read the section on {topic} {suffix}",
        "{prefix} explain the document about {topic} {suffix}",
        "{prefix} summarize chapter {chapter_word} {suffix}",
        "{prefix} what does page {page_word} say {suffix}",
    ],
}

RAG_TOPICS = [
    "pid controller",
    "machine learning",
    "convolution",
    "gradient descent",
    "overfitting",
    "neural networks",
    "data preprocessing",
    "logistic regression",
    "feature engineering",
    "confusion matrix",
    "normalization",
    "cross validation",
    "classification",
    "regression",
    "activation functions",
    "backpropagation",
    "linear regression",
    "optimization",
    "bias and variance",
    "reinforcement learning",
    "svm",
    "entropy",
    "accuracy and precision",
    "unsupervised learning",
    "supervised learning",
    "the uploaded document",
    "my notes",
    "the course pdf",
    "the lesson",
    "chapter overview",
    "the file",
]


@dataclass(frozen=True)
class GenerationStats:
    """Summary of a generated dataset."""

    raw_count: int
    unique_count: int
    duplicate_count: int
    duplicate_rate: float
    per_intent_counts: Dict[str, int]


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def deduplicate_samples(samples: Iterable[str]) -> List[str]:
    """Remove duplicates using exact and normalized text matches."""
    unique: List[str] = []
    seen = set()
    for sample in samples:
        normalized = _normalize_text(sample)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _apply_template(template: str, slot_values: Dict[str, str]) -> str:
    return _normalize_text(template.format(**slot_values))


def _build_quantity_phrases(
    values: Sequence[float],
    word_map: Dict[float, List[str]],
    singular_unit: str,
    plural_unit: str,
) -> List[str]:
    phrases: List[str] = []
    for value in values:
        unit = singular_unit if float(value) == 1 else plural_unit
        number_tokens = word_map.get(value, [str(value)])
        for token in number_tokens:
            phrases.append(_normalize_text(f"{token} {unit}"))
    return deduplicate_samples(phrases)


DISTANCE_PHRASES = _build_quantity_phrases(DISTANCES, DISTANCE_WORDS, "meter", "meters")
ANGLE_PHRASES = _build_quantity_phrases(ANGLES, ANGLE_WORDS, "degree", "degrees")
PAGE_PHRASES = deduplicate_samples(
    [
        _normalize_text(f"{token} page")
        for value in PAGE_VALUES
        for token in NUMBER_WORDS.get(value, [str(value)])
    ]
)
CHAPTER_PHRASES = deduplicate_samples(
    [
        _normalize_text(f"chapter {token}")
        for value in CHAPTER_VALUES
        for token in NUMBER_WORDS.get(value, [str(value)])
    ]
)

WORD_TO_NUMBER = {
    variant: str(value)
    for value, variants in NUMBER_WORDS.items()
    for variant in variants
    if not variant.isdigit()
}

NUMBER_TO_WORD = {
    str(value): [variant for variant in variants if not variant.isdigit()]
    for value, variants in NUMBER_WORDS.items()
}


def apply_synonyms(
    sentence: str,
    allowed_words: set[str] | None = None,
    max_depth: int = 2,
    max_variants: int = 16,
) -> List[str]:
    """Return synonym-based sentence variants."""
    normalized = _normalize_text(sentence)
    variants = [normalized]
    seen = {normalized}
    queue: List[Tuple[str, int]] = [(normalized, 0)]

    while queue and len(variants) < max_variants:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        tokens = current.split()
        for index, token in enumerate(tokens):
            if allowed_words is not None and token not in allowed_words:
                continue
            synonyms = SYNONYMS.get(token, [])
            for synonym in synonyms:
                candidate_tokens = list(tokens)
                candidate_tokens[index] = synonym
                candidate = _normalize_text(" ".join(candidate_tokens))
                if candidate in seen:
                    continue
                seen.add(candidate)
                variants.append(candidate)
                queue.append((candidate, depth + 1))
                if len(variants) >= max_variants:
                    break
            if len(variants) >= max_variants:
                break

    return deduplicate_samples(variants)


def _number_surface_variants(sentence: str) -> List[str]:
    """Generate number and unit wording variants."""
    tokens = sentence.split()
    variants = [sentence]

    for index, token in enumerate(tokens):
        lower = token.lower()

        if lower in WORD_TO_NUMBER:
            candidate = list(tokens)
            candidate[index] = WORD_TO_NUMBER[lower]
            variants.append(_normalize_text(" ".join(candidate)))

        if lower in NUMBER_TO_WORD:
            for word_variant in NUMBER_TO_WORD[lower]:
                candidate = list(tokens)
                candidate[index] = word_variant
                variants.append(_normalize_text(" ".join(candidate)))

        if lower in UNIT_VARIATIONS:
            for unit_variant in UNIT_VARIATIONS[lower]:
                candidate = list(tokens)
                candidate[index] = unit_variant
                variants.append(_normalize_text(" ".join(candidate)))

    return deduplicate_samples(variants)


def apply_permutation(sentence: str) -> List[str]:
    """Generate NAVIGATE clause reorderings."""
    normalized = _normalize_text(sentence)
    if not any(connector in normalized for connector in NAVIGATION_CONNECTORS):
        return [normalized]

    split_pattern = r"\b(?:and then|after that|then|and)\b"
    clauses = [part.strip() for part in re.split(split_pattern, normalized) if part.strip()]
    if len(clauses) < 2:
        return [normalized]

    first_clause = clauses[0]
    second_clause = clauses[1]
    variants = [
        normalized,
        f"{second_clause} then {first_clause}",
        f"{first_clause} then {second_clause}",
        f"{first_clause} and then {second_clause}",
        f"{second_clause} after that {first_clause}",
    ]
    return deduplicate_samples(variants)


def _is_number_token(token: str) -> bool:
    token = token.strip().lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return True
    for variants in NUMBER_WORDS.values():
        if token in variants:
            return True
    return False


def _insert_filler_word(sentence: str, rng: random.Random) -> str:
    tokens = sentence.split()
    if not tokens:
        return sentence
    position = 1 if len(tokens) > 1 else 0
    if len(tokens) > 2:
        position = rng.randint(1, len(tokens) - 1)
    filler = rng.choice(FILLER_WORDS)
    tokens.insert(position, filler)
    return _normalize_text(" ".join(tokens))


def _drop_optional_word(sentence: str, rng: random.Random) -> str:
    tokens = sentence.split()
    removable_indices = [index for index, token in enumerate(tokens) if token in OPTIONAL_DROP_WORDS]
    if not removable_indices:
        return sentence
    index = rng.choice(removable_indices)
    del tokens[index]
    return _normalize_text(" ".join(tokens))


def _drop_unit(sentence: str, rng: random.Random) -> str:
    """Remove a distance/angle unit to mimic ASR omissions."""
    tokens = sentence.split()
    unit_indices = [index for index, token in enumerate(tokens) if token in UNIT_VARIATIONS]
    if not unit_indices:
        return sentence
    index = rng.choice(unit_indices)
    del tokens[index]
    return _normalize_text(" ".join(tokens))


def _repeat_word(sentence: str, rng: random.Random) -> str:
    """Repeat a frequent connector/action term, e.g. 'and and'."""
    tokens = sentence.split()
    candidate_indices = [index for index, token in enumerate(tokens) if token in REPEATABLE_WORDS]
    if not candidate_indices:
        return sentence
    index = rng.choice(candidate_indices)
    tokens.insert(index, tokens[index])
    return _normalize_text(" ".join(tokens))


def _apply_spelling_variation(sentence: str, rng: random.Random) -> str:
    tokens = sentence.split()
    candidate_indices = [
        index for index, token in enumerate(tokens) if token in SPELLING_VARIATIONS
    ]
    if not candidate_indices:
        return sentence
    index = rng.choice(candidate_indices)
    replacement = rng.choice(SPELLING_VARIATIONS[tokens[index]])
    tokens[index] = replacement
    return _normalize_text(" ".join(tokens))


def _apply_number_confusion(sentence: str, rng: random.Random) -> str:
    tokens = sentence.split()
    candidate_indices = [
        index for index, token in enumerate(tokens) if _is_number_token(token)
    ]
    if not candidate_indices:
        return sentence

    index = rng.choice(candidate_indices)
    token = tokens[index]
    lower = token.lower()
    replacement = token

    if re.fullmatch(r"\d+(?:\.\d+)?", lower):
        number_value = lower
        for value, variants in NUMBER_WORDS.items():
            if number_value == str(value):
                replacement = rng.choice(variants)
                break
        if replacement == token:
            replacement = token
    else:
        for value, variants in NUMBER_WORDS.items():
            if lower in variants:
                replacement = str(value)
                break

    tokens[index] = replacement
    return _normalize_text(" ".join(tokens))


def inject_noise(
    sentence: str,
    rng: random.Random,
    noise_probability_range: Tuple[float, float] = NOISE_PROBABILITY_RANGE,
    force: bool = False,
) -> str:
    """Inject speech-like noise into a sentence."""
    normalized = _normalize_text(sentence)
    probability = rng.uniform(*noise_probability_range)
    if not force and rng.random() > probability:
        return normalized

    transforms = [
        lambda text: _insert_filler_word(text, rng),
        lambda text: _drop_optional_word(text, rng),
        lambda text: _drop_unit(text, rng),
        lambda text: _repeat_word(text, rng),
        lambda text: _apply_spelling_variation(text, rng),
        lambda text: _apply_number_confusion(text, rng),
    ]

    noisy = normalized
    applied = False
    max_steps = 1 if force else 2
    steps = rng.randint(1, max_steps)

    for _ in range(steps):
        rng.shuffle(transforms)
        for transform in transforms:
            candidate = transform(noisy)
            candidate = _normalize_text(candidate)
            if candidate and candidate != noisy:
                noisy = candidate
                applied = True
                break

    return noisy if applied else normalized


def _iter_structured_samples(intent: str, templates: Sequence[str]) -> Iterable[str]:
    if intent in {"START_SESSION", "STOP_SESSION", "GET_STATS", "SMALL_TALK", "BREAK"}:
        slot_names = ["prefix", "suffix"]
        slot_values = {
            "prefix": PREFIXES,
            "suffix": SUFFIXES,
        }
    elif intent == "NAVIGATE":
        slot_names = ["prefix", "suffix", "distance_phrase", "angle_phrase"]
        slot_values = {
            "prefix": PREFIXES,
            "suffix": SUFFIXES,
            "distance_phrase": DISTANCE_PHRASES,
            "angle_phrase": ANGLE_PHRASES,
        }
    elif intent == "RAG_QUERY":
        slot_names = ["prefix", "suffix", "topic", "page_word", "chapter_word"]
        slot_values = {
            "prefix": PREFIXES,
            "suffix": SUFFIXES,
            "topic": RAG_TOPICS,
            "page_word": [token for value in PAGE_VALUES for token in NUMBER_WORDS.get(value, [str(value)])],
            "chapter_word": [token for value in CHAPTER_VALUES for token in NUMBER_WORDS.get(value, [str(value)])],
        }
    else:
        slot_names = []
        slot_values = {}

    if not slot_names:
        for template in templates:
            yield _normalize_text(template)
        return

    for template in templates:
        pools = [slot_values[name] for name in slot_names]
        for combination in product(*pools):
            context = dict(zip(slot_names, combination))
            yield _apply_template(template, context)


def _token_jaccard_similarity(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(len(set_a | set_b), 1)


def _bigram_set(tokens: Sequence[str]) -> set[Tuple[str, ...]]:
    if len(tokens) < 2:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + 2]) for index in range(len(tokens) - 1)}


def _bigram_jaccard_similarity(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    bigrams_a = _bigram_set(tokens_a)
    bigrams_b = _bigram_set(tokens_b)
    if not bigrams_a and not bigrams_b:
        return 1.0
    return len(bigrams_a & bigrams_b) / max(len(bigrams_a | bigrams_b), 1)


def _is_near_duplicate(candidate: str, existing: str) -> bool:
    candidate_tokens = candidate.split()
    existing_tokens = existing.split()

    if abs(len(candidate_tokens) - len(existing_tokens)) > 2:
        return False

    token_similarity = _token_jaccard_similarity(candidate_tokens, existing_tokens)
    if token_similarity < NEAR_DUPLICATE_TOKEN_THRESHOLD:
        return False

    bigram_similarity = _bigram_jaccard_similarity(candidate_tokens, existing_tokens)
    return bigram_similarity >= NEAR_DUPLICATE_BIGRAM_THRESHOLD


def _augment_sentence(intent: str, sentence: str, rng: random.Random) -> List[str]:
    normalized = _normalize_text(sentence)
    allowed_synonyms = INTENT_SYNONYM_KEYS.get(intent)

    seen = {normalized}
    frontier = [normalized]

    for _ in range(2):
        if not frontier or len(seen) >= MAX_AUGMENTED_VARIANTS_PER_BASE:
            break

        next_frontier: List[str] = []
        for variant in frontier:
            transformed = [variant]
            transformed.extend(
                apply_synonyms(
                    variant,
                    allowed_words=allowed_synonyms,
                    max_depth=1,
                    max_variants=12,
                )
            )
            transformed.extend(_number_surface_variants(variant))

            if intent == "NAVIGATE":
                permutation_variants = apply_permutation(variant)
                transformed.extend(permutation_variants)
                for permuted in permutation_variants:
                    transformed.extend(
                        apply_synonyms(
                            permuted,
                            allowed_words=allowed_synonyms,
                            max_depth=1,
                            max_variants=8,
                        )
                    )

            rng.shuffle(transformed)
            for candidate in transformed:
                normalized_candidate = _normalize_text(candidate)
                if not normalized_candidate or normalized_candidate in seen:
                    continue
                seen.add(normalized_candidate)
                next_frontier.append(normalized_candidate)
                if len(seen) >= MAX_AUGMENTED_VARIANTS_PER_BASE:
                    break

            if len(seen) >= MAX_AUGMENTED_VARIANTS_PER_BASE:
                break

        frontier = next_frontier[:32]

    return list(seen)


def _generate_intent_samples(
    intent: str,
    templates: Sequence[str],
    target_count: int,
    rng: random.Random,
) -> Tuple[List[str], int]:
    """Generate a balanced sample list for a single intent."""
    unique: List[str] = []
    seen = set()
    near_dup_buckets: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    raw_count = 0

    def _bucket_keys_for(candidate: str) -> List[Tuple[str, int]]:
        tokens = candidate.split()
        if not tokens:
            return []
        first = tokens[0]
        length = len(tokens)
        return [(first, length - 1), (first, length), (first, length + 1)]

    def add_candidate(candidate: str) -> None:
        nonlocal raw_count
        raw_count += 1
        normalized = _normalize_text(candidate)
        if not normalized or normalized in seen:
            return

        bucket_keys = _bucket_keys_for(normalized)
        for bucket_key in bucket_keys:
            for existing in near_dup_buckets.get(bucket_key, [])[-40:]:
                if _is_near_duplicate(normalized, existing):
                    return

        seen.add(normalized)
        unique.append(normalized)
        for bucket_key in bucket_keys:
            near_dup_buckets[bucket_key].append(normalized)
            if len(near_dup_buckets[bucket_key]) > 300:
                near_dup_buckets[bucket_key] = near_dup_buckets[bucket_key][-300:]

    for structured in _iter_structured_samples(intent, templates):
        for augmented in _augment_sentence(intent, structured, rng):
            add_candidate(augmented)
            if len(unique) >= target_count:
                return unique[:target_count], raw_count

            if rng.random() < rng.uniform(*NOISE_PROBABILITY_RANGE):
                noisy_variant = inject_noise(augmented, rng, force=True)
                add_candidate(noisy_variant)
                if len(unique) >= target_count:
                    return unique[:target_count], raw_count

                if rng.random() < 0.45:
                    chained_noisy = inject_noise(noisy_variant, rng, force=True)
                    add_candidate(chained_noisy)
                    if len(unique) >= target_count:
                        return unique[:target_count], raw_count

    fallback_pool = list(unique) or list(deduplicate_samples(templates))
    fallback_attempts = 0
    max_fallback_attempts = max(target_count * 40, 1000)
    while len(unique) < target_count and fallback_attempts < max_fallback_attempts:
        source = rng.choice(fallback_pool)
        noisy_variant = inject_noise(source, rng, force=True)
        add_candidate(noisy_variant)
        if rng.random() < 0.35:
            add_candidate(inject_noise(noisy_variant, rng, force=True))
        fallback_pool.append(noisy_variant)
        fallback_attempts += 1

    return unique[:target_count], raw_count


def _save_dataset(dataset: Sequence[Tuple[str, str]], save_path: str | Path) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"text": text, "intent": intent} for text, intent in dataset]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_generated_dataset(
    seed_templates: Dict[str, Sequence[str]] | None = None,
    target_per_intent: int = TARGET_SAMPLES_PER_INTENT,
    random_seed: int = DEFAULT_RANDOM_SEED,
    save_path: str | Path | None = None,
) -> Tuple[List[Tuple[str, str]], GenerationStats]:
    """Generate a balanced dataset plus generation statistics."""
    templates = seed_templates or TEMPLATES
    rng = random.Random(random_seed)

    dataset: List[Tuple[str, str]] = []
    raw_count = 0
    per_intent_counts: Dict[str, int] = {}

    for intent in INTENT_ORDER:
        intent_templates = templates.get(intent, [])
        intent_samples, intent_raw_count = _generate_intent_samples(
            intent=intent,
            templates=intent_templates,
            target_count=target_per_intent,
            rng=rng,
        )
        raw_count += intent_raw_count
        per_intent_counts[intent] = len(intent_samples)
        dataset.extend((sample, intent) for sample in intent_samples)

    rng.shuffle(dataset)

    stats = GenerationStats(
        raw_count=raw_count,
        unique_count=len(dataset),
        duplicate_count=max(raw_count - len(dataset), 0),
        duplicate_rate=(raw_count - len(dataset)) / raw_count if raw_count else 0.0,
        per_intent_counts=per_intent_counts,
    )

    if save_path is not None:
        _save_dataset(dataset, save_path)

    return dataset, stats


def generate_dataset(
    seed_templates: Dict[str, Sequence[str]] | None = None,
    target_per_intent: int = TARGET_SAMPLES_PER_INTENT,
    random_seed: int = DEFAULT_RANDOM_SEED,
    save_path: str | Path | None = None,
) -> List[Tuple[str, str]]:
    """Generate a balanced dataset of (text, intent) pairs."""
    dataset, _ = build_generated_dataset(
        seed_templates=seed_templates,
        target_per_intent=target_per_intent,
        random_seed=random_seed,
        save_path=save_path,
    )
    return dataset


if __name__ == "__main__":
    dataset, stats = build_generated_dataset(save_path=Path(__file__).with_name("generated_dataset.json"))
    print(f"Generated {len(dataset)} samples.")
    print(f"Raw candidates: {stats.raw_count}")
    print(f"Duplicate rate: {stats.duplicate_rate:.2%}")
    print(f"Per intent: {stats.per_intent_counts}")
