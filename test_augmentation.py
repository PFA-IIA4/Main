"""
Quick inspection script for the intent augmentation pipeline.
Prints sample outputs, dataset sizes, duplicate rate, and noise examples.
"""

from __future__ import annotations

from collections import defaultdict
import random

from intent.data_augmentation import (
    DEFAULT_RANDOM_SEED,
    INTENT_ORDER,
    TARGET_SAMPLES_PER_INTENT,
    build_generated_dataset,
    inject_noise,
)


def main() -> None:
    dataset, stats = build_generated_dataset(
        target_per_intent=TARGET_SAMPLES_PER_INTENT,
        random_seed=DEFAULT_RANDOM_SEED,
        save_path=None,
    )

    grouped = defaultdict(list)
    for text, intent in dataset:
        grouped[intent].append(text)

    print(f"Dataset size: {len(dataset)}")
    print(f"Duplicate rate: {stats.duplicate_rate:.2%}")
    print(f"Raw candidates: {stats.raw_count}")
    print(f"Unique candidates: {stats.unique_count}")
    print()

    print("Samples per intent:")
    for intent in INTENT_ORDER:
        samples = grouped[intent]
        print(f"- {intent}: {len(samples)} samples")
        for sample in samples[:3]:
            print(f"  • {sample}")
        print()

    print("Noise injection examples:")
    noise_seed = DEFAULT_RANDOM_SEED + 7
    for index, intent in enumerate(INTENT_ORDER):
        if not grouped[intent]:
            continue
        source = grouped[intent][0]
        noisy = inject_noise(source, random.Random(noise_seed + index), force=True)
        print(f"- {intent}: {source} -> {noisy}")


if __name__ == "__main__":
    main()
