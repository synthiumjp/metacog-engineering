"""
Step 2b: Rebuild training set without modal filter.
Uses existing calibration data from Step 2 — no re-sampling needed.
Post-hoc follow-up after Phase 0 Stop result.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer

MODEL_ID = "google/gemma-3-4b-it"
SEED = 42
SHUFFLED_SEED = 43
DATA_DIR = Path(r"D:\metacog\data")

TRIVIAQA_PROMPT = (
    "You are answering trivia questions. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
)

CONFIDENCE_TARGET_MAP = {
    0: 5, 1: 15, 2: 25, 3: 35, 4: 45, 5: 50,
    6: 60, 7: 70, 8: 80, 9: 90, 10: 95,
}


def build_sft_example(item, tokenizer):
    user_msg = TRIVIAQA_PROMPT.format(question=item["question"])
    assistant_msg = (
        f"{item['modal_answer']}\n\n"
        f"Confidence: {item['confidence_target']}%"
    )
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    return {
        "question_id": item["question_id"],
        "ds_index": item["ds_index"],
        "n_correct": item["n_correct"],
        "confidence_target": item["confidence_target"],
        "modal_correct": item["modal_correct"],
        "difficulty_bin": item["difficulty_bin"],
        "messages": messages,
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        ),
    }


def main():
    # Load existing calibration data
    with open(DATA_DIR / "step2_calibration.json") as f:
        tcal_records = json.load(f)
    print(f"Loaded {len(tcal_records)} calibration records")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # NO modal filter — use all items
    training_items = tcal_records
    n_correct_true = sum(1 for r in tcal_records if r["modal_correct"])
    n_correct_false = sum(1 for r in tcal_records if not r["modal_correct"])
    print(f"Training set: {len(training_items)} items "
          f"(modal_correct={n_correct_true}, modal_incorrect={n_correct_false})")

    # Build SFT examples
    sft_examples = [build_sft_example(item, tokenizer) for item in training_items]

    with open(DATA_DIR / "step2_training_set.json", "w") as f:
        json.dump(sft_examples, f, indent=2)
    print(f"Training set -> {DATA_DIR / 'step2_training_set.json'}")

    # Empty conflict set for Step 4 compatibility
    with open(DATA_DIR / "step2_conflict_set.json", "w") as f:
        json.dump([], f, indent=2)
    print(f"Conflict set (empty) -> {DATA_DIR / 'step2_conflict_set.json'}")

    # Shuffled targets
    real_targets = [item["confidence_target"] for item in training_items]
    rng = np.random.RandomState(SHUFFLED_SEED)
    shuffled_targets = real_targets.copy()
    rng.shuffle(shuffled_targets)

    from scipy.stats import pearsonr, spearmanr
    r_p, _ = pearsonr(real_targets, shuffled_targets)
    r_s, _ = spearmanr(real_targets, shuffled_targets)
    print(f"E7: Pearson r={r_p:.4f}, Spearman rho={r_s:.4f}")

    shuffled_sft = []
    for item, shuf_target in zip(training_items, shuffled_targets):
        item_copy = dict(item)
        item_copy["confidence_target"] = shuf_target
        shuffled_sft.append(build_sft_example(item_copy, tokenizer))

    with open(DATA_DIR / "step2_shuffled_training_set.json", "w") as f:
        json.dump(shuffled_sft, f, indent=2)
    print(f"Shuffled training set -> {DATA_DIR / 'step2_shuffled_training_set.json'}")

    # Target distribution
    from collections import Counter
    ct = Counter(real_targets)
    print(f"\nTarget distribution:")
    for t in sorted(CONFIDENCE_TARGET_MAP.values()):
        bar = "#" * (ct.get(t, 0) // 5)
        print(f"  {t:3d}%: {ct.get(t, 0):4d} {bar}")


if __name__ == "__main__":
    main()
