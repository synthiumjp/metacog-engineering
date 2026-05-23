#!/usr/bin/env python3
"""prep_e2e_70b_triviaqa.py — Prepare end-to-end training data for 70B TriviaQA.

Uses 70B T-cal baseline responses with binary confidence targets (100/0).
Balances correct/incorrect by downsampling the majority class.

Usage:
    python3 prep_e2e_70b_triviaqa.py \
        --tcal-responses ~/jpwork/metacog-engineering/phase1/results_raw/step1/tcal_greedy_responses_Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED.json \
        --output-dir ~/jpwork/metacog-engineering/phase1/finetune/Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED/e2e_triviaqa \
        --seed 42
"""

import json
import random
import argparse
from pathlib import Path


TRIVIAQA_PROMPT = (
    "You are answering trivia questions. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcal-responses", required=True, type=Path,
                        help="Path to tcal_greedy_responses_*.json from 70B baseline")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-size", type=int, default=30)
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Cap per class (default: min of correct/incorrect count)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load T-cal responses
    with open(args.tcal_responses) as f:
        items = json.load(f)
    print(f"Loaded {len(items)} T-cal items")

    # Filter to items with valid parsed answers (non-empty)
    valid_items = [it for it in items
                   if it.get("parsed_answer") and len(it["parsed_answer"].strip()) > 0]
    print(f"  Valid parsed answers: {len(valid_items)}")

    # Split by correctness
    correct = [it for it in valid_items if it.get("correct", False)]
    incorrect = [it for it in valid_items if not it.get("correct", False)]
    print(f"  Correct: {len(correct)}, Incorrect: {len(incorrect)}")

    # Balance: downsample correct to match incorrect (no oversampling!)
    n_per_class = min(len(correct), len(incorrect))
    if args.max_per_class and args.max_per_class < n_per_class:
        n_per_class = args.max_per_class
    rng.shuffle(correct)
    rng.shuffle(incorrect)
    balanced_correct = correct[:n_per_class]
    balanced_incorrect = incorrect[:n_per_class]
    print(f"  Balanced: {n_per_class} per class = {2 * n_per_class} total")

    # Format as end-to-end messages
    # Use the SAME prompt as eval (TRIVIAQA_PROMPT) so train/eval formats match.
    # Assistant response = model's actual parsed_answer + binary confidence target.
    def format_item(item, is_correct):
        question = item["question"]
        # Use parsed_answer (text before confidence line in raw output)
        answer = item["parsed_answer"].strip()
        conf = 100 if is_correct else 0

        user_msg = TRIVIAQA_PROMPT.format(question=question)
        assistant_msg = f"{answer}\nConfidence: {conf}%"

        return {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        }

    formatted = []
    for it in balanced_correct:
        formatted.append(format_item(it, is_correct=True))
    for it in balanced_incorrect:
        formatted.append(format_item(it, is_correct=False))

    rng.shuffle(formatted)

    # Split: train / valid / test
    n_valid = args.valid_size
    n_test = args.test_size
    n_train = len(formatted) - n_valid - n_test

    if n_train < 100:
        print(f"  WARNING: only {n_train} train items. Consider reducing valid/test size.")

    train = formatted[:n_train]
    valid = formatted[n_train:n_train + n_valid]
    test = formatted[n_train + n_valid:]

    print(f"  Split: train={len(train)}, valid={len(valid)}, test={len(test)}")

    # Save as JSONL
    for split_name, split_data in [("train", train), ("valid", valid), ("test", test)]:
        path = args.output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")
        print(f"  Saved {path} ({len(split_data)} items)")

    # Summary stats
    train_correct = sum(1 for it in train
                        if "Confidence: 100%" in it["messages"][1]["content"])
    print(f"\n  Train balance: {train_correct} correct, {len(train) - train_correct} incorrect")

    # Compute iters for config
    grad_accum = 16
    for epochs in [3, 6, 9]:
        iters = (n_train * epochs) // grad_accum
        print(f"  {epochs} epochs → {iters} iters (grad_accum={grad_accum})")

    # Show a sample
    print(f"\n  Sample train item:")
    sample = train[0]
    print(f"    User: {sample['messages'][0]['content'][:100]}...")
    print(f"    Asst: {sample['messages'][1]['content'][:100]}...")


if __name__ == "__main__":
    main()
