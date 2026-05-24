#!/usr/bin/env python3
"""gen_gsm8k_responses.py — Generate greedy responses on GSM8K test set.

Produces response file in the same format as the Gemma 12B responses:
[{id, raw_output, gold, correct, confidence}, ...]

Usage:
    python3 gen_gsm8k_responses.py \
        --model-path ~/mnt/models-lan/foresight/synthesis-archive/Meta-Llama-3.1-8B-Instruct-bf16 \
        --output ~/jpwork/metacog-engineering/phase1/results_raw/domain_gen/responses_gsm8k_Meta-Llama-3.1-8B-Instruct-bf16.json
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate

SEED = 42

GSM8K_PROMPT = (
    "Solve this math problem step by step.\n\n"
    "Question: {question}"
)


def load_gsm8k(seed=SEED):
    """Load GSM8K test set with same ID scheme as probe_check_domain.py."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    items = []
    for row in ds:
        match = re.search(r"####\s*(.+)", row["answer"])
        gold = match.group(1).strip().replace(",", "") if match else ""
        items.append({
            "id": f"gsm8k_{len(items)}",
            "question": row["question"],
            "gold_answer": gold,
        })
    random.seed(seed)
    random.shuffle(items)
    return items


def extract_final_number(text: str) -> str:
    """Extract the final number from a CoT response."""
    # Look for "the answer is X" patterns
    patterns = [
        r"(?:the\s+)?answer\s+is\s*:?\s*\$?([+-]?\d[\d,]*\.?\d*)",
        r"####\s*([+-]?\d[\d,]*\.?\d*)",
        r"=\s*\$?([+-]?\d[\d,]*\.?\d*)\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].replace(",", "")

    # Fallback: last number in text
    numbers = re.findall(r"[+-]?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def is_correct(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold."""
    try:
        return abs(float(predicted) - float(gold)) < 1e-5
    except (ValueError, TypeError):
        return predicted.strip() == gold.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}")
    model, tokenizer = load(args.model_path)

    items = load_gsm8k()
    if args.max_items:
        items = items[:args.max_items]
    print(f"Generating on {len(items)} items")

    results = []
    t0 = time.time()

    for i, item in enumerate(items):
        user_msg = GSM8K_PROMPT.format(question=item["question"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=args.max_tokens, verbose=False,
        )

        predicted = extract_final_number(response)
        correct = is_correct(predicted, item["gold_answer"])

        results.append({
            "id": item["id"],
            "raw_output": response,
            "gold": item["gold_answer"],
            "correct": correct,
            "confidence": None,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            n_correct = sum(1 for r in results if r["correct"])
            print(f"  {i+1}/{len(items)} ({elapsed:.0f}s) acc={n_correct/(i+1):.3f}")

    elapsed = time.time() - t0
    n_correct = sum(1 for r in results if r["correct"])
    print(f"\nDone: {len(results)} items in {elapsed:.0f}s")
    print(f"Accuracy: {n_correct}/{len(results)} = {n_correct/len(results):.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
