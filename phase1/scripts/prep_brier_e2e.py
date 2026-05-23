#!/usr/bin/env python3
"""
prep_brier_e2e.py — Prepare balanced end-to-end training data for Brier-score PT-CSFT.

Loads GSM8K from HuggingFace, joins with existing greedy responses by ID,
formats as end-to-end chat with coarse confidence suffix (0-9),
balances by correctness, and saves as JSONL.

Usage (Gemma 12B GSM8K, binary targets, coarse mode):
    python3 prep_brier_e2e.py \
        --responses ~/jpwork/metacog-engineering/phase1/results_raw/domain_gen/responses_gsm8k_gemma-3-12b-it.json \
        --domain gsm8k \
        --target-mode binary \
        --coarse \
        --output ~/jpwork/metacog-engineering/phase1/finetune/gemma-3-12b-it/brier_e2e_gsm8k/train.jsonl

Usage (Llama 70B TriviaQA, binary targets, fine mode):
    python3 prep_brier_e2e.py \
        --responses ~/jpwork/metacog-engineering/phase1/results_raw/step1/responses_*.json \
        --domain triviaqa \
        --target-mode binary \
        --output ~/jpwork/metacog-engineering/phase1/finetune/.../train.jsonl
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

SEED = 42

# ──────────────────────────────────────────────────────────────────
# Question loading (mirrors probe_check_domain.py exactly)
# ──────────────────────────────────────────────────────────────────

def load_gsm8k_questions(seed=SEED) -> dict:
    """Load GSM8K test set, shuffle with seed, assign IDs.
    Returns dict: id → question text.
    Mirrors probe_check_domain.py load_gsm8k() exactly.
    """
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    items = []
    for row in ds:
        match = re.search(r"####\s*(.+)", row["answer"])
        gold = match.group(1).strip().replace(",", "") if match else ""
        items.append({
            "id": f"gsm8k_{len(items)}",  # ID assigned BEFORE shuffle
            "question": row["question"],
            "gold_answer": gold,
        })
    random.seed(seed)
    random.shuffle(items)
    # IDs stay as assigned — gsm8k_588 is the 589th row in HF dataset
    return {it["id"]: it["question"] for it in items}


def load_triviaqa_questions(responses: list) -> dict:
    """For TriviaQA, questions should be in the response file already.
    If not, load from T-cal partition.
    """
    # Try to get from responses first
    q_map = {}
    for r in responses:
        qid = r.get("question_id", r.get("id", ""))
        q = r.get("question", r.get("question_text", ""))
        if q:
            q_map[qid] = q
    if q_map:
        return q_map
    raise ValueError("TriviaQA questions not found in response file. "
                     "Need to load from dataset partition.")


# ──────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────

GSM8K_PROMPT_COARSE = (
    "Solve this math problem step by step, then state your confidence "
    "as a single digit from 0 (no confidence) to 9 (very confident) "
    "on the last line in the format 'Confidence: X'.\n\nQuestion: {question}"
)

GSM8K_PROMPT_FINE = (
    "Solve this math problem step by step, then state your confidence "
    "as an integer from 0 to 100 on the last line in the format "
    "'Confidence: X%'.\n\nQuestion: {question}"
)

TRIVIAQA_PROMPT_COARSE = (
    "Answer the following question, then state your confidence "
    "as a single digit from 0 (no confidence) to 9 (very confident) "
    "on the last line in the format 'Confidence: X'.\n\nQuestion: {question}"
)

TRIVIAQA_PROMPT_FINE = (
    "Answer the following question, then state your confidence "
    "as an integer from 0 to 100 on the last line in the format "
    "'Confidence: X%'.\n\nQuestion: {question}"
)


# ──────────────────────────────────────────────────────────────────
# Formatting
# ──────────────────────────────────────────────────────────────────

def strip_existing_confidence(text: str) -> str:
    """Remove any existing 'Confidence: X%' from model output."""
    return re.split(r"(?i)\bconfidence\b", text)[0].rstrip()


def format_item(question: str, raw_output: str, conf_value: int,
                coarse: bool, domain: str) -> dict:
    """Format a single item as end-to-end chat with confidence suffix."""
    answer_clean = strip_existing_confidence(raw_output).strip()

    if domain == "gsm8k":
        prompt_tpl = GSM8K_PROMPT_COARSE if coarse else GSM8K_PROMPT_FINE
    else:
        prompt_tpl = TRIVIAQA_PROMPT_COARSE if coarse else TRIVIAQA_PROMPT_FINE

    prompt = prompt_tpl.format(question=question)

    if coarse:
        assistant_text = f"{answer_clean}\nConfidence: {conf_value}"
    else:
        assistant_text = f"{answer_clean}\nConfidence: {conf_value}%"

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_text},
        ]
    }


# ──────────────────────────────────────────────────────────────────
# Balancing
# ──────────────────────────────────────────────────────────────────

def balance_by_correctness(items: list, seed: int = 42) -> list:
    """Balance dataset to equal correct/incorrect items."""
    correct = [it for it in items if it["_correct"]]
    incorrect = [it for it in items if not it["_correct"]]

    rng = random.Random(seed)
    n_min = min(len(correct), len(incorrect))

    if n_min == 0:
        print(f"WARNING: one class is empty (correct={len(correct)}, "
              f"incorrect={len(incorrect)}). Returning unbalanced.",
              file=sys.stderr)
        return items

    if len(correct) > n_min:
        correct = rng.sample(correct, n_min)
    if len(incorrect) > n_min:
        incorrect = rng.sample(incorrect, n_min)

    balanced = correct + incorrect
    rng.shuffle(balanced)
    print(f"Balanced: {n_min} correct + {n_min} incorrect = {len(balanced)} items")
    return balanced


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare balanced end-to-end data for Brier-score PT-CSFT"
    )
    parser.add_argument("--responses", required=True, type=Path,
                        help="Path to greedy response JSON")
    parser.add_argument("--domain", choices=["gsm8k", "triviaqa"], required=True)
    parser.add_argument("--target-mode", choices=["binary"], default="binary",
                        help="binary: 0/max from correctness")
    parser.add_argument("--coarse", action="store_true",
                        help="Coarse mode: confidence 0-9 (required for Gemma)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=int, default=50,
                        help="Number of items for validation split")
    parser.add_argument("--use-cal-only", action="store_true",
                        help="Use only calibration items (first 800) for training")
    args = parser.parse_args()

    max_conf = 9 if args.coarse else 100

    # ── Load responses ──
    with open(args.responses) as f:
        responses = json.load(f)
    print(f"Loaded {len(responses)} responses from {args.responses}")

    # ── Load questions ──
    if args.domain == "gsm8k":
        print("Loading GSM8K questions from HuggingFace...")
        questions = load_gsm8k_questions(seed=SEED)
        print(f"Loaded {len(questions)} questions")
    else:
        questions = load_triviaqa_questions(responses)
        print(f"Loaded {len(questions)} questions from response file")

    # ── Join and format ──
    items = []
    skipped = 0

    for r in responses:
        rid = r.get("id", r.get("question_id", ""))
        question = questions.get(rid)
        if question is None:
            skipped += 1
            continue

        correct = bool(r.get("correct", False))
        raw_output = r.get("raw_output", r.get("raw_response", r.get("response", "")))

        if not raw_output:
            skipped += 1
            continue

        # Confidence target
        conf_value = max_conf if correct else 0

        formatted = format_item(question, raw_output, conf_value,
                                coarse=args.coarse, domain=args.domain)
        formatted["_id"] = rid
        formatted["_correct"] = correct
        formatted["_conf_target"] = conf_value
        items.append(formatted)

    print(f"Formatted {len(items)} items (skipped {skipped})")
    print(f"  Correct: {sum(1 for it in items if it['_correct'])}")
    print(f"  Incorrect: {sum(1 for it in items if not it['_correct'])}")

    # ── Balance ──
    items = balance_by_correctness(items, args.seed)

    # ── Train/val split ──
    rng = random.Random(args.seed + 1)
    rng.shuffle(items)

    val_n = min(args.val_split, len(items) // 5)
    val_items = items[:val_n]
    train_items = items[val_n:]
    print(f"Split: {len(train_items)} train, {len(val_items)} val")

    # ── Save ──
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        for it in train_items:
            out = {
                "messages": it["messages"],
                "metadata": {
                    "id": it["_id"],
                    "correct": it["_correct"],
                    "conf_target": it["_conf_target"],
                    "coarse": args.coarse,
                },
            }
            f.write(json.dumps(out) + "\n")

    val_path = args.output.with_name(args.output.stem + "_val" + args.output.suffix)
    with open(val_path, "w") as f:
        for it in val_items:
            out = {
                "messages": it["messages"],
                "metadata": {
                    "id": it["_id"],
                    "correct": it["_correct"],
                    "conf_target": it["_conf_target"],
                    "coarse": args.coarse,
                },
            }
            f.write(json.dumps(out) + "\n")

    print(f"\nSaved: {args.output} ({len(train_items)} items)")
    print(f"Saved: {val_path} ({len(val_items)} items)")

    # ── Stats ──
    targets = [it["_conf_target"] for it in train_items]
    correct_rate = sum(1 for it in train_items if it["_correct"]) / len(train_items)
    print(f"\nTrain stats:")
    print(f"  Correct rate: {correct_rate:.3f}")
    print(f"  Conf target distribution: {Counter(targets)}")
    print(f"  Mode: {'coarse (0-9)' if args.coarse else 'fine (0-100)'}")

    # Spot check
    print(f"\n--- Spot check (first item) ---")
    first = train_items[0]
    user_msg = first["messages"][0]["content"]
    asst_msg = first["messages"][1]["content"]
    print(f"User (last 80 chars): ...{user_msg[-80:]}")
    print(f"Assistant (last 80 chars): ...{asst_msg[-80:]}")
    print(f"Correct: {first['_correct']}, Target: {first['_conf_target']}")


if __name__ == "__main__":
    main()
