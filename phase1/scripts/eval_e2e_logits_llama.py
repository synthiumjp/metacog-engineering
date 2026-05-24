#!/usr/bin/env python3
"""
eval_e2e_logits.py — Evaluate end-to-end PT-CSFT using logit probabilities
over confidence digit tokens, not the argmax text output.

The model generates as normal, but instead of parsing "Confidence: 9" from text,
we feed the full response back through the model and read P(digit=d) at the
confidence token position. Expected confidence = Σ (d/9) × P(d).

This tests whether the model's uncertainty is already calibrated at the logit level
even when the argmax output is binary (0 or 9).

Usage:
    python3 eval_e2e_logits.py \
        --model-path ~/mnt/models-lan/.../gemma-3-12b-it \
        --adapter-path ~/jpwork/.../adapter_e2e_ce \
        --benchmark gsm8k \
        --output ~/jpwork/.../e2e_ce_logits_gsm8k.json
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load, generate

SEED = 42

# Gemma digit token IDs (index = digit value)
DIGIT_TOKEN_IDS = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # Llama 3.1

GSM8K_PROMPT = (
    "Solve this math problem step by step, then state your confidence "
    "as a single digit from 0 (no confidence) to 9 (very confident) "
    "on the last line in the format 'Confidence: X'.\n\nQuestion: {question}"
)


def load_gsm8k_eval(seed=SEED):
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
    return items[800:]


def is_correct_gsm8k(predicted, gold):
    numbers = re.findall(r"[-+]?\d*\.?\d+", predicted.replace(",", ""))
    if not numbers:
        return False
    try:
        return float(numbers[-1]) == float(gold)
    except ValueError:
        return numbers[-1].strip() == gold.strip()


def get_logit_confidence(model, tokenizer, full_text):
    """Feed full text through model, read logit distribution at confidence position.

    Returns: (expected_confidence_0_100, argmax_digit, digit_probs)
    """
    tokens = tokenizer.encode(full_text)
    digit_set = set(DIGIT_TOKEN_IDS)

    # Find last digit token position in the sequence
    conf_pos = None
    for i in range(len(tokens) - 1, max(0, len(tokens) - 15), -1):
        if tokens[i] in digit_set:
            conf_pos = i
            break

    if conf_pos is None:
        return float("nan"), -1, []

    # Forward pass up to conf_pos (we need logits at conf_pos-1 predicting conf_pos)
    input_tokens = mx.array([tokens[:conf_pos]])
    logits = model(input_tokens)  # (1, seq_len, vocab_size)

    # Logits at the last position predict the confidence token
    conf_logits = logits[0, -1, :]  # (vocab_size,)

    # Extract logits for digit tokens
    digit_ids_mx = mx.array(DIGIT_TOKEN_IDS)
    digit_logits = conf_logits[digit_ids_mx]  # (10,)

    # Softmax → probability distribution over digits
    digit_probs = mx.softmax(digit_logits)
    digit_probs_np = np.array(digit_probs.astype(mx.float32))

    # Expected confidence = Σ (d/9) × P(d), scaled to 0-100
    scores = np.arange(10) / 9.0
    expected_conf = float(np.sum(digit_probs_np * scores) * 100.0)

    # Argmax digit
    argmax_digit = int(np.argmax(digit_probs_np))

    mx.eval(logits)
    return expected_conf, argmax_digit, digit_probs_np.tolist()


def parse_confidence_text(text):
    """Parse confidence from generated text (argmax method)."""
    patterns = [
        re.compile(r"[Cc]onfidence\s*:?\s*(\d{1,3})\s*%?"),
        re.compile(r"(\d{1,3})\s*%\s*$"),
        re.compile(r"\b(\d{1,3})\s*$"),
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 9:
                return val * (100.0 / 9.0)
    return float("nan")


def parse_answer_before_confidence(text):
    parts = re.split(r"(?i)\bconfidence\b", text)
    return parts[0].strip()


def auroc2(confidence, correct):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(correct)) < 2:
        return float("nan")
    return roc_auc_score(correct, confidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--benchmark", default="gsm8k")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}")
    print(f"Loading adapter from {args.adapter_path}")
    model, tokenizer = load(args.model_path, adapter_path=args.adapter_path)

    eval_items = load_gsm8k_eval()
    if args.max_items:
        eval_items = eval_items[:args.max_items]
    print(f"Evaluating on {len(eval_items)} items")

    def greedy(logits):
        return mx.argmax(logits, axis=-1)

    results = []
    start = time.time()

    for i, item in enumerate(eval_items):
        prompt = GSM8K_PROMPT.format(question=item["question"])
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate response
        response = generate(
            model, tokenizer, prompt=chat_prompt,
            max_tokens=args.max_tokens, sampler=greedy, verbose=False,
        )

        # Parse answer + text-based confidence
        answer_text = parse_answer_before_confidence(response)
        correct = is_correct_gsm8k(answer_text, item["gold_answer"])
        text_conf = parse_confidence_text(response)

        # Get logit-based confidence
        full_text = chat_prompt + response
        logit_conf, argmax_digit, digit_probs = get_logit_confidence(
            model, tokenizer, full_text
        )

        results.append({
            "id": item["id"],
            "gold": item["gold_answer"],
            "correct": correct,
            "text_confidence": text_conf,
            "logit_confidence": logit_conf,
            "argmax_digit": argmax_digit,
            "digit_probs": digit_probs,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  {i+1}/{len(eval_items)} ({elapsed:.0f}s)")

        if (i + 1) % 100 == 0:
            mx.clear_cache()

    elapsed = time.time() - start
    print(f"Generation complete: {len(results)} items in {elapsed:.0f}s")

    # ── Compute metrics ──
    correct_arr = np.array([r["correct"] for r in results], dtype=int)

    # Text-based (argmax) confidence
    text_conf = np.array([r["text_confidence"] for r in results])
    text_valid = ~np.isnan(text_conf)

    # Logit-based (expected value) confidence
    logit_conf = np.array([r["logit_confidence"] for r in results])
    logit_valid = ~np.isnan(logit_conf)

    # AUROC₂ for both methods
    text_auroc = auroc2(text_conf[text_valid], correct_arr[text_valid])
    logit_auroc = auroc2(logit_conf[logit_valid], correct_arr[logit_valid])

    metrics = {
        "benchmark": "gsm8k",
        "model": args.model_path,
        "adapter": args.adapter_path,
        "n_eval": len(results),
        "accuracy": round(float(np.mean(correct_arr)), 3),
        "text_auroc2": round(float(text_auroc), 3) if not np.isnan(text_auroc) else None,
        "text_conf_mean": round(float(np.nanmean(text_conf)), 1),
        "text_conf_std": round(float(np.nanstd(text_conf)), 1),
        "logit_auroc2": round(float(logit_auroc), 3) if not np.isnan(logit_auroc) else None,
        "logit_conf_mean": round(float(np.nanmean(logit_conf)), 1),
        "logit_conf_std": round(float(np.nanstd(logit_conf)), 1),
        "elapsed_s": round(elapsed, 1),
    }

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Accuracy:          {metrics['accuracy']}")
    print(f"  --- Argmax (text-parsed) ---")
    print(f"  AUROC₂:           {metrics['text_auroc2']}")
    print(f"  Conf mean:         {metrics['text_conf_mean']}")
    print(f"  Conf std:          {metrics['text_conf_std']}")
    print(f"  --- Logit (expected value) ---")
    print(f"  AUROC₂:           {metrics['logit_auroc2']}")
    print(f"  Conf mean:         {metrics['logit_conf_mean']}")
    print(f"  Conf std:          {metrics['logit_conf_std']}")
    print(f"  --- Baseline ---")
    print(f"  Baseline AUROC₂:   0.546 (verbal), 0.769 (probe)")
    print(f"{'='*60}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    resp_path = args.output.with_name(args.output.stem + "_responses.json")
    with open(resp_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {args.output}")
    print(f"Saved: {resp_path}")


if __name__ == "__main__":
    main()
