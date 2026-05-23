#!/usr/bin/env python3
"""
eval_e2e.py — Evaluate end-to-end PT-CSFT on GSM8K or TriviaQA.

Single-pass: model generates CoT + answer + confidence in one pass
with the adapter loaded. Parses answer + confidence from the output.

Usage:
    python3 eval_e2e.py \
        --model-path ~/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it \
        --adapter-path ~/jpwork/metacog-engineering/phase1/finetune/gemma-3-12b-it/brier_e2e_gsm8k/adapter_e2e_ce \
        --benchmark gsm8k \
        --coarse \
        --output ~/jpwork/metacog-engineering/phase1/results_raw/domain_gen/e2e_ce_gsm8k_gemma-3-12b-it.json
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

SEED = 42

# ──────────────────────────────────────────────────────────────────
# Data loading (mirrors probe_check_domain.py)
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


def load_gsm8k_eval(seed=SEED):
    """Load GSM8K eval split (items 800+)."""
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
    # Cal: 0-799, Eval: 800+
    return items[800:]


def is_correct_gsm8k(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold (last number extraction)."""
    # Extract last number from predicted
    numbers = re.findall(r"[-+]?\d*\.?\d+", predicted.replace(",", ""))
    if not numbers:
        return False
    pred_num = numbers[-1]
    # Normalise
    try:
        return float(pred_num) == float(gold)
    except ValueError:
        return pred_num.strip() == gold.strip()


# ──────────────────────────────────────────────────────────────────
# Confidence parsing
# ──────────────────────────────────────────────────────────────────

def parse_confidence(text: str, coarse: bool) -> float:
    """Parse confidence from model output. Returns value in [0, 100] or NaN."""
    # Strip everything after confidence for answer parsing
    # Pattern: "Confidence: X" (coarse) or "Confidence: X%" (fine)
    patterns = [
        re.compile(r"[Cc]onfidence\s*:?\s*(\d{1,3})\s*%?"),
        re.compile(r"(\d{1,3})\s*%\s*$"),
        re.compile(r"\b(\d{1,3})\s*$"),
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            val = int(m.group(1))
            if coarse:
                # Map 0-9 to 0-100 scale for AUROC₂ comparison
                if 0 <= val <= 9:
                    return val * (100.0 / 9.0)
            else:
                if 0 <= val <= 100:
                    return float(val)
    return float("nan")


def parse_answer_before_confidence(text: str) -> str:
    """Extract the answer portion before any confidence statement."""
    # Split on confidence
    parts = re.split(r"(?i)\bconfidence\b", text)
    return parts[0].strip()


# ──────────────────────────────────────────────────────────────────
# AUROC₂
# ──────────────────────────────────────────────────────────────────

def auroc2(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Compute Type-2 AUROC."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(correct)) < 2:
        return float("nan")
    return roc_auc_score(correct, confidence)


# ──────────────────────────────────────────────────────────────────
# VRS screening
# ──────────────────────────────────────────────────────────────────

def vrs_screen(confidence: np.ndarray, correct: np.ndarray) -> dict:
    """Quick VRS screening: L, TRIN, r."""
    # Ceiling rate L: fraction at max confidence
    max_conf = np.max(confidence)
    L = np.mean(confidence >= max_conf - 1)

    # TRIN: top-response index of non-variability
    # Fraction of responses using the top 2 values
    sorted_conf = np.sort(np.unique(confidence))
    if len(sorted_conf) >= 2:
        top2 = sorted_conf[-2:]
        TRIN = np.mean(np.isin(confidence, top2))
    else:
        TRIN = 1.0

    # Correlation
    r = float(np.corrcoef(confidence, correct)[0, 1]) if len(np.unique(confidence)) > 1 else 0.0

    return {"L": round(L, 3), "TRIN": round(TRIN, 3), "r": round(r, 3)}


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--benchmark", default="gsm8k", choices=["gsm8k"])
    parser.add_argument("--coarse", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-items", type=int, default=None,
                        help="Limit eval items (for quick testing)")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load, generate

    # Load model + adapter
    print(f"Loading model from {args.model_path}")
    print(f"Loading adapter from {args.adapter_path}")
    model, tokenizer = load(
        args.model_path,
        adapter_path=args.adapter_path,
    )

    # Load eval items
    eval_items = load_gsm8k_eval()
    if args.max_items:
        eval_items = eval_items[:args.max_items]
    print(f"Evaluating on {len(eval_items)} items")

    # Greedy sampler
    def greedy(logits):
        return mx.argmax(logits, axis=-1)

    # Generate
    results = []
    start = time.time()

    for i, item in enumerate(eval_items):
        if args.coarse:
            prompt = GSM8K_PROMPT_COARSE.format(question=item["question"])
        else:
            prompt = GSM8K_PROMPT_FINE.format(question=item["question"])

        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = generate(
            model, tokenizer, prompt=chat_prompt,
            max_tokens=args.max_tokens, sampler=greedy, verbose=False,
        )

        # Parse
        answer_text = parse_answer_before_confidence(response)
        correct = is_correct_gsm8k(answer_text, item["gold_answer"])
        confidence = parse_confidence(response, coarse=args.coarse)

        results.append({
            "id": item["id"],
            "gold": item["gold_answer"],
            "raw_output": response,
            "correct": correct,
            "confidence": confidence,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  {i+1}/{len(eval_items)} ({elapsed:.0f}s)")

    elapsed = time.time() - start
    print(f"Generation complete: {len(results)} items in {elapsed:.0f}s")

    # ── Compute metrics ──
    correct_arr = np.array([r["correct"] for r in results], dtype=int)
    conf_arr = np.array([r["confidence"] for r in results], dtype=float)
    parseable = ~np.isnan(conf_arr)

    accuracy = np.mean(correct_arr)
    parse_rate = np.mean(parseable)

    conf_valid = conf_arr[parseable]
    correct_valid = correct_arr[parseable]

    if len(conf_valid) > 0 and len(np.unique(correct_valid)) == 2:
        auc = auroc2(conf_valid, correct_valid)
    else:
        auc = float("nan")

    vrs = vrs_screen(conf_valid, correct_valid) if len(conf_valid) > 10 else {}

    metrics = {
        "benchmark": "gsm8k",
        "model": args.model_path,
        "adapter": args.adapter_path,
        "mode": "coarse" if args.coarse else "fine",
        "n_eval": len(results),
        "accuracy": round(float(accuracy), 3),
        "parse_rate": round(float(parse_rate), 3),
        "auroc2": round(float(auc), 3) if not np.isnan(auc) else None,
        "conf_mean": round(float(np.mean(conf_valid)), 1) if len(conf_valid) > 0 else None,
        "conf_std": round(float(np.std(conf_valid)), 1) if len(conf_valid) > 0 else None,
        "vrs": vrs,
        "elapsed_s": round(elapsed, 1),
    }

    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Accuracy:  {metrics['accuracy']}")
    print(f"  Parse rate: {metrics['parse_rate']}")
    print(f"  AUROC₂:   {metrics['auroc2']}")
    print(f"  Conf mean: {metrics['conf_mean']}")
    print(f"  Conf std:  {metrics['conf_std']}")
    print(f"  VRS:       {metrics['vrs']}")
    print(f"  Baseline:  0.546 (verbal), 0.769 (probe)")
    print(f"{'='*50}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save per-item responses
    resp_path = args.output.with_name(args.output.stem + "_responses.json")
    with open(resp_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {args.output}")
    print(f"Saved: {resp_path}")


if __name__ == "__main__":
    main()
