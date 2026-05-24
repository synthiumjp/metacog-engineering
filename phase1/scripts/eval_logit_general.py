#!/usr/bin/env python3
"""eval_logit_general.py — Generalised logit-based confidence eval.

Supports:
  - Coarse mode (0-9): Gemma GSM8K style
  - Fine mode (0-100): Llama/Qwen TriviaQA style
  - Auto-detects digit token IDs from tokenizer
  - GSM8K and TriviaQA benchmarks

Usage (Gemma 12B, GSM8K, coarse):
    python3 eval_logit_general.py \
        --model-path ~/models/gemma-3-12b-it \
        --adapter-path ~/adapters/adapter_e2e_ce \
        --benchmark gsm8k --mode coarse \
        --output results.json

Usage (Llama 8B, GSM8K, coarse):
    python3 eval_logit_general.py \
        --model-path ~/models/Meta-Llama-3.1-8B-Instruct-bf16 \
        --adapter-path ~/adapters/adapter_e2e_ce \
        --benchmark gsm8k --mode coarse \
        --output results.json

Usage (Qwen 32B, TriviaQA, fine):
    python3 eval_logit_general.py \
        --model-path ~/models/Qwen2.5-32B-Instruct-bf16 \
        --adapter-path ~/adapters/probe_target \
        --benchmark triviaqa --mode fine \
        --output results.json
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load, generate

SEED = 42

# ──────────────────────────────────────────────────────────────
# Token ID detection
# ──────────────────────────────────────────────────────────────

def detect_digit_token_ids(tokenizer, mode="coarse"):
    """Auto-detect token IDs for confidence values from tokenizer.

    coarse: digits 0-9 (10 tokens)
    fine: integers 0-100 (up to 101 tokens, only single-token ones)
    """
    if mode == "coarse":
        values = list(range(10))
    else:
        values = list(range(101))

    token_map = {}  # value -> token_id
    for v in values:
        ids = tokenizer.encode(str(v), add_special_tokens=False)
        if len(ids) == 1:
            token_map[v] = ids[0]
        # Multi-token values are skipped — can't read P(value) from one position

    print(f"  Detected {len(token_map)}/{len(values)} single-token confidence values")
    if mode == "coarse":
        assert len(token_map) == 10, f"Expected 10 coarse tokens, got {len(token_map)}"
        print(f"  Token IDs: {[token_map[d] for d in range(10)]}")
    else:
        missing = [v for v in values if v not in token_map]
        if missing:
            print(f"  Multi-token (skipped): {missing[:10]}{'...' if len(missing) > 10 else ''}")

    return token_map


def compute_expected_confidence(logits_at_pos, token_map, mode="coarse"):
    """Compute expected confidence from logits at the confidence token position.

    Returns: (expected_conf_0_100, argmax_value, value_probs_dict)
    """
    max_val = 9 if mode == "coarse" else 100

    # Extract logits for valid confidence tokens
    token_ids = mx.array([token_map[v] for v in sorted(token_map.keys())])
    values = sorted(token_map.keys())

    conf_logits = logits_at_pos[token_ids]
    probs = mx.softmax(conf_logits)
    probs_np = np.array(probs.astype(mx.float32))

    # Expected value
    values_np = np.array(values, dtype=np.float64)
    expected_raw = float(np.sum(probs_np * values_np))
    expected_conf = expected_raw / max_val * 100.0  # normalise to 0-100

    # Argmax
    argmax_idx = int(np.argmax(probs_np))
    argmax_value = values[argmax_idx]

    # Build probs dict for saving
    value_probs = {str(v): float(probs_np[i]) for i, v in enumerate(values)}

    return expected_conf, argmax_value, value_probs


# ──────────────────────────────────────────────────────────────
# Benchmark loading
# ──────────────────────────────────────────────────────────────

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
    return items


def load_triviaqa_eval():
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    items = []
    for row in ds:
        items.append({
            "id": row["question_id"],
            "question": row["question"],
            "aliases": row["answer"]["aliases"],
            "gold_answer": row["answer"]["value"],
        })
    random.seed(SEED)
    random.shuffle(items)
    return items[:1000]  # T-eval: first 1000 after shuffle


def is_correct_gsm8k(predicted, gold):
    try:
        return abs(float(predicted.replace(",", "")) - float(gold)) < 1e-5
    except (ValueError, TypeError):
        return False


def is_correct_triviaqa(predicted, aliases):
    if not predicted:
        return False
    pred_lower = predicted.lower().strip()
    for alias in aliases:
        alias_lower = alias.lower().strip()
        if pred_lower in alias_lower or alias_lower in pred_lower:
            if min(len(pred_lower), len(alias_lower)) >= 2:
                return True
    return False


# ──────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────

GSM8K_PROMPT_COARSE = (
    "Solve this math problem step by step. After your solution, "
    "state your confidence as a single digit from 0 (no confidence) to 9 (very confident) "
    "in the format 'Confidence: D'.\n\n"
    "Question: {question}"
)

GSM8K_PROMPT_FINE = (
    "Solve this math problem step by step. After your solution, "
    "state your confidence as a percentage from 0 to 100 "
    "in the format 'Confidence: N%'.\n\n"
    "Question: {question}"
)

TRIVIAQA_PROMPT = (
    "You are answering trivia questions. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
)


def get_prompt(benchmark, mode):
    if benchmark == "gsm8k":
        return GSM8K_PROMPT_COARSE if mode == "coarse" else GSM8K_PROMPT_FINE
    else:
        return TRIVIAQA_PROMPT


# ──────────────────────────────────────────────────────────────
# Answer/confidence parsing
# ──────────────────────────────────────────────────────────────

def parse_gsm8k_answer(text):
    """Extract final numeric answer from CoT."""
    patterns = [
        r"(?:the\s+)?answer\s+is\s*:?\s*\$?([+-]?\d[\d,]*\.?\d*)",
        r"####\s*([+-]?\d[\d,]*\.?\d*)",
    ]
    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        if matches:
            return matches[-1].replace(",", "")
    numbers = re.findall(r"[+-]?\d[\d,]*\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else ""


def parse_triviaqa_answer(text):
    """Extract answer before confidence line."""
    lines = text.strip().split("\n")
    answer_lines = []
    for line in lines:
        if re.match(r"^\s*confidence\s*:", line, re.IGNORECASE):
            break
        answer_lines.append(line)
    return "\n".join(answer_lines).strip()


def parse_confidence_text(text, mode="coarse"):
    """Parse confidence from generated text (argmax)."""
    if mode == "coarse":
        match = re.search(r"[Cc]onfidence:\s*(\d)", text)
        if match:
            val = int(match.group(1))
            if 0 <= val <= 9:
                return val * (100.0 / 9.0)
    else:
        match = re.search(r"[Cc]onfidence:\s*(\d+)\s*%?", text)
        if match:
            val = int(match.group(1))
            if 0 <= val <= 100:
                return float(val)
    return float("nan")


# ──────────────────────────────────────────────────────────────
# Logit extraction
# ──────────────────────────────────────────────────────────────

def find_confidence_token_position(tokens, token_map, mode):
    """Find the position of the confidence digit/number token in the sequence."""
    digit_set = set(token_map.values())

    # Search backwards for the last confidence token
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] in digit_set:
            return i
    return None


def get_logit_confidence(model, tokenizer, response_text, token_map, mode):
    """Get logit-based confidence by reading P(digit) at confidence position.

    1. Encode the full response
    2. Forward pass to get logits at every position
    3. Find the confidence token position
    4. Extract digit probabilities and compute expected value
    """
    # Encode
    tokens = tokenizer.encode(response_text, add_special_tokens=False)
    pos = find_confidence_token_position(tokens, token_map, mode)

    if pos is None or pos < 1:
        return None, None, None

    # Forward pass on tokens up to (but not including) the confidence token
    # to get logits that predict the confidence token
    input_ids = mx.array(tokens[:pos])[None, :]  # (1, seq_len)
    logits = model(input_ids)  # (1, seq_len, vocab)

    # Logits at the last position predict the next token (= confidence token)
    conf_logits = logits[0, -1, :]  # (vocab,)

    expected_conf, argmax_val, value_probs = compute_expected_confidence(
        conf_logits, token_map, mode
    )

    return expected_conf, argmax_val, value_probs


# ──────────────────────────────────────────────────────────────
# AUROC₂
# ──────────────────────────────────────────────────────────────

def auroc2(confidences, corrects):
    from sklearn.metrics import roc_auc_score
    valid = ~np.isnan(confidences)
    if valid.sum() < 10:
        return float("nan")
    try:
        return roc_auc_score(corrects[valid], confidences[valid])
    except ValueError:
        return float("nan")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--benchmark", default="gsm8k", choices=["gsm8k", "triviaqa"])
    parser.add_argument("--mode", default="coarse", choices=["coarse", "fine"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}")
    print(f"Loading adapter from {args.adapter_path}")
    model, tokenizer = load(args.model_path, adapter_path=args.adapter_path)

    # Detect token IDs
    print(f"Mode: {args.mode}")
    token_map = detect_digit_token_ids(tokenizer, args.mode)

    # Load eval data
    if args.benchmark == "gsm8k":
        eval_items = load_gsm8k_eval()
    else:
        eval_items = load_triviaqa_eval()

    if args.max_items:
        eval_items = eval_items[:args.max_items]
    print(f"Evaluating on {len(eval_items)} items")

    prompt_template = get_prompt(args.benchmark, args.mode)
    results = []
    t0 = time.time()

    for i, item in enumerate(eval_items):
        user_msg = prompt_template.format(question=item["question"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )

        # Generate
        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=args.max_tokens, verbose=False,
        )

        # Parse answer and check correctness
        if args.benchmark == "gsm8k":
            answer_text = parse_gsm8k_answer(response)
            correct = is_correct_gsm8k(answer_text, item["gold_answer"])
        else:
            answer_text = parse_triviaqa_answer(response)
            correct = is_correct_triviaqa(answer_text, item.get("aliases", []))

        # Parse text confidence (argmax)
        text_conf = parse_confidence_text(response, args.mode)

        # Get logit confidence
        full_text = prompt + response
        logit_conf, logit_argmax, value_probs = get_logit_confidence(
            model, tokenizer, full_text, token_map, args.mode
        )

        results.append({
            "id": item.get("id", item.get("question_id", f"item_{i}")),
            "correct": correct,
            "text_confidence": text_conf,
            "logit_confidence": logit_conf,
            "logit_argmax_value": logit_argmax,
            "value_probs": value_probs,
            "answer": answer_text,
            "raw_output": response,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(eval_items)} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"Generation complete: {len(results)} items in {elapsed:.0f}s")

    # ── Compute metrics ──
    corrects = np.array([r["correct"] for r in results], dtype=int)
    text_confs = np.array([r["text_confidence"] for r in results])
    logit_confs = np.array([r["logit_confidence"] if r["logit_confidence"] is not None
                            else float("nan") for r in results])

    accuracy = corrects.mean()
    text_auroc = auroc2(text_confs, corrects)
    logit_auroc = auroc2(logit_confs, corrects)

    text_mean = float(np.nanmean(text_confs))
    text_std = float(np.nanstd(text_confs))
    logit_mean = float(np.nanmean(logit_confs))
    logit_std = float(np.nanstd(logit_confs))
    logit_valid = int(np.sum(~np.isnan(logit_confs)))

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Accuracy:          {accuracy:.3f}")
    print(f"  --- Argmax (text-parsed) ---")
    print(f"  AUROC₂:           {text_auroc:.3f}")
    print(f"  Conf mean:         {text_mean:.1f}")
    print(f"  Conf std:          {text_std:.1f}")
    print(f"  --- Logit (expected value) ---")
    print(f"  AUROC₂:           {logit_auroc:.3f}")
    print(f"  Conf mean:         {logit_mean:.1f}")
    print(f"  Conf std:          {logit_std:.1f}")
    print(f"  Valid logit items: {logit_valid}/{len(results)}")
    print(f"  --- Info ---")
    print(f"  Benchmark:         {args.benchmark}")
    print(f"  Mode:              {args.mode}")
    print(f"{'='*60}")

    # Save metrics
    metrics = {
        "benchmark": args.benchmark,
        "mode": args.mode,
        "model": args.model_path,
        "adapter": args.adapter_path,
        "n_eval": len(results),
        "accuracy": float(accuracy),
        "text_auroc2": float(text_auroc),
        "text_conf_mean": text_mean,
        "text_conf_std": text_std,
        "logit_auroc2": float(logit_auroc),
        "logit_conf_mean": logit_mean,
        "logit_conf_std": logit_std,
        "logit_valid": logit_valid,
        "elapsed_s": elapsed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {args.output}")

    # Save per-item responses
    resp_path = args.output.with_name(args.output.stem + "_responses.json")
    with open(resp_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {resp_path}")


if __name__ == "__main__":
    main()
