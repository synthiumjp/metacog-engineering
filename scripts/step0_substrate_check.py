"""
Step 0: Substrate Pre-Check
============================

Phase 0 v4, pre-reg v2. Runs 10 × T=0.7 samples on 500 TriviaQA items
disjoint from T-eval, T-cal, and the saturation paper's 524.

Purpose: verify that Gemma 3 4B-it produces parseable answer + confidence
responses and exhibits non-degenerate answer variability on TriviaQA under
the Phase 0 prompt and sampling configuration.

Pass criteria:
    - Confidence parse rate >= 80%
    - Answer parse rate >= 90%
    - At least 20% of items show answer variability (not all 10 samples
      produce the same normalised answer)

If fail: one re-run permitted with substrate filtering (question-length or
entity-rarity threshold). If that also fails, Phase 0 is reported as
technically incomplete.

Outputs:
    D:\\metacog\\data\\step0_substrate_check.json
        - per-item: question_id, 10 raw responses, parsed answers,
          parsed confidences, n_unique_answers, n_correct
        - aggregate: pass/fail, parse rates, variability rate

Runtime: ~1 hour (500 items × 10 samples × ~0.7s per sample).
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from utils_phase0 import (
    is_correct_triviaqa,
    parse_response,
    partition_triviaqa_pool,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "google/gemma-3-4b-it"
SEED = 42
N_STEP0 = 500
N_SAMPLES = 10
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 64

PROJECT_ROOT = Path(r"D:\metacog")
OUTPUT_PATH = PROJECT_ROOT / "data" / "step0_substrate_check.json"

# Pass criteria
MIN_CONFIDENCE_PARSE_RATE = 0.80
MIN_ANSWER_PARSE_RATE = 0.90
MIN_VARIABILITY_RATE = 0.20

TRIVIAQA_PROMPT = (
    "You are answering trivia questions. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
)


def build_prompt(tokenizer, question: str) -> str:
    user_msg = TRIVIAQA_PROMPT.format(question=question)
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_step0_items(seed: int) -> list[dict]:
    """Load the Step 0 pre-check set (500 items), disjoint from everything."""
    partition = partition_triviaqa_pool(seed=seed)
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")

    assert len(ds) == 17_944, (
        f"TriviaQA rc.nocontext validation has {len(ds)} items, expected 17,944."
    )

    items = []
    for i in partition["step0"]:
        ex = ds[i]
        aliases = ex["answer"]["aliases"] + [ex["answer"]["value"]]
        items.append({
            "ds_index": i,
            "question_id": ex["question_id"],
            "question": ex["question"],
            "aliases": [a for a in aliases if a],
        })

    print(f"[disjointness] saturation excluded: {len(partition['saturation'])}")
    print(f"[disjointness] Step 0 items: {len(items)} "
          f"(disjoint from T-eval={len(partition['teval'])}, "
          f"T-cal={len(partition['tcal'])})")

    return items


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_responses(
    model,
    tokenizer,
    prompt: str,
    n_samples: int,
    temperature: float,
    max_new_tokens: int,
    device: str,
) -> list[str]:
    """Generate n_samples responses at the given temperature."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    responses = []
    for _ in range(n_samples):
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        responses.append(text)
    return responses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run on 3 items only for pipeline verification.")
    args = parser.parse_args()

    # Env check
    print(f"[env] HSA_OVERRIDE_GFX_VERSION="
          f"{os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'UNSET')}")
    assert torch.cuda.is_available(), "ROCm not detected"
    print(f"[env] device: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"[load] {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()
    device = next(model.parameters()).device
    print(f"[load] VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Load data
    items = load_step0_items(SEED)
    if args.dry_run:
        items = items[:3]
    print(f"[data] Step 0 items: {len(items)}")

    # Run sampling
    records = []
    start = time.time()

    for idx, item in enumerate(items):
        prompt = build_prompt(tokenizer, item["question"])
        raw_responses = sample_responses(
            model, tokenizer, prompt,
            n_samples=N_SAMPLES,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
            device=device,
        )

        parsed = [parse_response(r, prompt_format="triviaqa") for r in raw_responses]
        answers = [p.answer for p in parsed]
        confidences = [p.confidence for p in parsed]

        # Normalise answers for variability check
        norm_answers = []
        for a in answers:
            na = a.lower().strip()
            if na:
                norm_answers.append(na)

        n_unique = len(set(norm_answers)) if norm_answers else 0

        # Correctness per sample
        n_correct = sum(
            1 for a in answers
            if a and is_correct_triviaqa(a, item["aliases"])
        )

        records.append({
            "question_id": item["question_id"],
            "ds_index": item["ds_index"],
            "question": item["question"],
            "raw_responses": raw_responses,
            "parsed_answers": answers,
            "parsed_confidences": confidences,
            "n_unique_answers": n_unique,
            "n_correct": n_correct,
            "has_variability": n_unique > 1,
        })

        if (idx + 1) % 25 == 0 or idx == len(items) - 1:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed
            eta = (len(items) - idx - 1) / rate if rate > 0 else 0
            print(f"[step0] {idx+1}/{len(items)}  "
                  f"elapsed={elapsed:.0f}s  rate={rate:.2f}/s  eta={eta:.0f}s")

    # Aggregate
    n_items = len(records)
    total_samples = n_items * N_SAMPLES

    all_confidences = [c for r in records for c in r["parsed_confidences"]]
    all_answers = [a for r in records for a in r["parsed_answers"]]

    confidence_parsed = sum(1 for c in all_confidences if not np.isnan(c))
    answer_parsed = sum(1 for a in all_answers if a.strip())
    variable_items = sum(1 for r in records if r["has_variability"])

    conf_parse_rate = confidence_parsed / total_samples if total_samples > 0 else 0
    ans_parse_rate = answer_parsed / total_samples if total_samples > 0 else 0
    variability_rate = variable_items / n_items if n_items > 0 else 0

    passed = (
        conf_parse_rate >= MIN_CONFIDENCE_PARSE_RATE
        and ans_parse_rate >= MIN_ANSWER_PARSE_RATE
        and variability_rate >= MIN_VARIABILITY_RATE
    )

    result = {
        "status": "PASS" if passed else "FAIL",
        "n_items": n_items,
        "n_samples_per_item": N_SAMPLES,
        "total_samples": total_samples,
        "confidence_parse_rate": conf_parse_rate,
        "answer_parse_rate": ans_parse_rate,
        "variability_rate": variability_rate,
        "criteria": {
            "min_confidence_parse_rate": MIN_CONFIDENCE_PARSE_RATE,
            "min_answer_parse_rate": MIN_ANSWER_PARSE_RATE,
            "min_variability_rate": MIN_VARIABILITY_RATE,
        },
        "model_id": MODEL_ID,
        "seed": SEED,
        "temperature": TEMPERATURE,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": time.time() - start,
        "items": records,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=float)

    print(f"\n=== Step 0 {'PASS' if passed else 'FAIL'} ===")
    print(f"  Confidence parse rate: {conf_parse_rate:.3f} "
          f"(threshold: {MIN_CONFIDENCE_PARSE_RATE})")
    print(f"  Answer parse rate:     {ans_parse_rate:.3f} "
          f"(threshold: {MIN_ANSWER_PARSE_RATE})")
    print(f"  Variability rate:      {variability_rate:.3f} "
          f"(threshold: {MIN_VARIABILITY_RATE})")
    print(f"  Output: {OUTPUT_PATH}")

    if not passed:
        print("\n[FAIL] Step 0 did not pass. Pre-reg permits one re-run with "
              "substrate filtering. See protocol v4 §7.1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
