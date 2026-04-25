"""
Step 2: Calibration Sampling & Confidence Target Derivation
============================================================

Phase 0 v4, pre-reg v2. Runs 10 × T=0.7 samples on T-cal (2,000 TriviaQA
items) and derives per-item confidence targets from n_correct counts.

This step produces:
    1. Per-item n_correct (how many of 10 samples are correct)
    2. Per-item confidence target via the pre-registered mapping
    3. Per-item modal_correct flag (is the modal answer correct?)
    4. Difficulty bins (Easy/Medium/Hard) for T-cal items
    5. The modal-filter partition: training set (modal_correct=True)
       and conflict set (modal_correct=False, held out for H5)
    6. SFT training examples in chat format

Also derives T-eval difficulty bins via the same 10-sample process
(needed for within-bin analysis in Step 4, H3).

Outputs:
    D:\\metacog\\data\\step2_calibration.json
        - per-item: question_id, n_correct, confidence_target,
          modal_correct, difficulty_bin, modal_answer, all answers
    D:\\metacog\\data\\step2_teval_difficulty.json
        - per T-eval item: n_correct_eval, difficulty_bin
    D:\\metacog\\data\\step2_training_set.json
        - modal-filtered training examples with confidence targets
    D:\\metacog\\data\\step2_conflict_set.json
        - held-out conflict items (modal_correct=False)
    D:\\metacog\\data\\step2_shuffled_training_set.json
        - same items as training set but targets permuted (seed=43)

Runtime: ~5.5 hours (2000 items × 10 samples + 1000 items × 10 samples).
"""

import argparse
import gc
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
SHUFFLED_SEED = 43
N_SAMPLES = 10
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 64

PROJECT_ROOT = Path(r"D:\metacog")
DATA_DIR = PROJECT_ROOT / "data"

# Pre-registered confidence target mapping (n_correct -> target %)
CONFIDENCE_TARGET_MAP = {
    0: 5, 1: 15, 2: 25, 3: 35, 4: 45, 5: 50,
    6: 60, 7: 70, 8: 80, 9: 90, 10: 95,
}

# Pre-registered difficulty bins
def difficulty_bin(n_correct: int) -> str:
    if n_correct >= 8:
        return "Easy"
    elif n_correct >= 4:
        return "Medium"
    else:
        return "Hard"

# Prompt (same as Step 0 and Step 1)
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

def load_split_items(seed: int, split: str) -> list[dict]:
    """Load T-cal or T-eval items."""
    partition = partition_triviaqa_pool(seed=seed)
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")

    assert len(ds) == 17_944, (
        f"TriviaQA rc.nocontext validation has {len(ds)} items, expected 17,944."
    )

    indices = partition[split]
    items = []
    for i in indices:
        ex = ds[i]
        aliases = ex["answer"]["aliases"] + [ex["answer"]["value"]]
        items.append({
            "ds_index": i,
            "question_id": ex["question_id"],
            "question": ex["question"],
            "aliases": [a for a in aliases if a],
        })

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
) -> list[str]:
    """Generate n_samples responses at the given temperature."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
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
# Target derivation
# ---------------------------------------------------------------------------

def derive_targets(items: list[dict], all_samples: list[list[str]]) -> list[dict]:
    """For each item, compute n_correct, modal answer, confidence target, etc."""
    records = []
    for item, samples in zip(items, all_samples):
        parsed = [parse_response(s, prompt_format="triviaqa") for s in samples]
        answers = [p.answer for p in parsed]

        # Correctness per sample
        correct_flags = []
        for a in answers:
            if a and a.strip():
                correct_flags.append(is_correct_triviaqa(a, item["aliases"]))
            else:
                correct_flags.append(False)

        n_correct = sum(correct_flags)

        # Modal answer (most common non-empty normalised answer)
        norm_answers = []
        for a in answers:
            na = a.lower().strip() if a else ""
            if na:
                norm_answers.append(na)

        if norm_answers:
            counter = Counter(norm_answers)
            modal_answer = counter.most_common(1)[0][0]
            # Check if modal answer is correct
            modal_correct = is_correct_triviaqa(modal_answer, item["aliases"])
        else:
            modal_answer = ""
            modal_correct = False

        confidence_target = CONFIDENCE_TARGET_MAP[n_correct]
        diff_bin = difficulty_bin(n_correct)

        records.append({
            "question_id": item["question_id"],
            "ds_index": item["ds_index"],
            "question": item["question"],
            "aliases": item["aliases"],
            "n_correct": n_correct,
            "confidence_target": confidence_target,
            "modal_answer": modal_answer,
            "modal_correct": modal_correct,
            "difficulty_bin": diff_bin,
            "raw_answers": answers,
            "correct_flags": correct_flags,
        })

    return records


# ---------------------------------------------------------------------------
# SFT example construction
# ---------------------------------------------------------------------------

def build_sft_example(item: dict, tokenizer) -> dict:
    """Build a chat-format SFT training example.

    Format:
        User: <trivia prompt>
        Assistant: <modal_answer>\n\nConfidence: <target>%
    """
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run on 5 items per split for pipeline verification.")
    parser.add_argument("--skip-teval", action="store_true",
                        help="Skip T-eval difficulty sampling (if already done).")
    args = parser.parse_args()

    # Env check
    print(f"[env] HSA_OVERRIDE_GFX_VERSION="
          f"{os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'UNSET')}")
    assert torch.cuda.is_available(), "ROCm not detected"
    print(f"[env] device: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

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
    print(f"[load] VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # ------------------------------------------------------------------
    # T-cal calibration sampling (2000 items × 10 samples)
    # ------------------------------------------------------------------
    print("\n=== Step 2: T-cal calibration sampling ===")
    tcal_items = load_split_items(SEED, "tcal")
    if args.dry_run:
        tcal_items = tcal_items[:5]
    print(f"[data] T-cal items: {len(tcal_items)}")

    tcal_all_samples = []
    start = time.time()

    for idx, item in enumerate(tcal_items):
        prompt = build_prompt(tokenizer, item["question"])
        samples = sample_responses(
            model, tokenizer, prompt,
            n_samples=N_SAMPLES,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        tcal_all_samples.append(samples)

        if (idx + 1) % 50 == 0 or idx == len(tcal_items) - 1:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed
            eta = (len(tcal_items) - idx - 1) / rate if rate > 0 else 0
            print(f"[T-cal] {idx+1}/{len(tcal_items)}  "
                  f"elapsed={elapsed:.0f}s  rate={rate:.2f}/s  eta={eta:.0f}s")

    # Derive targets
    tcal_records = derive_targets(tcal_items, tcal_all_samples)

    # Save full calibration data
    with open(DATA_DIR / "step2_calibration.json", "w") as f:
        json.dump(tcal_records, f, indent=2)
    print(f"[out] Calibration data -> {DATA_DIR / 'step2_calibration.json'}")

    # ------------------------------------------------------------------
    # T-eval difficulty sampling (1000 items × 10 samples)
    # ------------------------------------------------------------------
    if not args.skip_teval:
        print("\n=== Step 2: T-eval difficulty sampling ===")
        teval_items = load_split_items(SEED, "teval")
        if args.dry_run:
            teval_items = teval_items[:5]
        print(f"[data] T-eval items: {len(teval_items)}")

        teval_all_samples = []
        start2 = time.time()

        for idx, item in enumerate(teval_items):
            prompt = build_prompt(tokenizer, item["question"])
            samples = sample_responses(
                model, tokenizer, prompt,
                n_samples=N_SAMPLES,
                temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            teval_all_samples.append(samples)

            if (idx + 1) % 50 == 0 or idx == len(teval_items) - 1:
                elapsed = time.time() - start2
                rate = (idx + 1) / elapsed
                eta = (len(teval_items) - idx - 1) / rate if rate > 0 else 0
                print(f"[T-eval] {idx+1}/{len(teval_items)}  "
                      f"elapsed={elapsed:.0f}s  rate={rate:.2f}/s  eta={eta:.0f}s")

        teval_records = derive_targets(teval_items, teval_all_samples)

        # Save only the difficulty info for T-eval (no training targets)
        teval_difficulty = [
            {
                "question_id": r["question_id"],
                "ds_index": r["ds_index"],
                "n_correct_eval": r["n_correct"],
                "difficulty_bin": r["difficulty_bin"],
            }
            for r in teval_records
        ]
        with open(DATA_DIR / "step2_teval_difficulty.json", "w") as f:
            json.dump(teval_difficulty, f, indent=2)
        print(f"[out] T-eval difficulty -> {DATA_DIR / 'step2_teval_difficulty.json'}")

    # ------------------------------------------------------------------
    # Modal filter: partition into training set and conflict set
    # ------------------------------------------------------------------
    print("\n=== Step 2: Modal filter partitioning ===")

    training_items = [r for r in tcal_records if r["modal_correct"]]
    conflict_items = [r for r in tcal_records if not r["modal_correct"]]

    print(f"[filter] Training set (modal_correct=True): {len(training_items)}")
    print(f"[filter] Conflict set (modal_correct=False): {len(conflict_items)}")
    if len(conflict_items) < 200:
        print(f"[filter] WARNING: conflict set n={len(conflict_items)} < 200, "
              f"H5 will be power-limited per pre-reg §5.4")

    # Build SFT training examples
    sft_examples = [build_sft_example(item, tokenizer) for item in training_items]

    with open(DATA_DIR / "step2_training_set.json", "w") as f:
        json.dump(sft_examples, f, indent=2)
    print(f"[out] Training set -> {DATA_DIR / 'step2_training_set.json'}")

    # Save conflict set
    conflict_out = [
        {
            "question_id": r["question_id"],
            "ds_index": r["ds_index"],
            "question": r["question"],
            "aliases": r["aliases"],
            "n_correct": r["n_correct"],
            "confidence_target": r["confidence_target"],
            "modal_answer": r["modal_answer"],
            "difficulty_bin": r["difficulty_bin"],
        }
        for r in conflict_items
    ]
    with open(DATA_DIR / "step2_conflict_set.json", "w") as f:
        json.dump(conflict_out, f, indent=2)
    print(f"[out] Conflict set -> {DATA_DIR / 'step2_conflict_set.json'}")

    # ------------------------------------------------------------------
    # Shuffled-target training set (seed=43)
    # ------------------------------------------------------------------
    print("\n=== Step 2: Shuffled-target construction (E7) ===")

    # Extract real targets in training-set order
    real_targets = [item["confidence_target"] for item in training_items]

    # Permute with seed=43
    rng_shuffle = np.random.RandomState(SHUFFLED_SEED)
    shuffled_targets = real_targets.copy()
    rng_shuffle.shuffle(shuffled_targets)

    # E7: real-shuffled target correlation
    from scipy.stats import pearsonr, spearmanr
    if len(real_targets) > 2:
        r_pearson, _ = pearsonr(real_targets, shuffled_targets)
        r_spearman, _ = spearmanr(real_targets, shuffled_targets)
    else:
        r_pearson, r_spearman = float("nan"), float("nan")

    print(f"[E7] Real-shuffled target correlation: "
          f"Pearson r={r_pearson:.4f}, Spearman rho={r_spearman:.4f}")
    if abs(r_pearson) < 0.05 and abs(r_spearman) < 0.05:
        print(f"[E7] Clean: both |r| < 0.05")
    else:
        print(f"[E7] WARNING: correlation exceeds 0.05; shuffled adjustment "
              f"may under-correct")

    # Build shuffled SFT examples
    shuffled_sft = []
    for item, shuf_target in zip(training_items, shuffled_targets):
        item_copy = item.copy()
        item_copy["confidence_target"] = shuf_target
        shuffled_sft.append(build_sft_example(item_copy, tokenizer))

    with open(DATA_DIR / "step2_shuffled_training_set.json", "w") as f:
        json.dump(shuffled_sft, f, indent=2)
    print(f"[out] Shuffled training set -> "
          f"{DATA_DIR / 'step2_shuffled_training_set.json'}")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    print("\n=== Step 2 summary ===")

    # Difficulty distribution (T-cal)
    bin_counts = Counter(r["difficulty_bin"] for r in tcal_records)
    print(f"[dist] T-cal difficulty: "
          f"Easy={bin_counts.get('Easy', 0)} "
          f"Medium={bin_counts.get('Medium', 0)} "
          f"Hard={bin_counts.get('Hard', 0)}")

    # n_correct distribution
    nc_counts = Counter(r["n_correct"] for r in tcal_records)
    for nc in sorted(nc_counts.keys()):
        bar = "#" * (nc_counts[nc] // 5)
        print(f"  n_correct={nc:2d}: {nc_counts[nc]:4d} {bar}")

    # Confidence target distribution
    ct_counts = Counter(r["confidence_target"] for r in tcal_records)
    print(f"[dist] Target distribution: "
          + ", ".join(f"{t}%:{ct_counts.get(t,0)}"
                      for t in sorted(CONFIDENCE_TARGET_MAP.values())))

    # Training/conflict split
    print(f"[split] Training: {len(training_items)} | "
          f"Conflict: {len(conflict_items)} "
          f"({len(conflict_items)/len(tcal_records)*100:.1f}%)")

    # E7 summary
    print(f"[E7] Pearson r={r_pearson:.4f}, Spearman rho={r_spearman:.4f}")

    # Save summary
    summary = {
        "n_tcal": len(tcal_records),
        "n_training": len(training_items),
        "n_conflict": len(conflict_items),
        "difficulty_distribution_tcal": dict(bin_counts),
        "n_correct_distribution": {str(k): v for k, v in sorted(nc_counts.items())},
        "e7_pearson_r": float(r_pearson),
        "e7_spearman_rho": float(r_spearman),
        "model_id": MODEL_ID,
        "seed": SEED,
        "shuffled_seed": SHUFFLED_SEED,
        "n_samples": N_SAMPLES,
        "temperature": TEMPERATURE,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(DATA_DIR / "step2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[out] Summary -> {DATA_DIR / 'step2_summary.json'}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
