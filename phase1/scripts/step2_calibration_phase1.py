"""
Step 2: Self-Consistency Calibration Sampling (Phase 1)
=======================================================

Phase 1, pre-reg locked on OSF. Generates 10 samples at T=0.7 on T-cal
(2,000 TriviaQA items) for a given model. Computes n_correct per item,
maps to confidence targets, assigns difficulty bins. Builds the training
set for Step 3 LoRA fine-tuning (NO modal filter, per Phase 0 lesson).

Also optionally samples T-eval (1,000 items) for within-bin difficulty
analysis (E4). Use --include-teval flag.

Environment: M3 Ultra, MLX, Python 3.14, venv .venv_metacog.

Usage:
    python3 step2_calibration_phase1.py \
        --model-path ~/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it \
        --model-name gemma-3-12b-it

    # With T-eval sampling:
    python3 step2_calibration_phase1.py \
        --model-path ~/mnt/models-lan/foresight/synthesis-archive/gemma-3-27b-it \
        --model-name gemma-3-27b-it \
        --include-teval

Outputs (in ~/jpwork/results/step2/):
    step2_calibration_{model_name}.json      Full sampling data (2000 items x 10)
    step2_training_set_{model_name}.json     Training set for Step 3 (no filter)
    step2_teval_difficulty_{model_name}.json  T-eval difficulty bins (if --include-teval)
    step2_summary_{model_name}.json          Distribution summary + diagnostics

Runtime: ~6.8h (12B) / ~19h (27B) for T-cal. Add ~50% for T-eval.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# MLX imports
import mlx.core as mx
from mlx_lm import load
from gen_helpers import generate_greedy, generate_sampled


# ---------------------------------------------------------------------------
# Config (locked per pre-reg)
# ---------------------------------------------------------------------------

SEED = 42
N_TEVAL = 1000
N_TCAL = 2000
N_SAMPLES = 10
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 256

RESULTS_DIR = Path(os.path.expanduser("~/jpwork/results/step2"))
CHECKPOINT_DIR = Path(os.path.expanduser("~/jpwork/results/step2/checkpoints"))

# Confidence target mapping (pre-reg §4.4, same as Phase 0)
N_CORRECT_TO_TARGET = {
    0: 5, 1: 15, 2: 25, 3: 35, 4: 45,
    5: 50, 6: 60, 7: 70, 8: 80, 9: 90, 10: 95,
}

# Difficulty bins (pre-reg §4.4)
def difficulty_bin(n_correct: int) -> str:
    if n_correct >= 8:
        return "Easy"
    elif n_correct >= 4:
        return "Medium"
    else:
        return "Hard"


# ---------------------------------------------------------------------------
# TriviaQA correctness (ported from utils_phase0.py)
# ---------------------------------------------------------------------------

_ARTICLE_RE = re.compile(r"^(the|a|an)\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _normalise(s: str) -> str:
    s = s.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    s = _ARTICLE_RE.sub("", s)
    return s


def is_correct_triviaqa(pred: str, aliases: list[str]) -> bool:
    """TriviaQA correctness: alias appears in pred (or vice versa) after norm."""
    p = _normalise(pred)
    if not p:
        return False
    for a in aliases:
        an = _normalise(a)
        if not an:
            continue
        if an in p or p in an:
            return True
    return False


# ---------------------------------------------------------------------------
# Response parsing (simplified for Step 2 — we only need the answer)
# ---------------------------------------------------------------------------

def parse_answer(raw: str) -> str:
    """Extract the answer from a generated response. Simple first-line heuristic."""
    text = raw.strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return ""
    first = lines[0]
    m = re.match(r"^(?:answer\s*:?\s*)(.*)", first, flags=re.IGNORECASE)
    answer = m.group(1).strip() if m else first
    # Strip trailing confidence if leaked
    answer = re.sub(
        r"\s*[,;]?\s*confidence\s*:?.*$", "", answer, flags=re.IGNORECASE
    ).strip()
    return answer


# ---------------------------------------------------------------------------
# Data loading + partitioning (matches Phase 1 step0 + step1 scripts)
# ---------------------------------------------------------------------------

def load_triviaqa_partition():
    """Load TriviaQA and partition into T-eval, T-cal.

    Reproduces the exact Phase 1 partitioning (matches step0 + step1 scripts):
    1. Shuffle all indices with random.Random(42)
    2. T-eval = indices[0:1000], T-cal = indices[1000:3000]

    No saturation-paper exclusion (Phase 1 uses a different partition from Phase 0).
    """
    from datasets import load_dataset

    print("[data] Loading TriviaQA rc.nocontext validation...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")

    indices = list(range(len(ds)))
    rng = random.Random(SEED)
    rng.shuffle(indices)

    teval_idx = indices[0:N_TEVAL]
    tcal_idx = indices[N_TEVAL:N_TEVAL + N_TCAL]

    def _to_items(idxs):
        items = []
        for i in idxs:
            ex = ds[i]
            aliases = ex["answer"]["aliases"] + [ex["answer"]["value"]]
            items.append({
                "question_id": ex["question_id"],
                "question": ex["question"],
                "aliases": [a for a in aliases if a],
            })
        return items

    teval = _to_items(teval_idx)
    tcal = _to_items(tcal_idx)
    print(f"[data] T-eval: {len(teval)}, T-cal: {len(tcal)}")

    return teval, tcal


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

TRIVIAQA_PROMPT = (
    "You are answering trivia questions. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
)


def build_prompt(tokenizer, question: str) -> str:
    """Build chat-formatted prompt for a TriviaQA item."""
    user_msg = TRIVIAQA_PROMPT.format(question=question)
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# MLX generation with temperature sampling
# ---------------------------------------------------------------------------

def generate_sample(model, tokenizer, prompt: str, temp: float, max_tokens: int) -> str:
    """Generate a single sample at given temperature using MLX."""
    return generate_sampled(model, tokenizer, prompt, temp)


# ---------------------------------------------------------------------------
# Sampling loop with checkpointing
# ---------------------------------------------------------------------------

def sample_items(
    model, tokenizer, items: list[dict], n_samples: int, temp: float,
    checkpoint_path: Path, checkpoint_every: int = 50,
    label: str = "T-cal",
) -> list[dict]:
    """Run n_samples at T=temp on each item. Checkpoint periodically."""

    # Load checkpoint if exists
    results = []
    done_qids = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            results = json.load(f)
        done_qids = {r["question_id"] for r in results}
        print(f"[resume] Loaded checkpoint with {len(results)} items done")

    total = len(items)
    start_time = time.time()
    session_start_count = len(results)  # items from checkpoint

    for idx, item in enumerate(items):
        qid = item["question_id"]
        if qid in done_qids:
            continue

        question = item["question"]
        aliases = item["aliases"]
        prompt = build_prompt(tokenizer, question)

        # Generate n_samples
        samples = []
        n_correct_count = 0
        for s in range(n_samples):
            raw = generate_sample(model, tokenizer, prompt, temp, MAX_NEW_TOKENS)
            answer = parse_answer(raw)
            correct = is_correct_triviaqa(answer, aliases)
            if correct:
                n_correct_count += 1
            samples.append({
                "sample_idx": s,
                "raw": raw,
                "parsed_answer": answer,
                "correct": correct,
            })

        n_correct = n_correct_count
        target = N_CORRECT_TO_TARGET[n_correct]
        diff_bin = difficulty_bin(n_correct)

        # Modal answer
        answer_counts = Counter(
            _normalise(s["parsed_answer"]) for s in samples if s["parsed_answer"]
        )
        modal_answer_norm = answer_counts.most_common(1)[0][0] if answer_counts else ""
        modal_correct = is_correct_triviaqa(modal_answer_norm, aliases) if modal_answer_norm else False

        result = {
            "question_id": qid,
            "question": question,
            "aliases": aliases,
            "n_correct": n_correct,
            "confidence_target": target,
            "difficulty_bin": diff_bin,
            "modal_answer_normalised": modal_answer_norm,
            "modal_correct": modal_correct,
            "samples": samples,
        }
        results.append(result)
        done_qids.add(qid)

        # Progress
        elapsed = time.time() - start_time
        done_count = len(results)
        done_this_session = done_count - session_start_count
        if done_this_session > 0:
            rate = elapsed / done_this_session
        else:
            rate = 0
        remaining_items = total - done_count
        eta_h = (remaining_items * rate) / 3600 if rate > 0 else 0

        if (done_count % 10 == 0) or done_count == total:
            print(
                f"[{label}] {done_count}/{total}  "
                f"n_correct={n_correct}  target={target}%  "
                f"bin={diff_bin}  "
                f"ETA: {eta_h:.1f}h"
            )

        # Checkpoint
        if done_count % checkpoint_every == 0:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump(results, f)

    # Final save
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(results, f)

    return results


# ---------------------------------------------------------------------------
# Training set construction (no modal filter)
# ---------------------------------------------------------------------------

def build_training_set(calibration_data: list[dict], tokenizer) -> list[dict]:
    """Build CSFT training set from calibration data. NO modal filter.

    Each training example:
    - prompt: chat-formatted question
    - completion: modal_answer + confidence target
    - target: confidence percentage

    Uses all 2000 items regardless of modal_correct status.
    """
    training_set = []

    for item in calibration_data:
        qid = item["question_id"]
        question = item["question"]
        target = item["confidence_target"]
        modal_answer = item["modal_answer_normalised"]
        modal_correct = item["modal_correct"]
        n_correct = item["n_correct"]

        # Build the prompt (same as evaluation)
        prompt = build_prompt(tokenizer, question)

        # Build the completion (answer + confidence)
        # Use the most common raw answer (not normalised) for natural text
        # Fall back to normalised if no raw available
        raw_answers = [
            s["parsed_answer"] for s in item["samples"] if s["parsed_answer"]
        ]
        if raw_answers:
            answer_counts = Counter(raw_answers)
            modal_answer_raw = answer_counts.most_common(1)[0][0]
        else:
            modal_answer_raw = modal_answer

        completion = f"{modal_answer_raw}\nConfidence: {target}%"

        training_set.append({
            "question_id": qid,
            "prompt": prompt,
            "completion": completion,
            "modal_answer": modal_answer_raw,
            "confidence_target": target,
            "n_correct": n_correct,
            "difficulty_bin": item["difficulty_bin"],
            "modal_correct": modal_correct,
        })

    return training_set


def build_shuffled_training_set(training_set: list[dict], shuffle_seed: int = 43) -> tuple[list[dict], float]:
    """Build shuffled-target control. Permute confidence targets across items.

    Returns (shuffled_set, correlation) where correlation is r(real, shuffled).
    E7 pre-check: |r| must be < 0.05.
    """
    rng = np.random.default_rng(shuffle_seed)
    targets = [item["confidence_target"] for item in training_set]
    shuffled_targets = list(targets)
    rng.shuffle(shuffled_targets)

    # E7: verify decorrelation
    r = float(np.corrcoef(targets, shuffled_targets)[0, 1])

    shuffled_set = []
    for item, new_target in zip(training_set, shuffled_targets):
        s = dict(item)
        s["confidence_target"] = int(new_target)
        s["completion"] = f"{item['modal_answer']}\nConfidence: {new_target}%"
        shuffled_set.append(s)

    return shuffled_set, r


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(calibration_data: list[dict]) -> dict:
    """Compute distribution summary for reporting."""
    n_correct_dist = Counter(item["n_correct"] for item in calibration_data)
    bin_dist = Counter(item["difficulty_bin"] for item in calibration_data)
    modal_correct_rate = np.mean([item["modal_correct"] for item in calibration_data])

    # Bimodality check
    n_extreme = n_correct_dist.get(0, 0) + n_correct_dist.get(10, 0)
    n_intermediate = sum(v for k, v in n_correct_dist.items() if 1 <= k <= 9)
    pct_extreme = n_extreme / len(calibration_data) * 100
    pct_intermediate = n_intermediate / len(calibration_data) * 100

    return {
        "n_items": len(calibration_data),
        "n_correct_distribution": dict(sorted(n_correct_dist.items())),
        "difficulty_bin_distribution": dict(bin_dist),
        "modal_correct_rate": float(modal_correct_rate),
        "pct_extreme": float(pct_extreme),
        "pct_intermediate": float(pct_intermediate),
        "bimodal": pct_extreme > 70,
        "distribution_fork": "sharpening" if pct_intermediate <= 30 else "smoothing",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 2: Self-consistency calibration sampling")
    parser.add_argument("--model-path", required=True, help="Path to model directory")
    parser.add_argument("--model-name", required=True, help="Model name for output files")
    parser.add_argument("--include-teval", action="store_true",
                        help="Also sample T-eval for difficulty bins (adds ~50% runtime)")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save checkpoint every N items (default: 50)")
    args = parser.parse_args()

    model_name = args.model_name
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(f"[model] Loading {args.model_path}...")
    t0 = time.time()
    model, tokenizer = load(args.model_path)
    print(f"[model] Loaded in {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # Load data partition
    # -----------------------------------------------------------------------
    teval_items, tcal_items = load_triviaqa_partition()

    # Verify against Step 1 saved data if available
    step1_path = Path(os.path.expanduser(
        f"~/jpwork/results/step1/tcal_greedy_responses_{model_name}.json"
    ))
    if step1_path.exists():
        with open(step1_path) as f:
            step1_data = json.load(f)
        step1_qids = {r["question_id"] for r in step1_data}
        tcal_qids = {item["question_id"] for item in tcal_items}
        overlap = step1_qids & tcal_qids
        print(f"[verify] Step 1 T-cal QIDs: {len(step1_qids)}, "
              f"Step 2 T-cal QIDs: {len(tcal_qids)}, "
              f"overlap: {len(overlap)}")
        if len(overlap) < len(step1_qids) * 0.95:
            print("[WARN] Low overlap with Step 1 T-cal — partition mismatch?")
            print("       Proceeding, but verify manually.")
    else:
        print(f"[info] No Step 1 T-cal file found at {step1_path}; skipping verification")

    # -----------------------------------------------------------------------
    # T-cal sampling (critical path)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 2: T-cal sampling — {len(tcal_items)} items × {N_SAMPLES} samples @ T={TEMPERATURE}")
    print(f"{'='*60}\n")

    tcal_checkpoint = CHECKPOINT_DIR / f"tcal_checkpoint_{model_name}.json"
    tcal_results = sample_items(
        model, tokenizer, tcal_items,
        n_samples=N_SAMPLES, temp=TEMPERATURE,
        checkpoint_path=tcal_checkpoint,
        checkpoint_every=args.checkpoint_every,
        label="T-cal",
    )

    # Save full calibration data
    cal_path = RESULTS_DIR / f"step2_calibration_{model_name}.json"
    with open(cal_path, "w") as f:
        json.dump(tcal_results, f, indent=2)
    print(f"\n[save] Calibration data: {cal_path}")

    # Build training set (no modal filter)
    training_set = build_training_set(tcal_results, tokenizer)
    train_path = RESULTS_DIR / f"step2_training_set_{model_name}.json"
    with open(train_path, "w") as f:
        json.dump(training_set, f, indent=2)
    print(f"[save] Training set ({len(training_set)} items, no filter): {train_path}")

    # Build shuffled-target control
    shuffled_set, e7_corr = build_shuffled_training_set(training_set)
    shuffled_path = RESULTS_DIR / f"step2_shuffled_training_set_{model_name}.json"
    with open(shuffled_path, "w") as f:
        json.dump(shuffled_set, f, indent=2)
    print(f"[save] Shuffled training set: {shuffled_path}")
    print(f"[E7]  Real-shuffled target correlation: r = {e7_corr:.4f}  "
          f"{'PASS' if abs(e7_corr) < 0.05 else 'WARN: |r| >= 0.05'}")

    # Summary
    summary = compute_summary(tcal_results)
    summary["e7_real_shuffled_correlation"] = e7_corr
    summary["model_name"] = model_name
    summary["temperature"] = TEMPERATURE
    summary["n_samples"] = N_SAMPLES

    print(f"\n=== T-cal Distribution Summary ===")
    print(f"  n_correct distribution:")
    for k in sorted(summary["n_correct_distribution"]):
        v = summary["n_correct_distribution"][k]
        pct = v / summary["n_items"] * 100
        bar = "█" * int(pct / 2)
        print(f"    n={k:2d}: {v:5d} ({pct:5.1f}%)  {bar}")
    print(f"  Extreme (0 or 10): {summary['pct_extreme']:.1f}%")
    print(f"  Intermediate (1-9): {summary['pct_intermediate']:.1f}%")
    print(f"  Distribution fork: {summary['distribution_fork']}")
    print(f"  Difficulty bins: {summary['difficulty_bin_distribution']}")
    print(f"  Modal correct rate: {summary['modal_correct_rate']:.3f}")

    # -----------------------------------------------------------------------
    # T-eval sampling (optional, for E4 within-bin analysis)
    # -----------------------------------------------------------------------
    if args.include_teval:
        print(f"\n{'='*60}")
        print(f"Step 2: T-eval difficulty sampling — {len(teval_items)} items × {N_SAMPLES} samples")
        print(f"{'='*60}\n")

        teval_checkpoint = CHECKPOINT_DIR / f"teval_checkpoint_{model_name}.json"
        teval_results = sample_items(
            model, tokenizer, teval_items,
            n_samples=N_SAMPLES, temp=TEMPERATURE,
            checkpoint_path=teval_checkpoint,
            checkpoint_every=args.checkpoint_every,
            label="T-eval",
        )

        # Save T-eval difficulty data (just qid + n_correct + bin, no raw samples)
        teval_difficulty = []
        for item in teval_results:
            teval_difficulty.append({
                "question_id": item["question_id"],
                "n_correct": item["n_correct"],
                "difficulty_bin": item["difficulty_bin"],
                "confidence_target": item["confidence_target"],
            })

        teval_diff_path = RESULTS_DIR / f"step2_teval_difficulty_{model_name}.json"
        with open(teval_diff_path, "w") as f:
            json.dump(teval_difficulty, f, indent=2)
        print(f"[save] T-eval difficulty: {teval_diff_path}")

        # T-eval summary
        teval_summary = compute_summary(teval_results)
        summary["teval_distribution"] = teval_summary
        print(f"\n=== T-eval Distribution Summary ===")
        print(f"  Extreme (0 or 10): {teval_summary['pct_extreme']:.1f}%")
        print(f"  Intermediate (1-9): {teval_summary['pct_intermediate']:.1f}%")
        print(f"  Difficulty bins: {teval_summary['difficulty_bin_distribution']}")

    # Save summary
    summary_path = RESULTS_DIR / f"step2_summary_{model_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[save] Summary: {summary_path}")

    print(f"\n=== Step 2 complete for {model_name} ===")


if __name__ == "__main__":
    main()
