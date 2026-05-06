"""
Step 0: Phase 1 Diagnostic Spike (MLX on M3 Ultra)
====================================================

Pre-reg §5: Before committing to the full protocol, run a diagnostic spike
on 200 T-cal items at both 12B and 27B:

1. Verify model loads and runs in bfloat16.
2. Run greedy generation on 200 T-cal items → verify accuracy > 40%.
3. Run 10-sample self-consistency at T=0.7 on same 200 items → record
   n_correct distribution.
4. Run 10-sample at T={0.5, 1.0} on 50-item subset (E11).
5. Compute single-item inference time to estimate total pipeline duration.
6. Verify LoRA target module names.

The diagnostic spike is not a hypothesis test. It is a hardware and
distribution check. Results are recorded but do not gate the full protocol.

Usage:
    python step0_spike_phase1.py --model_path /path/to/gemma-3-12b-it
    python step0_spike_phase1.py --model_path /path/to/gemma-3-27b-it
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from gen_helpers import generate_greedy, generate_sampled


# ---------------------------------------------------------------------------
# Constants (locked in pre-reg)
# ---------------------------------------------------------------------------
SEED = 42
N_SPIKE = 200           # items for greedy + self-consistency
N_E11 = 50              # subset for temperature sweep (E11)
N_SAMPLES = 10          # self-consistency samples per item
T_DEFAULT = 0.7         # default sampling temperature
T_SWEEP = [0.5, 0.7, 1.0]  # E11 temperatures
MAX_TOKENS = 256

TRIVIAQA_PROMPT = (
    "You are answering trivia questions. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
)


# ---------------------------------------------------------------------------
# Data loading — identical partition to Phase 0
# ---------------------------------------------------------------------------
def load_tcal_spike(seed: int, n: int) -> list:
    """Load first n items from T-cal (indices[1000:3000]) partition."""
    from datasets import load_dataset
    
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    
    # T-cal = indices[1000:3000], take first n
    tcal_idx = indices[1000:1000 + n]
    
    items = []
    for i in tcal_idx:
        ex = ds[i]
        aliases = ex["answer"]["aliases"] + [ex["answer"]["value"]]
        items.append({
            "question_id": ex["question_id"],
            "question": ex["question"],
            "aliases": [a for a in aliases if a],
        })
    return items


# ---------------------------------------------------------------------------
# Correctness check (from utils_phase0)
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

def is_correct_triviaqa(pred: str, aliases: list) -> bool:
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
# Response parsing
# ---------------------------------------------------------------------------
def parse_answer(raw: str) -> str:
    """Extract the answer from a generated response."""
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    if not lines:
        return ""
    first = lines[0]
    m = re.match(r"^(?:answer\s*:?\s*)(.*)", first, flags=re.IGNORECASE)
    answer = m.group(1).strip() if m else first
    answer = re.sub(
        r"\s*[,;]?\s*confidence\s*:?.*$", "", answer, flags=re.IGNORECASE
    ).strip()
    return answer


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def build_prompt(tokenizer, question: str) -> str:
    """Build chat-template prompt for TriviaQA."""
    user_msg = TRIVIAQA_PROMPT.format(question=question)
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )




# ---------------------------------------------------------------------------
# LoRA module discovery
# ---------------------------------------------------------------------------
def discover_lora_targets(model) -> list:
    """List candidate LoRA target module names."""
    targets = []
    for name, _ in model.named_modules():
        if any(k in name for k in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                     'gate_proj', 'up_proj', 'down_proj']):
            targets.append(name)
    return sorted(set(targets))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 1 Step 0 diagnostic spike")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to local model directory")
    parser.add_argument("--output_dir", type=str, default="./results/step0",
                        help="Output directory for results")
    args = parser.parse_args()

    model_name = Path(args.model_path).name
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Phase 1 Step 0: Diagnostic Spike")
    print(f"Model: {model_name}")
    print(f"Model path: {args.model_path}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("Loading model...")
    t0 = time.time()
    model, tokenizer = load(args.model_path)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s\n")

    # ------------------------------------------------------------------
    # 2. LoRA target module discovery
    # ------------------------------------------------------------------
    print("Discovering LoRA target modules...")
    lora_targets = discover_lora_targets(model)
    # Deduplicate to unique suffixes
    suffixes = sorted(set(name.split(".")[-1] for name in lora_targets))
    print(f"  Unique target suffixes: {suffixes}")
    print(f"  Total matchable modules: {len(lora_targets)}")
    print(f"  Sample paths: {lora_targets[:5]}\n")

    # ------------------------------------------------------------------
    # 3. Load data
    # ------------------------------------------------------------------
    print(f"Loading T-cal spike data ({N_SPIKE} items, seed={SEED})...")
    items = load_tcal_spike(SEED, N_SPIKE)
    print(f"  Loaded {len(items)} items\n")

    # ------------------------------------------------------------------
    # 4. Greedy generation — accuracy check
    # ------------------------------------------------------------------
    print(f"Running greedy generation on {N_SPIKE} items...")
    greedy_results = []
    t_start = time.time()

    for i, item in enumerate(items):
        prompt = build_prompt(tokenizer, item["question"])
        t_item = time.time()
        raw = generate_greedy(model, tokenizer, prompt)
        elapsed = time.time() - t_item

        answer = parse_answer(raw)
        correct = is_correct_triviaqa(answer, item["aliases"])

        greedy_results.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "raw_output": raw,
            "parsed_answer": answer,
            "correct": correct,
            "elapsed_s": round(elapsed, 3),
        })

        if (i + 1) % 20 == 0:
            acc_so_far = sum(r["correct"] for r in greedy_results) / len(greedy_results)
            print(f"  [{i+1}/{N_SPIKE}] acc={acc_so_far:.3f}, "
                  f"last_time={elapsed:.2f}s")

    greedy_time = time.time() - t_start
    accuracy = sum(r["correct"] for r in greedy_results) / len(greedy_results)
    mean_item_time = greedy_time / N_SPIKE

    print(f"\n  Greedy results:")
    print(f"    Accuracy: {accuracy:.3f} ({sum(r['correct'] for r in greedy_results)}/{N_SPIKE})")
    print(f"    Total time: {greedy_time:.1f}s")
    print(f"    Mean per item: {mean_item_time:.2f}s")
    print(f"    PASS criterion (>0.40): {'PASS' if accuracy > 0.40 else 'FAIL'}\n")

    # ------------------------------------------------------------------
    # 5. Self-consistency sampling at T=0.7
    # ------------------------------------------------------------------
    print(f"Running {N_SAMPLES}-sample self-consistency at T={T_DEFAULT} "
          f"on {N_SPIKE} items...")
    sc_results = []
    t_start = time.time()

    for i, item in enumerate(items):
        prompt = build_prompt(tokenizer, item["question"])
        samples = []
        for s in range(N_SAMPLES):
            raw = generate_sampled(model, tokenizer, prompt, T_DEFAULT)
            answer = parse_answer(raw)
            correct = is_correct_triviaqa(answer, item["aliases"])
            samples.append({"raw": raw, "answer": answer, "correct": correct})

        n_correct = sum(s["correct"] for s in samples)
        answers = [s["answer"] for s in samples]
        answer_counts = Counter(answers)
        modal_answer = answer_counts.most_common(1)[0][0] if answer_counts else ""
        modal_correct = is_correct_triviaqa(modal_answer, item["aliases"])

        sc_results.append({
            "question_id": item["question_id"],
            "n_correct": n_correct,
            "modal_answer": modal_answer,
            "modal_correct": modal_correct,
            "answer_distribution": dict(answer_counts),
        })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed * 60
            print(f"  [{i+1}/{N_SPIKE}] {rate:.1f} items/min")

    sc_time = time.time() - t_start

    # n_correct distribution
    nc_dist = Counter(r["n_correct"] for r in sc_results)
    n_extreme = nc_dist.get(0, 0) + nc_dist.get(10, 0)
    pct_extreme = n_extreme / N_SPIKE * 100
    n_intermediate = N_SPIKE - n_extreme
    pct_intermediate = n_intermediate / N_SPIKE * 100

    print(f"\n  Self-consistency results (T={T_DEFAULT}):")
    print(f"    Total time: {sc_time:.1f}s ({sc_time/60:.1f}min)")
    print(f"    Mean per item (10 samples): {sc_time/N_SPIKE:.2f}s")
    print(f"    n_correct distribution:")
    for k in range(11):
        count = nc_dist.get(k, 0)
        bar = "#" * (count * 40 // N_SPIKE) if N_SPIKE > 0 else ""
        print(f"      {k:2d}: {count:4d} ({count/N_SPIKE*100:5.1f}%) {bar}")
    print(f"    Extreme (0 or 10): {n_extreme} ({pct_extreme:.1f}%)")
    print(f"    Intermediate (1-9): {n_intermediate} ({pct_intermediate:.1f}%)")
    print(f"    Bimodal threshold (>80% extreme): "
          f"{'BIMODAL' if pct_extreme > 80 else 'SMOOTHED'}\n")

    # ------------------------------------------------------------------
    # 6. Temperature sweep on 50-item subset (E11)
    # ------------------------------------------------------------------
    print(f"Running temperature sweep (E11) on {N_E11} items...")
    e11_items = items[:N_E11]
    e11_results = {}

    for temp in T_SWEEP:
        print(f"  T={temp}...")
        temp_nc = []
        t_start = time.time()
        for item in e11_items:
            prompt = build_prompt(tokenizer, item["question"])
            n_corr = 0
            for _ in range(N_SAMPLES):
                raw = generate_sampled(model, tokenizer, prompt, temp)
                answer = parse_answer(raw)
                if is_correct_triviaqa(answer, item["aliases"]):
                    n_corr += 1
            temp_nc.append(n_corr)
        elapsed = time.time() - t_start

        dist = Counter(temp_nc)
        extreme = dist.get(0, 0) + dist.get(10, 0)
        print(f"    Time: {elapsed:.1f}s, Extreme: {extreme}/{N_E11} "
              f"({extreme/N_E11*100:.0f}%)")
        e11_results[str(temp)] = {
            "n_correct_list": temp_nc,
            "distribution": {str(k): dist.get(k, 0) for k in range(11)},
            "pct_extreme": extreme / N_E11 * 100,
            "time_s": round(elapsed, 1),
        }

    # ------------------------------------------------------------------
    # 7. Time estimates for full pipeline
    # ------------------------------------------------------------------
    time_per_greedy = mean_item_time
    time_per_sc_item = sc_time / N_SPIKE  # 10 samples
    
    estimates = {
        "step1_baseline_teval_1000": round(time_per_greedy * 1000 / 3600, 2),
        "step1_baseline_tcal_2000": round(time_per_greedy * 2000 / 3600, 2),
        "step2_sc_tcal_2000": round(time_per_sc_item * 2000 / 3600, 2),
        "step4_eval_teval_1000": round(time_per_greedy * 1000 / 3600, 2),
    }
    total_est = sum(estimates.values())

    print(f"\n  Pipeline time estimates (hours):")
    for k, v in estimates.items():
        print(f"    {k}: {v:.2f}h")
    print(f"    Total (excl. training): {total_est:.2f}h\n")

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    output = {
        "model": model_name,
        "model_path": args.model_path,
        "seed": SEED,
        "n_spike": N_SPIKE,
        "load_time_s": round(load_time, 1),
        "greedy": {
            "accuracy": round(accuracy, 4),
            "n_correct": sum(r["correct"] for r in greedy_results),
            "n_total": N_SPIKE,
            "pass": accuracy > 0.40,
            "total_time_s": round(greedy_time, 1),
            "mean_item_time_s": round(mean_item_time, 3),
        },
        "self_consistency": {
            "temperature": T_DEFAULT,
            "n_samples": N_SAMPLES,
            "distribution": {str(k): nc_dist.get(k, 0) for k in range(11)},
            "pct_extreme": round(pct_extreme, 1),
            "pct_intermediate": round(pct_intermediate, 1),
            "bimodal": pct_extreme > 80,
            "total_time_s": round(sc_time, 1),
        },
        "e11_temperature_sweep": e11_results,
        "time_estimates_hours": estimates,
        "total_estimate_hours": round(total_est, 2),
        "lora_targets": {
            "unique_suffixes": suffixes,
            "n_matchable": len(lora_targets),
            "sample_paths": lora_targets[:10],
        },
    }

    outfile = os.path.join(args.output_dir, f"step0_{model_name}.json")
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {outfile}")

    # Also save raw greedy + SC results
    raw_file = os.path.join(args.output_dir, f"step0_{model_name}_raw.json")
    with open(raw_file, "w") as f:
        json.dump({
            "greedy_results": greedy_results,
            "sc_results": sc_results,
        }, f, indent=2)
    print(f"Raw results saved to {raw_file}")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 0 SUMMARY: {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy:        {accuracy:.3f} ({'PASS' if accuracy > 0.40 else 'FAIL'})")
    print(f"  SC distribution: {'BIMODAL' if pct_extreme > 80 else 'SMOOTHED'} "
          f"({pct_extreme:.0f}% extreme)")
    print(f"  Mean greedy time: {mean_item_time:.2f}s/item")
    print(f"  Est. full pipeline: {total_est:.1f}h")
    print(f"  LoRA suffixes: {suffixes}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
