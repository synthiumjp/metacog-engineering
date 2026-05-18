"""
eval_zeroshot_transfer.py — Test whether TriviaQA PT-CSFT adapter transfers
zero-shot to GSM8K or ARC-Challenge.

The question: Does PT-CSFT teach a general confidence calibration capability,
or is it task-specific? If verbal AUROC₂ improves over the baseline on an
unseen domain, the answer is "train once, calibrate everywhere."

Usage:
    # Gemma 12B TriviaQA adapter → GSM8K
    python3 eval_zeroshot_transfer.py \
        --benchmark gsm8k \
        --model-path ~/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it \
        --model-name gemma-3-12b-it \
        --adapter-path ~/jpwork/results/finetune/gemma-3-12b-it/probe_target/adapters \
        --baseline-auroc2 0.546

    # Qwen 7B TriviaQA adapter → ARC
    python3 eval_zeroshot_transfer.py \
        --benchmark arc \
        --model-path ~/mnt/models-lan/foresight/synthesis-archive/Qwen2.5-7B-Instruct-bf16 \
        --model-name Qwen2.5-7B-Instruct-bf16 \
        --adapter-path ~/jpwork/results/finetune/Qwen2.5-7B-Instruct-bf16/probe_target/adapters \
        --baseline-auroc2 0.620

    --baseline-auroc2: verbal AUROC₂ from the probe_check_domain.py run (no adapter).
                       Used for the delta comparison. If omitted, just reports raw AUROC₂.

Requires: mlx_lm, datasets, sklearn, numpy
Run from: ~/jpwork/metacog-engineering/phase1/scripts/
"""
import argparse, json, os, random, re, sys, time
import numpy as np
import mlx.core as mx
from mlx_lm import load
from sklearn.metrics import roc_auc_score

SEED = 42
MAX_TOKENS = 1024  # GSM8K chain-of-thought needs room

# ---------------------------------------------------------------------------
# Prompts (identical to probe_check_domain.py)
# ---------------------------------------------------------------------------
GSM8K_PROMPT = (
    "Solve this math problem step by step. "
    "After your final answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
)

ARC_PROMPT = (
    "Answer the following science question by selecting the letter of the correct answer. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
    "{choices}\n"
)

# ---------------------------------------------------------------------------
# Data loading (identical to probe_check_domain.py)
# ---------------------------------------------------------------------------
def load_gsm8k(seed=SEED):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    items = []
    for row in ds:
        answer_text = row["answer"]
        match = re.search(r"####\s*(.+)", answer_text)
        gold = match.group(1).strip().replace(",", "") if match else ""
        items.append({
            "id": f"gsm8k_{len(items)}",
            "question": row["question"],
            "gold_answer": gold,
            "prompt_template": GSM8K_PROMPT,
        })
    random.seed(seed)
    random.shuffle(items)
    n_cal = 800
    return items[:n_cal], items[n_cal:]


def load_arc(seed=SEED):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for row in ds:
        choices_text = "\n".join(
            f"{label}) {text}"
            for label, text in zip(row["choices"]["label"], row["choices"]["text"])
        )
        items.append({
            "id": row["id"],
            "question": row["question"],
            "gold_answer": row["answerKey"],
            "choices_text": choices_text,
            "prompt_template": ARC_PROMPT,
        })
    random.seed(seed)
    random.shuffle(items)
    n_cal = 700
    return items[:n_cal], items[n_cal:]


# ---------------------------------------------------------------------------
# Correctness (identical to probe_check_domain.py)
# ---------------------------------------------------------------------------
def is_correct_gsm8k(predicted, gold):
    if not predicted or not gold:
        return False
    match = re.search(r"####\s*(.+?)(?:\n|$)", predicted)
    if match:
        pred_num = match.group(1).strip().replace(",", "")
    else:
        numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", predicted)
        if not numbers:
            return False
        pred_num = numbers[-1].replace(",", "")
    gold_clean = gold.replace(",", "").strip()
    try:
        return float(pred_num) == float(gold_clean)
    except ValueError:
        return pred_num.strip() == gold_clean.strip()


def is_correct_arc(predicted, gold):
    if not predicted or not gold:
        return False
    pred = predicted.strip().upper()
    match = re.match(r"^([A-D])", pred)
    if match:
        return match.group(1) == gold.upper()
    for letter in ["A", "B", "C", "D"]:
        if f"answer is {letter}" in pred.upper() or f"answer: {letter}" in pred.upper():
            return letter == gold.upper()
    return False


# ---------------------------------------------------------------------------
# Confidence parsing (identical to probe_check_domain.py)
# ---------------------------------------------------------------------------
_CONFIDENCE_PATTERNS = [
    re.compile(r"confidence\s*:?\s*(\d{1,3})\s*%?", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s*%"),
]

def parse_confidence(text):
    for pat in _CONFIDENCE_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                v = int(m.group(1))
                if 0 <= v <= 100:
                    return float(v)
            except ValueError:
                pass
    return float("nan")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_greedy(model, tokenizer, prompt, max_tokens=MAX_TOKENS):
    """Greedy generation using gen_helpers."""
    from gen_helpers import generate_greedy as _gen
    return _gen(model, tokenizer, prompt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot transfer: evaluate TriviaQA PT-CSFT adapter on GSM8K/ARC")
    parser.add_argument("--benchmark", required=True, choices=["gsm8k", "arc"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-path", required=True,
                        help="Path to TriviaQA PT-CSFT LoRA adapter")
    parser.add_argument("--baseline-auroc2", type=float, default=None,
                        help="Baseline verbal AUROC₂ from probe_check_domain.py (for delta)")
    parser.add_argument("--output-dir", default=os.path.expanduser(
        "~/jpwork/metacog-engineering/phase1/results_raw/domain_gen"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Zero-Shot Transfer Test")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Model: {args.model_name}")
    print(f"  Adapter: {args.adapter_path}")
    print(f"{'='*60}\n")

    # Load data — eval set only (same split as probe_check_domain.py)
    print(f"[data] Loading {args.benchmark}...")
    if args.benchmark == "gsm8k":
        _, eval_items = load_gsm8k()
    else:
        _, eval_items = load_arc()
    print(f"  Eval: {len(eval_items)} items\n")

    # Load model with adapter
    print(f"[model] Loading {args.model_path} + adapter...")
    model, tokenizer = load(args.model_path, adapter_path=args.adapter_path)
    print(f"  Loaded.\n")

    # Run generation on eval set
    is_correct_fn = is_correct_gsm8k if args.benchmark == "gsm8k" else is_correct_arc
    results = []
    t0 = time.time()

    print(f"=== Eval set ({len(eval_items)} items) ===")
    for i, item in enumerate(eval_items):
        if args.benchmark == "gsm8k":
            user_msg = item["prompt_template"].format(question=item["question"])
        else:
            user_msg = item["prompt_template"].format(
                question=item["question"],
                choices=item["choices_text"],
            )

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
        raw = generate_greedy(model, tokenizer, prompt)
        conf = parse_confidence(raw)
        correct = is_correct_fn(raw, item["gold_answer"])

        results.append({
            "id": item["id"],
            "raw_output": raw,
            "gold": item["gold_answer"],
            "correct": correct,
            "confidence": conf,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            acc = np.mean([r["correct"] for r in results])
            confs_so_far = np.array([r["confidence"] for r in results])
            parse_rate = (~np.isnan(confs_so_far)).mean()
            print(f"  [{i+1}/{len(eval_items)}]  acc={acc:.3f}  "
                  f"parse={parse_rate:.3f}  elapsed={elapsed:.0f}s")

    # Compute metrics
    corrects = np.array([r["correct"] for r in results], dtype=int)
    confs = np.array([r["confidence"] for r in results])
    mask = ~np.isnan(confs)

    acc = corrects.mean()
    parse_rate = mask.mean()
    conf_mean = float(np.nanmean(confs)) if mask.sum() > 0 else float("nan")
    conf_std = float(np.nanstd(confs)) if mask.sum() > 0 else float("nan")

    if mask.sum() > 10 and corrects[mask].sum() > 0 and corrects[mask].sum() < mask.sum():
        auroc2 = roc_auc_score(corrects[mask], confs[mask])
    else:
        auroc2 = float("nan")

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  RESULTS: {args.benchmark.upper()} zero-shot transfer")
    print(f"  Model: {args.model_name}")
    print(f"  Adapter: {os.path.basename(os.path.dirname(args.adapter_path))}")
    print(f"{'='*60}")
    print(f"  Accuracy:       {acc:.3f}")
    print(f"  Parse rate:     {parse_rate:.3f}")
    print(f"  Verbal AUROC₂:  {auroc2:.3f}")
    print(f"  Conf mean:      {conf_mean:.1f}")
    print(f"  Conf std:       {conf_std:.1f}")

    if args.baseline_auroc2 is not None and not np.isnan(auroc2):
        delta = auroc2 - args.baseline_auroc2
        print(f"\n  Baseline AUROC₂: {args.baseline_auroc2:.3f}")
        print(f"  Delta:           {delta:+.3f}")
        if delta > 0.02:
            print(f"\n  >>> TRANSFER DETECTED: verbal AUROC₂ improved by {delta:+.3f}")
            print(f"  >>> Zero-shot calibration transfer from TriviaQA to {args.benchmark.upper()}")
        elif delta > -0.02:
            print(f"\n  >>> NO CLEAR TRANSFER: delta within noise ({delta:+.3f})")
        else:
            print(f"\n  >>> NEGATIVE TRANSFER: verbal AUROC₂ degraded ({delta:+.3f})")
    else:
        print(f"\n  (No baseline provided — report raw AUROC₂ only)")

    print(f"\n  Total time: {elapsed:.0f}s")

    # Save results
    summary = {
        "benchmark": args.benchmark,
        "model": args.model_name,
        "adapter_path": args.adapter_path,
        "accuracy": round(acc, 3),
        "parse_rate": round(parse_rate, 3),
        "verbal_auroc2": round(auroc2, 3),
        "conf_mean": round(conf_mean, 1),
        "conf_std": round(conf_std, 1),
        "baseline_auroc2": args.baseline_auroc2,
        "delta": round(auroc2 - args.baseline_auroc2, 3) if args.baseline_auroc2 and not np.isnan(auroc2) else None,
        "n_eval": len(eval_items),
        "n_correct": int(corrects.sum()),
        "n_parseable": int(mask.sum()),
        "elapsed_s": round(elapsed, 0),
    }

    label = f"zeroshot_{args.benchmark}_{args.model_name}"
    outpath = os.path.join(args.output_dir, f"{label}.json")
    with open(outpath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {outpath}")

    # Save raw responses
    resp_path = os.path.join(args.output_dir, f"{label}_responses.json")
    with open(resp_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved responses: {resp_path}")


if __name__ == "__main__":
    main()
