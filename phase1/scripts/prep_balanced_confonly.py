"""
prep_balanced_confonly.py — Prepare BALANCED confidence-only training data.

Addresses two hypothesized causes of LoRA failure at Llama 70B:

1. DATA BIAS: At 87% baseline accuracy, ~87% of training items have probe
   targets near 100%. The model can satisfy the loss by always outputting
   high confidence. Solution: rebalance so low-confidence items are
   over-represented.

2. GRADIENT DILUTION: Even on short TriviaQA answers, the answer text
   competes with the confidence tokens for gradient. Solution: reformat
   to confidence-rating-only (model rates an already-given answer).

Input: existing TriviaQA train.jsonl with probe-derived confidence targets
Output: balanced confidence-only train/valid/test JSONL + LoRA config

Strategy:
- Parse confidence target from each existing item
- Bin items by confidence target into 5 bins (0-20, 20-40, 40-60, 60-80, 80-100)
- Resample to balanced distribution across bins
- Reformat to confidence-rating-only

Usage:
    python3 prep_balanced_confonly.py \
        --input-jsonl ~/jpwork/metacog-engineering/phase1/results_raw/finetune/Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED/probe_target/data/train.jsonl \
        --output-dir ~/jpwork/metacog-engineering/phase1/results_raw/finetune/Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED/balanced_confonly/data \
        --model-path ~/mnt/models-lan/foresight/synthesis-archive/Meta-Llama-3.1-70B-Instruct-bf16-CORRECTED \
        --lr 5e-5 --rank 16 --lora-layers 80
"""
import argparse, json, os, re, random
from collections import defaultdict
import yaml

SEED = 42

# Confidence bins (low to high)
BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]
BIN_LABELS = ["very_low", "low", "medium", "high", "very_high"]


def extract_question(user_content):
    """Extract the question from a TriviaQA or GSM8K prompt."""
    match = re.search(r"Question:\s*(.+?)(?:\n|$)", user_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return user_content.strip()


def detect_domain(user_content):
    """Detect whether an item is TriviaQA or GSM8K based on prompt text."""
    lower = user_content.lower()
    if "math problem" in lower or "step by step" in lower:
        return "gsm8k"
    if "trivia" in lower:
        return "triviaqa"
    if "science question" in lower or "correct answer" in lower:
        return "arc"
    return "unknown"


DOMAIN_LABELS = {
    "triviaqa": "trivia question",
    "gsm8k": "math question",
    "arc": "science question",
    "unknown": "question",
}


def extract_answer_and_confidence(assistant_content):
    """Split assistant response into answer and confidence target."""
    parts = re.split(r"(?i)\bconfidence\b", assistant_content)
    if len(parts) >= 2:
        answer = parts[0].strip()
        conf_match = re.search(r"(\d{1,3})\s*%?", parts[-1])
        if conf_match:
            conf = int(conf_match.group(1))
            return answer, conf
    return None, None


def extract_final_answer(answer_text):
    """Extract just the final numeric answer from a CoT response."""
    # Try #### format first (GSM8K standard)
    m = re.search(r"####\s*(.+?)(?:\n|$)", answer_text)
    if m:
        return m.group(1).strip()
    # Fall back to last number in the text
    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", answer_text)
    if numbers:
        return numbers[-1].replace(",", "")
    # Last resort: last non-empty line
    lines = [l.strip() for l in answer_text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else answer_text.strip()


def reformat_as_confonly(question, answer, conf, domain="triviaqa", strip_cot=False):
    """Reformat as a confidence-rating task with domain-appropriate prompt."""
    domain_label = DOMAIN_LABELS.get(domain, "question")

    if strip_cot and domain in ("gsm8k", "arc"):
        final_answer = extract_final_answer(answer)
        user = (
            f"You solved a {domain_label}.\n"
            f"Question: {question}\n"
            f"Your final answer: {final_answer}\n"
            f"How confident are you that {final_answer} is correct? "
            "State your confidence as a percentage from 0 to 100."
        )
    else:
        user = (
            f"You answered the following {domain_label}.\n"
            f"Question: {question}\n"
            f"Your answer: {answer}\n"
            "How confident are you that your answer is correct? "
            "State your confidence as a percentage from 0 to 100."
        )

    assistant = f"Confidence: {conf}%"
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def bin_of(conf):
    """Return the bin index for a confidence value."""
    for i, (lo, hi) in enumerate(BINS):
        if lo <= conf < hi:
            return i
    return len(BINS) - 1  # cap at last bin


def main():
    parser = argparse.ArgumentParser(
        description="Prepare balanced confidence-only training data")
    parser.add_argument("--input-jsonl", required=True,
                        help="Existing PT-CSFT train.jsonl with probe targets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-layers", type=int, default=80)
    parser.add_argument("--target-per-bin", type=int, default=None,
                        help="Target items per bin (default: max possible)")
    parser.add_argument("--max-multiplier", type=float, default=5.0,
                        help="Max oversampling multiplier for rare bins")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--domain", choices=["triviaqa", "gsm8k", "arc", "all"],
                        default="all",
                        help="Filter to specific domain (default: all)")
    parser.add_argument("--strip-cot", action="store_true",
                        help="For GSM8K/ARC: use only the final answer, not the full CoT")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    parent_dir = os.path.dirname(args.output_dir)

    # Load existing data
    print(f"[load] {args.input_jsonl}")
    raw_items = [json.loads(l) for l in open(args.input_jsonl)]
    print(f"  Loaded: {len(raw_items)} items")

    # Parse and bin
    parsed = []
    parse_fail = 0
    domain_counts = defaultdict(int)
    for item in raw_items:
        user_content = item["messages"][0]["content"]
        assistant_content = item["messages"][1]["content"]
        domain = detect_domain(user_content)
        domain_counts[domain] += 1

        # Filter by domain if requested
        if args.domain != "all" and domain != args.domain:
            continue

        question = extract_question(user_content)
        answer, conf = extract_answer_and_confidence(assistant_content)
        if answer is None or conf is None:
            parse_fail += 1
            continue
        parsed.append({"question": question, "answer": answer, "conf": conf,
                        "domain": domain})

    print(f"  Parsed: {len(parsed)} items ({parse_fail} parse failures)")
    print(f"  Domain distribution: {dict(domain_counts)}")
    if args.domain != "all":
        print(f"  Filtered to: {args.domain} ({len(parsed)} items)")

    # Show original distribution
    bins_orig = defaultdict(list)
    for item in parsed:
        bins_orig[bin_of(item["conf"])].append(item)

    print(f"\n[orig] Confidence distribution (5 bins):")
    for i, label in enumerate(BIN_LABELS):
        n = len(bins_orig[i])
        pct = 100 * n / len(parsed)
        bar = "█" * int(pct / 2)
        print(f"  {label:>10} ({BINS[i][0]:>3}-{BINS[i][1]-1:>3}%): {n:>5} ({pct:>5.1f}%) {bar}")

    # Rebalance
    target = args.target_per_bin
    if target is None:
        # Default: target = median bin size (so we don't over-shrink common bins)
        sizes = sorted([len(v) for v in bins_orig.values()])
        target = sizes[len(sizes) // 2]
    print(f"\n[balance] Target items per bin: {target}")

    random.seed(SEED)
    balanced = []
    for i in range(len(BINS)):
        items = bins_orig[i]
        n_orig = len(items)
        if n_orig == 0:
            print(f"  {BIN_LABELS[i]}: empty, skipping")
            continue
        if n_orig >= target:
            # Undersample
            sampled = random.sample(items, target)
            print(f"  {BIN_LABELS[i]}: {n_orig} → {target} (undersample)")
        else:
            # Oversample with cap
            max_oversample = int(n_orig * args.max_multiplier)
            n_target = min(target, max_oversample)
            sampled = items.copy()
            while len(sampled) < n_target:
                sampled.append(random.choice(items))
            print(f"  {BIN_LABELS[i]}: {n_orig} → {len(sampled)} "
                  f"({len(sampled)/n_orig:.1f}× oversample)")
        balanced.extend(sampled)

    print(f"\n  Balanced total: {len(balanced)} items")

    # Reformat
    print(f"\n[reformat] confidence-rating-only format{' (CoT stripped)' if args.strip_cot else ''}")
    reformatted = [reformat_as_confonly(it["question"], it["answer"], it["conf"],
                                        it["domain"], strip_cot=args.strip_cot)
                   for it in balanced]

    # Shuffle and split (90/5/5)
    random.shuffle(reformatted)
    n = len(reformatted)
    n_valid = max(20, n // 20)
    n_test = max(20, n // 20)
    n_train = n - n_valid - n_test

    splits = {
        "train": reformatted[:n_train],
        "valid": reformatted[n_train:n_train + n_valid],
        "test": reformatted[n_train + n_valid:],
    }

    for split_name, items in splits.items():
        path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        print(f"  {split_name}: {len(items)} items → {path}")

    # Show example
    ex = splits["train"][0]
    print(f"\n[example] Reformatted item:")
    print(f"  User (last 200 chars): ...{ex['messages'][0]['content'][-200:]}")
    print(f"  Assistant: {ex['messages'][1]['content']}")

    # Generate LoRA config
    config = {
        "model": args.model_path,
        "train": True,
        "data": args.output_dir,
        "seed": SEED,
        "fine_tune_type": "lora",
        "num_layers": args.lora_layers,
        "batch_size": 1,
        "grad_accumulation_steps": 16,
        "iters": (n_train * args.epochs) // 16,  # effective batches
        "val_batches": 25,
        "learning_rate": args.lr,
        "lr_schedule": {
            "name": "cosine_decay",
            "arguments": [args.lr, (n_train * args.epochs) // 16],
        },
        "steps_per_report": 21,
        "steps_per_eval": max(50, (n_train * args.epochs) // 48),
        "adapter_path": os.path.join(parent_dir, "adapters"),
        "save_every": (n_train * args.epochs) // 16,
        "grad_checkpoint": True,
        "mask_prompt": True,
        "max_seq_length": 512,
        "optimizer": "adam",
        "test": False,
        "lora_parameters": {
            "rank": args.rank,
            "scale": 2.0,
            "dropout": 0.05,
        },
    }
    config_path = os.path.join(parent_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n[config] {config_path}")
    print(f"  iters: {config['iters']}, lr: {args.lr}, rank: {args.rank}")
    print(f"\n[next] Train:")
    print(f"  python3 -m mlx_lm lora --config {config_path}")
    print(f"\n[then] Two-pass eval:")
    print(f"  python3 eval_confonly_triviaqa.py \\")
    print(f"      --model-path {args.model_path} \\")
    print(f"      --model-name $(basename {args.model_path}) \\")
    print(f"      --adapter-path {os.path.join(parent_dir, 'adapters')}")


if __name__ == "__main__":
    main()
