"""
reformat_confidence_only.py — Reformat GSM8K training items from
CoT+confidence to confidence-rating-only format.

Problem: In the original format, the assistant response is 400 tokens of CoT
plus 5 tokens of confidence. The confidence gets negligible gradient.

Fix: Separate the CoT from the confidence. The assistant response is ONLY
the confidence tag. The CoT answer goes into the user turn as context.

Training format:
  User: "You answered the following math question.
         Question: [question]
         Your answer: [model's CoT answer]
         How confident are you that your answer is correct? (0-100)"
  Assistant: "Confidence: 42%"

At inference time (two-pass):
  1. Generate answer normally (no adapter)
  2. Load adapter, present question+answer, get confidence

Usage:
    python3 reformat_confidence_only.py \
        --input-dir ~/jpwork/metacog-engineering/phase1/results_raw/finetune/gemma-3-12b-it/multitask_probe_target/data \
        --output-dir ~/jpwork/metacog-engineering/phase1/results_raw/finetune/gemma-3-12b-it/confonly_probe_target/data \
        --model-path ~/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it \
        --lr 2e-4 --rank 16
"""
import argparse, json, os, re, random
import yaml

SEED = 42

# Markers that identify GSM8K items in the training data
GSM8K_MARKERS = ["math problem", "step by step"]
TRIVIA_MARKERS = ["trivia question"]


def is_gsm8k_item(item):
    user_content = item["messages"][0]["content"].lower()
    return any(m in user_content for m in GSM8K_MARKERS)


def extract_question_from_prompt(user_content):
    """Extract the question text from the GSM8K prompt."""
    match = re.search(r"Question:\s*(.+?)$", user_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return user_content


def extract_answer_and_confidence(assistant_content):
    """Split assistant response into answer (CoT) and confidence target."""
    # Split on last occurrence of Confidence (case-insensitive)
    parts = re.split(r"(?i)\bconfidence\b", assistant_content)
    if len(parts) >= 2:
        answer = parts[0].strip()
        # Extract the number from the confidence part
        conf_part = parts[-1]
        conf_match = re.search(r"(\d{1,3})\s*%?", conf_part)
        if conf_match:
            conf = int(conf_match.group(1))
            return answer, conf
    return assistant_content.strip(), None


def reformat_gsm8k_item(item):
    """Convert a GSM8K CoT+confidence item to confidence-rating format."""
    user_content = item["messages"][0]["content"]
    assistant_content = item["messages"][1]["content"]

    question = extract_question_from_prompt(user_content)
    answer, conf = extract_answer_and_confidence(assistant_content)

    if conf is None:
        return None  # Skip items where we can't parse confidence

    # New format: model rates its own answer
    new_user = (
        "You answered the following math question.\n"
        f"Question: {question}\n"
        f"Your answer: {answer}\n"
        "How confident are you that your answer is correct? "
        "State your confidence as a percentage from 0 to 100."
    )
    new_assistant = f"Confidence: {conf}%"

    return {
        "messages": [
            {"role": "user", "content": new_user},
            {"role": "assistant", "content": new_assistant},
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reformat GSM8K items to confidence-rating-only format")
    parser.add_argument("--input-dir", required=True,
                        help="Directory with existing train/valid/test.jsonl")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for reformatted data")
    parser.add_argument("--model-path", required=True,
                        help="Model path for LoRA config")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-layers", type=int, default=48,
                        help="Number of LoRA layers (model-specific)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Process each split
    for split in ["train", "valid", "test"]:
        input_path = os.path.join(args.input_dir, f"{split}.jsonl")
        if not os.path.exists(input_path):
            print(f"  Skipping {split} (not found)")
            continue

        items = [json.loads(l) for l in open(input_path)]
        trivia_items = []
        gsm8k_items = []
        gsm8k_failed = 0

        for item in items:
            if is_gsm8k_item(item):
                reformatted = reformat_gsm8k_item(item)
                if reformatted:
                    gsm8k_items.append(reformatted)
                else:
                    gsm8k_failed += 1
            else:
                trivia_items.append(item)

        combined = trivia_items + gsm8k_items
        random.seed(SEED)
        random.shuffle(combined)

        output_path = os.path.join(args.output_dir, f"{split}.jsonl")
        with open(output_path, "w") as f:
            for item in combined:
                f.write(json.dumps(item) + "\n")

        print(f"  {split}: {len(trivia_items)} TriviaQA + {len(gsm8k_items)} GSM8K "
              f"(reformatted) = {len(combined)} total"
              f"{f' ({gsm8k_failed} GSM8K parse failures)' if gsm8k_failed else ''}")

        # Show example
        if split == "train" and gsm8k_items:
            ex = gsm8k_items[0]
            print(f"\n  Example GSM8K (reformatted):")
            print(f"    User (last 150 chars): ...{ex['messages'][0]['content'][-150:]}")
            print(f"    Assistant: {ex['messages'][1]['content']}")
            print()

    # Generate LoRA config
    n_train = len([json.loads(l) for l in open(os.path.join(args.output_dir, "train.jsonl"))])
    parent_dir = os.path.dirname(args.output_dir)
    config = {
        "model": args.model_path,
        "train": True,
        "data": args.output_dir,
        "seed": SEED,
        "lora_layers": args.lora_layers,
        "batch_size": 1,
        "iters": n_train * 3,  # 3 epochs
        "val_batches": 25,
        "learning_rate": args.lr,
        "steps_per_report": 50,
        "steps_per_eval": n_train // 2,
        "adapter_path": os.path.join(parent_dir, "adapters"),
        "save_every": n_train,
        "grad_checkpoint": True,
        "mask_prompt": True,
        "lora_parameters": {
            "rank": args.rank,
            "scale": 2.0,
            "dropout": 0.05,
            "keys": ["self_attn.q_proj", "self_attn.k_proj",
                     "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        },
    }
    config_path = os.path.join(parent_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"  Config: {config_path}")
    print(f"  Training items: {n_train}")
    print(f"  Iters (3 epochs): {n_train * 3}")
    print(f"\n  Run training:")
    print(f"    python3 -m mlx_lm lora --config {config_path}")
    print(f"\n  Then eval (two-pass inference needed — see eval_confonly.py)")


if __name__ == "__main__":
    main()
