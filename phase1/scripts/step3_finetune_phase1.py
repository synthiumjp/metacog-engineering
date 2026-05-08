"""
Step 3: LoRA Fine-Tuning (Phase 1 — MLX on M3 Ultra)
======================================================

Converts Step 2 training set to MLX chat JSONL format, generates YAML config,
and runs LoRA fine-tuning via `python3 -m mlx_lm lora`.

Also runs Step 3b (shuffled-target control) with seed 43.

Pre-reg parameters (locked):
    LoRA: rank 16, alpha 32 (scale=2.0), dropout 0.05
    All layers, all proj modules (q, k, v, o, gate, up, down)
    LR: 2e-4, cosine schedule, 3 epochs, effective batch size 16
    Training seed: 42, Shuffled seed: 43
    mask_prompt: true

Usage:
    # Prep data + generate configs:
    python3 step3_finetune_phase1.py \
        --model-name gemma-3-12b-it \
        --model-path ~/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it \
        --prep-only

    # Then run training:
    python3 -m mlx_lm lora --config ~/jpwork/results/finetune/gemma-3-12b-it/real/config.yaml
    python3 -m mlx_lm lora --config ~/jpwork/results/finetune/gemma-3-12b-it/shuffled/config.yaml
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Config (locked per pre-reg)
# ---------------------------------------------------------------------------

SEED = 42
SHUFFLE_SEED = 43
LORA_RANK = 16
LORA_ALPHA = 32
LORA_SCALE = LORA_ALPHA / LORA_RANK  # 2.0
LORA_DROPOUT = 0.05
LR = 2e-4
N_EPOCHS = 3
EFFECTIVE_BATCH = 16
BATCH_SIZE = 4
GRAD_ACCUM = EFFECTIVE_BATCH // BATCH_SIZE  # 4
MAX_SEQ_LENGTH = 512

# Layer counts per model
MODEL_LAYERS = {
    "gemma-3-12b-it": 48,
    "gemma-3-27b-it": 62,
}

STEP2_DIR = Path(os.path.expanduser("~/jpwork/results/step2"))
FINETUNE_DIR = Path(os.path.expanduser("~/jpwork/results/finetune"))

TRIVIAQA_PROMPT = (
    "You are answering trivia questions. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
)


# ---------------------------------------------------------------------------
# Data conversion
# ---------------------------------------------------------------------------

def convert_training_set_to_jsonl(training_set: list[dict], output_path: Path):
    """Convert Step 2 training set to MLX chat JSONL format.

    Each item becomes:
    {"messages": [
        {"role": "user", "content": "<trivia prompt>"},
        {"role": "assistant", "content": "<answer>\nConfidence: <target>%"}
    ]}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in training_set:
            user_content = TRIVIAQA_PROMPT.format(question=item["question"])
            # Use completion field which has "answer\nConfidence: X%"
            assistant_content = item["completion"]

            record = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
            f.write(json.dumps(record) + "\n")
    print(f"  Wrote {len(training_set)} items to {output_path}")


def make_valid_split(training_set: list[dict], n_valid: int = 50) -> tuple[list, list]:
    """Split off a small validation set for training-loop loss monitoring.

    This is NOT the pre-reg evaluation (that's Step 4 on T-eval).
    Just for MLX to report val loss during training.
    """
    import random
    rng = random.Random(SEED)
    indices = list(range(len(training_set)))
    rng.shuffle(indices)

    valid_idx = set(indices[:n_valid])
    train_split = [training_set[i] for i in range(len(training_set)) if i not in valid_idx]
    valid_split = [training_set[i] for i in range(len(training_set)) if i in valid_idx]

    return train_split, valid_split


# ---------------------------------------------------------------------------
# YAML config generation
# ---------------------------------------------------------------------------

def generate_config(
    model_path: str,
    model_name: str,
    data_dir: str,
    adapter_path: str,
    seed: int,
    n_train: int,
) -> dict:
    """Generate MLX LoRA training config matching pre-reg parameters."""

    n_layers = MODEL_LAYERS.get(model_name)
    if n_layers is None:
        raise ValueError(f"Unknown model {model_name}. Add to MODEL_LAYERS dict.")

    # Compute iterations: 3 epochs over n_train items at effective batch 16
    steps_per_epoch = (n_train + EFFECTIVE_BATCH - 1) // EFFECTIVE_BATCH
    total_iters = steps_per_epoch * N_EPOCHS

    config = {
        "model": model_path,
        "data": data_dir,
        "train": True,
        "test": False,
        "seed": seed,
        "fine_tune_type": "lora",
        "optimizer": "adam",
        "batch_size": BATCH_SIZE,
        "grad_accumulation_steps": GRAD_ACCUM,
        "iters": total_iters,
        "learning_rate": LR,
        "lr_schedule": {
            "name": "cosine_decay",
            "arguments": [LR, total_iters],
        },
        "num_layers": n_layers,
        "max_seq_length": MAX_SEQ_LENGTH,
        "mask_prompt": True,
        "steps_per_report": max(1, steps_per_epoch // 5),
        "steps_per_eval": steps_per_epoch,  # eval once per epoch
        "save_every": total_iters,  # save only at end
        "adapter_path": adapter_path,
        "grad_checkpoint": False,
        "lora_parameters": {
            "rank": LORA_RANK,
            "dropout": LORA_DROPOUT,
            "scale": LORA_SCALE,
        },
    }
    return config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 3: LoRA fine-tuning")
    parser.add_argument("--model-name", required=True,
                        help="Model name (e.g. gemma-3-12b-it)")
    parser.add_argument("--model-path", required=True,
                        help="Full path to model directory")
    parser.add_argument("--prep-only", action="store_true",
                        help="Only prepare data + configs, don't run training")
    parser.add_argument("--run-real", action="store_true",
                        help="Run real-target training")
    parser.add_argument("--run-shuffled", action="store_true",
                        help="Run shuffled-target training (Step 3b)")
    parser.add_argument("--n-valid", type=int, default=50,
                        help="Number of items for validation split (default: 50)")
    args = parser.parse_args()

    model_name = args.model_name
    model_path = os.path.expanduser(args.model_path)

    # Paths
    real_dir = FINETUNE_DIR / model_name / "real"
    shuffled_dir = FINETUNE_DIR / model_name / "shuffled"
    real_data_dir = real_dir / "data"
    shuffled_data_dir = shuffled_dir / "data"

    # -----------------------------------------------------------------------
    # Load Step 2 outputs
    # -----------------------------------------------------------------------
    train_path = STEP2_DIR / f"step2_training_set_{model_name}.json"
    shuffled_path = STEP2_DIR / f"step2_shuffled_training_set_{model_name}.json"

    if not train_path.exists():
        print(f"[fatal] Step 2 training set not found: {train_path}")
        print("        Run Step 2 first.")
        sys.exit(2)

    with open(train_path) as f:
        training_set = json.load(f)
    print(f"[data] Loaded real training set: {len(training_set)} items")

    if not shuffled_path.exists():
        print(f"[fatal] Step 2 shuffled training set not found: {shuffled_path}")
        sys.exit(2)

    with open(shuffled_path) as f:
        shuffled_set = json.load(f)
    print(f"[data] Loaded shuffled training set: {len(shuffled_set)} items")

    # Check that training set has 'question' field (needed for JSONL conversion)
    # Step 2 training_set has 'prompt' (chat-formatted) but we need raw question
    # for the messages format. Check if 'question' is available via calibration data.
    if "question" not in training_set[0]:
        # Load from calibration data to get questions
        cal_path = STEP2_DIR / f"step2_calibration_{model_name}.json"
        if cal_path.exists():
            with open(cal_path) as f:
                cal_data = json.load(f)
            q_by_qid = {item["question_id"]: item["question"] for item in cal_data}
            for item in training_set:
                item["question"] = q_by_qid.get(item["question_id"], "")
            for item in shuffled_set:
                item["question"] = q_by_qid.get(item["question_id"], "")
            print(f"[data] Added questions from calibration data")
        else:
            print(f"[warn] No calibration data found; using prompt field directly")
            # Fall back: extract question from prompt field
            # The prompt has the chat template applied, so we need to strip it
            print("[fatal] Cannot extract raw questions. Add 'question' to training set.")
            sys.exit(2)

    # -----------------------------------------------------------------------
    # Split and convert data
    # -----------------------------------------------------------------------
    print(f"\n[prep] Preparing real-target data...")
    train_split, valid_split = make_valid_split(training_set, n_valid=args.n_valid)
    convert_training_set_to_jsonl(train_split, real_data_dir / "train.jsonl")
    convert_training_set_to_jsonl(valid_split, real_data_dir / "valid.jsonl")
    # MLX needs test.jsonl even if test=False
    convert_training_set_to_jsonl(valid_split[:10], real_data_dir / "test.jsonl")

    print(f"\n[prep] Preparing shuffled-target data...")
    shuf_train_split, shuf_valid_split = make_valid_split(shuffled_set, n_valid=args.n_valid)
    convert_training_set_to_jsonl(shuf_train_split, shuffled_data_dir / "train.jsonl")
    convert_training_set_to_jsonl(shuf_valid_split, shuffled_data_dir / "valid.jsonl")
    convert_training_set_to_jsonl(shuf_valid_split[:10], shuffled_data_dir / "test.jsonl")

    # -----------------------------------------------------------------------
    # Generate YAML configs
    # -----------------------------------------------------------------------
    n_train_actual = len(train_split)

    real_config = generate_config(
        model_path=model_path,
        model_name=model_name,
        data_dir=str(real_data_dir),
        adapter_path=str(real_dir / "adapters"),
        seed=SEED,
        n_train=n_train_actual,
    )
    real_config_path = real_dir / "config.yaml"
    with open(real_config_path, "w") as f:
        yaml.dump(real_config, f, default_flow_style=False)
    print(f"\n[config] Real config: {real_config_path}")

    shuffled_config = generate_config(
        model_path=model_path,
        model_name=model_name,
        data_dir=str(shuffled_data_dir),
        adapter_path=str(shuffled_dir / "adapters"),
        seed=SHUFFLE_SEED,
        n_train=len(shuf_train_split),
    )
    shuffled_config_path = shuffled_dir / "config.yaml"
    with open(shuffled_config_path, "w") as f:
        yaml.dump(shuffled_config, f, default_flow_style=False)
    print(f"[config] Shuffled config: {shuffled_config_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    steps_per_epoch = (n_train_actual + EFFECTIVE_BATCH - 1) // EFFECTIVE_BATCH
    total_iters = steps_per_epoch * N_EPOCHS

    print(f"\n{'='*60}")
    print(f"STEP 3 PREP COMPLETE: {model_name}")
    print(f"{'='*60}")
    print(f"  Training items (real): {n_train_actual} (+ {args.n_valid} valid)")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total iters (3 epochs): {total_iters}")
    print(f"  Effective batch: {BATCH_SIZE} × {GRAD_ACCUM} = {EFFECTIVE_BATCH}")
    print(f"  LoRA: rank={LORA_RANK}, scale={LORA_SCALE}, dropout={LORA_DROPOUT}")
    print(f"  LR: {LR}, cosine schedule")
    print(f"  mask_prompt: True")
    print(f"\nTo run training:")
    print(f"  python3 -m mlx_lm lora --config {real_config_path}")
    print(f"  python3 -m mlx_lm lora --config {shuffled_config_path}")

    # -----------------------------------------------------------------------
    # Optionally run training
    # -----------------------------------------------------------------------
    if args.run_real:
        print(f"\n{'='*60}")
        print(f"Running real-target training...")
        print(f"{'='*60}\n")
        t0 = time.time()
        subprocess.run(
            ["python3", "-m", "mlx_lm", "lora", "--config", str(real_config_path)],
            check=True,
        )
        print(f"\nReal-target training complete in {(time.time()-t0)/60:.1f} min")

    if args.run_shuffled:
        print(f"\n{'='*60}")
        print(f"Running shuffled-target training (Step 3b)...")
        print(f"{'='*60}\n")
        t0 = time.time()
        subprocess.run(
            ["python3", "-m", "mlx_lm", "lora", "--config", str(shuffled_config_path)],
            check=True,
        )
        print(f"\nShuffled-target training complete in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
