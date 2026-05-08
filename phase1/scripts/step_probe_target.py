"""
Step 2b/3b: Probe-Target CSFT (Exploratory)
=============================================

Ad-hoc exploration motivated by the negative CSFT result: self-consistency
targets are bimodal (91% extreme), so CSFT has no graded signal to learn.

Instead, use the fitted linear probe's P(correct) as continuous confidence
targets. The probe (Step 1b) achieves AUROC₂ = 0.857 on hidden states,
proving the internal signal exists. This experiment tests whether LoRA
fine-tuning can teach the model to VERBALIZE that internal signal.

Workflow:
    1. Load fitted probe (from Step 1b) and T-cal hidden states (from Step 1)
    2. Compute P(correct) per T-cal item using probe
    3. Map to confidence percentage (continuous, 0-100)
    4. Build training set with probe-derived targets
    5. Run LoRA fine-tuning (same hyperparameters as pre-reg Step 3)

Usage:
    # Prep data + config:
    python3 step_probe_target.py \
        --model-name gemma-3-12b-it \
        --model-path /Users/chrismarmo/mnt/models-lan/foresight/synthesis-archive/gemma-3-12b-it \
        --prep-only

    # Then run training:
    python3 -m mlx_lm lora --config ~/jpwork/results/finetune/gemma-3-12b-it/probe_target/config.yaml

    # Then evaluate (use step4_eval_phase1.py with --adapter-label probe_target)
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Config (same LoRA hyperparameters as pre-reg Step 3)
# ---------------------------------------------------------------------------

SEED = 42
N_TEVAL = 1000
N_TCAL = 2000

LORA_RANK = 16
LORA_ALPHA = 32
LORA_SCALE = LORA_ALPHA / LORA_RANK  # 2.0
LORA_DROPOUT = 0.05
LR = 2e-4
N_EPOCHS = 3
EFFECTIVE_BATCH = 16
BATCH_SIZE = 4
GRAD_ACCUM = EFFECTIVE_BATCH // BATCH_SIZE
MAX_SEQ_LENGTH = 512

MODEL_LAYERS = {
    "gemma-3-12b-it": 48,
    "gemma-3-27b-it": 62,
}

STEP1_DIR = Path(os.path.expanduser("~/jpwork/results/step1"))
PROBE_DIR = Path(os.path.expanduser("~/jpwork/results/probe"))
FINETUNE_DIR = Path(os.path.expanduser("~/jpwork/results/finetune"))

PRIMARY_PROBE_CONFIG = "last_last_answer_token"

TRIVIAQA_PROMPT = (
    "You are answering trivia questions. "
    "After your answer, state your confidence as a percentage from 0 to 100.\n"
    "Question: {question}\n"
)


# ---------------------------------------------------------------------------
# Probe application
# ---------------------------------------------------------------------------

def load_and_apply_probe(
    probe_fits_path: Path,
    hidden_states_path: Path,
    responses_path: Path,
    config_key: str = PRIMARY_PROBE_CONFIG,
) -> dict[str, float]:
    """Load fitted probe, apply to hidden states, return {qid: P(correct)}.

    Uses the primary probe config (last layer, last answer token).
    """
    # Load probe parameters
    fits = np.load(probe_fits_path)

    coef_key = f"{config_key}__coef"
    intercept_key = f"{config_key}__intercept"
    mean_key = f"{config_key}__scaler_mean"
    scale_key = f"{config_key}__scaler_scale"

    coef = fits[coef_key]
    intercept = fits[intercept_key]
    scaler_mean = fits[mean_key]
    scaler_scale = fits[scale_key]

    print(f"[probe] Loaded probe: coef shape={coef.shape}, "
          f"intercept={float(intercept.flat[0]):.4f}")

    # Load hidden states
    npz = np.load(hidden_states_path)

    # Parse config key into layer/position
    parts = config_key.split("_", 1)  # "last" and "last_answer_token"
    layer = parts[0]
    position = parts[1] if len(parts) > 1 else parts[0]
    # Actually the config key is like "last_last_answer_token"
    # which means layer=last, position=last_answer_token
    # Need to split properly
    if config_key == "last_last_answer_token":
        layer, position = "last", "last_answer_token"
    elif config_key == "last_pre_answer_token":
        layer, position = "last", "pre_answer_token"
    elif config_key == "middle_last_answer_token":
        layer, position = "middle", "last_answer_token"
    elif config_key == "middle_pre_answer_token":
        layer, position = "middle", "pre_answer_token"
    elif config_key == "first_last_answer_token":
        layer, position = "first", "last_answer_token"
    elif config_key == "first_pre_answer_token":
        layer, position = "first", "pre_answer_token"
    else:
        raise ValueError(f"Unknown probe config: {config_key}")

    suffix = f"__{layer}__{position}"

    # Apply probe to each item
    scores = {}
    for key in npz.files:
        if not key.endswith(suffix):
            continue
        qid = key[:key.index("__")]
        x = npz[key].reshape(1, -1)
        x_s = (x - scaler_mean) / scaler_scale
        logit = x_s @ coef.T + intercept
        p_correct = 1.0 / (1.0 + np.exp(-logit))
        scores[qid] = float(p_correct[0, 0])

    print(f"[probe] Applied to {len(scores)} items")

    # Distribution summary
    vals = np.array(list(scores.values()))
    print(f"[probe] P(correct) distribution: "
          f"mean={vals.mean():.3f}, std={vals.std():.3f}, "
          f"min={vals.min():.3f}, max={vals.max():.3f}")

    return scores


# ---------------------------------------------------------------------------
# Training set construction
# ---------------------------------------------------------------------------

def build_probe_target_training_set(
    probe_scores: dict[str, float],
    responses_path: Path,
    calibration_path: Path | None = None,
) -> list[dict]:
    """Build training set with probe-derived continuous confidence targets.

    For each T-cal item:
    - confidence_target = round(P(correct) * 100)  (continuous, 0-100)
    - answer = greedy answer from Step 1 baseline
    - question = original question
    """
    # Load Step 1 greedy responses for answers
    with open(responses_path) as f:
        responses = json.load(f)

    resp_by_qid = {r["question_id"]: r for r in responses}

    # Try to get questions from calibration data if available
    q_by_qid = {}
    if calibration_path and calibration_path.exists():
        with open(calibration_path) as f:
            cal_data = json.load(f)
        q_by_qid = {item["question_id"]: item["question"] for item in cal_data}

    training_set = []
    for qid, p_correct in probe_scores.items():
        if qid not in resp_by_qid:
            continue

        resp = resp_by_qid[qid]
        question = q_by_qid.get(qid, resp.get("question", ""))
        answer = resp.get("parsed_answer", "")

        if not question or not answer:
            continue

        # Continuous confidence target from probe
        confidence_target = int(round(p_correct * 100))
        confidence_target = max(0, min(100, confidence_target))

        training_set.append({
            "question_id": qid,
            "question": question,
            "answer": answer,
            "confidence_target": confidence_target,
            "probe_p_correct": round(p_correct, 4),
            "correct": resp.get("correct", False),
        })

    print(f"[data] Built training set: {len(training_set)} items")

    # Distribution of targets
    targets = np.array([item["confidence_target"] for item in training_set])
    print(f"[data] Target distribution: "
          f"mean={targets.mean():.1f}, std={targets.std():.1f}")
    print(f"[data] Targets in [20,80] (intermediate): "
          f"{np.sum((targets >= 20) & (targets <= 80))} "
          f"({np.mean((targets >= 20) & (targets <= 80))*100:.1f}%)")

    return training_set


def convert_to_jsonl(training_set: list[dict], output_path: Path):
    """Convert training set to MLX chat JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in training_set:
            user_content = TRIVIAQA_PROMPT.format(question=item["question"])
            assistant_content = (
                f"{item['answer']}\nConfidence: {item['confidence_target']}%"
            )
            record = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
            f.write(json.dumps(record) + "\n")
    print(f"  Wrote {len(training_set)} items to {output_path}")


def make_valid_split(training_set: list[dict], n_valid: int = 50):
    rng = random.Random(SEED)
    indices = list(range(len(training_set)))
    rng.shuffle(indices)
    valid_idx = set(indices[:n_valid])
    train = [training_set[i] for i in range(len(training_set)) if i not in valid_idx]
    valid = [training_set[i] for i in range(len(training_set)) if i in valid_idx]
    return train, valid


def generate_config(
    model_path: str,
    model_name: str,
    data_dir: str,
    adapter_path: str,
    n_train: int,
) -> dict:
    n_layers = MODEL_LAYERS[model_name]
    steps_per_epoch = (n_train + EFFECTIVE_BATCH - 1) // EFFECTIVE_BATCH
    total_iters = steps_per_epoch * N_EPOCHS

    return {
        "model": model_path,
        "data": data_dir,
        "train": True,
        "test": False,
        "seed": SEED,
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
        "steps_per_eval": steps_per_epoch,
        "save_every": total_iters,
        "adapter_path": adapter_path,
        "grad_checkpoint": False,
        "lora_parameters": {
            "rank": LORA_RANK,
            "dropout": LORA_DROPOUT,
            "scale": LORA_SCALE,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Probe-target CSFT (exploratory)"
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prep-only", action="store_true")
    parser.add_argument("--n-valid", type=int, default=50)
    args = parser.parse_args()

    model_name = args.model_name
    model_path = os.path.expanduser(args.model_path)

    # Input paths
    probe_fits = PROBE_DIR / f"probe_fits_{model_name}.npz"
    tcal_hidden = STEP1_DIR / f"hidden_states_tcal_{model_name}.npz"
    tcal_resp = STEP1_DIR / f"tcal_greedy_responses_{model_name}.json"
    calibration = Path(os.path.expanduser(
        f"~/jpwork/results/step2/step2_calibration_{model_name}.json"
    ))

    for p, label in [
        (probe_fits, "probe fits"),
        (tcal_hidden, "T-cal hidden states"),
        (tcal_resp, "T-cal responses"),
    ]:
        if not p.exists():
            print(f"[fatal] Missing: {p} ({label})")
            sys.exit(2)

    # Output paths
    pt_dir = FINETUNE_DIR / model_name / "probe_target"
    data_dir = pt_dir / "data"
    adapter_dir = pt_dir / "adapters"

    print(f"{'='*60}")
    print(f"Probe-Target CSFT (Exploratory): {model_name}")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # 1. Apply probe to T-cal hidden states
    # -----------------------------------------------------------------------
    print("[Step 1] Applying fitted probe to T-cal hidden states...")
    probe_scores = load_and_apply_probe(
        probe_fits, tcal_hidden, tcal_resp,
        config_key=PRIMARY_PROBE_CONFIG,
    )

    # -----------------------------------------------------------------------
    # 2. Build training set with probe-derived targets
    # -----------------------------------------------------------------------
    print("\n[Step 2] Building probe-target training set...")
    training_set = build_probe_target_training_set(
        probe_scores, tcal_resp, calibration,
    )

    # Save training set for reference
    pt_dir.mkdir(parents=True, exist_ok=True)
    ts_path = pt_dir / f"probe_target_training_set_{model_name}.json"
    with open(ts_path, "w") as f:
        json.dump(training_set, f, indent=2)
    print(f"[save] Training set: {ts_path}")

    # -----------------------------------------------------------------------
    # 3. Convert to JSONL and generate config
    # -----------------------------------------------------------------------
    print("\n[Step 3] Preparing JSONL data + config...")
    train_split, valid_split = make_valid_split(training_set, n_valid=args.n_valid)
    convert_to_jsonl(train_split, data_dir / "train.jsonl")
    convert_to_jsonl(valid_split, data_dir / "valid.jsonl")
    convert_to_jsonl(valid_split[:10], data_dir / "test.jsonl")

    config = generate_config(
        model_path=model_path,
        model_name=model_name,
        data_dir=str(data_dir),
        adapter_path=str(adapter_dir),
        n_train=len(train_split),
    )
    config_path = pt_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[save] Config: {config_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    steps_per_epoch = (len(train_split) + EFFECTIVE_BATCH - 1) // EFFECTIVE_BATCH
    total_iters = steps_per_epoch * N_EPOCHS

    print(f"\n{'='*60}")
    print(f"PROBE-TARGET CSFT PREP COMPLETE: {model_name}")
    print(f"{'='*60}")
    print(f"  Training items: {len(train_split)} (+ {args.n_valid} valid)")
    print(f"  Probe config: {PRIMARY_PROBE_CONFIG}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total iters (3 epochs): {total_iters}")

    # Compare target distributions
    sc_targets = []
    sc_path = Path(os.path.expanduser(
        f"~/jpwork/results/step2/step2_training_set_{model_name}.json"
    ))
    if sc_path.exists():
        with open(sc_path) as f:
            sc_data = json.load(f)
        sc_targets = [item["confidence_target"] for item in sc_data]

    probe_targets = [item["confidence_target"] for item in training_set]

    print(f"\n  Target distribution comparison:")
    print(f"  {'':20s} {'SC (original)':>15s} {'Probe (new)':>15s}")
    if sc_targets:
        sc_arr = np.array(sc_targets)
        pt_arr = np.array(probe_targets)
        print(f"  {'Mean':20s} {sc_arr.mean():>15.1f} {pt_arr.mean():>15.1f}")
        print(f"  {'Std':20s} {sc_arr.std():>15.1f} {pt_arr.std():>15.1f}")
        sc_inter = np.sum((sc_arr >= 20) & (sc_arr <= 80))
        pt_inter = np.sum((pt_arr >= 20) & (pt_arr <= 80))
        print(f"  {'Intermediate [20-80]':20s} {sc_inter:>15d} {pt_inter:>15d}")
        sc_extreme = np.sum((sc_arr <= 5) | (sc_arr >= 95))
        pt_extreme = np.sum((pt_arr <= 5) | (pt_arr >= 95))
        print(f"  {'Extreme [≤5 or ≥95]':20s} {sc_extreme:>15d} {pt_extreme:>15d}")

    print(f"\nTo run training:")
    print(f"  python3 -m mlx_lm lora --config {config_path}")
    print(f"\nTo evaluate (after training):")
    print(f"  # Use step4_eval_phase1.py but point to probe_target adapter")


if __name__ == "__main__":
    main()
