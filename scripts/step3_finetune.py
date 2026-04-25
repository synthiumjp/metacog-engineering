"""
Step 3 / 3b: LoRA Fine-Tuning
===============================

Phase 0 v4, pre-reg v2.

Step 3:  Fine-tune on T-cal training set (modal_correct=True) with real
         confidence targets from Step 2.
Step 3b: Fine-tune on the same items with shuffled targets (seed=43)
         as the format-learning control for H2.

Both runs use identical architecture, hyperparameters, and training
procedure. Only the confidence target values differ.

Pre-registered hyperparameters:
    LoRA rank=16, alpha=32, dropout=0.05
    Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    LR: 2e-4, cosine schedule, 3 epochs
    Effective batch size: 16 (micro-batch × gradient accumulation)
    Seed: 42

Hardware fixes (from environment setup session):
    - dtype: bfloat16 (fp16 produces NaN on Gemma 3)
    - No device_map="auto" (accelerate hooks trigger HIP errors)
    - autocast_adapter_dtype=False (PEFT fp32 cast fails on ROCm)
    - attn_implementation="eager" (SDPA not compiled)

Outputs:
    D:\\metacog\\results\\finetune\\lora_real\\       (adapter weights)
    D:\\metacog\\results\\finetune\\lora_shuffled\\    (adapter weights)
    D:\\metacog\\results\\finetune\\training_log.json  (loss curves)

Runtime: ~1-2 hours per run (depends on training set size).
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "google/gemma-3-4b-it"
SEED = 42

# LoRA config (pre-registered)
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training config (pre-registered)
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
EFFECTIVE_BATCH_SIZE = 16
MICRO_BATCH_SIZE = 1  # Constrained by VRAM; accumulate to reach 16
MAX_SEQ_LENGTH = 512

PROJECT_ROOT = Path(r"D:\metacog")
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "finetune"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """Simple dataset that returns pre-tokenised SFT examples."""

    def __init__(self, examples: list[dict], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        for ex in examples:
            encoded = tokenizer(
                ex["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            input_ids = encoded["input_ids"]
            # Labels = input_ids (causal LM); mask prompt tokens with -100
            # Find where the assistant turn begins
            labels = input_ids.copy()

            # Mask everything up to and including the model turn token
            # For Gemma 3: the assistant turn starts after "<start_of_turn>model\n"
            model_turn_text = "<start_of_turn>model\n"
            model_turn_ids = tokenizer.encode(
                model_turn_text, add_special_tokens=False
            )
            # Find the start of assistant content
            prompt_end = self._find_subsequence(input_ids, model_turn_ids)
            if prompt_end is not None:
                mask_end = prompt_end + len(model_turn_ids)
                labels[:mask_end] = [-100] * mask_end
            # else: no masking (shouldn't happen with valid data)

            self.data.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": [1] * len(input_ids),
            })

    @staticmethod
    def _find_subsequence(seq: list, subseq: list) -> int | None:
        """Find the starting index of subseq in seq."""
        for i in range(len(seq) - len(subseq) + 1):
            if seq[i:i + len(subseq)] == subseq:
                return i
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(
                item["attention_mask"], dtype=torch.long
            ),
        }


def collate_fn(batch):
    """Pad batch to max length in batch."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = []
    labels = []
    attention_mask = []

    for b in batch:
        pad_len = max_len - b["input_ids"].shape[0]
        input_ids.append(
            torch.cat([b["input_ids"],
                       torch.zeros(pad_len, dtype=torch.long)])
        )
        labels.append(
            torch.cat([b["labels"],
                       torch.full((pad_len,), -100, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([b["attention_mask"],
                       torch.zeros(pad_len, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_lora(
    model,
    tokenizer,
    train_data: list[dict],
    output_dir: Path,
    run_label: str,
    seed: int = SEED,
    dry_run: bool = False,
) -> dict:
    """Run LoRA fine-tuning and save adapter weights."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed everything
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build dataset
    dataset = SFTDataset(train_data, tokenizer, MAX_SEQ_LENGTH)
    if dry_run:
        # Use only first 4 examples
        dataset.data = dataset.data[:4]

    grad_accum_steps = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE
    dataloader = DataLoader(
        dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )

    total_steps = len(dataloader) * NUM_EPOCHS
    warmup_steps = max(1, total_steps // 10)

    # Optimiser
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\n[train] {run_label}")
    print(f"  examples: {len(dataset)}")
    print(f"  epochs: {NUM_EPOCHS}")
    print(f"  micro_batch: {MICRO_BATCH_SIZE}")
    print(f"  grad_accum: {grad_accum_steps}")
    print(f"  effective_batch: {EFFECTIVE_BATCH_SIZE}")
    print(f"  total_steps: {total_steps}")
    print(f"  warmup_steps: {warmup_steps}")
    print(f"  lr: {LEARNING_RATE}")
    print(f"  max_seq_length: {MAX_SEQ_LENGTH}")

    model.train()
    log = {"losses": [], "epoch_losses": []}
    global_step = 0
    start = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            step_loss = outputs.loss.item()
            epoch_loss += step_loss
            epoch_steps += 1
            global_step += 1

            if global_step % grad_accum_steps == 0 or step == len(dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            log["losses"].append({
                "global_step": global_step,
                "epoch": epoch,
                "loss": step_loss,
                "lr": scheduler.get_last_lr()[0],
            })

            if global_step % 100 == 0 or global_step == total_steps:
                elapsed = time.time() - start
                print(f"  [{run_label}] step {global_step}/{total_steps}  "
                      f"loss={step_loss:.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"elapsed={elapsed:.0f}s")

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        log["epoch_losses"].append({
            "epoch": epoch,
            "avg_loss": avg_epoch_loss,
            "n_steps": epoch_steps,
        })
        print(f"  [{run_label}] epoch {epoch} complete  "
              f"avg_loss={avg_epoch_loss:.4f}")

    # Check loss decreased across epochs
    if len(log["epoch_losses"]) >= 2:
        first_loss = log["epoch_losses"][0]["avg_loss"]
        last_loss = log["epoch_losses"][-1]["avg_loss"]
        if last_loss >= first_loss:
            print(f"  [{run_label}] WARNING: loss did not decrease across "
                  f"epochs ({first_loss:.4f} -> {last_loss:.4f}). "
                  f"Pre-reg §7 stopping rule may apply.")

    elapsed = time.time() - start
    print(f"  [{run_label}] training complete in {elapsed:.0f}s")
    log["total_time_s"] = elapsed

    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  [{run_label}] adapter saved -> {output_dir}")

    # Save training log
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    return log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Train on 4 examples for 3 epochs.")
    parser.add_argument("--mode", choices=["real", "shuffled", "both"],
                        default="both",
                        help="Which training run(s) to execute.")
    args = parser.parse_args()

    # Env check
    print(f"[env] HSA_OVERRIDE_GFX_VERSION="
          f"{os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'UNSET')}")
    assert torch.cuda.is_available(), "ROCm not detected"
    print(f"[env] device: {torch.cuda.get_device_name(0)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load training data from Step 2
    real_path = DATA_DIR / "step2_training_set.json"
    shuffled_path = DATA_DIR / "step2_shuffled_training_set.json"

    if not real_path.exists():
        print(f"ERROR: {real_path} not found. Run Step 2 first.")
        sys.exit(1)

    if args.mode in ("real", "both"):
        # ----- Step 3: Real-target fine-tuning -----
        print("\n=== Step 3: Real-target LoRA fine-tuning ===")

        with open(real_path) as f:
            real_data = json.load(f)
        print(f"[data] Training examples (real): {len(real_data)}")

        # Load fresh model for each run
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to("cuda")
        print(f"[load] VRAM after model: "
              f"{torch.cuda.memory_allocated() / 1e9:.1f} GB")

        # Apply LoRA
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(
            model, lora_config, autocast_adapter_dtype=False
        )
        model.print_trainable_parameters()
        print(f"[load] VRAM after LoRA: "
              f"{torch.cuda.memory_allocated() / 1e9:.1f} GB")

        real_log = train_lora(
            model, tokenizer, real_data,
            output_dir=RESULTS_DIR / "lora_real",
            run_label="real-target",
            seed=SEED,
            dry_run=args.dry_run,
        )

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if args.mode in ("shuffled", "both"):
        # ----- Step 3b: Shuffled-target fine-tuning -----
        print("\n=== Step 3b: Shuffled-target LoRA fine-tuning ===")

        if not shuffled_path.exists():
            print(f"ERROR: {shuffled_path} not found. Run Step 2 first.")
            sys.exit(1)

        with open(shuffled_path) as f:
            shuffled_data = json.load(f)
        print(f"[data] Training examples (shuffled): {len(shuffled_data)}")

        # Load fresh model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to("cuda")

        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(
            model, lora_config, autocast_adapter_dtype=False
        )
        model.print_trainable_parameters()

        shuffled_log = train_lora(
            model, tokenizer, shuffled_data,
            output_dir=RESULTS_DIR / "lora_shuffled",
            run_label="shuffled-target",
            seed=SEED,
            dry_run=args.dry_run,
        )

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ----- Combined summary -----
    print("\n=== Step 3 complete ===")
    if args.mode in ("real", "both"):
        el = real_log["epoch_losses"]
        print(f"  Real-target:    loss {el[0]['avg_loss']:.4f} -> "
              f"{el[-1]['avg_loss']:.4f}  ({real_log['total_time_s']:.0f}s)")
    if args.mode in ("shuffled", "both"):
        el = shuffled_log["epoch_losses"]
        print(f"  Shuffled-target: loss {el[0]['avg_loss']:.4f} -> "
              f"{el[-1]['avg_loss']:.4f}  ({shuffled_log['total_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
